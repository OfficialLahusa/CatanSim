using Agents;
using Common;
using Common.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using Action = Common.Actions.Action;

namespace DatasetCollector
{
    public class DatasetCollector
    {
        private const uint UNWINNABLE_ROUND_THRESHOLD = 50_000;

        private readonly uint _threadCount;
        private readonly uint _matchCount;
        private readonly uint _samplesPerMatch;
        private readonly uint _playoutsPerSample;

        // Map generator options
        private bool _centerDesert = false;

        // Agents (stateless, so the workers share them)
        private Agent[] _agents;
        public const int PLAYER_COUNT = 4;

        // File output
        string _datasetName;
        string _stateDirectory;
        string _evaluationDirectory;
        string _vectorDirectory;

        // Progress and problems of all workers
        private int _nextMatchIdx = -1;
        private int _finishedMatches;
        private int _writtenSamples;
        private int _skippedSamples;
        private int _errorCount;
        private Exception? _firstError;

        public DatasetCollector(uint threadCount, uint matchCount, uint samplesPerMatch, uint playoutsPerSample)
        {
            _threadCount = threadCount;
            _matchCount = matchCount;
            _samplesPerMatch = samplesPerMatch;
            _playoutsPerSample = playoutsPerSample;

            _agents = new Agent[PLAYER_COUNT];

            for (sbyte i = 0; i < PLAYER_COUNT; i++)
            {
                _agents[i] = new RandomAgent(i);
            }

            _datasetName = $"m{_matchCount}_s{_samplesPerMatch}_p{_playoutsPerSample}_{DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss")}";
            _stateDirectory = Path.Combine("datasets", _datasetName, "input");
            _evaluationDirectory = Path.Combine("datasets", _datasetName, "output");
            _vectorDirectory = Path.Combine("datasets", _datasetName, "vectors");
        }

        public void Collect()
        {
            // Create output directories
            Directory.CreateDirectory(_stateDirectory);
            Directory.CreateDirectory(_evaluationDirectory);
            Directory.CreateDirectory(_vectorDirectory);

            Console.WriteLine("\nDataset directory:\n" + Path.GetFullPath(Path.Combine("datasets", _datasetName)));
            Console.WriteLine("\nCollecting Dataset:");

            // Each worker takes whole matches: it plays the match, then all playouts of its samples on its own thread and
            // writes their files, so the workers never wait for each other
            Stopwatch stopwatch = Stopwatch.StartNew();
            Task[] workers = Enumerable.Range(0, (int)Math.Max(1, _threadCount))
                .Select(workerIdx => Task.Factory.StartNew(() => RunWorker(workerIdx), TaskCreationOptions.LongRunning))
                .ToArray();

            while (!Task.WaitAll(workers, TimeSpan.FromSeconds(1)))
                ReportProgress(stopwatch);
            ReportProgress(stopwatch);

            if (_skippedSamples > 0)
                Console.WriteLine($"\n\nWarning: {_skippedSamples} sample(s) skipped because none of their playouts finished.");

            if (_errorCount > 0)
            {
                Console.WriteLine($"\n\nWarning: {_errorCount} error(s) during collection, the first one:");
                Console.WriteLine(_firstError);
            }

            Console.WriteLine("\nDone!");
        }

        private void RunWorker(int workerIdx)
        {
            using VectorChunkWriter vectors = new(_vectorDirectory, workerIdx);

            int matchIdx;
            while ((matchIdx = Interlocked.Increment(ref _nextMatchIdx)) < _matchCount)
            {
                try
                {
                    CollectMatch(matchIdx, vectors);
                }
                catch (Exception e)
                {
                    RecordError(e);
                }

                Interlocked.Increment(ref _finishedMatches);
            }
        }

        private void ReportProgress(Stopwatch stopwatch)
        {
            double seconds = stopwatch.Elapsed.TotalSeconds;
            Console.Write($"\x000DMatches {_finishedMatches}/{_matchCount}, samples {_writtenSamples} ({_writtenSamples / Math.Max(seconds, 1e-9):F1}/s), {seconds:F0}s          ");
        }

        private void CollectMatch(int matchIdx, VectorChunkWriter vectors)
        {
            (GameState baseState, List<Action> playedActions) = PlayMatch(matchIdx);

            // Action Indices sammeln bis Sample Count erreicht ist
            // Immer garantiert den ersten State (vor Placement) und den letzten State (Termination) in den Datensatz schreiben
            // Für die übrigen States dazwischen einen random Index Shuffle samplen
            HashSet<int> sampleIndices = [0, playedActions.Count];
            int remainingSampleCount = Math.Min((int)_samplesPerMatch - 2, playedActions.Count - 1);

            List<int> possibleSampleIndices = Enumerable.Range(1, playedActions.Count - 1).ToList();
            possibleSampleIndices.Shuffle();

            sampleIndices.UnionWith(possibleSampleIndices.Take(remainingSampleCount));

            // States bei den gegebenen Indices kopieren, jeweils mit den bis dahin gespielten Actions
            GameState playState = new GameState(baseState);
            List<(GameState state, List<Action> previousActions)> statesToSample = [(new GameState(playState), [])];

            for (int actionIdx = 0; actionIdx < playedActions.Count; actionIdx++)
            {
                Action playedAction = playedActions[actionIdx];
                if (!playedAction.IsValidFor(playState)) throw new InvalidOperationException();
                playedAction.Apply(playState);

                if (sampleIndices.Contains(actionIdx + 1))
                {
                    statesToSample.Add((new GameState(playState), playedActions.GetRange(0, actionIdx + 1)));
                }
            }

            // Dann für diese States jeweils die gegebene Anzahl Playouts spielen und Input sowie normalisierten Output als Dateien speichern
            for (int sampleIdx = 0; sampleIdx < statesToSample.Count; sampleIdx++)
            {
                (GameState sampleState, List<Action> previousActions) = statesToSample[sampleIdx];
                CollectSample(sampleState, previousActions, matchIdx, sampleIdx, vectors);
            }
        }

        /// <summary>
        /// Plays a match with the agents until it ends, starting over on a new map if it takes too many rounds
        /// </summary>
        private (GameState baseState, List<Action> playedActions) PlayMatch(int matchIdx)
        {
            while (true)
            {
                GameState baseState = new GameState(MapGenerator.GenerateRandomClassic(_centerDesert), PLAYER_COUNT);
                GameState playState = new GameState(baseState);
                List<Action> playedActions = [];

                while (!playState.HasEnded && playState.Turn.RoundCounter < UNWINNABLE_ROUND_THRESHOLD)
                {
                    Action playedAction = _agents[ActingPlayer(playState)].Act(playState);
                    if (!playedAction.IsValidFor(playState)) throw new InvalidOperationException();

                    playedAction.Apply(playState);
                    playedActions.Add(playedAction);
                }

                if (playState.HasEnded)
                    return (baseState, playedActions);

                Console.WriteLine($"\nTerminating unwinnable match (idx {matchIdx}) after {playState.Turn.RoundCounter} rounds. Retrying.\n");
            }
        }

        private void CollectSample(GameState sampleState, List<Action> previousActions, int matchIdx, int sampleIdx, VectorChunkWriter vectors)
        {
            int[] timesWon = new int[PLAYER_COUNT];

            for (int runIdx = 0; runIdx < _playoutsPerSample; runIdx++)
            {
                try
                {
                    GameState simState = new GameState(sampleState);

                    // Play until game has ended or unwinnable round threshold is reached
                    while (!simState.HasEnded && simState.Turn.RoundCounter < UNWINNABLE_ROUND_THRESHOLD)
                    {
                        _agents[ActingPlayer(simState)].Act(simState).Apply(simState);
                    }

                    // If the playout was aborted due to timeout, don't collect a result
                    if (!simState.HasEnded) continue;

                    // Since players can only win on their own turn, the winner is the current turn player
                    timesWon[simState.Turn.PlayerIndex]++;
                }
                catch (Exception e)
                {
                    RecordError(e);
                }
            }

            // Without a single finished playout there is no label; skip the sample instead of writing NaNs
            double[]? normalizedEvaluation = NormalizeWinCounts(timesWon);
            if (normalizedEvaluation == null)
            {
                Interlocked.Increment(ref _skippedSamples);
                return;
            }

            // Write GameState to file
            string sampleName = $"r{matchIdx}_s{sampleIdx}";
            SaveFile saveFile = new SaveFile(sampleState, previousActions, []);
            string saveData = SaveFileSerializer.Serialize(saveFile);
            File.WriteAllText(Path.Combine(_stateDirectory, sampleName + ".yaml"), saveData);

            // Write evaluation to file
            StringBuilder evalStringBuilder = new StringBuilder();
            foreach (double evalEntry in normalizedEvaluation)
            {
                evalStringBuilder.Append(evalEntry.ToString(CultureInfo.InvariantCulture) + "\n");
            }
            File.WriteAllText(Path.Combine(_evaluationDirectory, sampleName + ".txt"), evalStringBuilder.ToString());

            // Write the state vector, so the dataset doesn't need to be vectorized from the YAML files afterwards
            vectors.Write(sampleName, sampleState);

            Interlocked.Increment(ref _writtenSamples);
        }

        private static sbyte ActingPlayer(GameState state)
        {
            // Find first player that is allowed to act on state
            for (sbyte playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
            {
                if (state.CanPlayerAct(playerIdx))
                    return playerIdx;
            }

            throw new InvalidOperationException();
        }

        private void RecordError(Exception e)
        {
            if (Interlocked.Increment(ref _errorCount) == 1)
                _firstError = e;
        }

        /// <summary>
        /// Win rates from the playout win counts per player, or null if no playout finished
        /// </summary>
        internal static double[]? NormalizeWinCounts(IReadOnlyList<int> timesWon)
        {
            int evalSum = timesWon.Sum();
            if (evalSum == 0) return null;

            return timesWon.Select(wins => (double)wins / evalSum).ToArray();
        }
    }
}
