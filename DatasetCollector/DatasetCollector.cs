using Agents;
using Common;
using Common.Actions;
using Common.Agents;
using Common.Serialization;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
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

        private GameState _baseState;
        private Stack<Action> _playedActions;

        // Map generator options
        private bool _centerDesert = false;

        // Agents
        private Agent[] _agents;
        public const int PLAYER_COUNT = 4;

        // File output
        string _datasetName;
        string _stateDirectory;
        string _evaluationDirectory;

        public DatasetCollector(uint threadCount, uint matchCount, uint samplesPerMatch, uint playoutsPerSample)
        {
            _threadCount = threadCount;
            _matchCount = matchCount;
            _samplesPerMatch = samplesPerMatch;
            _playoutsPerSample = playoutsPerSample;

            _baseState = new GameState(MapGenerator.GenerateRandomClassic(_centerDesert), PLAYER_COUNT);
            _playedActions = [];

            _agents = new Agent[PLAYER_COUNT];

            for (sbyte i = 0; i < PLAYER_COUNT; i++)
            {
                _agents[i] = new RandomAgent(i);
            }

            _datasetName = $"m{_matchCount}_s{_samplesPerMatch}_p{_playoutsPerSample}_{DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss")}";
            _stateDirectory = Path.Combine("datasets", _datasetName, "input");
            _evaluationDirectory = Path.Combine("datasets", _datasetName, "output");
        }

        public void Collect()
        {
            // Create output directories
            
            Directory.CreateDirectory(_stateDirectory);
            Directory.CreateDirectory(_evaluationDirectory);

            Console.WriteLine("\nDataset directory:\n" + Path.GetFullPath(Path.Combine("datasets", _datasetName)));
            Console.WriteLine("\nCollecting Dataset:");

            for (int matchIdx = 0; matchIdx < _matchCount; matchIdx++)
            {
                CollectMatch(matchIdx);
            }

            Console.WriteLine("\nDone!");
        }

        private void CollectMatch(int matchIdx)
        {
            // GameState neu generieren
            _baseState = new GameState(MapGenerator.GenerateRandomClassic(), PLAYER_COUNT);
            _playedActions = [];

            // Kopie erstellen und bis zum Ende durchspielen, dabei Actions sammeln
            GameState playState = new GameState(_baseState);
            while (!playState.HasEnded)
            {
                // Terminate unwinnable rounds and retry
                if (playState.Turn.RoundCounter >= UNWINNABLE_ROUND_THRESHOLD)
                {
                    Console.WriteLine($"\nTerminating unwinnable match (idx {matchIdx}) after {playState.Turn.RoundCounter} rounds. Retrying.\n");
                    CollectMatch(matchIdx);
                    return;
                }

                // Find first player that is allowed to act on state
                int? actingPlayerIdx = null;
                for (int playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                {
                    if (playState.CanPlayerAct(playerIdx))
                    {
                        actingPlayerIdx = playerIdx;
                        break;
                    }
                }

                if (!actingPlayerIdx.HasValue) throw new InvalidOperationException();

                // Query agent for current state
                Action playedAction = _agents[actingPlayerIdx.Value].Act(playState, (uint)_playedActions.Count);

                if (!playedAction.IsValidFor(playState)) throw new InvalidOperationException();

                // Apply action to state
                playedAction.Apply(playState);

                // Push to action stack
                _playedActions.Push(playedAction);
            }

            // Action Indices sammeln bis Sample Count erreicht ist
            // Immer garantiert den ersten State (vor Placement) und den letzten State (Termination) in den Datensatz schreiben
            // Für die übrigen States dazwischen einen random Index Shuffle samplen
            HashSet<int> sampleIndices = [0, _playedActions.Count];
            int remainingSampleCount = Math.Min((int)_samplesPerMatch - 2, _playedActions.Count - 1);

            List<int> possibleSampleIndices = Enumerable.Range(1, _playedActions.Count - 1).ToList();
            possibleSampleIndices.Shuffle();

            sampleIndices.UnionWith(possibleSampleIndices.Take(remainingSampleCount));

            // States bei den gegebenen Indices kopieren und sammeln
            playState = new GameState(_baseState);
            Stack<Action> replayActions = new(_playedActions);
            List<(GameState, Stack<Action>)> statesToSample = [(new GameState(playState), new (_playedActions.Reverse()))];
            int stateIdx = 0;

            while (!playState.HasEnded)
            {
                // Query agent for current state
                Action playedAction = replayActions.Pop();

                if (!playedAction.IsValidFor(playState)) throw new InvalidOperationException();

                // Apply action to state
                playedAction.Apply(playState);
                _playedActions.Push(playedAction);

                stateIdx++;

                if (sampleIndices.Contains(stateIdx))
                {
                    statesToSample.Add((new GameState(playState), new(_playedActions.Reverse())));
                }
            }

            // Dann für diese States jeweils die gegebene Anzahl Playouts parallel spielen und Input sowie normalisierten Output als Dateien speichern
            int sampleRunIdx = 0;
            foreach ((GameState sampleState, Stack<Action> samplePlayedActions) in statesToSample)
            {
                Console.Write($"\x000DMatch {matchIdx + 1}/{_matchCount}, Sample {sampleRunIdx+1}/{_samplesPerMatch}                                 ");
                CollectSample(sampleState, samplePlayedActions, matchIdx, sampleRunIdx).GetAwaiter().GetResult();
                sampleRunIdx++;
            }
        }

        private async Task CollectSample(GameState sampleState, Stack<Action> previousActions, int roundIdx, int sampleIdx)
        {
            DateTime startTime = DateTime.Now;

            // Precompute Adjacency Cache
            sampleState.Board.Adjacency.PrecomputeAll();

            ConcurrentQueue<int> runs = new ConcurrentQueue<int>(Enumerable.Range(0, (int)_playoutsPerSample));
            CountdownEvent countdown = new CountdownEvent((int)_playoutsPerSample);

            object lockObj = new object();
            List<int> timesWon = new List<int>();
            for (int i = 0; i < PLAYER_COUNT; i++)
                timesWon.Add(0);

            ConcurrentBag<Exception> workerExceptions = new ConcurrentBag<Exception>();

            System.Action doPlayout = () =>
            {
                while (runs.TryDequeue(out int runIdx))
                {
                    try
                    {
                        GameState simState = new GameState(sampleState);
                        uint actionsPlayedSinceFork = 0;

                        // Play until game has ended or unwinnable round threshold is reached
                        while (!simState.HasEnded && simState.Turn.RoundCounter < UNWINNABLE_ROUND_THRESHOLD)
                        {
                            // Find first player that is allowed to act on state
                            sbyte? actingPlayerIdx = null;
                            for (sbyte playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                            {
                                if (simState.CanPlayerAct(playerIdx))
                                {
                                    actingPlayerIdx = playerIdx;
                                    break;
                                }
                            }

                            if (!actingPlayerIdx.HasValue) throw new InvalidOperationException();

                            // Pick action
                            Action playedAction = _agents[actingPlayerIdx.Value].Act(simState, (uint)previousActions.Count + actionsPlayedSinceFork);

                            // Apply action to state
                            playedAction.Apply(simState);
                            actionsPlayedSinceFork++;
                        }

                        // If the playout was aborted due to timeout, don't collect a result
                        if (!simState.HasEnded) break;

                        // Since players can only win on their own turn, the winner is the current turn player
                        int winnerIdx = simState.Turn.PlayerIndex;

                        // Collect match result thread-safely
                        lock (lockObj)
                        {
                            timesWon[winnerIdx]++;
                        }
                    }
                    catch (Exception e)
                    {
                        workerExceptions.Add(e);
                    }
                    finally
                    {
                        countdown.Signal();
                    }
                }
            };


            List<Task> tasks = new List<Task>();
            for (int t = 0; t < _threadCount; t++)
            {
                tasks.Add(Task.Run(doPlayout));
            }

            countdown.Wait();

            // Not strictly necessary, but ensures all tasks are complete
            await Task.WhenAll(tasks);

            countdown.Dispose();

            if (!workerExceptions.IsEmpty)
            {
                Console.WriteLine($"\n\nWarning: {workerExceptions.Count} playout(s) threw exceptions:");
                foreach (var e in workerExceptions)
                {
                    Console.WriteLine(e.ToString());
                    break;
                }
            }

            // Write GameState to file
            string sampleName = $"r{roundIdx}_s{sampleIdx}";
            SaveFile saveFile = new SaveFile(sampleState, previousActions.Reverse().ToList(), []);
            string saveData = SaveFileSerializer.Serialize(saveFile);
            File.WriteAllText(Path.Combine(_stateDirectory, sampleName + ".yaml"), saveData);

            // Write evaluation to file
            double[] normalizedEvaluation = new double[PLAYER_COUNT];
            int evalSum = timesWon.Sum();
            for (int i = 0; i < normalizedEvaluation.Length; i++)
            {
                normalizedEvaluation[i] = (double)timesWon[i] / evalSum;
            }

            StringBuilder evalStringBuilder = new StringBuilder();
            foreach (double evalEntry in  normalizedEvaluation)
            {
                evalStringBuilder.Append(evalEntry.ToString(CultureInfo.InvariantCulture) + "\n");
            }
            File.WriteAllText(Path.Combine(_evaluationDirectory, sampleName + ".txt"), evalStringBuilder.ToString());
        }
    }
}
