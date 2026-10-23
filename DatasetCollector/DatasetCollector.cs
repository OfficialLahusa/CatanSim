using Agents;
using Common;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Action = Common.Actions.Action;

namespace DatasetCollector
{
    /// <summary>
    /// Plays matches with a mixed agent population, samples states of each match uniformly over all positions and labels them
    /// with playouts. Per sample it writes the save file (input/*.yaml, with the actions that led to the state), the win rates
    /// (output/*.txt), the state vector (vectors/) and the raw playout results (samples/, playouts/), plus metadata.json for
    /// the run. Every match is played from its own seed, derived from the run seed, so it can be reproduced.
    /// </summary>
    public class DatasetCollector
    {
        private const int UNWINNABLE_ROUND_THRESHOLD = 50_000;
        public const int PLAYER_COUNT = 4;

        private readonly CollectorSettings _settings;
        private readonly TextWriter _log;

        // File output
        private readonly string _datasetDirectory;
        private readonly string _stateDirectory;
        private readonly string _evaluationDirectory;
        private readonly string _vectorDirectory;

        // Progress and problems of all workers
        private int _nextMatchIdx = -1;
        private int _finishedMatches;
        private int _writtenSamples;
        private int _skippedSamples;
        private long _playouts;
        private int _errorCount;
        private Exception? _firstError;

        public DatasetCollector(CollectorSettings settings, TextWriter? log = null)
        {
            settings.Validate();
            _settings = settings;
            _log = log ?? Console.Out;

            _datasetDirectory = Path.Combine(settings.OutputDirectory, settings.DatasetName);
            _stateDirectory = Path.Combine(_datasetDirectory, "input");
            _evaluationDirectory = Path.Combine(_datasetDirectory, "output");
            _vectorDirectory = Path.Combine(_datasetDirectory, "vectors");
        }

        public string DatasetDirectory => _datasetDirectory;

        /// <summary>
        /// Collects the dataset; a cancelled run stops soon and abandons the matches in progress (their remaining samples
        /// are not written)
        /// </summary>
        public CollectionResult Collect(CancellationToken cancellation = default)
        {
            Directory.CreateDirectory(_stateDirectory);
            Directory.CreateDirectory(_evaluationDirectory);
            Directory.CreateDirectory(_vectorDirectory);

            DateTime started = DateTime.Now;
            WriteMetadata(started, null, null);

            _log.WriteLine("\nDataset directory:\n" + Path.GetFullPath(_datasetDirectory));
            _log.WriteLine($"Seed {_settings.Seed}, {_settings.Threads} threads, {_settings.Matches} matches, {_settings.SamplesPerMatch} samples per match, {_settings.PlayoutsPerSample} playouts per sample");
            _log.WriteLine($"Agents: {_settings.Agents}");
            _log.WriteLine("\nCollecting Dataset:");

            // Each worker takes whole matches: it plays the match, then all playouts of its samples on its own thread and
            // writes their files, so the workers never wait for each other
            Stopwatch stopwatch = Stopwatch.StartNew();
            Task[] workers = Enumerable.Range(0, _settings.Threads)
                .Select(workerIdx => Task.Factory.StartNew(() => RunWorker(workerIdx, cancellation), TaskCreationOptions.LongRunning))
                .ToArray();

            while (!Task.WaitAll(workers, TimeSpan.FromSeconds(1)))
                ReportProgress(stopwatch);
            ReportProgress(stopwatch);
            stopwatch.Stop();

            CollectionResult result = new(_writtenSamples, _skippedSamples, _playouts, _errorCount, stopwatch.Elapsed, cancellation.IsCancellationRequested);
            WriteMetadata(started, DateTime.Now, result);

            if (_skippedSamples > 0)
                _log.WriteLine($"\n\nWarning: {_skippedSamples} sample(s) skipped because none of their playouts finished.");

            if (_errorCount > 0)
            {
                _log.WriteLine($"\n\nWarning: {_errorCount} error(s) during collection, the first one:");
                _log.WriteLine(_firstError);
            }

            _log.WriteLine(result.Cancelled ? "\nCancelled." : "\nDone!");
            return result;
        }

        private void RunWorker(int workerIdx, CancellationToken cancellation)
        {
            using VectorChunkWriter vectors = new(_vectorDirectory, workerIdx);
            using SampleRecordWriter records = new(_datasetDirectory, workerIdx);

            int matchIdx;
            while (!cancellation.IsCancellationRequested && (matchIdx = Interlocked.Increment(ref _nextMatchIdx)) < _settings.Matches)
            {
                try
                {
                    CollectMatch(matchIdx, vectors, records, cancellation);
                }
                catch (Exception e)
                {
                    records.DiscardPlayouts();
                    RecordError(e);
                }

                Interlocked.Increment(ref _finishedMatches);
            }
        }

        private void ReportProgress(Stopwatch stopwatch)
        {
            double seconds = Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9);
            _log.Write($"\x000DMatches {_finishedMatches}/{_settings.Matches}, samples {_writtenSamples} ({_writtenSamples / seconds:F1}/s), playouts {Interlocked.Read(ref _playouts) / seconds:F0}/s, {seconds:F0}s          ");
        }

        private void CollectMatch(int matchIdx, VectorChunkWriter vectors, SampleRecordWriter records, CancellationToken cancellation)
        {
            ulong matchSeed = Seeds.Derive(_settings.Seed, (ulong)matchIdx);
            (GameState baseState, List<Action> playedActions, MatchInfo match) = PlayMatch(matchSeed, matchIdx, _settings.Agents, _settings.CenterDesert);

            // k positions uniformly over 0..N (empty board to ended match), each with the same chance k/(N+1)
            int[] positions = SamplePositions(playedActions.Count, _settings.SamplesPerMatch);

            // Copy the states at these positions, each with the number of actions played until then
            GameState playState = new GameState(baseState);
            List<(GameState state, int position)> statesToSample = [];
            int nextPosition = 0;
            for (int position = 0; nextPosition < positions.Length; position++)
            {
                if (position > 0)
                {
                    Action playedAction = playedActions[position - 1];
                    if (!playedAction.IsValidFor(playState)) throw new InvalidOperationException();
                    playedAction.Apply(playState);
                }

                if (positions[nextPosition] == position)
                {
                    statesToSample.Add((new GameState(playState), position));
                    nextPosition++;
                }
            }

            // The actions of the match are serialized once for the save files of all its samples
            MatchSaveFiles saveFiles = new MatchSaveFiles(playedActions);

            for (int sampleIdx = 0; sampleIdx < statesToSample.Count; sampleIdx++)
            {
                (GameState sampleState, int position) = statesToSample[sampleIdx];
                SampleInfo sample = new($"r{matchIdx}_s{sampleIdx}", match, sampleIdx, position, sampleState.Turn.RoundCounter);
                if (!CollectSample(sampleState, sample, saveFiles, vectors, records, cancellation))
                    return;
            }
        }

        /// <summary>
        /// Plays a match from its seed until it ends, starting over on a new map if it takes too many rounds. Everything random
        /// in the match (map, agents, rolls) comes from the seed, so the same seed always gives the same match.
        /// </summary>
        internal static (GameState baseState, List<Action> playedActions, MatchInfo match) PlayMatch(ulong matchSeed, int matchIdx, AgentMix agentMix, bool centerDesert)
        {
            Utils.Random.SeedCurrentThread(Seeds.ToInt32(matchSeed));

            while (true)
            {
                AgentKind[] kinds = new AgentKind[PLAYER_COUNT];
                Agent[] agents = agentMix.Draw(kinds);

                GameState baseState = new GameState(MapGenerator.GenerateRandomClassic(centerDesert), PLAYER_COUNT);
                GameState playState = new GameState(baseState);
                List<Action> playedActions = [];
                foreach (Agent agent in agents) agent.ResetToState(playState);

                while (!playState.HasEnded && playState.Turn.RoundCounter < UNWINNABLE_ROUND_THRESHOLD)
                {
                    Action playedAction = agents[ActingPlayer(playState)].Act(playState);
                    if (!playedAction.IsValidFor(playState)) throw new InvalidOperationException();

                    playedAction.Apply(playState);
                    playedActions.Add(playedAction);
                    foreach (Agent agent in agents) agent.HandleAction(playedAction);
                }

                if (playState.HasEnded)
                {
                    int[] victoryPoints = playState.Players.Select(player => player.VictoryPoints.Total).ToArray();
                    MatchInfo match = new(matchSeed, matchIdx, playedActions.Count, playState.Turn.PlayerIndex, victoryPoints, playState.Turn.RoundCounter, AgentMix.Letters(kinds));
                    return (baseState, playedActions, match);
                }

                Console.WriteLine($"\nTerminating unwinnable match (idx {matchIdx}) after {playState.Turn.RoundCounter} rounds. Retrying.\n");
            }
        }

        /// <summary>
        /// count distinct positions out of 0..lastPosition, uniformly (Floyd's algorithm), sorted; all if there are fewer
        /// </summary>
        internal static int[] SamplePositions(int lastPosition, int count)
        {
            int positionCount = lastPosition + 1;
            if (count >= positionCount) return Enumerable.Range(0, positionCount).ToArray();

            HashSet<int> chosen = new(count);
            for (int j = positionCount - count; j < positionCount; j++)
            {
                int candidate = Utils.Random.Next(j + 1);
                if (!chosen.Add(candidate)) chosen.Add(j);
            }
            return chosen.Order().ToArray();
        }

        /// <summary>
        /// Labels the sample with playouts (an ended match needs none) and writes its files. Returns false if the run was cancelled.
        /// </summary>
        private bool CollectSample(GameState sampleState, in SampleInfo sample, MatchSaveFiles saveFiles, VectorChunkWriter vectors, SampleRecordWriter records, CancellationToken cancellation)
        {
            double[] winRates;
            if (sampleState.HasEnded)
            {
                // The ended match's label goes entirely to its winner
                winRates = new double[PLAYER_COUNT];
                winRates[sampleState.Turn.PlayerIndex] = 1;
            }
            else
            {
                int[] timesWon = new int[PLAYER_COUNT];
                Span<AgentKind> kinds = stackalloc AgentKind[PLAYER_COUNT];
                ulong sampleSeed = Seeds.Derive(sample.Match.Seed, (ulong)sample.Position);

                for (int runIdx = 0; runIdx < _settings.PlayoutsPerSample; runIdx++)
                {
                    if (cancellation.IsCancellationRequested)
                    {
                        records.DiscardPlayouts();
                        return false;
                    }

                    try
                    {
                        GameState simState = Playout(sampleState, Seeds.Derive(sampleSeed, (ulong)runIdx), kinds);
                        records.AddPlayout(simState, kinds);
                        Interlocked.Increment(ref _playouts);

                        // Since players can only win on their own turn, the winner is the current turn player
                        if (simState.HasEnded) timesWon[simState.Turn.PlayerIndex]++;
                    }
                    catch (Exception e)
                    {
                        RecordError(e);
                    }
                }

                // Without a single finished playout there is no label; skip the sample instead of writing NaNs
                double[]? normalized = NormalizeWinCounts(timesWon);
                if (normalized == null)
                {
                    records.DiscardPlayouts();
                    Interlocked.Increment(ref _skippedSamples);
                    return true;
                }
                winRates = normalized;
            }

            // Write GameState to file, with the actions that led to it
            File.WriteAllText(Path.Combine(_stateDirectory, sample.Name + ".yaml"), saveFiles.Build(sampleState, sample.Position));

            // Write evaluation to file
            StringBuilder evalStringBuilder = new StringBuilder();
            foreach (double evalEntry in winRates)
            {
                evalStringBuilder.Append(evalEntry.ToString(CultureInfo.InvariantCulture) + "\n");
            }
            File.WriteAllText(Path.Combine(_evaluationDirectory, sample.Name + ".txt"), evalStringBuilder.ToString());

            // Write the state vector, so the dataset doesn't need to be vectorized from the YAML files afterwards
            vectors.Write(sample.Name, sampleState);

            // And the raw results, last, so a sample listed there has all its files
            records.WriteSample(sample);

            Interlocked.Increment(ref _writtenSamples);
            return true;
        }

        /// <summary>
        /// Plays the state to the end (or the round limit) with agents drawn from the population, all from the playout's seed
        /// </summary>
        private GameState Playout(GameState sampleState, ulong playoutSeed, Span<AgentKind> kinds)
        {
            Utils.Random.SeedCurrentThread(Seeds.ToInt32(playoutSeed));
            Agent[] agents = _settings.Agents.Draw(kinds);

            GameState simState = new GameState(sampleState);
            foreach (Agent agent in agents) agent.ResetToState(simState);

            while (!simState.HasEnded && simState.Turn.RoundCounter < UNWINNABLE_ROUND_THRESHOLD)
            {
                Action action = agents[ActingPlayer(simState)].Act(simState);
                action.Apply(simState);
                foreach (Agent agent in agents) agent.HandleAction(action);
            }

            return simState;
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

        private void WriteMetadata(DateTime started, DateTime? finished, CollectionResult? result)
        {
            var metadata = new
            {
                format = 2,
                seed = _settings.Seed.ToString(CultureInfo.InvariantCulture),
                commit = CommitHash(),
                started = started.ToString("o", CultureInfo.InvariantCulture),
                finished = finished?.ToString("o", CultureInfo.InvariantCulture),
                status = result == null ? "running" : result.Cancelled ? "cancelled" : "finished",
                settings = new
                {
                    threads = _settings.Threads,
                    matches = _settings.Matches,
                    samples_per_match = _settings.SamplesPerMatch,
                    playouts_per_sample = _settings.PlayoutsPerSample,
                    center_desert = _settings.CenterDesert,
                    agents = new
                    {
                        random = _settings.Agents.RandomWeight,
                        simple = _settings.Agents.SimpleWeight,
                        builder = _settings.Agents.BuilderWeight,
                        min_exploration = _settings.Agents.MinExploration,
                        max_exploration = _settings.Agents.MaxExploration,
                    },
                    unwinnable_round_threshold = UNWINNABLE_ROUND_THRESHOLD,
                },
                feature_dim = StateVectorizer.FeatureDim,
                playout_record_size = SampleRecordWriter.PlayoutRecordSize,
                result = result == null ? null : new
                {
                    samples = result.Samples,
                    skipped_samples = result.SkippedSamples,
                    playouts = result.Playouts,
                    errors = result.Errors,
                    seconds = Math.Round(result.Elapsed.TotalSeconds, 1),
                },
            };

            JsonSerializerOptions options = new() { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
            File.WriteAllText(Path.Combine(_datasetDirectory, "metadata.json"), JsonSerializer.Serialize(metadata, options));
        }

        /// <summary>
        /// The commit the collector was built from (the SDK appends it to the informational version), null if unknown
        /// </summary>
        private static string? CommitHash()
        {
            string? version = typeof(DatasetCollector).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            int plus = version?.IndexOf('+') ?? -1;
            return plus >= 0 ? version![(plus + 1)..] : null;
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

    public sealed record CollectionResult(int Samples, int SkippedSamples, long Playouts, int Errors, TimeSpan Elapsed, bool Cancelled)
    {
        public double PlayoutsPerSecond => Playouts / Math.Max(Elapsed.TotalSeconds, 1e-9);
    }
}
