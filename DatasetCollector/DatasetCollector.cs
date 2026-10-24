using Agents;
using Agents.MCTS;
using Common;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Common.Actions;
using Common.Serialization;
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

            int localIdx;
            while (!cancellation.IsCancellationRequested && (localIdx = Interlocked.Increment(ref _nextMatchIdx)) < _settings.Matches)
            {
                try
                {
                    // Match indices (and with them the match seeds and sample names) continue those of earlier parts of the run
                    CollectMatch(_settings.FirstMatch + localIdx, vectors, records, cancellation);
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

            // Sibling groups go to sampled states (in random order) where the acting player has at least two distinct moves
            Dictionary<int, List<Sibling>> groups = [];
            foreach (int sampleIdx in Enumerable.Range(0, statesToSample.Count).OrderBy(_ => Utils.Random.Next()).ToList())
            {
                if (groups.Count == _settings.SiblingGroupsPerMatch) break;

                (GameState state, int position) = statesToSample[sampleIdx];
                Action? matchMove = position < playedActions.Count ? playedActions[position] : null;
                ulong groupSeed = Seeds.Derive(SampleSeed(matchSeed, position), GroupSeedIndex);
                List<Sibling> siblings = SiblingStates(state, matchMove, _settings.SiblingsPerGroup, groupSeed);
                if (siblings.Count > 0) groups[sampleIdx] = siblings;
            }

            // The actions of the match are serialized once for the save files of all its samples
            MatchSaveFiles saveFiles = new MatchSaveFiles(playedActions);

            for (int sampleIdx = 0; sampleIdx < statesToSample.Count; sampleIdx++)
            {
                (GameState sampleState, int position) = statesToSample[sampleIdx];
                ulong sampleSeed = SampleSeed(matchSeed, position);
                string name = $"r{matchIdx}_s{sampleIdx}";
                groups.TryGetValue(sampleIdx, out List<Sibling>? siblings);
                int actor = siblings == null ? -1 : ActingPlayer(sampleState);

                SampleInfo sample = NewSample(name, match, sampleIdx, position, sampleState,
                    siblings == null ? null : new SiblingInfo(name, "root", actor, "", false));
                SampleOutcome rootOutcome = CollectSample(sampleState, sample, sampleSeed, saveFiles, null, null, vectors, records, cancellation);
                if (rootOutcome == SampleOutcome.Cancelled)
                    return;

                // Sibling files take the history from the root's file, if it was written (it isn't without a finished playout)
                string? historyReference = rootOutcome == SampleOutcome.Written ? name + ".yaml" : null;

                // The siblings use the same playout seeds as their sampled state (common random numbers)
                for (int siblingIdx = 0; siblings != null && siblingIdx < siblings.Count; siblingIdx++)
                {
                    Sibling sibling = siblings[siblingIdx];
                    SampleInfo siblingSample = NewSample($"{name}_c{siblingIdx}", match, sampleIdx, position + 1, sibling.State,
                        new SiblingInfo(name, sibling.Source, actor, sibling.Move.GetType().Name, sibling.Chance));
                    if (CollectSample(sibling.State, siblingSample, sampleSeed, saveFiles, sibling.Move, historyReference, vectors, records, cancellation) == SampleOutcome.Cancelled)
                        return;
                }
            }
        }

        // Seeds of a sampled state: its playouts use Derive(sampleSeed, playoutIdx), its sibling group the index below
        private const ulong GroupSeedIndex = 1UL << 40;

        private static ulong SampleSeed(ulong matchSeed, int position)
        {
            return Seeds.Derive(matchSeed, (ulong)position);
        }

        private static SampleInfo NewSample(string name, MatchInfo match, int sampleIdx, int position, GameState state, SiblingInfo? sibling)
        {
            int[]? endVictoryPoints = state.HasEnded ? state.Players.Select(player => player.VictoryPoints.Total).ToArray() : null;
            return new SampleInfo(name, match, sampleIdx, position, state.Turn.RoundCounter, state.HasEnded,
                state.HasEnded ? state.Turn.PlayerIndex : -1, endVictoryPoints, sibling);
        }

        /// <summary>
        /// A state one move away from a sampled state: the move as applied (with its history) and how it was chosen
        /// </summary>
        internal sealed record Sibling(GameState State, Action Move, string Source, bool Chance);

        /// <summary>
        /// The sibling group of a state: the states after up to count distinct moves of the acting player, or none if there
        /// aren't at least two. The move played in the match comes first (its recorded outcome included), then BuilderAgent's
        /// choice, then random moves (action type uniformly, then a move of that type), all from the group seed.
        /// Rolls are left out: they are no choice between states, only chance. Moves with a random outcome (development card,
        /// robber steal) get one outcome, realized from the group seed, and are marked. Moves leading to the same state as an
        /// earlier one are skipped. A move may end the match; that sibling is labeled by its result.
        /// </summary>
        internal static List<Sibling> SiblingStates(GameState root, Action? matchMove, int count, ulong groupSeed)
        {
            if (root.HasEnded || count < 2) return [];

            Utils.Random.SeedCurrentThread(Seeds.ToInt32(groupSeed));
            sbyte actor = ActingPlayer(root);
            List<Action> legalMoves = MCTSTree.GetValidActions(root).Where(move => move is not RollAction && move.PlayerIndex == actor).ToList();
            List<Sibling> siblings = [];

            // Asked of the unresolved legal move, since the match's move already carries its outcome
            bool HasRandomOutcome(Action move) => legalMoves.FirstOrDefault(legal => legal.IsSameDecision(move)) is IOutputRandomnessAction random
                && random.GetOutcomeVariants(root, actor).Count > 1;

            void TryAdd(Action candidate, string source)
            {
                if (siblings.Count >= count || candidate is RollAction || candidate.PlayerIndex != actor || !candidate.IsValidFor(root)) return;
                if (siblings.Any(sibling => sibling.Move.IsSameDecision(candidate))) return;

                bool chance = HasRandomOutcome(candidate);
                GameState state = new GameState(root);
                Action move = candidate.Clone();
                move.Apply(state);
                if (siblings.Any(sibling => sibling.State.Equals(state))) return;

                siblings.Add(new Sibling(state, move, source, chance));
            }

            if (matchMove != null) TryAdd(matchMove, "match");
            TryAdd(new BuilderAgent(actor).Act(root), "builder");

            List<List<Action>> movesByType = legalMoves.GroupBy(move => move.GetType()).Select(group => group.ToList()).ToList();
            while (siblings.Count < count && movesByType.Count > 0)
            {
                int typeIdx = Utils.Random.Next(movesByType.Count);
                List<Action> moves = movesByType[typeIdx];
                int moveIdx = Utils.Random.Next(moves.Count);
                Action move = moves[moveIdx];
                moves[moveIdx] = moves[^1];
                moves.RemoveAt(moves.Count - 1);
                if (moves.Count == 0) movesByType.RemoveAt(typeIdx);

                TryAdd(move, "random");
            }

            return siblings.Count >= 2 ? siblings : [];
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
        /// Labels the sample with playouts from the sample seed (an ended match needs none) and writes its files. A sibling's
        /// save file holds only its move and references the root's file for the history before it (SaveFileSerializer.Load
        /// puts them together), or, without a written root file, the whole history.
        /// </summary>
        private SampleOutcome CollectSample(GameState sampleState, in SampleInfo sample, ulong sampleSeed, MatchSaveFiles saveFiles, Action? siblingMove,
            string? historyReference, VectorChunkWriter vectors, SampleRecordWriter records, CancellationToken cancellation)
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

                for (int runIdx = 0; runIdx < _settings.PlayoutsPerSample; runIdx++)
                {
                    if (cancellation.IsCancellationRequested)
                    {
                        records.DiscardPlayouts();
                        return SampleOutcome.Cancelled;
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
                    return SampleOutcome.Skipped;
                }
                winRates = normalized;
            }

            // Write GameState to file, with the actions that led to it
            string saveFile = siblingMove != null && historyReference != null
                ? SaveFileSerializer.Serialize(new SaveFile(sampleState, [siblingMove], []) { HistoryReference = historyReference })
                : saveFiles.Build(sampleState, siblingMove == null ? sample.Position : sample.Position - 1, siblingMove);
            File.WriteAllText(Path.Combine(_stateDirectory, sample.Name + ".yaml"), saveFile);

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
            return SampleOutcome.Written;
        }

        private enum SampleOutcome { Written, Skipped, Cancelled }

        /// <summary>
        /// Plays the state to the end (or the round limit) with agents drawn from the population, all from the playout's seed
        /// </summary>
        private GameState Playout(GameState sampleState, ulong playoutSeed, Span<AgentKind> kinds)
        {
            Utils.Random.SeedCurrentThread(Seeds.ToInt32(playoutSeed));
            Agent[] agents = _settings.Agents.Draw(kinds);

            GameState simState = new GameState(sampleState);
            foreach (Agent agent in agents) agent.ResetToState(simState);

            // Every seat draws from its own stream (dice have one anyway), so playouts of siblings with the same seed stay
            // alike for longer after their states differ: the difference between siblings has 2.4x less variance than with
            // independent playouts, 2.2x with one shared stream
            Random[] seatRandoms = Enumerable.Range(0, PLAYER_COUNT).Select(seat => new Random(Seeds.ToInt32(Seeds.Derive(playoutSeed, (ulong)seat)))).ToArray();

            while (!simState.HasEnded && simState.Turn.RoundCounter < UNWINNABLE_ROUND_THRESHOLD)
            {
                sbyte acting = ActingPlayer(simState);
                Utils.Random.UseOnCurrentThread(seatRandoms[acting]);
                Action action = agents[acting].Act(simState);
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
                    first_match = _settings.FirstMatch,
                    matches = _settings.Matches,
                    samples_per_match = _settings.SamplesPerMatch,
                    playouts_per_sample = _settings.PlayoutsPerSample,
                    sibling_groups_per_match = _settings.SiblingGroupsPerMatch,
                    siblings_per_group = _settings.SiblingsPerGroup,
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
