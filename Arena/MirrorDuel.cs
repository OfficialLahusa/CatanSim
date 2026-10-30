using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using Action = Common.Actions.Action;

namespace Arena
{
    internal sealed record MirrorSettings
    {
        public required AgentSpec Challenger { get; init; }
        public required AgentSpec Baseline { get; init; }

        // Iterations per MCTS decision, or 0 for the time budget in Seconds
        public long Iterations { get; init; } = 5000;
        public double Seconds { get; init; } = 0.25;

        // Games played at once (an iteration budget only; a time budget plays one game at a time)
        public int Parallel { get; init; } = 16;
        public int MaxGames { get; init; } = 4000;

        // Map pair n is generated from seed SeedBase + n
        public int SeedBase { get; init; } = 30_000;

        // The SPRT between H0 "the gain is Elo0" and H1 "the gain is Elo1"
        public double Elo0 { get; init; } = 0;
        public double Elo1 { get; init; } = 40;
        public double Alpha { get; init; } = 0.05;
        public double Beta { get; init; } = 0.05;

        // The duel stops here even if the SPRT is still open
        public TimeSpan MaxDuration { get; init; } = TimeSpan.FromHours(2);

        public InferenceProvider Provider { get; init; } = InferenceProvider.Auto;
    }

    /// <summary>
    /// Mirror duel of two MCTS agents, two seats each: every map is played twice with the sides swapped (seats C B C B,
    /// then B C B C), so the map and the seat advantage cancel within a pair. Both games of a pair use the same seed, which
    /// also fixes the dice and the development card deck as long as both games draw the same random numbers. It stops when
    /// the SPRT on the pair scores accepts H0 or H1, after MaxGames, or after MaxDuration; games still running then are
    /// cancelled and not counted. With an iteration budget several games run at once (the search quality per move doesn't
    /// depend on the load), with a time budget one after another.
    /// </summary>
    internal sealed class MirrorDuel(MirrorSettings settings, TextWriter log)
    {
        private readonly object _lock = new();
        private readonly Dictionary<int, string?[]> _pairWinners = [];
        private readonly List<double> _scores = [];
        private int _wins, _losses, _games;

        public PairStatistics Statistics { get; private set; } = PairStatistics.Of([], 0, 40);

        /// <summary>
        /// Plays the duel and writes one line per game, one per completed pair and a last line with the result
        /// </summary>
        public PairStatistics Run(CancellationToken cancellation = default)
        {
            var challengerNet = new StateValueNet(settings.Challenger.ModelPath, provider: settings.Provider);
            var baselineNet = settings.Baseline.ModelPath == settings.Challenger.ModelPath
                ? challengerNet : new StateValueNet(settings.Baseline.ModelPath, provider: settings.Provider);
            var nets = new Dictionary<AgentSpec, StateValueNet>
            {
                [settings.Challenger] = challengerNet,
                [settings.Baseline] = baselineNet,
            };
            bool iterationBudget = settings.Iterations > 0;
            int parallel = iterationBudget ? Math.Max(1, settings.Parallel) : 1;
            Statistics = PairStatistics.Of([], settings.Elo0, settings.Elo1, settings.Alpha, settings.Beta);

            Write($"[mirror duel] {settings.Challenger} against {settings.Baseline}; maps from seed {settings.SeedBase}, "
                + (iterationBudget ? $"{settings.Iterations} iterations per decision, {parallel} games at once" : $"{settings.Seconds.ToString(CultureInfo.InvariantCulture)}s per decision")
                + $"; SPRT [{settings.Elo0:+0;-0;0}, {settings.Elo1:+0;-0;0}], at most {settings.MaxGames} games and {settings.MaxDuration.TotalMinutes:F0} min");

            // The searches print every decision; the duel's lines go to the log
            TextWriter console = Console.Out;
            Console.SetOut(TextWriter.Null);
            using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            stop.CancelAfter(settings.MaxDuration);
            string reason = "all games played";
            try
            {
                var queue = new ConcurrentQueue<int>(Enumerable.Range(0, settings.MaxGames));
                // Dedicated threads: the searches block on their evaluations, which would starve the thread pool
                var threads = Enumerable.Range(0, parallel).Select(_ => new Thread(() =>
                {
                    while (!stop.IsCancellationRequested && queue.TryDequeue(out int game))
                    {
                        try
                        {
                            string? winner = PlayGame(game, nets, stop.Token, out GameState state, out double seconds);
                            if (!stop.IsCancellationRequested || winner != null && state.HasEnded)
                                Record(game, winner, state, seconds, stop);
                        }
                        catch (OperationCanceledException) { }
                        catch (Exception e) { Write($"game {game + 1} failed: {e.Message}"); }
                    }
                })).ToList();
                threads.ForEach(t => t.Start());
                threads.ForEach(t => t.Join());
            }
            finally
            {
                Console.SetOut(console);
            }

            if (Statistics.Verdict != 0) reason = "decided";
            else if (cancellation.IsCancellationRequested) reason = "cancelled";
            else if (stop.IsCancellationRequested) reason = "time limit";
            Write($"{reason}: {Summary()}");
            return Statistics;
        }

        private string? PlayGame(int game, Dictionary<AgentSpec, StateValueNet> nets, CancellationToken cancellation,
            out GameState state, out double seconds)
        {
            AgentSpec[] seats = game % 2 == 0
                ? [settings.Challenger, settings.Baseline, settings.Challenger, settings.Baseline]
                : [settings.Baseline, settings.Challenger, settings.Baseline, settings.Challenger];
            Utils.Random.SeedCurrentThread(settings.SeedBase + game / 2);
            state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            GameState start = state;
            Agent[] agents = Enumerable.Range(0, 4).Select(seat =>
            {
                AgentSpec spec = seats[seat];
                // The agent picks the encoding of its net (see Vectorizers): a belief-aware net sees its seat's view
                MCTSAgent search = new MCTSAgent((sbyte)seat, start, explorationParameter: spec.Exploration, timeLimitSeconds: settings.Seconds,
                    valueNet: nets[spec], placementWeight: spec.PlacementWeight, opponentModel: spec.OpponentModel,
                    iterationLimit: settings.Iterations, leavesPerBatch: spec.LeavesPerBatch)
                { UniformPriors = spec.UniformPriors, FirstPlayReduction = spec.FirstPlayReduction };

                // Under hidden information the agent sees only its seat's view: one tree over its information sets, or the
                // iterations split between its determinizations
                if (spec.InformationSets) return new InformationSetMCTSAgent(search, start);
                return spec.Determinizations > 0 ? (Agent)new DeterminizedMCTSAgent(search, start, spec.Determinizations) : search;
            }).ToArray();

            var stopwatch = Stopwatch.StartNew();
            while (!state.HasEnded && state.Turn.RoundCounter < 300)
            {
                cancellation.ThrowIfCancellationRequested();
                sbyte acting = 0;
                for (sbyte player = 0; player < 4; player++)
                    if (state.CanPlayerAct(player)) { acting = player; break; }
                Action action = agents[acting].Act(state, cancellation);
                cancellation.ThrowIfCancellationRequested();
                action.Apply(state);
                foreach (Agent agent in agents) agent.HandleAction(action);
            }
            seconds = stopwatch.Elapsed.TotalSeconds;
            return state.HasEnded ? seats[state.Turn.PlayerIndex].Name : null;
        }

        private void Record(int game, string? winner, GameState state, double seconds, CancellationTokenSource stop)
        {
            lock (_lock)
            {
                _games++;
                if (winner == settings.Challenger.Name) _wins++; else _losses++;
                string seats = game % 2 == 0 ? "C B C B" : "B C B C";
                Write($"game {game + 1} ({_games} done): seats {seats}, winner {winner ?? "none"}, VP {string.Join("/", state.Players.Select(p => p.VictoryPoints.Total))}, "
                    + $"{state.Turn.RoundCounter} rounds, {seconds:F0}s");

                string?[] pair = _pairWinners.TryGetValue(game / 2, out var known) ? known : _pairWinners[game / 2] = new string?[2];
                pair[game % 2] = winner ?? "";
                if (pair[0] == null || pair[1] == null) return;

                _scores.Add(pair.Count(w => w == settings.Challenger.Name) / 2.0);
                Statistics = PairStatistics.Of(_scores, settings.Elo0, settings.Elo1, settings.Alpha, settings.Beta);
                Write($"pair {game / 2 + 1}: {Summary()}");
                if (Statistics.Verdict != 0) stop.Cancel();
            }
        }

        /// <summary>
        /// Games, pairs, the Elo estimate with its interval and the SPRT state
        /// </summary>
        public string Summary()
        {
            PairStatistics s = Statistics;
            string text = $"{_games} games {_wins}:{_losses} | {s.Pairs} pairs: won twice {s.ChallengerTwice}:{s.BaselineTwice}, split {s.Split}";
            if (s.Pairs < 2) return text;
            string verdict = s.Verdict switch { 1 => $"H1 accepted (gain >= {settings.Elo1:+0;-0;0} Elo)", -1 => $"H0 accepted (gain <= {settings.Elo0:+0;-0;0} Elo)", _ => "open" };
            return FormattableString.Invariant($"{text} | {s.Elo:+0;-0;0} Elo [{s.EloLow:+0;-0;0}, {s.EloHigh:+0;-0;0}] | SPRT [{settings.Elo0:+0;-0;0}, {settings.Elo1:+0;-0;0}] LLR {s.Llr:+0.00;-0.00} ({s.LowerBound:0.00}, {s.UpperBound:0.00}) {verdict}");
        }

        private void Write(string line)
        {
            lock (log)
            {
                log.WriteLine($"{DateTime.Now:HH:mm} {line}");
                log.Flush();
            }
        }
    }
}
