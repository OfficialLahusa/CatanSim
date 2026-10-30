using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    /// <summary>
    /// Plays under hidden information by Perfect Information Monte Carlo (PIMC): it follows the game as its seat sees it
    /// (BeliefTracker), searches several determinizations of its view with the MCTS agent and plays the move with the most
    /// visits over all of them. The true state it is given is only used for its public parts.
    /// </summary>
    public sealed class DeterminizedMCTSAgent : Agent
    {
        private readonly MCTSAgent _search;
        private readonly Random _random;

        // The game as the agent followed it, to give the tracker the positions before and after each move
        private GameState _mirror;
        private BeliefTracker _beliefs;

        /// <summary>
        /// Full states searched per decision when something is hidden; the iteration limit is split between them
        /// </summary>
        public int Determinizations { get; set; }

        /// <param name="search">The MCTS agent of the same seat that searches each determinization</param>
        /// <param name="seed">Seed of the determinizations, random if null</param>
        public DeterminizedMCTSAgent(MCTSAgent search, GameState initialState, int determinizations = 8, int? seed = null)
            : base(search.PlayerIndex)
        {
            if (determinizations < 1) throw new ArgumentOutOfRangeException(nameof(determinizations), "At least one determinization is needed.");
            _search = search;
            Determinizations = determinizations;
            _random = seed is int s ? new Random(s) : new Random();
            _mirror = new GameState(initialState);
            _beliefs = new BeliefTracker(PlayerIndex, initialState);
        }

        /// <summary>
        /// What the agent knows about the game
        /// </summary>
        public BeliefTracker Beliefs => _beliefs;

        public MCTSAgent Search => _search;

        /// <summary>
        /// The full states the next decision would search: one if nothing is hidden, Determinizations otherwise
        /// </summary>
        public IReadOnlyList<GameState> Determinize(GameState state)
        {
            GameState view = _beliefs.View(state);
            if (!_beliefs.HasHiddenInformation) return [_beliefs.Determinize(view, _random)];
            return Enumerable.Range(0, Determinizations).Select(_ => _beliefs.Determinize(view, _random)).ToList();
        }

        public override Action Act(GameState state) => Act(state, CancellationToken.None);

        public override Action Act(GameState state, CancellationToken cancellation)
        {
            IReadOnlyList<GameState> worlds = Determinize(state);

            // Each world gets its share of the iterations; without an iteration limit, of the time
            long iterations = _search.IterationLimit, perWorldIterations = iterations > 0 ? Math.Max(1, iterations / worlds.Count) : 0;
            double seconds = _search.TimeLimitSeconds;
            _search.IterationLimit = perWorldIterations;
            _search.TimeLimitSeconds = seconds / worlds.Count;

            List<(Action move, double weight)> votes = [];
            try
            {
                foreach (GameState world in worlds)
                {
                    if (cancellation.IsCancellationRequested) break;
                    _search.ResetToState(world);
                    Action chosen = _search.Act(world, cancellation);

                    // The visits of the root moves; without a tree of this world (a trade answer, a chance node) its pick counts once
                    (GameState? root, IReadOnlyList<MCTSAgent.RootMove> moves) = _search.GetRootMoves();
                    if (root != null && root.Equals(world) && moves.Count > 0 && moves.Any(m => m.Move.IsSameDecision(chosen)))
                    {
                        long total = moves.Sum(m => (long)m.Visits);
                        foreach (MCTSAgent.RootMove move in moves)
                            votes.Add((move.Move, total > 0 ? (double)move.Visits / total : 1.0 / moves.Count));
                    }
                    else
                    {
                        votes.Add((chosen, 1));
                    }
                }
            }
            finally
            {
                _search.IterationLimit = iterations;
                _search.TimeLimitSeconds = seconds;
                _search.ResetToState(state);
            }

            // The move with the most weight over all worlds; its outcome is the true game's to decide
            List<(Action move, double weight)> totals = [];
            foreach ((Action move, double weight) in votes)
            {
                int index = totals.FindIndex(t => t.move.IsSameDecision(move));
                if (index < 0) totals.Add((move, weight));
                else totals[index] = (totals[index].move, totals[index].weight + weight);
            }
            Action best = totals.OrderByDescending(t => t.weight).First().move.Clone();
            if (best is IReplayAction replay) replay.ClearHistory();
            return best;
        }

        public override void HandleAction(Action action)
        {
            // Called after the action was applied to the real game, so its outcome is in its history
            GameState before = new GameState(_mirror);
            Action replayed = action.Clone();
            if (!replayed.IsValidFor(_mirror))
            {
                // Lost track of the game (missed moves): start over from what the game shows now, as if everything were known
                ResetToState(_mirror);
                return;
            }
            replayed.Apply(_mirror);
            _beliefs.Observe(before, replayed, _mirror);
        }

        /// <summary>
        /// Goes on from a position with what the seat knows there (the match keeps the beliefs along its history, so after an
        /// undo, a load or a seat change the agent knows no more than the seat could)
        /// </summary>
        public void ResetTo(GameState state, BeliefTracker beliefs)
        {
            if (beliefs.Observer != PlayerIndex) throw new ArgumentException($"The beliefs are seat {beliefs.Observer}'s, not seat {PlayerIndex}'s.", nameof(beliefs));
            _mirror = new GameState(state);
            _beliefs = new BeliefTracker(beliefs);
            _search.ResetToState(state);
        }

        public override void ResetToState(GameState state)
        {
            // After an undo or an edit the agent knows the hands of the given position (see BeliefTracker's constructor)
            _mirror = new GameState(state);
            _beliefs = new BeliefTracker(PlayerIndex, state);
            _search.ResetToState(state);
        }
    }
}
