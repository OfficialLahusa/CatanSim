using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    /// <summary>
    /// Plays under hidden information with information set MCTS (ISMCTSTree): it follows the game as its seat sees it
    /// (BeliefTracker) and searches one tree over its information sets, a world drawn for every iteration, with the net, the
    /// time or iteration budget and the settings of the MCTS agent of its seat. Trades between players are decided outside
    /// the tree, on a world drawn from the view. The true state it is given is only used for its public parts.
    /// </summary>
    public sealed class InformationSetMCTSAgent : Agent
    {
        private readonly MCTSAgent _search;
        private readonly Random _random;

        // The game as the agent followed it, to give the tracker the positions before and after each move
        private GameState _mirror;
        private BeliefTracker _beliefs;

        /// <param name="search">The MCTS agent of the same seat, whose net, budget and settings the search uses</param>
        /// <param name="seed">Seed of the worlds, random if null</param>
        public InformationSetMCTSAgent(MCTSAgent search, GameState initialState, int? seed = null)
            : base(search.PlayerIndex)
        {
            _search = search;
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
        /// Searches the true position instead of worlds drawn from the beliefs (an ablation: the same search without hidden
        /// information)
        /// </summary>
        public bool FullKnowledge { get; init; }

        /// <summary>
        /// Ablations of the tree (see ISMCTSTree): branching on outcomes the seat can't see, normalized values
        /// </summary>
        public bool BranchHiddenOutcomes { get; init; }

        public bool NormalizeValues { get; init; }

        /// <summary>
        /// The tree of the last decision, for showing the search
        /// </summary>
        public ISMCTSTree? LastTree { get; private set; }

        public override Action Act(GameState state) => Act(state, CancellationToken.None);

        public override Action Act(GameState state, CancellationToken cancellation)
        {
            if (_search.DecideTrade(FullKnowledge ? state : _beliefs.Determinize(_beliefs.View(state), _random)) is Action trade) return trade;

            List<Action> legal = MCTSTree.GetValidActions(state);
            if (legal.Count == 1) return legal[0].Clone();

            ISMCTSTree tree = new ISMCTSTree(state, _beliefs, _search.ExplorationParameter, _random.Next(), FullKnowledge)
            {
                FirstPlayReduction = _search.FirstPlayReduction,
                BranchHiddenOutcomes = BranchHiddenOutcomes,
                NormalizeValues = NormalizeValues
            };
            LastTree = tree;
            _search.RunSearch(tree, cancellation);
            return tree.GetMostPromisingMove();
        }

        public override void HandleAction(Action action)
        {
            // Called after the action was applied to the real game, so its outcome is in its history
            GameState before = new GameState(_mirror);
            Action replayed = action.Clone();
            if (!replayed.IsValidFor(_mirror))
            {
                ResetToState(_mirror);
                return;
            }
            replayed.Apply(_mirror);
            _beliefs.Observe(before, replayed, _mirror);
        }

        /// <summary>
        /// Goes on from a position with what the seat knows there (see DeterminizedMCTSAgent.ResetTo)
        /// </summary>
        public void ResetTo(GameState state, BeliefTracker beliefs)
        {
            if (beliefs.Observer != PlayerIndex) throw new ArgumentException($"The beliefs are seat {beliefs.Observer}'s, not seat {PlayerIndex}'s.", nameof(beliefs));
            _mirror = new GameState(state);
            _beliefs = new BeliefTracker(beliefs);
        }

        public override void ResetToState(GameState state)
        {
            // Without the history the agent knows the hands of the given position (see BeliefTracker's constructor)
            _mirror = new GameState(state);
            _beliefs = new BeliefTracker(PlayerIndex, state);
        }
    }
}
