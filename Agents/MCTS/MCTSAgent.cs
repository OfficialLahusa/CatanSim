using Agents.Inference;
using Common;
using Common.Actions;
using Common.Serialization;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    public class MCTSAgent : Agent
    {
        public const double TIME_LIMIT_SECONDS = 5;
        // Leaves selected before their evaluation, unless the constructor is given another number
        public const int DefaultLeavesPerBatch = 256;

        // Search tree, kept across moves by HandleAction (null until the next Act builds a new one)
        protected MCTSTree? _tree;
        protected double _explorationParameter;
        protected double _timeLimitSeconds;
        protected long _iterationLimit;
        protected Random _random = Random.Shared;

        // The default net is shared by all agents that aren't given another one, and loaded on first use
        private static readonly Lazy<StateValueNet> DefaultValueNet = new(() => new StateValueNet(StateValueNet.DefaultModelFile));

        protected readonly StateValueNet _stateValueNet;
        protected readonly IStateVectorizer _stateVectorizer;

        // Value of places 1 to 4 on [0, 1], the same as PLACEMENT_SCORES in state_value_net.py
        private static readonly float[] PlacementScores = [1f, 2f / 3f, 1f / 3f, 0f];
        private double _placementWeight;
        private OpponentModel _opponentModel;

        protected readonly object _treeLock = new object();

        /// <summary>
        /// The encoding of the leaves for the agent's net (see Vectorizers)
        /// </summary>
        public IStateVectorizer Vectorizer => _stateVectorizer;

        public double ExplorationParameter => _explorationParameter;

        /// <summary>
        /// Searches another tree (an information set tree, see InformationSetMCTSAgent) with this agent's net, budget and
        /// settings
        /// </summary>
        public void RunSearch(ISearchTree tree, CancellationToken cancellation = default)
        {
            if (_iterationLimit > 0) _search.Run(tree, double.PositiveInfinity, _iterationLimit, cancellation);
            else _search.Run(tree, _timeLimitSeconds, 0, cancellation);
        }

        /// <summary>
        /// The agent's answer or pick in an open trade between players, decided outside the tree (see TradeDecisions), or null
        /// </summary>
        public Action? DecideTrade(GameState state) => TradeDecisions.Decide(state, PlayerIndex, TradeUtility);

        /// <summary>
        /// Statistics of the last search (RunForTime or RunBatchedIterations)
        /// </summary>
        public SearchStatistics? LastSearch => _search.LastSearch;

        // The batched search, with large batches selected on all cores and evaluated by the agent's net
        private readonly MCTSSearch _search;


        /// <summary>
        /// How much lower than its parent an unvisited move counts in PUCT (see MCTSTree.FirstPlayReduction)
        /// </summary>
        public double FirstPlayReduction { get; set; } = 0.1;

        /// <summary>
        /// Search by PUCT with uniform priors instead of the policy head's (see MCTSSearch.UniformPriors)
        /// </summary>
        public bool UniformPriors
        {
            get => _search.UniformPriors;
            set => _search.UniformPriors = value;
        }

        /// <param name="valueNet">Value net for the leaves, the default model if null</param>
        /// <param name="vectorizer">Encoding the value net was trained on, the StateVectorizer for the net's input size if null</param>
        /// <summary>
        /// Weight of the placement score in the search value: the value of a state is the win probability plus this weight
        /// times the expected placement score (1 for first place to 0 for last), so the search can tell options apart when it
        /// is far behind and all of them have a win probability of about 0. 0 searches for the win probability alone; above 0
        /// needs a value net with placements (HasPlacements). Can be changed between decisions.
        /// </summary>
        public double PlacementWeight
        {
            get => _placementWeight;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "The placement weight must not be negative.");
                if (value > 0 && !_stateValueNet.HasPlacements)
                    throw new ArgumentException("A placement weight above 0 needs a value net with a placements output.", nameof(value));
                _placementWeight = value;
                _search.PlacementWeight = value;
            }
        }

        /// <summary>
        /// How the search picks the other players' moves (see OpponentModel). Changing it starts a new tree at the next decision,
        /// so it can be set while the agent is thinking.
        /// </summary>
        public OpponentModel OpponentModel
        {
            get => _opponentModel;
            set => _opponentModel = value;
        }

        /// <summary>
        /// Search time per decision, can be changed between decisions; not used while IterationLimit is set
        /// </summary>
        public double TimeLimitSeconds
        {
            get => _timeLimitSeconds;
            set => _timeLimitSeconds = value;
        }

        /// <summary>
        /// Iterations per decision instead of the time limit (0 searches for TimeLimitSeconds). Unlike time, the search quality
        /// then doesn't depend on the load, e.g. with several games running at once. Rounded up to whole batches; can be
        /// changed between decisions.
        /// </summary>
        public long IterationLimit
        {
            get => _iterationLimit;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "The iteration limit must not be negative.");
                _iterationLimit = value;
            }
        }

        public MCTSAgent(sbyte playerIndex, GameState initialState, double explorationParameter = 1.414, double timeLimitSeconds = TIME_LIMIT_SECONDS,
            StateValueNet? valueNet = null, IStateVectorizer? vectorizer = null, double placementWeight = 0, OpponentModel opponentModel = OpponentModel.MaxN,
            long iterationLimit = 0, int leavesPerBatch = DefaultLeavesPerBatch)
            : base(playerIndex)
        {
            _explorationParameter = explorationParameter;
            _timeLimitSeconds = timeLimitSeconds;
            IterationLimit = iterationLimit;
            _stateValueNet = valueNet ?? DefaultValueNet.Value;
            _stateVectorizer = vectorizer ?? Vectorizers.For(_stateValueNet, playerIndex);
            _search = new MCTSSearch(new DirectEvaluator(_stateValueNet), _stateVectorizer, leavesPerBatch, parallelSelection: true);
            PlacementWeight = placementWeight;
            _opponentModel = opponentModel;

            ResetToState(initialState);
        }

        public override Action Act(GameState state)
        {
            return Act(state, CancellationToken.None);
        }

        /// <summary>
        /// Searches until the time or iteration limit or until the token is cancelled (after the batch in progress)
        /// </summary>
        public override Action Act(GameState state, CancellationToken cancellation)
        {
            // Trades between players are decided outside the tree: the value of the position with and without the trade
            if (TradeDecisions.Decide(state, PlayerIndex, TradeUtility) is Action trade) return trade;

            // Reuse the tree kept up to date by HandleAction as long as it describes exactly this state.
            // Otherwise (first move, missed or manual actions, undo, another opponent model) start over on a private copy of the state.
            if (_tree == null || _tree.Model != _opponentModel || !_tree.RootState.Equals(state))
            {
                _tree = new MCTSTree(new GameState(state), PlayerIndex, _explorationParameter, model: _opponentModel);
            }
            _tree.FirstPlayReduction = FirstPlayReduction;

            if (_iterationLimit > 0) RunBatchedIterations(_iterationLimit, cancellation);
            else RunForTime(_timeLimitSeconds, cancellation);

            return _tree.GetMostPromisingMove();
        }

        public override void HandleAction(Action action)
        {
            // Called after the action was applied to the real game, so its output randomness is resolved in its history
            if (_tree == null) return;

            MCTSNode? nextRoot = _tree.FindChildForAction(action);
            GameState nextState = new GameState(_tree.RootState);
            Action replay = action.Clone();

            // Without a matching explored child, or if the tree lost track of the real game, the next Act starts over
            if (nextRoot == null || !replay.IsValidFor(nextState))
            {
                _tree = null;
                return;
            }

            // Advance the root state with the real action instead of the tree's: equivalent outcomes keep their actual
            // details (e.g. the dice pair behind a roll total), so the root state stays identical to the real game state
            replay.Apply(nextState);
            _tree.ReRoot(nextRoot, nextState);
        }

        public override void ResetToState(GameState state)
        {
            // The tree is rebuilt for the given state by the next Act
            _tree = null;
        }

        /// <summary>
        /// A root move of the search: its visits, its prior and its mean value for each player
        /// </summary>
        public sealed record RootMove(Action Move, uint Visits, float Prior, double[] MeanValues);

        /// <summary>
        /// The root moves of the current tree and the state they belong to, for showing the search (also while it runs: the
        /// counts are read without a lock, so they can lag a batch behind); empty without a tree or for a chance node
        /// </summary>
        public (GameState? state, IReadOnlyList<RootMove> moves) GetRootMoves()
        {
            MCTSTree? tree = _tree;
            if (tree == null) return (null, []);

            List<RootMove> moves = [];
            foreach (MCTSNode child in tree.RootNode.Children)
            {
                if (child.LastAction == null) return (tree.RootState, []);
                uint resolved = child.ResolvedVisitCount;
                double[] means = new double[MCTSNode.PlayerCount];
                for (int player = 0; player < means.Length; player++)
                    means[player] = resolved == 0 ? 0 : child.PlayerScore(player) / resolved;
                moves.Add(new RootMove(child.LastAction, child.VisitCount, child.Prior, means));
            }
            return (tree.RootState, moves);
        }

        public void RunIterations(int iterations)
        {
            Console.WriteLine($"Running {iterations} MCTS iterations:");
            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < iterations; i++)
            {
                RunOneIteration();
            }

            TimeSpan runtime = stopwatch.Elapsed;

            Console.WriteLine($"Simulation took {runtime.TotalSeconds}s.");
        }

        public void RunForTime(double timeLimitSeconds, CancellationToken cancellation = default)
        {
            Console.WriteLine($"Running MCTS for {timeLimitSeconds}s:");
            RunSearch(timeLimitSeconds, 0, cancellation);
        }

        /// <summary>
        /// The batched search of RunForTime, for a number of iterations (rounded up to whole batches) instead of a time
        /// </summary>
        public void RunBatchedIterations(long iterations, CancellationToken cancellation = default)
        {
            Console.WriteLine($"Running MCTS for {iterations} iterations:");
            RunSearch(double.PositiveInfinity, iterations, cancellation);
        }

        private void RunSearch(double timeLimitSeconds, long iterationLimit, CancellationToken cancellation)
        {
            // Only called while Act holds a tree
            _search.Run(_tree!, timeLimitSeconds, iterationLimit, cancellation);
            Console.WriteLine($"Simulation ran for {LastSearch!.Iterations} iterations.");
        }

        protected void RunOneIteration()
        {
            // Only called while Act holds a tree
            MCTSTree tree = _tree!;

            // Selection
            // Find a leaf node to expand, starting from the root node and traversing down the tree using the UCT formula
            (MCTSNode leafNode, GameState stateAtLeaf) = tree.SelectLeafNode();

            // Expansion
            // If the selected leaf isn't terminal, expand its children and select a leaf node among them
            if (!leafNode.IsTerminal)
            {
                tree.ExpandNode(leafNode, stateAtLeaf);
                (leafNode, stateAtLeaf) = tree.SelectLeafNode(leafNode, stateAtLeaf, suppressStateCopy: true);
            }

            // Simulation
            // Ended matches have a known result, everything else is estimated by the value net
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            if (stateAtLeaf.HasEnded) TerminalUtilities(stateAtLeaf, utilities);
            else SimulateUtilities(stateAtLeaf, utilities);

            // Backpropagation
            // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
            if (tree.Model == OpponentModel.Random) tree.Backpropagate(leafNode, utilities[PlayerIndex]);
            else tree.Backpropagate(leafNode, utilities);
        }

        private void LeafUtilities(ReadOnlySpan<float> winProbabilities, ReadOnlySpan<float> placements, Span<double> destination)
        {
            MCTSSearch.LeafUtilities(winProbabilities, placements, _placementWeight, destination);
        }

        private void TerminalUtilities(GameState state, Span<double> destination)
        {
            MCTSSearch.TerminalUtilities(state, _placementWeight, destination);
        }

        protected double TerminalResult(GameState state)
        {
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            TerminalUtilities(state, utilities);
            return utilities[PlayerIndex];
        }

        /// <summary>
        /// Expected placement score of a placement distribution over places 1 to 4
        /// </summary>
        public static float PlacementScore(ReadOnlySpan<float> placementDistribution)
        {
            float score = 0;
            for (int place = 0; place < PlacementScores.Length; place++)
                score += placementDistribution[place] * PlacementScores[place];
            return score;
        }

        /// <summary>
        /// Placement score of a player in an ended match: the winner is first, the others are ranked by victory points, and
        /// players with equal victory points share their places (the same rule as the placement labels in playout_results.py)
        /// </summary>
        public static float TerminalPlacementScore(GameState state, int player)
        {
            int winner = state.Turn.PlayerIndex;
            if (player == winner) return PlacementScores[0];

            int points = state.Players[player].VictoryPoints.Total;
            int better = 1, tied = 0;
            for (int p = 0; p < state.Players.Length; p++)
            {
                if (p == winner) continue;
                int other = state.Players[p].VictoryPoints.Total;
                if (other > points) better++;
                else if (other == points) tied++;
            }

            float score = 0;
            for (int place = better; place < better + tied; place++)
                score += PlacementScores[place];
            return score / tied;
        }

        protected double SimulatePlayout(GameState state)
        {
            //Console.WriteLine("Sim state hash: " + state.GetHashCode());

            while (!state.HasEnded)
            {
                List<Action> actions = MCTSTree.GetValidActions(state);
                if (actions.Count == 0)
                {
                    Console.WriteLine("No actions available but not over: Saving state");

                    // Write GameState to file
                    SaveFile saveFile = new SaveFile(state, [], []);
                    string saveData = SaveFileSerializer.Serialize(saveFile);
                    File.WriteAllText("messup.yaml", saveData);

                    throw new InvalidOperationException("No valid actions available.");
                }
                Action action = actions[_random.Next(actions.Count)];
                action.Apply(state);
                //Console.WriteLine("In sim state hash: " + state.GetHashCode());
            }
            return state.Turn.PlayerIndex == PlayerIndex ? 1 : 0;
        }

        protected double SimulateML(GameState state)
        {
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            SimulateUtilities(state, utilities);
            return utilities[PlayerIndex];
        }

        private void SimulateUtilities(GameState state, Span<double> destination)
        {
            float[] features = new float[_stateVectorizer.FeatureDim];
            _stateVectorizer.VectorizeInto(state, features);
            var input = new DenseTensor<float>(features, [1, features.Length]);

            if (_placementWeight == 0)
            {
                LeafUtilities(_stateValueNet.Run(input), default, destination);
                return;
            }

            (float[] winProbabilities, float[] placements) = _stateValueNet.RunWithPlacements(input);
            LeafUtilities(winProbabilities, placements, destination);
        }

        /// <summary>
        /// A player's utility of a position by the value net, as the search scores its leaves
        /// </summary>
        private float TradeUtility(GameState state, int player)
        {
            Span<double> utilities = stackalloc double[MCTSNode.PlayerCount];
            SimulateUtilities(state, utilities);
            return (float)utilities[player];
        }

        protected float ValuationToScore(ReadOnlySpan<float> valuation)
        {
            float sum = 0;
            foreach (float value in valuation)
                sum += value;

            return valuation[PlayerIndex] / sum;
        }

        protected float[] StateValueFunc(GameState state)
        {
            float[] features = new float[_stateVectorizer.FeatureDim];
            _stateVectorizer.VectorizeInto(state, features);
            return _stateValueNet.Run(new DenseTensor<float>(features, [1, features.Length]));
        }
    }
}
