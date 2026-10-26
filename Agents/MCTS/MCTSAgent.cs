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
        protected const uint BATCH_SIZE = 256;

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
        /// Counters of a search: iterations, distinct leaves evaluated (several iterations of a batch can arrive at the same leaf),
        /// the time spent selecting, evaluating and backpropagating (evaluation overlaps the other two) and waiting for evaluations
        /// </summary>
        public record SearchStatistics(long Iterations, long DistinctLeaves, double Seconds, double SelectionSeconds, double EvaluationSeconds, double BackpropagationSeconds,
            double EvaluationWaitSeconds)
        {
            public double IterationsPerSecond => Iterations / Seconds;
            public double DistinctLeafShare => Iterations == 0 ? 0 : (double)DistinctLeaves / Iterations;
        }

        /// <summary>
        /// Statistics of the last search (RunForTime or RunBatchedIterations)
        /// </summary>
        public SearchStatistics? LastSearch { get; private set; }

        // Leaves of a batch with their features, written by the selecting threads so the evaluation only runs the value net
        private sealed record LeafBatch((MCTSNode leafNode, GameState stateAtLeaf)[] Leaves, DenseTensor<float> Features);

        private long _distinctLeaves;
        private readonly Stopwatch _selectionTime = new(), _evaluationTime = new(), _backpropagationTime = new(), _evaluationWaitTime = new();


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
            long iterationLimit = 0)
            : base(playerIndex)
        {
            _explorationParameter = explorationParameter;
            _timeLimitSeconds = timeLimitSeconds;
            IterationLimit = iterationLimit;
            _stateValueNet = valueNet ?? DefaultValueNet.Value;
            _stateVectorizer = vectorizer ?? StateVectorizer.For(_stateValueNet);
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
            // Reuse the tree kept up to date by HandleAction as long as it describes exactly this state.
            // Otherwise (first move, missed or manual actions, undo, another opponent model) start over on a private copy of the state.
            if (_tree == null || _tree.Model != _opponentModel || !_tree.RootState.Equals(state))
            {
                _tree = new MCTSTree(new GameState(state), PlayerIndex, _explorationParameter, model: _opponentModel);
            }

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
            Stopwatch stopwatch = Stopwatch.StartNew();
            _distinctLeaves = 0;
            _selectionTime.Reset();
            _evaluationTime.Reset();
            _backpropagationTime.Reset();
            _evaluationWaitTime.Reset();

            // Only called while Act holds a tree
            MCTSTree tree = _tree!;

            // Pipelined batches: while the value net evaluates one batch, the previous one is backpropagated and the next one selected.
            // Selection therefore runs one batch behind on results, but the visits of the batch in evaluation are already counted.
            LeafBatch? evaluatedBatch = null;
            Task<double[]>? evaluation = null; // utilities per leaf and player

            uint iterationCounter = 0;
            do
            {
                LeafBatch selectedBatch = SelectBatch(tree, BATCH_SIZE);

                _evaluationWaitTime.Start();
                double[]? previousResults = evaluation?.Result;
                _evaluationWaitTime.Stop();

                LeafBatch? previousBatch = evaluatedBatch;
                evaluatedBatch = selectedBatch;
                evaluation = Task.Run(() => EvaluateBatch(selectedBatch));

                if (previousBatch != null)
                    BackpropagateBatch(tree, previousBatch, previousResults!);

                iterationCounter += BATCH_SIZE;
            }
            while ((iterationLimit > 0 ? iterationCounter < iterationLimit : stopwatch.Elapsed.TotalSeconds < timeLimitSeconds)
                && !cancellation.IsCancellationRequested);

            _evaluationWaitTime.Start();
            BackpropagateBatch(tree, evaluatedBatch, evaluation.Result);
            _evaluationWaitTime.Stop();

            LastSearch = new SearchStatistics(iterationCounter, _distinctLeaves, stopwatch.Elapsed.TotalSeconds,
                _selectionTime.Elapsed.TotalSeconds, _evaluationTime.Elapsed.TotalSeconds, _backpropagationTime.Elapsed.TotalSeconds,
                _evaluationWaitTime.Elapsed.TotalSeconds);
            Console.WriteLine($"Simulation ran for {iterationCounter} iterations.");
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

        protected void RunBatchedIterations(uint batchSize)
        {
            // Only called while Act holds a tree
            MCTSTree tree = _tree!;

            LeafBatch batch = SelectBatch(tree, batchSize);
            BackpropagateBatch(tree, batch, EvaluateBatch(batch));
        }

        private LeafBatch SelectBatch(MCTSTree tree, uint batchSize)
        {
            var leaves = new (MCTSNode leafNode, GameState stateAtLeaf)[batchSize];
            int featureDim = _stateVectorizer.FeatureDim;
            DenseTensor<float> features = new DenseTensor<float>([(int)batchSize, featureDim]);
            Memory<float> featureBuffer = features.Buffer;

            _selectionTime.Start();
            Parallel.For(0, (int)batchSize, i =>
            {
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

                tree.MarkPendingPath(leafNode);

                leaves[i] = (leafNode, stateAtLeaf);

                // Featurize right away, on the same thread (ended matches don't need the value net, their row stays empty)
                if (!stateAtLeaf.HasEnded)
                    _stateVectorizer.VectorizeInto(stateAtLeaf, featureBuffer.Span.Slice(i * featureDim, featureDim));
            });
            _selectionTime.Stop();
            _distinctLeaves += leaves.Select(leaf => leaf.leafNode).Distinct().Count();

            return new LeafBatch(leaves, features);
        }

        /// <summary>
        /// Utilities of the leaves of a batch, one per player (flat: leaf * 4 + player). Only reads the batch, so it can run
        /// while the tree is in use.
        /// </summary>
        private double[] EvaluateBatch(LeafBatch batch)
        {
            // Simulation
            // Ended matches have a known result, everything else is estimated by the value net in one batched run
            _evaluationTime.Start();
            var leaves = batch.Leaves;
            double[] utilities = new double[leaves.Length * MCTSNode.PlayerCount];

            float[]? outputs = null, placements = null;
            if (leaves.Any(leaf => !leaf.stateAtLeaf.HasEnded))
            {
                if (_placementWeight > 0) (outputs, placements) = _stateValueNet.RunWithPlacements(batch.Features);
                else outputs = _stateValueNet.Run(batch.Features);
            }

            for (int i = 0; i < leaves.Length; i++)
            {
                Span<double> destination = utilities.AsSpan(i * MCTSNode.PlayerCount, MCTSNode.PlayerCount);
                if (leaves[i].stateAtLeaf.HasEnded)
                    TerminalUtilities(leaves[i].stateAtLeaf, destination);
                else
                    LeafUtilities(outputs.AsSpan(i * 4, 4), placements == null ? default : placements.AsSpan(i * 16, 16), destination);
            }
            _evaluationTime.Stop();

            return utilities;
        }

        private void BackpropagateBatch(MCTSTree tree, LeafBatch batch, double[] utilities)
        {
            var leaves = batch.Leaves;
            _backpropagationTime.Start();
            Parallel.For(0, leaves.Length, i =>
            {
                // Backpropagation
                // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root;
                // the opponent models that use every player's values get all four utilities, Random only the owner's
                if (tree.Model == OpponentModel.Random)
                    tree.Backpropagate(leaves[i].leafNode, utilities[i * MCTSNode.PlayerCount + PlayerIndex], pathAlreadyMarked: true);
                else
                    tree.Backpropagate(leaves[i].leafNode, utilities.AsSpan(i * MCTSNode.PlayerCount, MCTSNode.PlayerCount), pathAlreadyMarked: true);
            });
            _backpropagationTime.Stop();
        }

        /// <summary>
        /// Every player's utility from the value net: the win probability, plus the placement weight times the expected
        /// placement score if the placements are given
        /// </summary>
        private void LeafUtilities(ReadOnlySpan<float> winProbabilities, ReadOnlySpan<float> placements, Span<double> destination)
        {
            float sum = 0;
            foreach (float value in winProbabilities)
                sum += value;

            for (int player = 0; player < MCTSNode.PlayerCount; player++)
            {
                destination[player] = winProbabilities[player] / sum;
                if (!placements.IsEmpty)
                    destination[player] += _placementWeight * PlacementScore(placements.Slice(player * 4, 4));
            }
        }

        /// <summary>
        /// Every player's utility in an ended match: 1 for the winner, plus the placement weight times the actual placement score
        /// </summary>
        private void TerminalUtilities(GameState state, Span<double> destination)
        {
            // Players can only win on their own turn, so the turn player of an ended match is the winner
            for (int player = 0; player < MCTSNode.PlayerCount; player++)
            {
                destination[player] = state.Turn.PlayerIndex == player ? 1 : 0;
                if (_placementWeight > 0)
                    destination[player] += _placementWeight * TerminalPlacementScore(state, player);
            }
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
