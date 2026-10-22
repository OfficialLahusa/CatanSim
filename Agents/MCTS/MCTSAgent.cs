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
        protected const double TIME_LIMIT_SECONDS = 5;
        protected const uint BATCH_SIZE = 256;

        // Search tree, kept across moves by HandleAction (null until the next Act builds a new one)
        protected MCTSTree? _tree;
        protected double _explorationParameter;
        protected double _timeLimitSeconds;
        protected Random _random = Random.Shared;

        protected static StateValueNet _stateValueNet;
        protected static StateVectorizer _stateVectorizer;

        protected readonly object _treeLock = new object();

        /// <summary>
        /// Counters of a search: iterations, distinct leaves evaluated (several iterations of a batch can arrive at the same leaf)
        /// and the time spent selecting, evaluating and backpropagating
        /// </summary>
        public record SearchStatistics(long Iterations, long DistinctLeaves, double Seconds, double SelectionSeconds, double EvaluationSeconds, double BackpropagationSeconds)
        {
            public double IterationsPerSecond => Iterations / Seconds;
            public double DistinctLeafShare => Iterations == 0 ? 0 : (double)DistinctLeaves / Iterations;
        }

        /// <summary>
        /// Statistics of the last RunForTime call
        /// </summary>
        public SearchStatistics? LastSearch { get; private set; }

        private long _distinctLeaves;
        private readonly Stopwatch _selectionTime = new(), _evaluationTime = new(), _backpropagationTime = new();


        public MCTSAgent(sbyte playerIndex, GameState initialState, double explorationParameter = 1.414, double timeLimitSeconds = TIME_LIMIT_SECONDS)
            : base(playerIndex)
        {
            _explorationParameter = explorationParameter;
            _timeLimitSeconds = timeLimitSeconds;

            ResetToState(initialState);
        }

        static MCTSAgent()
        {
            _stateValueNet = new StateValueNet(StateValueNet.DefaultModelFile);
            _stateVectorizer = new StateVectorizer();
        }

        public override Action Act(GameState state)
        {
            // Reuse the tree kept up to date by HandleAction as long as it describes exactly this state.
            // Otherwise (first move, missed or manual actions, undo) start over on a private copy of the state.
            if (_tree == null || !_tree.RootState.Equals(state))
            {
                _tree = new MCTSTree(new GameState(state), PlayerIndex, _explorationParameter);
            }

            RunForTime(_timeLimitSeconds);

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

        public void RunForTime(double timeLimitSeconds)
        {
            Console.WriteLine($"Running MCTS for {timeLimitSeconds}s:");
            Stopwatch stopwatch = Stopwatch.StartNew();
            _distinctLeaves = 0;
            _selectionTime.Reset();
            _evaluationTime.Reset();
            _backpropagationTime.Reset();

            uint iterationCounter = 0;
            do
            {
                RunBatchedIterations(BATCH_SIZE);
                iterationCounter += BATCH_SIZE;
            }
            while (stopwatch.Elapsed.TotalSeconds < timeLimitSeconds);

            LastSearch = new SearchStatistics(iterationCounter, _distinctLeaves, stopwatch.Elapsed.TotalSeconds,
                _selectionTime.Elapsed.TotalSeconds, _evaluationTime.Elapsed.TotalSeconds, _backpropagationTime.Elapsed.TotalSeconds);
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
            double result = stateAtLeaf.HasEnded ? TerminalResult(stateAtLeaf) : SimulateML(stateAtLeaf);

            // Backpropagation
            // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
            tree.Backpropagate(leafNode, result);
        }

        protected void RunBatchedIterations(uint batchSize)
        {
            // Only called while Act holds a tree
            MCTSTree tree = _tree!;

            var leaves = new (MCTSNode leafNode, GameState stateAtLeaf)[batchSize];

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
            });
            _selectionTime.Stop();
            _distinctLeaves += leaves.Select(leaf => leaf.leafNode).Distinct().Count();

            // Simulation
            // Ended matches have a known result; only the remaining leaves go through the value net
            _evaluationTime.Start();
            double[] results = new double[batchSize];
            List<int> pendingIndices = [];

            for (int i = 0; i < batchSize; i++)
            {
                if (leaves[i].stateAtLeaf.HasEnded)
                    results[i] = TerminalResult(leaves[i].stateAtLeaf);
                else
                    pendingIndices.Add(i);
            }

            if (pendingIndices.Count > 0)
            {
                List<double> netResults = BatchSimulateML(pendingIndices.Select(i => leaves[i].stateAtLeaf).ToList());
                for (int k = 0; k < pendingIndices.Count; k++)
                    results[pendingIndices[k]] = netResults[k];
            }
            _evaluationTime.Stop();

            _backpropagationTime.Start();
            Parallel.For(0, (int)batchSize, i =>
            {
                // Backpropagation
                // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
                tree.Backpropagate(leaves[i].leafNode, results[i], pathAlreadyMarked: true);
            });
            _backpropagationTime.Stop();
        }

        protected double TerminalResult(GameState state)
        {
            // Players can only win on their own turn, so the turn player of an ended match is the winner
            return state.Turn.PlayerIndex == PlayerIndex ? 1 : 0;
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

        protected List<double> BatchSimulateML(List<GameState> states)
        {
            // Featurize all states
            DenseTensor<float> batchFeatures = new DenseTensor<float>([states.Count, StateVectorizer.FeatureDim]);
            Memory<float> memoryBuffer = batchFeatures.Buffer;

            // Parallel Featurization (CPU bound), written directly into the batch rows
            Parallel.For(0, states.Count, i =>
            {
                _stateVectorizer.VectorizeInto(states[i], memoryBuffer.Span.Slice(i * StateVectorizer.FeatureDim, StateVectorizer.FeatureDim));
            });

            // Run Batched NN Inference
            float[] outputs = _stateValueNet.Run(batchFeatures);

            // Compute final scores
            List<double> results = new(states.Count);
            for (int i = 0; i < states.Count; i++)
            {
                results.Add(ValuationToScore(outputs.AsSpan(i * 4, 4)));
            }

            return results;
        }

        protected double SimulateML(GameState state)
        {
            return ValuationToScore(StateValueFunc(state));
        }

        protected float ValuationToScore(ReadOnlySpan<float> valuation)
        {
            float sum = 0;
            foreach (float value in valuation)
                sum += value;

            return valuation[PlayerIndex] / sum;
        }

        protected static float[] StateValueFunc(GameState state)
        {
            var inputTensor = _stateVectorizer.Vectorize(state);
            return _stateValueNet.Run(inputTensor);
        }
    }
}
