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
        protected MCTSTree _tree;
        protected double _explorationParameter;
        protected Random _random = Random.Shared;

        protected static StateValueNet _stateValueNet;
        protected static StateVectorizer _stateVectorizer;

        protected readonly object _treeLock = new object();
        

        public MCTSAgent(sbyte playerIndex, GameState initialState, double explorationParameter = 1.414)
            : base(playerIndex)
        {
            _explorationParameter = explorationParameter;
        }

        static MCTSAgent()
        {
            _stateValueNet = new StateValueNet(StateValueNet.DefaultModelFile);
            _stateVectorizer = new StateVectorizer();
        }

        public override Action Act(GameState state, uint playedActionsCount)
        {
            _tree = new MCTSTree(state, PlayerIndex, _explorationParameter);

            RunForTime(TIME_LIMIT_SECONDS);

            return _tree.GetMostPromisingMove();
        }

        public override void HandleAction(Action action)
        {
            // TODO: Advance MCTSTree to child of root matching action
            // Enables reusing of relevant subtrees for future decisions
            // Requires careful consideration of grouped nodes
            // Unlink bidirectional parent <-> child references involving the old root node to allow for GC
        }

        public override void ResetToState(GameState state)
        {
            // TODO: Clear MCTS Tree and initialize new root node
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

            uint iterationCounter = 0;
            do
            {
                RunBatchedIterations(BATCH_SIZE);
                iterationCounter += BATCH_SIZE;
            }
            while (stopwatch.Elapsed.TotalSeconds < timeLimitSeconds);

            Console.WriteLine($"Simulation ran for {iterationCounter} iterations.");
        }

        protected void RunOneIteration()
        {
            // Selection
            // Find a leaf node to expand, starting from the root node and traversing down the tree using the UCT formula
            (MCTSNode leafNode, GameState stateAtLeaf) = _tree.SelectLeafNode();

            // Expansion
            // If the selected leaf isn't terminal, expand its children and select a leaf node among them
            if (!leafNode.IsTerminal)
            {
                _tree.ExpandNode(leafNode, stateAtLeaf);
                (leafNode, stateAtLeaf) = _tree.SelectLeafNode(leafNode, stateAtLeaf, suppressStateCopy: true);
            }

            // Simulation
            // Ended matches have a known result, everything else is estimated by the value net
            double result = stateAtLeaf.HasEnded ? TerminalResult(stateAtLeaf) : SimulateML(stateAtLeaf);

            // Backpropagation
            // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
            _tree.Backpropagate(leafNode, result);
        }

        protected void RunBatchedIterations(uint batchSize)
        {
            var leaves = new (MCTSNode leafNode, GameState stateAtLeaf)[batchSize];

            Parallel.For(0, (int)batchSize, i =>
            {
                // Selection
                // Find a leaf node to expand, starting from the root node and traversing down the tree using the UCT formula
                (MCTSNode leafNode, GameState stateAtLeaf) = _tree.SelectLeafNode();

                // Expansion
                // If the selected leaf isn't terminal, expand its children and select a leaf node among them
                if (!leafNode.IsTerminal)
                {
                    _tree.ExpandNode(leafNode, stateAtLeaf);
                    (leafNode, stateAtLeaf) = _tree.SelectLeafNode(leafNode, stateAtLeaf, suppressStateCopy: true);
                }

                _tree.MarkPendingPath(leafNode);

                leaves[i] = (leafNode, stateAtLeaf);
            });
            

            // Simulation
            // Ended matches have a known result; only the remaining leaves go through the value net
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

            Parallel.For(0, (int)batchSize, i =>
            {
                // Backpropagation
                // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
                _tree.Backpropagate(leaves[i].leafNode, results[i], pathAlreadyMarked: true);
            });
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

            // Parallel Featurization (CPU bound)
            Parallel.For(0, states.Count, i =>
            {
                Span<float> destinationSlice = memoryBuffer.Span.Slice((int)i * StateVectorizer.FeatureDim, StateVectorizer.FeatureDim);

                ReadOnlySpan<float> featureVector = _stateVectorizer.Vectorize(states[(int)i]).ToArray();
                featureVector.CopyTo(destinationSlice);
            });

            // Run Batched NN Inference
            float[] outputs = _stateValueNet.Run(batchFeatures);

            // Compute final scores
            List<double> results = [];
            for (int i = 0; i < states.Count; i++)
            {
                results.Add(ValuationToScore(new ArraySegment<float>(outputs, i * 4, 4).ToArray()));
            }

            return results;
        }

        protected double SimulateML(GameState state)
        {
            return ValuationToScore(StateValueFunc(state));
        }

        protected float ValuationToScore(float[] valuation)
        {
            return valuation[PlayerIndex] / valuation.Sum();
        }

        protected static float[] StateValueFunc(GameState state)
        {
            var inputTensor = _stateVectorizer.Vectorize(state);
            return _stateValueNet.Run(inputTensor);
        }
    }
}
