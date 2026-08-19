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
        

        public MCTSAgent(sbyte playerIndex, double explorationParameter = 1.414)
            : base(playerIndex)
        {
            _explorationParameter = explorationParameter;
        }

        static MCTSAgent()
        {
            _stateValueNet = new StateValueNet("state_value_net-0mw2c1bh.onnx");
            _stateVectorizer = new StateVectorizer();
        }

        public override Action Act(GameState state, uint playedActionsCount)
        {
            _tree = new MCTSTree(state, PlayerIndex, _explorationParameter);

            RunForTime(TIME_LIMIT_SECONDS);

            return _tree.GetMostPromisingMove();
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
            // Run simulation and get result (win percentage)
            double result = SimulateML(stateAtLeaf);

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
            // Run simulation and get result (win percentage)
            List<double> results = BatchSimulateML(leaves.Select(x => x.stateAtLeaf).ToList());

            Parallel.For(0, (int)batchSize, i =>
            {
                // Backpropagation
                // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
                _tree.Backpropagate(leaves[i].leafNode, results[i], pathAlreadyMarked: true);
            });
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
            DenseTensor<float> batchFeatures = new DenseTensor<float>([states.Count, 1611]);
            Memory<float> memoryBuffer = batchFeatures.Buffer;

            // Parallel Featurization (CPU bound)
            Parallel.For(0, states.Count, i =>
            {
                Span<float> destinationSlice = memoryBuffer.Span.Slice((int)i * 1611, 1611);

                ReadOnlySpan<float> featureVector = _stateVectorizer.Vectorize(states[(int)i], 0).ToArray();
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
            return ValuationToScore(StateValueFunc(state, 0));
        }

        protected float ValuationToScore(float[] valuation)
        {
            return valuation[PlayerIndex] / valuation.Sum();
        }

        protected static float[] StateValueFunc(GameState state, uint playedActionsCount)
        {
            var inputTensor = _stateVectorizer.Vectorize(state, playedActionsCount);
            return _stateValueNet.Run(inputTensor);
        }
    }
}
