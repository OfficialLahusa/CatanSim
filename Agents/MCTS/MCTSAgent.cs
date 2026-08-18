using Agents.Inference;
using Common;
using Common.Actions;
using Common.Serialization;
using System;
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
        protected MCTSTree _tree;
        protected double _explorationParameter;
        protected Random _random = new Random();

        protected static StateValueNet _stateValueNet;
        protected static StateVectorizer _stateVectorizer;
        

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
                RunOneIteration();
                iterationCounter++;
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
