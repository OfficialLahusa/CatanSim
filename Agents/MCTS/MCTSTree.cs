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
    public class MCTSTree
    {
        protected GameState _rootState;
        protected MCTSNode _rootNode;
        protected double _explorationParameter;
        protected sbyte _ownerIdx;
        protected Random _random = new Random();

        protected static StateValueNet _stateValueNet;
        protected static StateVectorizer _stateVectorizer;

        public MCTSTree(GameState rootState, sbyte ownerIdx, double explorationParameter = 1.414)
        {
            _rootState = rootState;
            _rootNode = new MCTSNode((sbyte)rootState.Turn.PlayerIndex, null, null, false);
            _explorationParameter = explorationParameter;
            _ownerIdx = ownerIdx;
        }

        static MCTSTree() {
            _stateValueNet = new StateValueNet("state_value_net-0mw2c1bh.onnx");
            _stateVectorizer = new StateVectorizer();
        }

        public Action GetMostPromisingMove()
        {
            // Return the child with the highest visit count
            MCTSNode bestChild = _rootNode.Children.OrderByDescending(c => c.VisitCount).First();

            return bestChild.LastAction!;
        }

        public void RunIterations(int iterations)
        {
            Console.WriteLine($"Running {iterations} MCTS iterations:");
            Stopwatch stopwatch = Stopwatch.StartNew();

            for (int i = 0; i < iterations; i++)
            {
                RunOneIteration();
                //Console.WriteLine($"Completed {i+1}/{iterations}.");
            }

            TimeSpan runtime = stopwatch.Elapsed;

            Console.WriteLine($"Simulation took {runtime.TotalMilliseconds} ms");
        }

        public int GetTreeSize()
        {
            return GetTreeSizeRec(_rootNode);
        }

        protected int GetTreeSizeRec(MCTSNode currentSubtreeRoot)
        {
            if (currentSubtreeRoot.IsTerminal || currentSubtreeRoot.Children.Count == 0)
            {
                return 1;
            }

            int sum = 0;
            foreach (MCTSNode child in currentSubtreeRoot.Children)
            {
                sum += GetTreeSizeRec(child);
            }
            return sum;
        }

        protected void RunOneIteration()
        {
            // Selection
            // Find a leaf node to expand, starting from the root node and traversing down the tree using the UCT formula
            MCTSNode node = _rootNode;
            GameState state = new GameState(_rootState);

            while (node.Children.Count > 0)
            {
                node = SelectChild(node);

                // Don't execute empty actions and actions leading into a randomness group (since their direct children are variations of that same action)
                if (node.LastAction != null && !node.IsOutputRandomnessGroup)
                {
                    node.LastAction!.Apply(state);
                }
            }

            // Expansion
            // Add all children to leaf node if node is not terminal
            if (!node.IsTerminal)
            {
                List<Action> actions = GetValidActions(state);
                foreach (var action in actions)
                {
                    // If action has output randomness, get all possible outcomes and group them into a subtree
                    if (action is IOutputRandomnessAction randomAction)
                    {
                        MCTSNode groupNode = new MCTSNode((sbyte)state.Turn.PlayerIndex, node, action, false, true);
                        node.Children.Add(groupNode);

                        List<Action> variants = randomAction.GetOutcomeVariants(state, (sbyte)state.Turn.PlayerIndex);

                        foreach (Action variant in variants)
                        {
                            // Check if node is terminal and who the active player is after the action
                            variant.Apply(state);
                            bool isTerminal = state.HasEnded;
                            sbyte nextActivePlayerIndex = (sbyte)state.Turn.PlayerIndex;
                            variant.Revert(state);

                            MCTSNode childNode = new MCTSNode(nextActivePlayerIndex, groupNode, variant, isTerminal);
                            groupNode.Children.Add(childNode);
                        }
                    }
                    // Otherwise just add the node directly as a child
                    else
                    {
                        // Check if node is terminal and who the active player is after the action
                        action.Apply(state);
                        bool isTerminal = state.HasEnded;
                        sbyte nextActivePlayerIndex = (sbyte)state.Turn.PlayerIndex;
                        action.Revert(state);

                        MCTSNode childNode = new MCTSNode(nextActivePlayerIndex, node, action, isTerminal);
                        node.Children.Add(childNode);
                    }
                }
            }

            // Simulation
            // Select a child to simulate
            MCTSNode nodeToSimulate;
            if(!node.IsTerminal)
            {
                nodeToSimulate = SelectChild(node);

                // Grouped output randomness node => Select child of child
                if (nodeToSimulate.IsOutputRandomnessGroup)
                {
                    nodeToSimulate = SelectChild(nodeToSimulate);
                    nodeToSimulate.LastAction!.Apply(state);
                }
                // Normal node => Select child
                else
                {
                    nodeToSimulate.LastAction!.Apply(state);
                }
            }
            else
            {
                nodeToSimulate = node;
            }

            // Run simulation and get result (win percentage)
            double result = SimulateML(state);

            // Backpropagation
            // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root
            Backpropagate(nodeToSimulate, result);
        }

        protected MCTSNode SelectChild(MCTSNode node)
        {
            // If the node is a grouped output randomness action, choose a child node with uniform random distribution
            if (node.IsOutputRandomnessGroup)
            {
                return node.Children[_random.Next(node.Children.Count)];
            }

            // Otherwise, determine if the active player this tree is modeling currently has deciding power over the next action
            // A player has deciding power if one of the following is true:
            // - It's the player's turn and nobody has to discard
            // - The player has to discard and nobody with a lower playerIndex has to discard
            bool isOwnersTurn = node.ActivePlayerIndex == _ownerIdx;
            bool discardRequired      = node.Children.Any(x => x.LastAction != null && x.LastAction is DiscardAction);
            bool ownerHasToDiscard    = node.Children.Any(x => x.LastAction != null && x.LastAction is DiscardAction discardAction && discardAction.PlayerIndex == _ownerIdx);
            bool previousHasToDiscard = node.Children.Any(x => x.LastAction != null && x.LastAction is DiscardAction discardAction && discardAction.PlayerIndex < _ownerIdx);
            bool hasDecidingPower = isOwnersTurn && !discardRequired || ownerHasToDiscard && !previousHasToDiscard;

            // Only select best move if the player has deciding power, otherwise select random move.
            // This is to avoid the AI from making bad moves on other players' turns.
            if (hasDecidingPower)
            {
                return node.Children.OrderByDescending(c => c.GetUCT(_explorationParameter)).First();
            }
            else
            {
                return node.Children[_random.Next(node.Children.Count)];
            }
        }

        protected List<Action> GetValidActions(GameState state)
        {
            sbyte activePlayerIdx = (sbyte)state.Turn.PlayerIndex;

            List<Action> firstInitialSettlementActions = FirstInitialSettlementAction.GetActionsForState(state, activePlayerIdx);
            if (firstInitialSettlementActions.Count > 0)
                return firstInitialSettlementActions;

            List<Action> secondInitialSettlementActions = SecondInitialSettlementAction.GetActionsForState(state, activePlayerIdx);
            if (secondInitialSettlementActions.Count > 0)
                return secondInitialSettlementActions;

            // Discards are assumed to be in player index order, since the outcome is order-invariant
            // Therefore only the discards of the next player in order are shown
            for (int playerToDiscardIdx = 0; playerToDiscardIdx < state.Players.Length; playerToDiscardIdx++)
            {
                if (DiscardAction.IsTurnValid(state.Turn, playerToDiscardIdx))
                {
                    List<Action> discardActions = DiscardAction.GetActionsForState(state, (sbyte)playerToDiscardIdx);

                    if (discardActions.Count > 0)
                        return discardActions;
                }
            }

            List<Action> actions = [
                .. EndTurnAction.GetActionsForState(state, activePlayerIdx),

                .. RollAction.GetActionsForState(state, activePlayerIdx),
                .. RobberAction.GetActionsForState(state, activePlayerIdx),

                .. FirstInitialRoadAction.GetActionsForState(state, activePlayerIdx),
                .. SecondInitialRoadAction.GetActionsForState(state, activePlayerIdx),

                .. RoadAction.GetActionsForState(state, activePlayerIdx),
                .. SettlementAction.GetActionsForState(state, activePlayerIdx),
                .. CityAction.GetActionsForState(state, activePlayerIdx),
                .. BuyDevelopmentCardAction.GetActionsForState(state, activePlayerIdx),

                .. KnightAction.GetActionsForState(state, activePlayerIdx),
                .. MonopolyAction.GetActionsForState(state, activePlayerIdx),
                .. RoadBuildingAction.GetActionsForState(state, activePlayerIdx),
                .. YearOfPlentyAction.GetActionsForState(state, activePlayerIdx),

                .. FourToOneTradeAction.GetActionsForState(state, activePlayerIdx),
                .. ThreeToOneTradeAction.GetActionsForState(state, activePlayerIdx),
                .. TwoToOneTradeAction.GetActionsForState(state, activePlayerIdx)
            ];

            return actions;
        }

        protected double SimulatePlayout(GameState state)
        {
            //Console.WriteLine("Sim state hash: " + state.GetHashCode());

            while (!state.HasEnded)
            {
                List<Action> actions = GetValidActions(state);
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
            return state.Turn.PlayerIndex == _ownerIdx ? 1 : 0;
        }

        protected double SimulateML(GameState state)
        {
            return ValuationToScore(StateValueFunc(state, 0));
        }

        protected float ValuationToScore(float[] valuation)
        {
            return valuation[_ownerIdx] / valuation.Sum();
        }

        protected static float[] StateValueFunc(GameState state, uint playedActionsCount)
        {
            var inputTensor = _stateVectorizer.Vectorize(state, playedActionsCount);
            return _stateValueNet.Run(inputTensor);
        }

        protected void Backpropagate(MCTSNode node, double result)
        {
            MCTSNode? currentNode = node;

            // Go up in tree until root is reached and add new result to each passed node
            while (currentNode != null)
            {
                currentNode.VisitCount++;
                currentNode.TotalScore += result;
                currentNode = currentNode.Parent;
            }
        }
    }
}
