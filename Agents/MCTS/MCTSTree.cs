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

        public MCTSTree(GameState rootState, sbyte ownerIdx, double explorationParameter = 1.414)
        {
            _rootState = rootState;
            _rootNode = new MCTSNode((sbyte)rootState.Turn.PlayerIndex, null, null, false);
            _explorationParameter = explorationParameter;
            _ownerIdx = ownerIdx;
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
                if (node.LastAction != null)
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
                    // Check if node is terminal
                    action.Apply(state);
                    bool isTerminal = state.HasEnded;
                    sbyte nextActivePlayerIndex = (sbyte)state.Turn.PlayerIndex;
                    action.Revert(state);

                    MCTSNode childNode = new MCTSNode(nextActivePlayerIndex, node, action, isTerminal);
                    node.Children.Add(childNode);
                }
            }

            // Simulation
            // Select a child to simulate
            MCTSNode nodeToSimulate;
            if(!node.IsTerminal)
            {
                nodeToSimulate = SelectChild(node);
                nodeToSimulate.LastAction!.Apply(state);
            }
            else
            {
                nodeToSimulate = node;
            }

            // Run simulation and get result (1 for win, 0 for loss)
            int result = Simulate(state);

            // Backpropagation
            // Update the win and visit counts for all nodes in the path from the simulated node back to the root
            Backpropagate(nodeToSimulate, result);
        }

        protected MCTSNode SelectChild(MCTSNode node)
        {
            // Determine if the active player this tree is modeling currently has deciding power over the next action
            bool isOwnersTurn = node.ActivePlayerIndex == _ownerIdx;
            bool ownerHasToDiscard = node.Children.Any(x => x.LastAction! is DiscardAction discardAction && discardAction.PlayerIndex == _ownerIdx);
            bool otherHasToDiscard = node.Children.Any(x => x.LastAction! is DiscardAction discardAction && discardAction.PlayerIndex != _ownerIdx);
            bool hasDecidingPower = (isOwnersTurn || ownerHasToDiscard) && !otherHasToDiscard;

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

            // Always assume the active player has to discard first
            if (DiscardAction.IsTurnValid(state.Turn, activePlayerIdx))
                return DiscardAction.GetActionsForState(state, activePlayerIdx);

            // Only show other players' discard actions if the active player is done
            List<Action> otherPlayersDiscardActions = new(); 
            for (int playerToDiscardIdx = 0; playerToDiscardIdx < state.Players.Length; playerToDiscardIdx++)
            {
                if (playerToDiscardIdx == activePlayerIdx)
                    continue;

                if (DiscardAction.IsTurnValid(state.Turn, playerToDiscardIdx))
                    otherPlayersDiscardActions.AddRange(DiscardAction.GetActionsForState(state, (sbyte)playerToDiscardIdx));
            }

            if (otherPlayersDiscardActions.Count > 0)
                return otherPlayersDiscardActions;


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

        protected int Simulate(GameState state)
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

        protected void Backpropagate(MCTSNode node, int result)
        {
            MCTSNode? currentNode = node;

            while (currentNode != null)
            {
                currentNode.VisitCount++;

                if (result == 1)
                {
                    currentNode.WinCount++;
                }

                currentNode = currentNode.Parent;
            }
        }
    }
}
