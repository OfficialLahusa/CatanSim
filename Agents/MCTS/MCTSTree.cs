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
using System.Xml.Linq;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    public class MCTSTree
    {
        public GameState RootState;
        public MCTSNode RootNode;
        protected sbyte _ownerIdx;
        protected double _explorationParameter;
        protected Random _random = new Random();

        public MCTSTree(GameState rootState, sbyte ownerIdx, double explorationParameter)
        {
            RootState = rootState;
            RootNode = new MCTSNode((sbyte)rootState.Turn.PlayerIndex, null, null, false);
            _ownerIdx = ownerIdx;
            _explorationParameter = explorationParameter;
        }

        public Action GetMostPromisingMove()
        {
            // Return the child with the highest visit count
            MCTSNode bestChild = RootNode.Children.OrderByDescending(c => c.VisitCount).First();

            return bestChild.LastAction!;
        }

        public int GetTreeSize()
        {
            return GetTreeSizeRec(RootNode);
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

        public (MCTSNode leafNode, GameState stateAtNode) SelectLeafNode(MCTSNode? subtreeRoot = null, GameState? subtreeRootState = null, bool suppressStateCopy = false)
        {
            // If no subtree was specified, start from root of entire tree
            MCTSNode node = subtreeRoot ?? RootNode;
            // (Optimization) Suppress GameState deep copy if flag was set. This directly mutates the given subtree root state
            GameState state = suppressStateCopy
                ? subtreeRootState ?? new GameState(RootState)
                : new GameState(subtreeRootState ?? RootState);

            while (node.Children.Count > 0)
            {
                node = SelectChild(node);

                // Don't execute empty actions and actions leading into a randomness group (since their direct children are variations of that same action)
                if (node.LastAction != null && !node.IsOutputRandomnessGroup)
                {
                    node.LastAction!.Apply(state);
                }
            }

            return (node, state);
        }

        public void ExpandNode(MCTSNode node, GameState state)
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

        public void Backpropagate(MCTSNode node, double result)
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

        public MCTSNode SelectChild(MCTSNode node)
        {
            // If the node is a grouped output randomness action, choose a child node with uniform random distribution
            if (node.IsOutputRandomnessGroup)
            {
                return node.Children[_random.Next(node.Children.Count)];
            }

            // Otherwise, determine if the player this agent is representing currently has deciding power over the next action            
            // Only select best move if the player has deciding power
            if (HasDecidingPower(node))
            {
                return node.Children.OrderByDescending(c => c.GetUCT(_explorationParameter)).First();
            }
            // Otherwise select random move
            // This is to avoid the AI from making intentional bad moves on other players' turns
            else
            {
                return node.Children[_random.Next(node.Children.Count)];
            }
        }

        private bool HasDecidingPower(MCTSNode node)
        {
            // A player has deciding power if one of the following is true:
            // - It's the player's turn and nobody has to discard
            // - The player has to discard and nobody with a lower playerIndex has to discard
            bool isOwnersTurn = node.ActivePlayerIndex == _ownerIdx;
            bool discardRequired = node.Children.Any(x => x.LastAction != null && x.LastAction is DiscardAction);
            bool ownerHasToDiscard = node.Children.Any(x => x.LastAction != null && x.LastAction is DiscardAction discardAction && discardAction.PlayerIndex == _ownerIdx);
            bool previousHasToDiscard = node.Children.Any(x => x.LastAction != null && x.LastAction is DiscardAction discardAction && discardAction.PlayerIndex < _ownerIdx);

            bool hasDecidingPower = isOwnersTurn && !discardRequired || ownerHasToDiscard && !previousHasToDiscard;
            return hasDecidingPower;
        }

        public static List<Action> GetValidActions(GameState state)
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
    }
}
