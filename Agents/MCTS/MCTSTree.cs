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
    /// <summary>
    /// How the search picks the moves of the other players:
    /// Random samples them uniformly (the owner's moves by UCT on its own value);
    /// MaxN picks every player's moves by UCT on that player's own value, from the value net's estimates for all players;
    /// Paranoid has the others pick the moves that are worst for the owner.
    /// MaxN is the agent default: 8:4 against Random in a duel with the same net (0.25 s per decision), at the same search speed.
    /// </summary>
    public enum OpponentModel { Random, MaxN, Paranoid }

    public class MCTSTree
    {
        public GameState RootState;
        public MCTSNode RootNode;
        protected sbyte _ownerIdx;
        protected double _explorationParameter;
        protected Random _random = Random.Shared;

        // Running value range of the tree, used to normalize Q values during selection (null disables normalization)
        public MinMaxStats? ValueBounds { get; private set; }

        // One value range per player for the opponent models that use every player's values (null without normalization)
        public MinMaxStats[]? PlayerBounds { get; private set; }

        public OpponentModel Model { get; }

        public MCTSTree(GameState rootState, sbyte ownerIdx, double explorationParameter, bool normalizeValues = true, OpponentModel model = OpponentModel.Random)
        {
            RootState = rootState;
            RootNode = new MCTSNode((sbyte)rootState.Turn.PlayerIndex, null, null, false);
            _ownerIdx = ownerIdx;
            _explorationParameter = explorationParameter;
            Model = model;
            ValueBounds = normalizeValues ? new MinMaxStats() : null;
            PlayerBounds = normalizeValues && model != OpponentModel.Random ? NewPlayerBounds() : null;
        }

        private static MinMaxStats[] NewPlayerBounds()
        {
            return Enumerable.Range(0, MCTSNode.PlayerCount).Select(_ => new MinMaxStats()).ToArray();
        }

        public Action GetMostPromisingMove()
        {
            // Return the child with the highest visit count
            MCTSNode bestChild = RootNode.Children.OrderByDescending(c => c.VisitCount).First();

            // Hand out a copy without history: the caller applies it to the real game (resolving output randomness anew),
            // while the tree's own instance stays untouched
            Action move = bestChild.LastAction!.Clone();
            if (move is IReplayAction replayMove)
                replayMove.ClearHistory();

            return move;
        }

        /// <summary>
        /// Finds the node the tree arrives at when the given, already applied action is played at the root:
        /// the matching decision child, or for output randomness actions the outcome child with an equivalent outcome.
        /// Returns null if the root was never expanded or no child matches.
        /// </summary>
        public MCTSNode? FindChildForAction(Action action)
        {
            foreach (MCTSNode child in RootNode.Children)
            {
                if (child.LastAction == null || !child.LastAction.IsSameDecision(action)) continue;

                if (!child.IsOutputRandomnessGroup) return child;

                // Outcomes are matched by equivalence, e.g. a roll of (6, 1) arrives at the outcome represented by (1, 6)
                return child.Children.FirstOrDefault(outcome => outcome.LastAction is IOutputRandomnessAction randomOutcome && randomOutcome.HasSameOutcome(action));
            }

            return null;
        }

        /// <summary>
        /// Makes a node below the root the new root, keeping the statistics of its subtree and discarding everything else.
        /// The new root state is the real game state after the move. It may differ from the state the subtree was explored
        /// with in details that do not affect the game (e.g. the dice pair behind a roll total).
        /// </summary>
        public void ReRoot(MCTSNode newRoot, GameState newRootState)
        {
            newRoot.DetachFromParent();
            RootNode = newRoot;
            RootState = newRootState;

            // Rebuild the value ranges from the kept subtree, so values of discarded branches no longer stretch them
            if (ValueBounds != null)
            {
                ValueBounds = new MinMaxStats();
                PlayerBounds = PlayerBounds != null ? NewPlayerBounds() : null;

                Stack<MCTSNode> pending = new([newRoot]);
                while (pending.TryPop(out MCTSNode? node))
                {
                    uint visits = node.VisitCount;
                    if (visits > 0)
                    {
                        ValueBounds.Update(node.TotalScore / visits);
                        for (int player = 0; PlayerBounds != null && player < MCTSNode.PlayerCount; player++)
                            PlayerBounds[player].Update(node.PlayerScore(player) / visits);
                    }

                    foreach (MCTSNode child in node.Children)
                        pending.Push(child);
                }
            }
        }

        public int GetTreeSize()
        {
            return GetTreeSizeRec(RootNode);
        }

        protected int GetTreeSizeRec(MCTSNode currentSubtreeRoot)
        {
            if (currentSubtreeRoot.IsTerminal || !currentSubtreeRoot.IsExpanded)
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

            while (node.IsExpanded)
            {
                node = SelectChild(node);

                // Don't execute empty actions and actions leading into a randomness group (since their direct children are variations of that same action)
                // Apply a copy: Apply rewrites the action's history, and the tree's instance is shared by all threads and iterations
                if (node.LastAction != null && !node.IsOutputRandomnessGroup)
                {
                    node.LastAction!.Clone().Apply(state);

                    // Whether a child ends the game is only known once its action was applied
                    if (state.HasEnded) node.MarkTerminal();
                }
            }

            return (node, state);
        }

        public void ExpandNode(MCTSNode node, GameState state)
        {
            // Guard against double expansion
            // If this node was already concurrently expanded by another thread, return early
            lock(node.ChildLock)
            {
                if(node.IsExpanded)
                {
                    return;
                }

                // Ended games have no actions
                if (state.HasEnded)
                {
                    node.MarkTerminal();
                    return;
                }

                // Children are created without applying their actions: whether they end the game and who acts next is
                // recorded when selection first applies them (see SelectLeafNode) and when they are expanded themselves
                node.ActivePlayerIndex = (sbyte)state.Turn.PlayerIndex;

                List<Action> actions = GetValidActions(state);
                List<MCTSNode> children = new(actions.Count);
                foreach (var action in actions)
                {
                    // If action has output randomness, get all possible outcomes and group them into a subtree
                    if (action is IOutputRandomnessAction randomAction)
                    {
                        MCTSNode groupNode = new MCTSNode((sbyte)state.Turn.PlayerIndex, node, action, false, true);
                        children.Add(groupNode);

                        List<OutcomeVariant> variants = randomAction.GetOutcomeVariants(state, (sbyte)state.Turn.PlayerIndex);
                        MCTSNode[] outcomes = new MCTSNode[variants.Count];

                        for (int i = 0; i < variants.Count; i++)
                        {
                            (Action variant, double probability) = variants[i];

                            MCTSNode childNode = new MCTSNode(MCTSNode.UnknownPlayer, groupNode, variant)
                            {
                                OutcomeProbability = probability
                            };
                            outcomes[i] = childNode;
                        }

                        // Outcomes are sampled, never chosen
                        groupNode.PublishChildren(outcomes, ownerDecides: false, MCTSNode.UnknownPlayer);
                    }
                    // Otherwise just add the node directly as a child
                    else
                    {
                        MCTSNode childNode = new MCTSNode(MCTSNode.UnknownPlayer, node, action);
                        children.Add(childNode);
                    }
                }

                node.PublishChildren([.. children], HasDecidingPower(node.ActivePlayerIndex, children),
                    actions.Count > 0 ? actions[0].PlayerIndex : MCTSNode.UnknownPlayer);
            }
        }

        public void Backpropagate(MCTSNode node, double result, bool pathAlreadyMarked = false)
        {
            MCTSNode? currentNode = node;

            // Go up in tree until root is reached and add new result to each passed node
            while (currentNode != null)
            {
                // Increment visit count, unless it was already increased by MarkPendingPath before
                if (!pathAlreadyMarked)
                    currentNode.IncrementVisitCount();

                currentNode.AddResult(result);

                // Keep the tree-wide value range up to date with the node's new mean value
                ValueBounds?.Update(currentNode.TotalScore / currentNode.VisitCount);

                currentNode = currentNode.Parent;
            }
        }

        /// <summary>
        /// Backpropagation of a result per player (the opponent models that use every player's values); the owner's result
        /// also goes into TotalScore
        /// </summary>
        public void Backpropagate(MCTSNode node, ReadOnlySpan<double> results, bool pathAlreadyMarked = false)
        {
            MCTSNode? currentNode = node;
            double ownerResult = results[_ownerIdx];

            while (currentNode != null)
            {
                if (!pathAlreadyMarked)
                    currentNode.IncrementVisitCount();

                currentNode.AddResults(results);
                currentNode.AddResult(ownerResult);

                uint visits = currentNode.VisitCount;
                ValueBounds?.Update(currentNode.TotalScore / visits);
                for (int player = 0; PlayerBounds != null && player < MCTSNode.PlayerCount; player++)
                    PlayerBounds[player].Update(currentNode.PlayerScore(player) / visits);

                currentNode = currentNode.Parent;
            }
        }

        public void MarkPendingPath(MCTSNode node)
        {
            MCTSNode? currentNode = node;

            // Go up in tree until root is reached and add new result to each passed node
            while (currentNode != null)
            {
                currentNode.IncrementVisitCount();
                currentNode = currentNode.Parent;
            }
        }

        public MCTSNode SelectChild(MCTSNode node)
        {
            // Children are published once, complete, so no lock is needed to read them
            MCTSNode[] children = node.ChildArray;

            // If the node is a grouped output randomness action, sample an outcome according to its probability
            if (node.IsOutputRandomnessGroup)
            {
                return SampleOutcome(children);
            }

            if (Model == OpponentModel.Random)
            {
                // Select a random move if the player this agent is representing has no deciding power over the next action
                // This is to avoid the AI from making intentional bad moves on other players' turns
                if (!node.OwnerDecides)
                {
                    return children[_random.Next(children.Length)];
                }

                // Only select best move if the player has deciding power, in one pass with values shared by all children
                double logParentVisits = Math.Log(node.VisitCount);
                MinMaxStats.Range? valueRange = ValueBounds?.GetRange();

                MCTSNode best = children[0];
                double bestValue = double.NegativeInfinity;
                foreach (MCTSNode child in children)
                {
                    // Slight random jitter to break ties (especially important for unexplored nodes)
                    double value = child.GetUCT(_explorationParameter, logParentVisits, valueRange) + _random.NextDouble() * 1e-5;
                    if (value > bestValue)
                    {
                        best = child;
                        bestValue = value;
                    }
                }

                return best;
            }

            // Every decision by UCT: on the deciding player's own value (MaxN), or for the others on the owner's value
            // turned around (Paranoid)
            int decider = node.DeciderIndex;
            int valuePlayer = Model == OpponentModel.MaxN ? decider : _ownerIdx;
            bool invert = Model == OpponentModel.Paranoid && decider != _ownerIdx;
            double logVisits = Math.Log(node.VisitCount);
            MinMaxStats.Range? range = PlayerBounds?[valuePlayer].GetRange();

            MCTSNode chosen = children[0];
            double chosenValue = double.NegativeInfinity;
            foreach (MCTSNode child in children)
            {
                double value = child.GetUCT(_explorationParameter, logVisits, range, valuePlayer, invert) + _random.NextDouble() * 1e-5;
                if (value > chosenValue)
                {
                    chosen = child;
                    chosenValue = value;
                }
            }

            return chosen;
        }

        private MCTSNode SampleOutcome(MCTSNode[] outcomes)
        {
            double total = 0;
            foreach (MCTSNode outcome in outcomes)
                total += outcome.OutcomeProbability;
            double remaining = _random.NextDouble() * total;

            foreach (MCTSNode outcome in outcomes)
            {
                remaining -= outcome.OutcomeProbability;
                if (remaining < 0) return outcome;
            }

            // Floating point leftovers
            return outcomes[^1];
        }

        private bool HasDecidingPower(sbyte activePlayerIdx, List<MCTSNode> children)
        {
            // A player has deciding power if one of the following is true:
            // - It's the player's turn and nobody has to discard
            // - The player has to discard and nobody with a lower playerIndex has to discard
            bool isOwnersTurn = activePlayerIdx == _ownerIdx;
            bool discardRequired = children.Any(x => x.LastAction != null && x.LastAction is DiscardAction);
            bool ownerHasToDiscard = children.Any(x => x.LastAction != null && x.LastAction is DiscardAction discardAction && discardAction.PlayerIndex == _ownerIdx);
            bool previousHasToDiscard = children.Any(x => x.LastAction != null && x.LastAction is DiscardAction discardAction && discardAction.PlayerIndex < _ownerIdx);

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
