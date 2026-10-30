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

    public class MCTSTree : ISearchTree
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

        /// <summary>
        /// Whether the search provides policy priors (set by MCTSSearch from its value net): nodes expanded before their
        /// priors arrive get uniform ones until then, and the opponent models that use every player's values select by PUCT
        /// </summary>
        public bool UsesPolicy { get; set; }

        /// <summary>
        /// How much lower than its parent's mean (normalized) an unvisited child counts in PUCT (first play urgency)
        /// </summary>
        public double FirstPlayReduction { get; set; } = 0.1;

        /// <summary>
        /// A leaf the search reaches for the first time is evaluated itself, and expanded only when it is reached again (with
        /// the priors of its evaluation). Otherwise every leaf is expanded at once and one of its moves is evaluated instead,
        /// picked with uniform priors, so a move that wasn't the first one tried under its parent is valued by a random reply
        /// (the old search: about 90 Elo weaker at 5000 iterations, 35 at 0.1 s per decision).
        /// </summary>
        public bool EvaluateNewLeaves { get; set; } = true;

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

        /// <summary>
        /// Player the tree searches for (whose result is TotalScore)
        /// </summary>
        public sbyte OwnerIndex => _ownerIdx;

        #region ISearchTree
        GameState ISearchTree.RootState => RootState;

        bool ISearchTree.RootHasPolicyPriors => RootNode.HasPolicyPriors;

        void ISearchTree.SetRootPriors(float[] priors) => ApplyPriors(RootNode, priors);

        SearchLeaf ISearchTree.SelectLeaf()
        {
            // Selection: a leaf down the tree by UCT or PUCT
            (MCTSNode leafNode, GameState stateAtLeaf) = SelectLeafNode();

            // Expansion: a leaf that isn't terminal gets its children, and selection goes one step further among them (with
            // EvaluateNewLeaves only a leaf that was visited before; a new one is evaluated itself)
            if (!leafNode.IsTerminal && (!EvaluateNewLeaves || leafNode.VisitCount > 0))
            {
                ExpandNode(leafNode, stateAtLeaf);
                (leafNode, stateAtLeaf) = SelectLeafNode(leafNode, stateAtLeaf, suppressStateCopy: true);
            }

            MarkPendingPath(leafNode);
            return new SearchLeaf(leafNode, stateAtLeaf);
        }

        bool ISearchTree.HasPolicyPriors(SearchLeaf leaf) => ((MCTSNode)leaf.Node).HasPolicyPriors;

        void ISearchTree.ApplyPriors(SearchLeaf leaf, float[] priors, List<Action> moves) => ApplyPriors((MCTSNode)leaf.Node, priors);

        void ISearchTree.Backpropagate(SearchLeaf leaf, ReadOnlySpan<double> utilities)
        {
            // The opponent models that use every player's values get all four utilities, Random only the owner's
            MCTSNode node = (MCTSNode)leaf.Node;
            if (Model == OpponentModel.Random) Backpropagate(node, utilities[_ownerIdx], pathAlreadyMarked: true);
            else Backpropagate(node, utilities, pathAlreadyMarked: true);
        }
        #endregion

        public Action GetMostPromisingMove()
        {
            // Return the child with the highest visit count
            MCTSNode bestChild = RootNode.Children.OrderByDescending(c => c.VisitCount).First();
            return HandOut(bestChild.LastAction!);
        }

        /// <summary>
        /// Samples a root move with probability proportional to visits^(1/temperature); temperature 0 is the most visited move
        /// </summary>
        public Action SampleMove(double temperature, Random random)
        {
            if (temperature <= 0) return GetMostPromisingMove();

            MCTSNode[] children = RootNode.ChildArray;
            uint maxVisits = children.Max(child => child.VisitCount);
            if (maxVisits == 0) return GetMostPromisingMove();

            // Relative to the most visited move, so the powers stay in range for low temperatures
            double[] weights = children.Select(child => Math.Pow((double)child.VisitCount / maxVisits, 1 / temperature)).ToArray();
            double remaining = random.NextDouble() * weights.Sum();
            for (int i = 0; i < children.Length; i++)
            {
                remaining -= weights[i];
                if (remaining < 0) return HandOut(children[i].LastAction!);
            }
            return HandOut(children.Where(child => child.VisitCount > 0).Last().LastAction!);
        }

        /// <summary>
        /// The root moves with their visit counts (copies without history, like GetMostPromisingMove)
        /// </summary>
        public IReadOnlyList<(Action move, uint visits)> RootVisits()
        {
            return RootNode.ChildArray.Select(child => (HandOut(child.LastAction!), child.VisitCount)).ToArray();
        }

        /// <summary>
        /// The root's visits without those that only exploration under noise gave a move (policy target pruning as in
        /// KataGo): every move but the most visited keeps the visits PUCT with the clean priors (in the order of the root's
        /// children) would have given it before its value fell below the most visited move's PUCT; moves left with at most
        /// one visit are dropped. Only for PUCT searches with every player's values (MaxN, Paranoid); otherwise the visits as
        /// they are.
        /// </summary>
        public uint[] PrunedRootVisits(ReadOnlySpan<float> cleanPriors)
        {
            MCTSNode[] children = RootNode.ChildArray;
            uint[] visits = children.Select(child => child.VisitCount).ToArray();
            if (!RootNode.HasPriors || PlayerBounds == null || cleanPriors.Length != children.Length || children.Length < 2)
                return visits;

            int decider = RootNode.DeciderIndex;
            bool invert = Model == OpponentModel.Paranoid && decider != _ownerIdx;
            int valuePlayer = Model == OpponentModel.MaxN ? decider : _ownerIdx;
            MinMaxStats.Range? range = PlayerBounds[valuePlayer].GetRange();
            double Mean(MCTSNode child)
            {
                double mean = child.PlayerScore(valuePlayer) / child.VisitCount;
                if (range != null) mean = range.Value.Normalize(mean);
                return invert ? 1 - mean : mean;
            }

            int best = Array.IndexOf(visits, visits.Max());
            double sqrtVisits = Math.Sqrt(Math.Max(RootNode.VisitCount, 1));
            double bestPuct = Mean(children[best]) + _explorationParameter * cleanPriors[best] * sqrtVisits / (1 + visits[best]);
            for (int i = 0; i < children.Length; i++)
            {
                if (i == best || visits[i] == 0) continue;
                double gap = bestPuct - Mean(children[i]);
                if (gap <= 0) continue;
                // PUCT keeps choosing the move while c * P * sqrt(N) / (1 + n) >= gap, i.e. for n up to this bound
                double bound = _explorationParameter * cleanPriors[i] * sqrtVisits / gap - 1;
                uint allowed = bound < 0 ? 0 : (uint)Math.Min(visits[i], Math.Floor(bound) + 1);
                visits[i] = allowed <= 1 ? 0 : allowed;
            }
            return visits;
        }

        /// <summary>
        /// Every player's mean result over the root's visits whose results are in, the search's value of the root state.
        /// Only the opponent models that use every player's values (MaxN, Paranoid) record them; 0 for an unvisited root.
        /// </summary>
        public double[] RootValues()
        {
            uint visits = RootNode.ResolvedVisitCount;
            return Enumerable.Range(0, MCTSNode.PlayerCount).Select(player => visits == 0 ? 0 : RootNode.PlayerScore(player) / visits).ToArray();
        }

        /// <summary>
        /// A copy of a tree action without history: the caller applies it to the real game (resolving output randomness anew),
        /// while the tree's own instance stays untouched
        /// </summary>
        private static Action HandOut(Action action)
        {
            Action move = action.Clone();
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

                // A move that wins at once is the only one worth searching
                List<Action> actions = GetValidActions(state);
                if (WinningMove(state, actions) is Action winning)
                    actions = [winning];
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

                // Priors from the policy head if the node was evaluated already, uniform until they arrive otherwise
                if (node.PendingPriors is float[] priors && priors.Length == children.Count)
                {
                    for (int i = 0; i < children.Count; i++) children[i].Prior = priors[i];
                    node.HasPriors = node.HasPolicyPriors = true;
                }
                else if (UsesPolicy && children.Count > 0)
                {
                    foreach (MCTSNode child in children) child.Prior = 1f / children.Count;
                    node.HasPriors = true;
                }
                node.PendingPriors = null;

                node.PublishChildren([.. children], HasDecidingPower(node.ActivePlayerIndex, children),
                    actions.Count > 0 ? actions[0].PlayerIndex : MCTSNode.UnknownPlayer);
            }
        }

        /// <summary>
        /// Sets the policy priors of a node's moves (in the order of GetValidActions): on its children if it is expanded
        /// already, otherwise for its expansion
        /// </summary>
        public void ApplyPriors(MCTSNode node, float[] priors)
        {
            lock (node.ChildLock)
            {
                if (!node.IsExpanded)
                {
                    node.PendingPriors = priors;
                    return;
                }
                MCTSNode[] children = node.ChildArray;
                if (children.Length != priors.Length) return;
                for (int i = 0; i < children.Length; i++) children[i].Prior = priors[i];
                node.HasPriors = node.HasPolicyPriors = true;
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
                if (pathAlreadyMarked)
                    currentNode.ResolvePendingVisit();

                // Keep the tree-wide value range up to date with the node's new mean value
                UpdateBounds(currentNode);

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
                if (pathAlreadyMarked)
                    currentNode.ResolvePendingVisit();

                UpdateBounds(currentNode);

                currentNode = currentNode.Parent;
            }
        }

        /// <summary>
        /// Widens the value ranges with the node's mean values. The means count only visits whose results are in: pending visits
        /// of other leaves in flight have no result yet and would pull the means (and so the minimums, which never recover)
        /// towards 0.
        /// </summary>
        private void UpdateBounds(MCTSNode node)
        {
            if (ValueBounds == null) return;

            uint visits = node.ResolvedVisitCount;
            if (visits == 0) return;

            ValueBounds.Update(node.TotalScore / visits);
            for (int player = 0; PlayerBounds != null && player < MCTSNode.PlayerCount; player++)
                PlayerBounds[player].Update(node.PlayerScore(player) / visits);
        }

        public void MarkPendingPath(MCTSNode node)
        {
            MCTSNode? currentNode = node;

            // Go up in tree until root is reached and count a visit at each passed node, whose result Backpropagate adds later
            while (currentNode != null)
            {
                currentNode.MarkPendingVisit();
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
            MinMaxStats.Range? range = PlayerBounds?[valuePlayer].GetRange();
            if (node.HasPriors)
                return SelectByPUCT(node, children, valuePlayer, invert, range);

            double logVisits = Math.Log(node.VisitCount);
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

        /// <summary>
        /// The child with the highest PUCT for the deciding player; unvisited children count as the node's own mean minus
        /// FirstPlayReduction
        /// </summary>
        private MCTSNode SelectByPUCT(MCTSNode node, MCTSNode[] children, int valuePlayer, bool invert, MinMaxStats.Range? range)
        {
            uint resolved = node.ResolvedVisitCount;
            double parentMean = resolved == 0 ? 0.5 : node.PlayerScore(valuePlayer) / resolved;
            if (range != null) parentMean = range.Value.Normalize(parentMean);
            if (invert) parentMean = 1 - parentMean;
            double firstPlayValue = parentMean - FirstPlayReduction;
            double sqrtVisits = Math.Sqrt(Math.Max(node.VisitCount, 1));

            MCTSNode chosen = children[0];
            double chosenValue = double.NegativeInfinity;
            foreach (MCTSNode child in children)
            {
                double value = child.GetPUCT(_explorationParameter, sqrtVisits, range, valuePlayer, invert, firstPlayValue) + _random.NextDouble() * 1e-5;
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

        /// <summary>
        /// A move that ends the game with a win for the player making it, or null. Only moves without a random outcome are
        /// tried, and only by a player at most 2 victory points short of winning (no single move gains more).
        /// </summary>
        public static Action? WinningMove(GameState state, IReadOnlyList<Action> actions)
        {
            if (actions.Count < 2) return null;
            sbyte mover = actions[0].PlayerIndex;
            if (state.Players[mover].VictoryPoints.Total < state.Settings.VictoryPoints - 2) return null;

            foreach (Action action in actions)
            {
                if (action is IOutputRandomnessAction || action.PlayerIndex != mover) continue;
                GameState after = new(state);
                action.Clone().Apply(after);
                if (after.HasEnded && after.Turn.PlayerIndex == mover) return action;
            }
            return null;
        }

        public static List<Action> GetValidActions(GameState state)
        {
            sbyte activePlayerIdx = (sbyte)state.Turn.PlayerIndex;

            // An open trade between players (only with free trade; agents decide those outside the tree): its answers or picks
            if (state.Turn.PendingTrade is TradeOffer offer)
                return LegalActionProvider.GetActionsForState(state, offer.NextResponder ?? offer.Offerer);

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
