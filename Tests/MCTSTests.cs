using Agents;
using Agents.MCTS;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The MCTS tree (without the value net) and the MCTSAgent (with the value net, skipped when it is missing)
    /// </summary>
    public class MCTSTests
    {
        // ================================================================ Value normalization

        [Fact]
        public void MinMaxStatsNormalizesToTheSeenRange()
        {
            MinMaxStats stats = new();
            Assert.Equal(0.3, stats.Normalize(0.3));

            stats.Update(0.4);
            Assert.Equal(0.3, stats.Normalize(0.3));

            stats.Update(0.2);
            stats.Update(0.6);
            Assert.Equal(0.5, stats.Normalize(0.4), 12);
            Assert.Equal(0.0, stats.Normalize(0.2), 12);
            Assert.Equal(1.0, stats.Normalize(0.6), 12);
        }

        [Fact]
        public void UctPrefersUnvisitedChildrenAndUsesNormalizedValues()
        {
            MCTSNode root = new(0);
            MCTSNode visited = new(0, root);
            MCTSNode unvisited = new(0, root);
            for (int i = 0; i < 10; i++) root.IncrementVisitCount();
            for (int i = 0; i < 5; i++) visited.IncrementVisitCount();
            visited.AddResult(0.05);

            Assert.Equal(1e9, unvisited.GetUCT(1.414));

            MinMaxStats bounds = new();
            bounds.Update(0.0);
            bounds.Update(0.02);
            double exploration = 1.414 * Math.Sqrt(Math.Log(10) / 5);
            Assert.Equal(0.01 + exploration, visited.GetUCT(1.414), 12);
            Assert.Equal(0.5 + exploration, visited.GetUCT(1.414, bounds), 12);
        }

        // ================================================================ Tree structure and chance nodes

        [Fact]
        public void RollGroupsHaveElevenOutcomesSampledWithTheDiceDistribution()
        {
            TestUtil.Seed(140);
            GameState state = PreRollState();
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);
            tree.ExpandNode(tree.RootNode, new GameState(state));

            MCTSNode group = tree.RootNode.Children.Single(child => child.IsOutputRandomnessGroup && child.LastAction is RollAction);
            Assert.Equal(11, group.Children.Count);

            const int Samples = 60_000;
            Dictionary<int, int> counts = [];
            for (int i = 0; i < Samples; i++)
            {
                int total = ((RollAction)tree.SelectChild(group).LastAction!).History!.RollResult.Total;
                counts[total] = counts.GetValueOrDefault(total) + 1;
            }

            for (int total = 2; total <= 12; total++)
                Assert.InRange(counts.GetValueOrDefault(total) / (double)Samples, (6 - Math.Abs(total - 7)) / 36.0 - 0.01, (6 - Math.Abs(total - 7)) / 36.0 + 0.01);
        }

        [Fact]
        public void PlayedActionsAreMatchedToTheirChildIncludingEquivalentRolls()
        {
            TestUtil.Seed(141);
            GameState state = PreRollState();
            sbyte roller = (sbyte)state.Turn.PlayerIndex;
            MCTSTree tree = new(new GameState(state), roller, 1.414);
            tree.ExpandNode(tree.RootNode, new GameState(state));

            for (byte first = 1; first <= 6; first++)
            {
                for (byte second = 1; second <= 6; second++)
                {
                    RollAction played = new(roller);
                    ActionTests.SetHistory(played, new RollAction.RollActionHistory(new RollResult { First = first, Second = second }, state.Turn.LastRoll));

                    MCTSNode? child = tree.FindChildForAction(played);
                    Assert.NotNull(child);
                    Assert.Equal(first + second, ((RollAction)child!.LastAction!).History!.RollResult.Total);
                }
            }

            // An action that is not among the children is not matched
            Assert.Null(tree.FindChildForAction(new EndTurnAction(roller)));
        }

        [Fact]
        public void ReRootingKeepsTheSubtreeAndDetachesIt()
        {
            TestUtil.Seed(142);
            GameState state = PreRollState();
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);
            RunFakeIterations(tree, 400);

            MCTSNode group = tree.RootNode.Children.Single(child => child.IsOutputRandomnessGroup);
            MCTSNode outcome = group.Children.OrderByDescending(child => child.VisitCount).First();
            uint visits = outcome.VisitCount;
            GameState next = new GameState(state);
            outcome.LastAction!.Clone().Apply(next);

            tree.ReRoot(outcome, next);

            Assert.Same(outcome, tree.RootNode);
            Assert.Null(tree.RootNode.Parent);
            Assert.Equal(visits, tree.RootNode.VisitCount);
            Assert.Same(next, tree.RootState);
        }

        [Fact]
        public void PendingVisitsDoNotWidenTheValueRange()
        {
            TestUtil.Seed(143);
            GameState state = TestUtil.EmptyNormalTurn(0);
            TestUtil.GiveCards(state, 0, ResourceCardType.Grain, 4);
            MCTSTree tree = new(new GameState(state), 0, 1.414);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            MCTSNode[] children = tree.RootNode.Children.ToArray();

            // Ten leaves in flight below the first child, then one result for the second: only that result counts for the means
            for (int i = 0; i < 10; i++)
                tree.MarkPendingPath(children[0]);
            tree.MarkPendingPath(children[1]);
            tree.Backpropagate(children[1], 0.5, pathAlreadyMarked: true);

            Assert.Equal(11u, tree.RootNode.VisitCount);
            Assert.Equal(1u, tree.RootNode.ResolvedVisitCount);
            Assert.Equal(new MinMaxStats.Range(0.5, 0.5), tree.ValueBounds!.GetRange());
        }

        [Fact]
        public void SearchNeverChangesTheTreesOwnActions()
        {
            // Regression test for shared tree actions being re-applied (and their history rewritten) by every iteration
            TestUtil.Seed(143);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && s.Players[s.Turn.PlayerIndex].ResourceCards.Count() >= 4));
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);

            RunFakeIterations(tree, 500);
            List<(Action action, object? history)> snapshot = AllNodes(tree.RootNode)
                .Where(node => node.LastAction != null)
                .Select(node => (node.LastAction!, node.LastAction!.GetType().GetProperty("History")?.GetValue(node.LastAction)))
                .ToList();

            RunFakeIterations(tree, 2000, parallel: true);

            Assert.True(snapshot.Count > 100);
            Assert.All(snapshot, entry => Assert.Same(entry.history, entry.action.GetType().GetProperty("History")?.GetValue(entry.action)));
        }

        [Fact]
        public void MostPromisingMoveIsACopyWithoutHistory()
        {
            TestUtil.Seed(144);
            GameState state = PreRollState();
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414);
            RunFakeIterations(tree, 300);

            Action move = tree.GetMostPromisingMove();

            Assert.DoesNotContain(tree.RootNode.Children, child => ReferenceEquals(child.LastAction, move));
            if (move is IReplayAction replay)
                Assert.False(replay.HasHistory());
        }

        // ================================================================ MCTSAgent with the value net

        [SkippableFact]
        public void AgentFindsAnImmediateWin()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(145);

            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => TestUtil.IsFreeTurn(s) && TestUtil.CountBuildings(s, s.Turn.PlayerIndex, Intersection.BuildingType.Settlement) > 0));
            int playerIdx = state.Turn.PlayerIndex;
            PlayerState player = state.Players[playerIdx];

            // One city away from ten points
            player.VictoryPoints.SettlementPoints += (byte)(9 - player.VictoryPoints.Total);
            foreach ((ResourceCardType type, uint amount) in new[] { (ResourceCardType.Ore, 3u), (ResourceCardType.Grain, 2u) })
                if (player.ResourceCards.Get(type) < amount)
                    TestUtil.GiveCards(state, playerIdx, type, amount - player.ResourceCards.Get(type));

            MCTSAgent agent = new((sbyte)playerIdx, state, timeLimitSeconds: 1.0);
            Action move = agent.Act(state);

            GameState after = new GameState(state);
            move.Apply(after);
            Assert.True(after.HasEnded && after.Turn.PlayerIndex == playerIdx, $"agent chose {move} instead of a winning move");
        }

        [SkippableFact]
        public void CancellingStopsTheSearchEarlyWithAValidMove()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(146);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, maxActions: 60);
            sbyte acting = TestUtil.ActingPlayer(state);

            MCTSAgent agent = new(acting, state, timeLimitSeconds: 60);
            using CancellationTokenSource cancellation = new(TimeSpan.FromMilliseconds(200));
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            Action move = agent.Act(state, cancellation.Token);

            Assert.True(stopwatch.Elapsed.TotalSeconds < 10, $"search ran {stopwatch.Elapsed.TotalSeconds:F1}s after the cancellation");
            Assert.True(move.IsValidFor(state));
        }

        [SkippableFact]
        public void AgentKeepsItsTreeInSyncWithTheGame()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(146);

            GameState state = TestUtil.NewGame();
            ProbeAgent mcts = new(0, state);
            Agent[] agents = [mcts, new RandomAgent(1), new RandomAgent(2), new RandomAgent(3)];
            int reusedAfterOwnMove = 0, ownFollowUps = 0, actions = 0;
            sbyte lastActor = -1;

            while (!state.HasEnded && actions < 400)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                if (acting == 0 && lastActor == 0)
                {
                    ownFollowUps++;
                    if (mcts.Tree != null && mcts.Tree.RootState.Equals(state)) reusedAfterOwnMove++;
                }

                Action action = agents[acting].Act(state);
                if (acting == 0)
                    Assert.DoesNotContain(mcts.Tree!.RootNode.Children, child => ReferenceEquals(child.LastAction, action));

                action.Apply(state);
                foreach (Agent agent in agents)
                    agent.HandleAction(action);

                if (mcts.Tree != null)
                {
                    Assert.True(mcts.Tree.RootState.Equals(state), $"tree root state differs from the game after {action}");
                    Assert.Null(mcts.Tree.RootNode.Parent);
                }

                lastActor = acting;
                actions++;
            }

            Assert.True(ownFollowUps > 0);
            Assert.Equal(ownFollowUps, reusedAfterOwnMove);
        }

        [SkippableFact]
        public void ResettingOrDivergingStatesStartANewTree()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(147);

            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, TestUtil.IsFreeTurn));
            ProbeAgent agent = new((sbyte)state.Turn.PlayerIndex, state);

            agent.Act(state);
            Assert.NotNull(agent.Tree);

            agent.ResetToState(state);
            Assert.Null(agent.Tree);

            // A state the tree does not know (e.g. after a manual edit) starts over on a copy of that state
            agent.Act(state);
            GameState edited = new GameState(state);
            TestUtil.GiveCards(edited, edited.Turn.PlayerIndex, ResourceCardType.Ore, 1);
            agent.Act(edited);
            Assert.True(agent.Tree!.RootState.Equals(edited));
            Assert.NotSame(edited, agent.Tree.RootState);
        }

        // ================================================================ Helpers

        private class ProbeAgent : MCTSAgent
        {
            public ProbeAgent(sbyte playerIdx, GameState state) : base(playerIdx, state, timeLimitSeconds: 0.05) { }
            public MCTSTree? Tree => _tree;
        }

        private static GameState PreRollState()
        {
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.TypeOfRound == TurnState.RoundType.Normal && s.Turn.MustRoll && RollAction.GetActionsForState(s, (sbyte)s.Turn.PlayerIndex).Count > 0));
            return state;
        }

        /// <summary>
        /// Selection, expansion and backpropagation with a constant leaf value instead of the value net
        /// </summary>
        private static void RunFakeIterations(MCTSTree tree, int iterations, bool parallel = false)
        {
            void Iterate(int _)
            {
                (MCTSNode leaf, GameState leafState) = tree.SelectLeafNode();
                if (!leaf.IsTerminal)
                {
                    tree.ExpandNode(leaf, leafState);
                    (leaf, leafState) = tree.SelectLeafNode(leaf, leafState, suppressStateCopy: true);
                }
                tree.Backpropagate(leaf, leafState.GetHashCode() % 2 == 0 ? 0.4 : 0.6);
            }

            if (parallel)
                Parallel.For(0, iterations, Iterate);
            else
                for (int i = 0; i < iterations; i++) Iterate(i);
        }

        private static IEnumerable<MCTSNode> AllNodes(MCTSNode root)
        {
            Stack<MCTSNode> pending = new([root]);
            while (pending.TryPop(out MCTSNode? node))
            {
                yield return node;
                foreach (MCTSNode child in node.Children)
                    pending.Push(child);
            }
        }
    }
}
