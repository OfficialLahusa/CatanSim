using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Common.Actions;
using Microsoft.ML.OnnxRuntime.Tensors;
using static Tests.HiddenInformationTests;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Information set MCTS (ISMCTSTree, single observer): one tree for the searching seat whose nodes are what it can tell
    /// apart, a world drawn from its beliefs for every iteration. Moves are keyed by what the seat sees, a child counts as
    /// available only in the worlds where it is legal, and the search depends on nothing the seat can't see.
    /// </summary>
    public class ISMCTSTests
    {
        /// <summary>
        /// A value net stand-in that gives every position the same deterministic utilities from its features, so searches
        /// repeat exactly
        /// </summary>
        private sealed class FixedEvaluator : ILeafEvaluator
        {
            public bool HasPlacements => false;
            public PolicyHead? Policy => null;

            public Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements, bool withPolicy = false)
            {
                int rows = features.Dimensions[0], width = features.Dimensions[1];
                float[] probabilities = new float[rows * 4];
                ReadOnlySpan<float> buffer = features.Buffer.Span;
                for (int row = 0; row < rows; row++)
                {
                    double sum = 0;
                    for (int f = 0; f < width; f += 7) sum += buffer[row * width + f] * (f % 13);
                    for (int p = 0; p < 4; p++) probabilities[row * 4 + p] = (float)(1 + Math.Abs(Math.Sin(sum * (p + 1))));
                }
                return Task.FromResult(new LeafValues(probabilities, null));
            }
        }

        private static ISMCTSTree Search(GameState truth, BeliefTracker beliefs, int iterations = 3000, int seed = 5)
        {
            ISMCTSTree tree = new ISMCTSTree(truth, beliefs, seed: seed);
            MCTSSearch search = new MCTSSearch(new FixedEvaluator(), new ViewVectorizer { Observer = beliefs.Observer }, 16, parallelSelection: false);
            search.Run(tree, double.PositiveInfinity, iterations);
            return tree;
        }

        /// <summary>
        /// Positions of a few random games where the given seat is about to decide, with its beliefs
        /// </summary>
        private static IEnumerable<(GameState truth, BeliefTracker beliefs)> Decisions(int seat, int count)
        {
            int found = 0;
            for (int seed = 1; found < count && seed < 50; seed++)
            {
                Followed game = new Followed(NewGame(seed));
                Agent[] agents = TestUtil.RandomAgents();
                while (!game.Truth.HasEnded && found < count && game.Played.Count < 3000)
                {
                    sbyte acting = TestUtil.ActingPlayer(game.Truth);
                    if (acting == seat && game.Played.Count > 60 && game.Played.Count % 23 == 0 && MCTSTree.GetValidActions(game.Truth).Count > 1)
                    {
                        found++;
                        yield return (new GameState(game.Truth), new BeliefTracker(game.Trackers[seat]));
                    }
                    game.Play(agents[acting].Act(game.Truth));
                }
            }
        }

        private static IEnumerable<ISNode> Nodes(ISNode node)
        {
            yield return node;
            foreach (ISNode child in node.Children)
                foreach (ISNode below in Nodes(child)) yield return below;
        }

        [Fact]
        public void TheRootsMovesAreTheSeatsLegalMoves()
        {
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(1, 6))
            {
                ISMCTSTree tree = Search(truth, beliefs, 800);
                List<Action> legal = MCTSTree.GetValidActions(truth);
                Assert.Equal(legal.Count, tree.RootNode.Children.Count);
                Assert.All(tree.RootNode.Children, child => Assert.Contains(legal, move => move.IsSameDecision(child.Move!)));
                Assert.True(legal.Any(move => move.IsSameDecision(tree.GetMostPromisingMove())));
                Assert.Null(tree.GetMostPromisingMove() is IReplayAction replay && replay.HasHistory() ? replay : null);
            }
        }

        [Fact]
        public void TheTreeKeysMovesByWhatTheSeatSees()
        {
            int hiddenDraws = 0, ownDraws = 0, availableInSomeWorlds = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(2, 10))
            {
                ISMCTSTree tree = Search(truth, beliefs);
                foreach (ISNode node in Nodes(tree.RootNode))
                {
                    // Visits never exceed the times a node was available, which never exceed its parent's visits
                    if (node.Parent != null && !node.Parent.IsChance)
                    {
                        Assert.True(node.Visits <= node.Availability, "a child was visited more often than it was available");
                        Assert.True(node.Availability <= node.Parent.Visits, "a child was available more often than its parent was visited");
                        // (a node's first visit is its own evaluation, before it has children: compare with the most available sibling)
                        if (node.Availability < node.Parent.Children.Max(sibling => sibling.Availability)) availableInSomeWorlds++;
                    }
                    Assert.True(node.Children.Sum(child => (long)child.Visits) <= node.Visits);

                    if (node.IsChance)
                    {
                        // An outcome the seat can't see is one child, whatever the world drew
                        bool hidden = node.Move!.PlayerIndex != 2 && node.Move is BuyDevelopmentCardAction
                            || node.Move is RobberAction { TargetPlayerIndex: sbyte victim } robber && robber.PlayerIndex != 2 && victim != 2;
                        if (hidden)
                        {
                            Assert.True(node.Children.Count <= 1, $"{node.Move} branches on what seat 2 can't see");
                            if (node.Move is BuyDevelopmentCardAction) hiddenDraws++;
                        }
                        else if (node.Move is BuyDevelopmentCardAction && node.Children.Count > 1) ownDraws++;
                        Assert.All(node.Children, outcome => Assert.Equal(node.Move.GetType(), outcome.Move!.GetType()));
                    }
                    else
                    {
                        // Children of a decision are distinct decisions
                        for (int a = 0; a < node.Children.Count; a++)
                            for (int b = a + 1; b < node.Children.Count; b++)
                                Assert.False(node.Children[a].Move!.IsSameDecision(node.Children[b].Move!));
                    }
                }
            }
            Assert.True(hiddenDraws > 0, "no other seat bought a development card in the searched trees");
            Assert.True(availableInSomeWorlds > 0, "no move was legal in only some of the worlds");
        }

        [Fact]
        public void TheSearchDependsOnlyOnWhatTheSeatSees()
        {
            // The same position played on with two different stolen cards: a seat outside the steal searches exactly alike
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action)
                && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2);
            (Followed a, Followed b) = TwoOutcomes(game, next, move => ((RobberAction)move).History!.StolenCard);
            int outsider = Enumerable.Range(0, 4).First(p => p != next.PlayerIndex && p != ((RobberAction)next).TargetPlayerIndex);

            ISMCTSTree treeA = Search(a.Truth, a.Trackers[outsider], 1500, seed: 9), treeB = Search(b.Truth, b.Trackers[outsider], 1500, seed: 9);
            Assert.Equal(treeA.RootVisits().Select(v => (v.move.ToString(), v.visits)), treeB.RootVisits().Select(v => (v.move.ToString(), v.visits)));
            Assert.Equal(treeA.RootValues(), treeB.RootValues());
        }

        [Fact]
        public void AFullKnowledgeTreeSearchesTheTruePosition()
        {
            int hidden = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(3, 4))
            {
                if (beliefs.HasHiddenInformation) hidden++;
                ISMCTSTree tree = new ISMCTSTree(truth, beliefs, seed: 1, fullKnowledge: true);
                for (int p = 0; p < 4; p++)
                {
                    Assert.Equal(truth.Players[p].ResourceCards, tree.RootState.Players[p].ResourceCards);
                    Assert.Equal(truth.Players[p].DevelopmentCards, tree.RootState.Players[p].DevelopmentCards);
                }
                Assert.Equal(truth.DevelopmentBank, tree.RootState.DevelopmentBank);
            }
            Assert.True(hidden > 0, "seat 3 never missed anything in the positions");
        }

        [Fact]
        public void TheSameSeedSearchesTheSame()
        {
            (GameState truth, BeliefTracker beliefs) = Decisions(0, 1).First();
            ISMCTSTree first = Search(truth, beliefs, 1000, seed: 3), second = Search(truth, beliefs, 1000, seed: 3);
            Assert.Equal(first.RootVisits().Select(v => v.visits), second.RootVisits().Select(v => v.visits));
        }

        [Fact]
        public void TheAgentPlaysLegalMovesUnderHiddenInformation()
        {
            GameState state = NewGame(36);
            StateValueNet net = new(TestUtil.FixturePath("ValueNet", "view_tiny.onnx"), provider: InferenceProvider.Cpu);
            InformationSetMCTSAgent agent = new(new MCTSAgent(1, state, valueNet: net, iterationLimit: 256, leavesPerBatch: 16), state, seed: 2);
            Agent[] agents = TestUtil.RandomAgents();
            int decisions = 0;
            while (!state.HasEnded && decisions < 40)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action = acting == 1 ? agent.Act(new GameState(state)) : agents[acting].Act(state);
                if (acting == 1)
                {
                    decisions++;
                    Assert.True(action.IsValidFor(state), $"{action} is not legal");
                }
                action.Apply(state);
                agent.HandleAction(action);
            }
            Assert.Equal(40, decisions);
            Assert.Equal(new BeliefTracker(agent.Beliefs), agent.Beliefs);
        }
    }
}
