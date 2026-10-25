using Agents.Inference;
using Agents.MCTS;
using Common;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The opponent models of the search: MaxN picks every player's moves by that player's own value, Paranoid has the others
    /// pick what is worst for the owner; both get one result per player from the backpropagation
    /// </summary>
    public class OpponentModelTests
    {
        /// <summary>
        /// A tree owned by player 0 whose root is a turn of player 1 after the roll, expanded, with every child visited once:
        /// child i scores i for player 1 and (children - i) for player 0, so the two players want opposite children
        /// </summary>
        private static (MCTSTree tree, MCTSNode[] children) OpponentTurn(OpponentModel model)
        {
            TestUtil.Seed(200);
            GameState state = TestUtil.EmptyNormalTurn(1);
            TestUtil.GiveCards(state, 1, ResourceCardType.Grain, 4);
            TestUtil.GiveCards(state, 1, ResourceCardType.Ore, 4);

            MCTSTree tree = new(state, ownerIdx: 0, explorationParameter: 0, normalizeValues: false, model: model);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            MCTSNode[] children = tree.RootNode.Children.ToArray();
            Assert.True(children.Length >= 2);
            Assert.Equal(1, tree.RootNode.DeciderIndex);

            for (int i = 0; i < children.Length; i++)
            {
                children[i].IncrementVisitCount();
                tree.RootNode.IncrementVisitCount();
                children[i].AddResults([children.Length - i, i, 0, 0]);
            }
            return (tree, children);
        }

        [Fact]
        public void MaxNPicksTheMoveThatIsBestForThePlayerToDecide()
        {
            (MCTSTree tree, MCTSNode[] children) = OpponentTurn(OpponentModel.MaxN);
            for (int repeat = 0; repeat < 20; repeat++)
                Assert.Same(children[^1], tree.SelectChild(tree.RootNode));
        }

        [Fact]
        public void ParanoidPicksTheMoveThatIsWorstForTheOwner()
        {
            (MCTSTree tree, MCTSNode[] children) = OpponentTurn(OpponentModel.Paranoid);
            for (int repeat = 0; repeat < 20; repeat++)
                Assert.Same(children[^1], tree.SelectChild(tree.RootNode));

            // On the owner's own turn it picks what is best for the owner
            TestUtil.Seed(201);
            GameState own = TestUtil.EmptyNormalTurn(0);
            TestUtil.GiveCards(own, 0, ResourceCardType.Grain, 4);
            MCTSTree ownTree = new(own, ownerIdx: 0, explorationParameter: 0, normalizeValues: false, model: OpponentModel.Paranoid);
            ownTree.ExpandNode(ownTree.RootNode, new GameState(own));
            MCTSNode[] ownChildren = ownTree.RootNode.Children.ToArray();
            for (int i = 0; i < ownChildren.Length; i++)
            {
                ownChildren[i].IncrementVisitCount();
                ownTree.RootNode.IncrementVisitCount();
                ownChildren[i].AddResults([i == 0 ? 1 : 0, 0, 0, 0]);
            }
            Assert.Same(ownChildren[0], ownTree.SelectChild(ownTree.RootNode));
        }

        [Fact]
        public void DiscardsAreDecidedByTheDiscardingPlayer()
        {
            TestUtil.Seed(202);
            GameState state = TestUtil.EmptyNormalTurn(0);
            TestUtil.GiveCards(state, 2, ResourceCardType.Lumber, 9);
            state.Turn.AwaitedPlayerDiscards[2] = true;

            MCTSTree tree = new(state, ownerIdx: 0, explorationParameter: 1.414, model: OpponentModel.MaxN);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            Assert.Equal(0, tree.RootNode.ActivePlayerIndex);
            Assert.Equal(2, tree.RootNode.DeciderIndex);
        }

        [Fact]
        public void EveryPlayersResultIsBackpropagatedToTheRoot()
        {
            (MCTSTree tree, MCTSNode[] children) = OpponentTurn(OpponentModel.MaxN);
            double rootOwner = tree.RootNode.TotalScore, rootPlayer2 = tree.RootNode.PlayerScore(2);

            tree.Backpropagate(children[0], [0.1, 0.2, 0.3, 0.4]);

            Assert.Equal(0.3, children[0].PlayerScore(2) - 0, 12);
            Assert.Equal(rootPlayer2 + 0.3, tree.RootNode.PlayerScore(2), 12);
            Assert.Equal(rootOwner + 0.1, tree.RootNode.TotalScore, 12);
            Assert.Equal(2u, children[0].VisitCount);
        }

        [SkippableTheory]
        [InlineData(OpponentModel.MaxN)]
        [InlineData(OpponentModel.Paranoid)]
        public void TheAgentPlaysValidMovesWithEveryModel(OpponentModel model)
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(203);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, maxActions: 60);

            sbyte acting = TestUtil.ActingPlayer(state);
            MCTSAgent agent = new(acting, state, timeLimitSeconds: 0.3, opponentModel: model);
            Assert.Equal(model, agent.OpponentModel);
            Action action = agent.Act(state);
            Assert.True(action.IsValidFor(state));
        }
    }
}
