using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The policy head: move encoding (ActionEncoding), priors on the tree and PUCT selection, and searches with a net that has
    /// a policy head. Uses a tiny untrained net (Fixtures/ValueNet/policy_tiny.onnx, exported by export_onnx.py from
    /// StateValueModule(1965, hidden=32, placement_weight=0.5, policy_weight=1, policy_types=19, policy_vocabulary=213,
    /// policy_dim=8)), so only the plumbing is tested, not playing strength.
    /// </summary>
    public class PolicyTests
    {
        private static string PolicyModelPath => TestUtil.FixturePath("ValueNet", "policy_tiny.onnx");
        private static string PlainModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        private static string Encoded(Action action)
        {
            short[] tokens = new short[ActionEncoding.MaxTokens];
            float[] weights = new float[ActionEncoding.MaxTokens];
            int count = ActionEncoding.Encode(action, tokens, weights);
            return string.Join(" ", Enumerable.Range(0, count).Select(i => $"{tokens[i]}x{weights[i]}"));
        }

        [Fact]
        public void EveryLegalMoveHasAValidEncodingDistinctFromItsSiblings()
        {
            short[] tokens = new short[ActionEncoding.MaxTokens];
            float[] weights = new float[ActionEncoding.MaxTokens];
            var typesSeen = new HashSet<int>();
            for (int game = 0; game < 4; game++)
            {
                TestUtil.Seed(400 + game);
                GameState state = TestUtil.NewGame();
                TestUtil.PlayRandomGame(state, beforeApply: (s, _) =>
                {
                    List<Action> moves = MCTSTree.GetValidActions(s);
                    var encodings = new HashSet<string>();
                    foreach (Action move in moves)
                    {
                        int count = ActionEncoding.Encode(move, tokens, weights);
                        Assert.InRange(count, 1, ActionEncoding.MaxTokens);
                        Assert.InRange(tokens[0], 0, ActionEncoding.MoveTypeCount - 1);
                        for (int i = 1; i < count; i++)
                        {
                            Assert.InRange(tokens[i], ActionEncoding.MoveTypeCount, ActionEncoding.VocabularySize - 1);
                            Assert.True(weights[i] >= 1);
                        }
                        typesSeen.Add(tokens[0]);
                        Assert.True(encodings.Add(Encoded(move)), $"{move} encodes like a sibling");
                    }
                }, maxActions: 2000);
            }
            // The random games reach most move types
            Assert.True(typesSeen.Count >= 14, $"only {typesSeen.Count} move types seen");
        }

        [Fact]
        public void TheRobbedPlayerIsEncodedRelativeToTheMover()
        {
            var robber1 = new Common.Actions.RobberAction(1, 12, 3);
            var robber2 = new Common.Actions.RobberAction(2, 12, 0);
            var noVictim = new Common.Actions.RobberAction(2, 12, null);
            Assert.Equal(Encoded(robber1), Encoded(robber2));
            Assert.NotEqual(Encoded(robber2), Encoded(noVictim));
            Assert.Contains($"{ActionEncoding.VictimStart + 2}x1", Encoded(robber1));
        }

        [Fact]
        public void LogitsAreTheTypeQueryTimesTheWeightedTokenEmbeddings()
        {
            const int Dim = 2;
            float[] queries = new float[ActionEncoding.MoveTypeCount * Dim];
            float[] embeddings = new float[ActionEncoding.VocabularySize * Dim];
            int discard = (int)ActionEncoding.MoveType.Discard;
            queries[discard * Dim] = 1; queries[discard * Dim + 1] = 2;
            embeddings[discard * Dim] = 0.5f;                                    // the type token: 0.5
            int wool = ActionEncoding.DiscardedResourceStart + 2;
            embeddings[wool * Dim + 1] = 1;                                       // wool: 2 per card

            var cards = new CardSet<ResourceCardType>();
            cards.Add(ResourceCardType.Wool, 3);
            Action[] moves = [new Common.Actions.DiscardAction(0, cards), new Common.Actions.EndTurnAction(0)];
            float[] logits = new float[2];
            ActionEncoding.Logits(queries, embeddings, Dim, moves, logits);
            Assert.Equal(0.5f + 3 * 2, logits[0], 5);
            Assert.Equal(0f, logits[1], 5);

            ActionEncoding.Softmax(logits);
            Assert.Equal(1f, logits.Sum(), 5);
            Assert.True(logits[0] > 0.99f);
        }

        [Fact]
        public void PriorsGoToTheChildrenBeforeOrAfterTheExpansion()
        {
            TestUtil.Seed(410);
            GameState state = TestUtil.EmptyNormalTurn(1);
            TestUtil.GiveCards(state, 1, ResourceCardType.Grain, 4);
            int moves = MCTSTree.GetValidActions(state).Count;
            float[] priors = Enumerable.Range(0, moves).Select(i => (i + 1f) / (moves * (moves + 1) / 2f)).ToArray();

            // Before: kept for the expansion
            MCTSTree before = new(new GameState(state), 1, 1.414, model: OpponentModel.MaxN);
            before.ApplyPriors(before.RootNode, priors);
            Assert.False(before.RootNode.IsExpanded);
            before.ExpandNode(before.RootNode, new GameState(state));
            Assert.Equal(priors, before.RootNode.Children.Select(child => child.Prior));
            Assert.True(before.RootNode.HasPolicyPriors);

            // After: uniform until they arrive
            MCTSTree after = new(new GameState(state), 1, 1.414, model: OpponentModel.MaxN) { UsesPolicy = true };
            after.ExpandNode(after.RootNode, new GameState(state));
            Assert.All(after.RootNode.Children, child => Assert.Equal(1f / moves, child.Prior, 5));
            Assert.False(after.RootNode.HasPolicyPriors);
            after.ApplyPriors(after.RootNode, priors);
            Assert.Equal(priors, after.RootNode.Children.Select(child => child.Prior));
            Assert.True(after.RootNode.HasPolicyPriors);
        }

        [Fact]
        public void PuctFollowsThePriorsAndThenTheValues()
        {
            TestUtil.Seed(411);
            GameState state = TestUtil.EmptyNormalTurn(1);
            TestUtil.GiveCards(state, 1, ResourceCardType.Grain, 4);
            MCTSTree tree = new(new GameState(state), 1, 1.414, model: OpponentModel.MaxN);
            int moves = MCTSTree.GetValidActions(state).Count;
            float[] priors = new float[moves];
            priors[^1] = 0.9f;
            for (int i = 0; i < moves - 1; i++) priors[i] = 0.1f / (moves - 1);
            tree.ApplyPriors(tree.RootNode, priors);
            tree.ExpandNode(tree.RootNode, new GameState(state));
            MCTSNode[] children = tree.RootNode.Children.ToArray();
            tree.RootNode.IncrementVisitCount();

            // Nothing visited: the move with the high prior first
            Assert.Same(children[^1], tree.SelectChild(tree.RootNode));

            // It keeps losing for player 1, another move keeps winning: PUCT turns to the winner
            for (int i = 0; i < 50; i++)
            {
                tree.Backpropagate(children[^1], [0.25, 0, 0.375, 0.375]);
                tree.Backpropagate(children[0], [0, 1, 0, 0]);
            }
            Assert.Same(children[0], tree.SelectChild(tree.RootNode));
        }

        [Fact]
        public void NetsWithAPolicyHeadGiveQueriesAndTheEmbeddingTable()
        {
            StateValueNet net = new(PolicyModelPath, provider: InferenceProvider.Cpu);
            Assert.NotNull(net.Policy);
            Assert.Equal(8, net.Policy!.Dim);
            Assert.Equal(ActionEncoding.VocabularySize * 8, net.Policy.Embeddings.Length);
            Assert.Null(new StateValueNet(PlainModelPath, provider: InferenceProvider.Cpu).Policy);

            TestUtil.Seed(412);
            GameState state = TestUtil.NewGame();
            DenseTensor<float> batch = new StateVectorizer().Vectorize(state);
            LeafValues values = net.Evaluate(batch, withPlacements: true, withPolicy: true);
            Assert.Equal(net.Policy.QueryLength, values.PolicyQueries!.Length);
            Assert.Equal(net.Run(batch), values.WinProbabilities);
        }

        [Fact]
        public void SearchesWithAPolicyHeadPutPriorsOnTheTreeAndPlayValidMoves()
        {
            StateValueNet net = new(PolicyModelPath, provider: InferenceProvider.Cpu);
            TestUtil.Seed(413);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayUntil(state, TestUtil.IsFreeTurn);

            var search = new MCTSSearch(new DirectEvaluator(net), new StateVectorizer(), 16, parallelSelection: true);
            MCTSTree tree = new(new GameState(state), (sbyte)state.Turn.PlayerIndex, 1.414, model: OpponentModel.MaxN);
            search.Run(tree, 0, 400);

            Assert.True(tree.UsesPolicy);
            Assert.True(tree.RootNode.HasPolicyPriors);
            Assert.Equal(1f, tree.RootNode.Children.Sum(child => child.Prior), 4);
            Assert.True(tree.RootNode.Children.Count(child => child.IsExpanded && child.HasPolicyPriors) > 0);
            Assert.True(tree.GetMostPromisingMove().IsValidFor(new GameState(state)));

            // Root noise changes the root's priors, which still sum to 1
            float[] clean = tree.RootNode.Children.Select(child => child.Prior).ToArray();
            search.RootNoiseWeight = 0.25;
            search.NoiseRandom = new Random(3);
            search.Run(tree, 0, 16);
            float[] noisy = tree.RootNode.Children.Select(child => child.Prior).ToArray();
            Assert.Equal(1f, noisy.Sum(), 4);
            Assert.NotEqual(clean, noisy);
        }

        [Fact]
        public void DirichletSamplesSumToOneWithTheExpectedSpread()
        {
            Random random = new(9);
            double[] mean = new double[4];
            const int Samples = 4000;
            for (int s = 0; s < Samples; s++)
            {
                double[] sample = MCTSSearch.Dirichlet(0.3, 4, random);
                Assert.Equal(1.0, sample.Sum(), 9);
                for (int i = 0; i < 4; i++) mean[i] += sample[i] / Samples;
            }
            Assert.All(mean, m => Assert.InRange(m, 0.22, 0.28));
        }

        [Fact]
        public void TheBatchingEvaluatorHandsEachSearchItsQueries()
        {
            StateValueNet net = new(PolicyModelPath, provider: InferenceProvider.Cpu);
            using var evaluator = new BatchingEvaluator([net], 64);
            Assert.NotNull(evaluator.Policy);

            TestUtil.Seed(414);
            var vectorizer = new StateVectorizer();
            GameState first = TestUtil.NewGame(), second = TestUtil.NewGame();
            TestUtil.PlayUntil(second, TestUtil.IsFreeTurn);
            LeafValues a = evaluator.EvaluateAsync(vectorizer.Vectorize(first), false, withPolicy: true).Result;
            LeafValues b = evaluator.EvaluateAsync(vectorizer.Vectorize(second), false, withPolicy: true).Result;
            Assert.Equal(net.Evaluate(vectorizer.Vectorize(first), false, true).PolicyQueries!, a.PolicyQueries!);
            Assert.Equal(net.Evaluate(vectorizer.Vectorize(second), false, true).PolicyQueries!, b.PolicyQueries!);
        }
    }
}
