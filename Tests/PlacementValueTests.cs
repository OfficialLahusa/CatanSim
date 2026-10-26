using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Value nets with a placements output and the placement tie-breaker of the MCTS agent. Uses tiny untrained nets with
    /// and without auxiliary heads (Fixtures/ValueNet/placements_tiny.onnx and plain_tiny.onnx, exported by export_onnx.py
    /// from StateValueNet(1764, hidden=16, aux_heads=True) and StateValueNet(1764, hidden=16)), so only the plumbing is
    /// tested, not playing strength.
    /// </summary>
    public class PlacementValueTests
    {
        private static readonly StateVectorizer Vectorizer = new();
        private static string TinyModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");
        private static string PlainTinyModelPath => TestUtil.FixturePath("ValueNet", "plain_tiny.onnx");

        [Fact]
        public void NetsWithPlacementsGiveADistributionPerPlayer()
        {
            StateValueNet net = new(TinyModelPath);
            Assert.True(net.HasPlacements);

            TestUtil.Seed(170);
            List<GameState> states = [];
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, afterApply: (s, _) => { if (states.Count < 3) states.Add(new GameState(s)); }, maxActions: 30);

            DenseTensor<float> batch = new([states.Count, StateVectorizer.FeatureDim]);
            for (int i = 0; i < states.Count; i++)
                Vectorizer.VectorizeInto(states[i], batch.Buffer.Span.Slice(i * StateVectorizer.FeatureDim, StateVectorizer.FeatureDim));

            (float[] winProbabilities, float[] placements) = net.RunWithPlacements(batch);
            Assert.Equal(states.Count * 4, winProbabilities.Length);
            Assert.Equal(states.Count * 16, placements.Length);
            for (int row = 0; row < states.Count * 4; row++)
                Assert.Equal(1f, placements.AsSpan(row * 4, 4).ToArray().Sum(), 4);

            // The win probabilities are the same as without the placements
            Assert.Equal(net.Run(batch), winProbabilities);
        }

        [Fact]
        public void ThePlacementWeightNeedsANetWithPlacements()
        {
            StateValueNet plain = new(PlainTinyModelPath);
            Assert.False(plain.HasPlacements);
            Assert.Throws<InvalidOperationException>(() => plain.RunWithPlacements(Vectorizer.Vectorize(TestUtil.NewGame())));

            GameState state = TestUtil.NewGame();
            Assert.Throws<ArgumentException>(() => new MCTSAgent(0, state, valueNet: plain, placementWeight: 0.1));
            MCTSAgent agent = new(0, state, valueNet: plain);
            Assert.Throws<ArgumentException>(() => agent.PlacementWeight = 0.1);
            Assert.Throws<ArgumentOutOfRangeException>(() => agent.PlacementWeight = -1);
        }

        [Fact]
        public void TheAgentSearchesWithThePlacementTieBreaker()
        {
            TestUtil.Seed(171);
            GameState state = TestUtil.NewGame();
            MCTSAgent agent = new(0, state, timeLimitSeconds: 0.3, valueNet: new StateValueNet(TinyModelPath), placementWeight: 0.1);
            Assert.Equal(0.1, agent.PlacementWeight);

            Action action = agent.Act(state);
            Assert.True(action.IsValidFor(state));
        }

        [Fact]
        public void PlacementScoresFollowThePlaces()
        {
            Assert.Equal(1f, MCTSAgent.PlacementScore([1, 0, 0, 0]), 6);
            Assert.Equal(2f / 3f, MCTSAgent.PlacementScore([0, 1, 0, 0]), 6);
            Assert.Equal(0.5f, MCTSAgent.PlacementScore([0.25f, 0.25f, 0.25f, 0.25f]), 6);

            // In ended games the winner scores 1, and the places (shared on equal victory points) always add up to 1 + 2/3 + 1/3
            TestUtil.Seed(172);
            for (int game = 0; game < 5; game++)
            {
                GameState state = TestUtil.NewGame();
                TestUtil.PlayRandomGame(state);
                Assert.True(state.HasEnded);

                Assert.Equal(1f, MCTSAgent.TerminalPlacementScore(state, state.Turn.PlayerIndex), 6);
                float total = Enumerable.Range(0, TestUtil.PlayerCount).Sum(p => MCTSAgent.TerminalPlacementScore(state, p));
                Assert.Equal(2f, total, 5);

                // Fewer victory points never give a better place
                for (int a = 0; a < TestUtil.PlayerCount; a++)
                    for (int b = 0; b < TestUtil.PlayerCount; b++)
                        if (a != state.Turn.PlayerIndex && b != state.Turn.PlayerIndex && state.Players[a].VictoryPoints.Total < state.Players[b].VictoryPoints.Total)
                            Assert.True(MCTSAgent.TerminalPlacementScore(state, a) < MCTSAgent.TerminalPlacementScore(state, b));
            }
        }
    }
}
