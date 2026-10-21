using Agents;
using Agents.Inference;
using Common;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Tests
{
    /// <summary>
    /// The deployed ONNX value net (skipped when the model file is not present)
    /// </summary>
    public class ValueNetTests
    {
        private static readonly StateVectorizer Vectorizer = new();

        [SkippableFact]
        public void ModelInputMatchesTheVectorizer()
        {
            TestUtil.SkipWithoutModel();

            using InferenceSession session = new(TestUtil.ModelPath);
            NodeMetadata input = session.InputMetadata["state"];
            NodeMetadata output = session.OutputMetadata["logits"];

            Assert.Equal(StateVectorizer.FeatureDim, input.Dimensions[1]);
            Assert.Equal(TestUtil.PlayerCount, output.Dimensions[1]);
        }

        [SkippableFact]
        public void FreshGameIsRoughlyEvenAndEndedGameGoesToTheWinner()
        {
            TestUtil.SkipWithoutModel();
            StateValueNet net = new(TestUtil.ModelPath);
            TestUtil.Seed(130);

            for (int game = 0; game < 5; game++)
            {
                GameState state = TestUtil.NewGame();
                float[] fresh = net.Run(Vectorizer.Vectorize(state));
                Assert.Equal(1f, fresh.Sum(), 3);
                Assert.All(fresh, p => Assert.InRange(p, 0.15f, 0.35f));

                TestUtil.PlayRandomGame(state);
                float[] ended = net.Run(Vectorizer.Vectorize(state));
                Assert.True(ended[state.Turn.PlayerIndex] > 0.9f, $"winner gets {ended[state.Turn.PlayerIndex]}");
            }
        }

        [SkippableFact]
        public void BatchedInferenceMatchesSingleInference()
        {
            TestUtil.SkipWithoutModel();
            StateValueNet net = new(TestUtil.ModelPath);
            TestUtil.Seed(131);

            List<GameState> states = [];
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, afterApply: (s, _) => { if (s.Turn.RoundCounter % 7 == 0 && states.Count < 8) states.Add(new GameState(s)); });

            DenseTensor<float> batch = new([states.Count, StateVectorizer.FeatureDim]);
            for (int i = 0; i < states.Count; i++)
                Vectorizer.Vectorize(states[i]).Buffer.Span.CopyTo(batch.Buffer.Span.Slice(i * StateVectorizer.FeatureDim, StateVectorizer.FeatureDim));

            float[] batched = net.Run(batch);
            for (int i = 0; i < states.Count; i++)
            {
                float[] single = net.Run(Vectorizer.Vectorize(states[i]));
                for (int p = 0; p < TestUtil.PlayerCount; p++)
                    Assert.Equal(single[p], batched[i * TestUtil.PlayerCount + p], 4);
            }
        }
    }
}
