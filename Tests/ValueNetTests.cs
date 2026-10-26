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
        [SkippableFact]
        public void ModelInputMatchesTheVectorizer()
        {
            TestUtil.SkipWithoutModel();

            using InferenceSession session = new(TestUtil.ModelPath);
            NodeMetadata input = session.InputMetadata["state"];
            NodeMetadata output = session.OutputMetadata["logits"];

            // The deployed net may still have an older layout, the agents then vectorize for it
            StateValueNet net = new(TestUtil.ModelPath);
            Assert.Equal(input.Dimensions[1], net.InputDim);
            Assert.Equal(net.InputDim, StateVectorizer.For(net).Dimension);
            Assert.Equal(TestUtil.PlayerCount, output.Dimensions[1]);
        }

        [SkippableFact]
        public void FreshGameIsRoughlyEvenAndEndedGameGoesToTheWinner()
        {
            TestUtil.SkipWithoutModel();
            StateValueNet net = new(TestUtil.ModelPath);
            StateVectorizer vectorizer = StateVectorizer.For(net);
            TestUtil.Seed(130);

            for (int game = 0; game < 5; game++)
            {
                GameState state = TestUtil.NewGame();
                float[] fresh = net.Run(vectorizer.Vectorize(state));
                Assert.Equal(1f, fresh.Sum(), 3);
                Assert.All(fresh, p => Assert.InRange(p, 0.15f, 0.35f));

                TestUtil.PlayRandomGame(state);
                float[] ended = net.Run(vectorizer.Vectorize(state));
                Assert.True(ended[state.Turn.PlayerIndex] > 0.9f, $"winner gets {ended[state.Turn.PlayerIndex]}");
            }
        }

        [SkippableFact]
        public void BatchedInferenceMatchesSingleInference()
        {
            TestUtil.SkipWithoutModel();
            StateValueNet net = new(TestUtil.ModelPath);
            StateVectorizer vectorizer = StateVectorizer.For(net);
            int dim = vectorizer.Dimension;
            TestUtil.Seed(131);

            List<GameState> states = [];
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, afterApply: (s, _) => { if (s.Turn.RoundCounter % 7 == 0 && states.Count < 8) states.Add(new GameState(s)); });

            DenseTensor<float> batch = new([states.Count, dim]);
            for (int i = 0; i < states.Count; i++)
                vectorizer.Vectorize(states[i]).Buffer.Span.CopyTo(batch.Buffer.Span.Slice(i * dim, dim));

            float[] batched = net.Run(batch);
            for (int i = 0; i < states.Count; i++)
            {
                float[] single = net.Run(vectorizer.Vectorize(states[i]));
                for (int p = 0; p < TestUtil.PlayerCount; p++)
                    Assert.Equal(single[p], batched[i * TestUtil.PlayerCount + p], 4);
            }
        }
    }
}
