using Agents;
using Agents.Inference;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Tests
{
    /// <summary>
    /// The value net on the CPU and with DirectML (GPU, Windows only; skipped where it isn't available): the same outputs,
    /// also for batches the fixed DirectML batch size has to pad or split
    /// </summary>
    public class InferenceProviderTests
    {
        private static readonly StateVectorizer Vectorizer = new();
        private static string TinyModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        private static DenseTensor<float> RandomGameBatch(int size, int seed)
        {
            TestUtil.Seed(seed);
            DenseTensor<float> batch = new([size, StateVectorizer.FeatureDim]);
            int row = 0;
            while (row < size)
            {
                GameState state = TestUtil.NewGame();
                TestUtil.PlayRandomGame(state, afterApply: (s, _) =>
                {
                    if (row < size && Random.Shared.Next(4) == 0)
                        Vectorizer.VectorizeInto(s, batch.Buffer.Span.Slice(row++ * StateVectorizer.FeatureDim, StateVectorizer.FeatureDim));
                }, maxActions: 400);
            }
            return batch;
        }

        private static StateValueNet DirectMLOrSkip(string modelPath)
        {
            Skip.IfNot(OperatingSystem.IsWindows(), "DirectML runs on Windows only.");
            StateValueNet net = new(modelPath, provider: InferenceProvider.DirectML);
            Skip.If(net.Provider != InferenceProvider.DirectML, "No DirectML device available.");
            return net;
        }

        [Fact]
        public void TheCpuProviderRunsAnyBatchAsItIs()
        {
            StateValueNet net = new(TinyModelPath, provider: InferenceProvider.Cpu);
            Assert.Equal(InferenceProvider.Cpu, net.Provider);
            Assert.Null(net.FixedBatchSize);
            Assert.Equal(3 * 4, net.Run(RandomGameBatch(3, 180)).Length);
        }

        [SkippableFact]
        public void DirectMLGivesTheSameValuesAsTheCpu()
        {
            StateValueNet gpu = DirectMLOrSkip(TinyModelPath);
            StateValueNet cpu = new(TinyModelPath, provider: InferenceProvider.Cpu);
            Assert.Equal(StateValueNet.DefaultFixedBatchSize, gpu.FixedBatchSize);

            // Exactly the fixed size, padded (1 and 37) and split (300 = 256 + 44)
            foreach (int size in new[] { 256, 1, 37, 300 })
            {
                DenseTensor<float> batch = RandomGameBatch(size, 181 + size);
                float[] expected = cpu.Run(batch), actual = gpu.Run(batch);
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-3, $"batch {size}, value {i}: CPU {expected[i]}, DirectML {actual[i]}");

                (float[] cpuWins, float[] cpuPlacements) = cpu.RunWithPlacements(batch);
                (float[] gpuWins, float[] gpuPlacements) = gpu.RunWithPlacements(batch);
                Assert.Equal(cpuPlacements.Length, gpuPlacements.Length);
                for (int i = 0; i < cpuPlacements.Length; i++)
                    Assert.True(Math.Abs(cpuPlacements[i] - gpuPlacements[i]) < 1e-3, $"batch {size}, placement {i}");
            }
        }

        [SkippableFact]
        public void TheDeployedNetRunsOnDirectML()
        {
            TestUtil.SkipWithoutModel();
            StateValueNet gpu = DirectMLOrSkip(TestUtil.ModelPath);
            StateValueNet cpu = new(TestUtil.ModelPath, provider: InferenceProvider.Cpu);

            DenseTensor<float> batch = RandomGameBatch(50, 190);
            float[] expected = cpu.Run(batch), actual = gpu.Run(batch);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-3, $"value {i}: CPU {expected[i]}, DirectML {actual[i]}");
        }
    }
}
