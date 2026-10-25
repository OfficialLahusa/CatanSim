using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Agents.Inference
{
    public class StateValueNet
    {
        // File name of the deployed model, set by StateValueNetModel in Agents.csproj, which also copies it next to the executable
        public static readonly string DefaultModelFile = typeof(StateValueNet).Assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .Single(attribute => attribute.Key == "StateValueNetModel").Value!;

        public static string DefaultModelPath => ResolveModelPath(DefaultModelFile);

        // The model file is not tracked in git, so a fresh clone builds without it
        public static bool IsDefaultModelAvailable => File.Exists(DefaultModelPath);

        // Threads ONNX Runtime uses inside one Run call. The MCTS agent evaluates one batch while it selects the next,
        // so half of the logical cores run the value net and half the search (more threads starve the search, at all
        // cores both slow down by half; measured with batches of 256 on 32 logical cores).
        public static readonly int DefaultIntraOpThreads = Math.Max(1, Environment.ProcessorCount / 2);

        private InferenceSession _session;

        public const int PlayerCount = 4;

        /// <summary>
        /// Whether the model also predicts each player's placement distribution (output "placements", models trained with
        /// auxiliary heads)
        /// </summary>
        public bool HasPlacements { get; }

        public StateValueNet(string modelPath, int intraOpThreads = 0)
        {
            string resolvedPath = ResolveModelPath(modelPath);
            if (!File.Exists(resolvedPath))
                throw new FileNotFoundException($"Value net not found at \"{resolvedPath}\". Place it in Agents/Artifacts (see StateValueNetModel in Agents.csproj) and rebuild.", resolvedPath);

            var options = new SessionOptions() {
                IntraOpNumThreads = intraOpThreads > 0 ? intraOpThreads : DefaultIntraOpThreads
            };

            _session = new InferenceSession(resolvedPath, options);
            HasPlacements = _session.OutputMetadata.ContainsKey("placements");
        }

        /// <summary>
        /// Relative model paths are resolved next to the executable instead of against the working directory
        /// </summary>
        private static string ResolveModelPath(string modelPath)
        {
            return Path.IsPathRooted(modelPath) ? modelPath : Path.Combine(AppContext.BaseDirectory, modelPath);
        }

        public float[] Run(DenseTensor<float> inputTensor)
        {
            float[] logits = RunOnlyLogits(inputTensor);
            return SoftmaxPerRow(logits, logits.Length / PlayerCount, PlayerCount);
        }

        public float[] RunOnlyLogits(DenseTensor<float> inputTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("state", inputTensor)
            };

            // Only the value output is computed, models with a second output skip it here
            using var results = _session.Run(inputs, ["logits"]);
            return results.First().AsTensor<float>().ToArray(); // row-major flat: [batch * NumPlayers], works for batch=1 or batched leaf-eval
        }

        /// <summary>
        /// Win probabilities [batch * players] and placement distributions [batch * players * places], each row of 4 places
        /// summing to 1, of a model with HasPlacements
        /// </summary>
        public (float[] winProbabilities, float[] placements) RunWithPlacements(DenseTensor<float> inputTensor)
        {
            if (!HasPlacements)
                throw new InvalidOperationException("The value net has no placements output (it was trained without auxiliary heads).");

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("state", inputTensor)
            };

            using var results = _session.Run(inputs, ["logits", "placements"]);
            float[] logits = results.First(result => result.Name == "logits").AsTensor<float>().ToArray();
            float[] placementLogits = results.First(result => result.Name == "placements").AsTensor<float>().ToArray();

            int batchSize = logits.Length / PlayerCount;
            return (SoftmaxPerRow(logits, batchSize, PlayerCount), SoftmaxPerRow(placementLogits, batchSize * PlayerCount, PlayerCount));
        }

        /// <summary>
        /// Batchable numerically stable softmax implementation for row-major flat arrays.
        /// </summary>
        /// <param name="flat">Row-major flat array of shape [batchSize * numClasses]</param>
        /// <param name="batchSize">Batch size</param>
        /// <param name="numClasses">Number of classes </param>
        /// <returns>Row-major flat array of shape [batchSize * numClasses] with softmax distributed outputs</returns>
        private static float[] SoftmaxPerRow(float[] flat, int batchSize, int numClasses)
        {
            var result = new float[flat.Length];

            for (int row = 0; row < batchSize; row++)
            {
                int offset = row * numClasses;

                float max = float.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                    max = Math.Max(max, flat[offset + c]);

                float sumExp = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float e = MathF.Exp(flat[offset + c] - max);
                    result[offset + c] = e;
                    sumExp += e;
                }

                for (int c = 0; c < numClasses; c++)
                    result[offset + c] /= sumExp;
            }

            return result;
        }
    }
}
