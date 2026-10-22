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

        public StateValueNet(string modelPath, int intraOpThreads = 0)
        {
            string resolvedPath = ResolveModelPath(modelPath);
            if (!File.Exists(resolvedPath))
                throw new FileNotFoundException($"Value net not found at \"{resolvedPath}\". Place it in Agents/Artifacts (see StateValueNetModel in Agents.csproj) and rebuild.", resolvedPath);

            var options = new SessionOptions() {
                IntraOpNumThreads = intraOpThreads > 0 ? intraOpThreads : DefaultIntraOpThreads
            };

            _session = new InferenceSession(resolvedPath, options);
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
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("state", inputTensor)
            };

            float[] logits;
            int batchSize;

            using (var results = _session.Run(inputs))
            {
                var outputTensor = results.First().AsTensor<float>();
                batchSize = outputTensor.Dimensions[0]; // works for batch=1 or batched leaf-eval
                logits = outputTensor.ToArray(); // row-major flat: [batch * NumPlayers]
            }

            var winProbabilities = SoftmaxPerRow(logits, batchSize, 4);

            return winProbabilities;
        }

        public float[] RunOnlyLogits(DenseTensor<float> inputTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("state", inputTensor)
            };

            float[] logits;
            int batchSize;

            using (var results = _session.Run(inputs))
            {
                var outputTensor = results.First().AsTensor<float>();
                batchSize = outputTensor.Dimensions[0]; // works for batch=1 or batched leaf-eval
                logits = outputTensor.ToArray(); // row-major flat: [batch * NumPlayers]
            }

            return logits;
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
