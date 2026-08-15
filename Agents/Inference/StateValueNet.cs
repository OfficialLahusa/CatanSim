using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Agents.Inference
{
    public class StateValueNet
    {
        private InferenceSession _session;

        public StateValueNet(string modelPath)
        {
            var options = new SessionOptions() {
                // Set the number of threads explicitly to avoid multithreading issues in parallel MCTS simulations
                IntraOpNumThreads = 1 
            };

            _session = new InferenceSession(modelPath, options);
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
