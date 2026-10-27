using Microsoft.ML.OnnxRuntime.Tensors;

namespace Agents.Inference
{
    /// <summary>
    /// Value net outputs of a batch of leaves: win probabilities [rows * players] and, if requested, placement distributions
    /// [rows * players * places] and policy queries [rows * move types * dim]
    /// </summary>
    public readonly record struct LeafValues(float[] WinProbabilities, float[]? Placements, float[]? PolicyQueries = null);

    /// <summary>
    /// The constant part of a policy head: the token embeddings [VocabularySize * Dim] (see ActionEncoding)
    /// </summary>
    public sealed record PolicyHead(float[] Embeddings, int Dim)
    {
        public int QueryLength => ActionEncoding.MoveTypeCount * Dim;
    }

    /// <summary>
    /// Runs the value net on the leaf features of a search. DirectEvaluator runs every batch on its own, BatchingEvaluator
    /// collects the batches of many searches into large GPU batches.
    /// </summary>
    public interface ILeafEvaluator
    {
        /// <summary>
        /// Whether the net also predicts the placements (see StateValueNet.HasPlacements)
        /// </summary>
        bool HasPlacements { get; }

        /// <summary>
        /// The net's policy head, or null for a value-only net
        /// </summary>
        PolicyHead? Policy { get; }

        /// <summary>
        /// Evaluates the rows of features [rows, featureDim]. The tensor is only read until the returned task completes, so the
        /// caller can reuse it afterwards.
        /// </summary>
        Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements, bool withPolicy = false);
    }

    /// <summary>
    /// Runs every batch on the net as it is, on a thread pool thread (the batched search of a single MCTSAgent)
    /// </summary>
    public sealed class DirectEvaluator : ILeafEvaluator
    {
        private readonly StateValueNet _net;

        public DirectEvaluator(StateValueNet net)
        {
            _net = net;
        }

        public bool HasPlacements => _net.HasPlacements;

        public PolicyHead? Policy => _net.Policy;

        public Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements, bool withPolicy = false)
        {
            return Task.Run(() => _net.Evaluate(features, withPlacements, withPolicy));
        }
    }
}
