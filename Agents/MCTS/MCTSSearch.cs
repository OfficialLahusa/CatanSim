using Agents.Inference;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace Agents.MCTS
{
    /// <summary>
    /// Counters of a search: iterations, distinct leaves evaluated (several iterations of a batch can arrive at the same leaf),
    /// the time spent selecting, evaluating and backpropagating (evaluation overlaps the other two) and waiting for evaluations
    /// </summary>
    public record SearchStatistics(long Iterations, long DistinctLeaves, double Seconds, double SelectionSeconds, double EvaluationSeconds, double BackpropagationSeconds,
        double EvaluationWaitSeconds)
    {
        public double IterationsPerSecond => Iterations / Seconds;
        public double DistinctLeafShare => Iterations == 0 ? 0 : (double)DistinctLeaves / Iterations;
    }

    /// <summary>
    /// The batched search on an MCTSTree: selects a batch of leaves, has the value net evaluate them through an ILeafEvaluator
    /// and backpropagates the results. Pipelined: while the value net evaluates one batch, the previous one is backpropagated
    /// and the next one selected, so selection runs one batch behind on results (the visits of the batch in evaluation are
    /// already counted). An MCTSAgent selects large batches on all cores; many searches at once (self-play) select small
    /// batches on one thread each and share a BatchingEvaluator. One search at a time per instance.
    /// </summary>
    public sealed class MCTSSearch
    {
        // Leaves of a batch with their features, written by selection so the evaluation only runs the value net
        private sealed record LeafBatch((MCTSNode leafNode, GameState stateAtLeaf)[] Leaves, DenseTensor<float> Features);

        private readonly ILeafEvaluator _evaluator;
        private readonly IStateVectorizer _vectorizer;
        private readonly bool _parallelSelection;
        private double _placementWeight;

        // Two feature buffers used in turn: one is evaluated while the next batch is selected into the other
        private readonly DenseTensor<float>[] _featureBuffers;
        private int _nextBuffer;

        private long _distinctLeaves;
        private readonly Stopwatch _selectionTime = new(), _evaluationTime = new(), _backpropagationTime = new(), _evaluationWaitTime = new();

        /// <param name="leavesPerBatch">Leaves selected before their evaluation (virtual loss spreads them out)</param>
        /// <param name="parallelSelection">Select a batch on all cores (one search at a time) instead of on the calling thread</param>
        public MCTSSearch(ILeafEvaluator evaluator, IStateVectorizer vectorizer, int leavesPerBatch, bool parallelSelection)
        {
            if (leavesPerBatch < 1) throw new ArgumentOutOfRangeException(nameof(leavesPerBatch));
            _evaluator = evaluator;
            _vectorizer = vectorizer;
            _parallelSelection = parallelSelection;
            LeavesPerBatch = leavesPerBatch;
            _featureBuffers = [new([leavesPerBatch, vectorizer.FeatureDim]), new([leavesPerBatch, vectorizer.FeatureDim])];
        }

        public int LeavesPerBatch { get; }

        /// <summary>
        /// Weight of the placement score in the search value (see MCTSAgent.PlacementWeight)
        /// </summary>
        public double PlacementWeight
        {
            get => _placementWeight;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "The placement weight must not be negative.");
                if (value > 0 && !_evaluator.HasPlacements)
                    throw new ArgumentException("A placement weight above 0 needs a value net with a placements output.", nameof(value));
                _placementWeight = value;
            }
        }

        /// <summary>
        /// Statistics of the last search
        /// </summary>
        public SearchStatistics? LastSearch { get; private set; }

        /// <summary>
        /// Searches until the time or iteration limit (0 for none, rounded up to whole batches) or until the token is cancelled
        /// (after the batch in progress)
        /// </summary>
        public void Run(MCTSTree tree, double timeLimitSeconds, long iterationLimit, CancellationToken cancellation = default)
        {
            RunAsync(tree, timeLimitSeconds, iterationLimit, cancellation).GetAwaiter().GetResult();
        }

        /// <inheritdoc cref="Run"/>
        public async Task RunAsync(MCTSTree tree, double timeLimitSeconds, long iterationLimit, CancellationToken cancellation = default)
        {
            Stopwatch stopwatch = Stopwatch.StartNew();
            _distinctLeaves = 0;
            _selectionTime.Reset();
            _evaluationTime.Reset();
            _backpropagationTime.Reset();
            _evaluationWaitTime.Reset();

            LeafBatch? evaluatedBatch = null;
            Task<double[]>? evaluation = null; // utilities per leaf and player

            long iterationCounter = 0;
            do
            {
                LeafBatch selectedBatch = SelectBatch(tree);

                double[]? previousResults = null;
                if (evaluation != null)
                {
                    _evaluationWaitTime.Start();
                    previousResults = await evaluation.ConfigureAwait(false);
                    _evaluationWaitTime.Stop();
                }

                LeafBatch? previousBatch = evaluatedBatch;
                evaluatedBatch = selectedBatch;
                evaluation = EvaluateBatchAsync(selectedBatch);

                if (previousBatch != null)
                    BackpropagateBatch(tree, previousBatch, previousResults!);

                iterationCounter += LeavesPerBatch;
            }
            while ((iterationLimit > 0 ? iterationCounter < iterationLimit : stopwatch.Elapsed.TotalSeconds < timeLimitSeconds)
                && !cancellation.IsCancellationRequested);

            _evaluationWaitTime.Start();
            double[] lastResults = await evaluation.ConfigureAwait(false);
            _evaluationWaitTime.Stop();
            BackpropagateBatch(tree, evaluatedBatch, lastResults);

            LastSearch = new SearchStatistics(iterationCounter, _distinctLeaves, stopwatch.Elapsed.TotalSeconds,
                _selectionTime.Elapsed.TotalSeconds, _evaluationTime.Elapsed.TotalSeconds, _backpropagationTime.Elapsed.TotalSeconds,
                _evaluationWaitTime.Elapsed.TotalSeconds);
        }

        private LeafBatch SelectBatch(MCTSTree tree)
        {
            var leaves = new (MCTSNode leafNode, GameState stateAtLeaf)[LeavesPerBatch];
            DenseTensor<float> features = _featureBuffers[_nextBuffer];
            _nextBuffer ^= 1;
            Memory<float> featureBuffer = features.Buffer;
            int featureDim = _vectorizer.FeatureDim;

            void SelectLeaf(int i)
            {
                // Selection
                // Find a leaf node to expand, starting from the root node and traversing down the tree using the UCT formula
                (MCTSNode leafNode, GameState stateAtLeaf) = tree.SelectLeafNode();

                // Expansion
                // If the selected leaf isn't terminal, expand its children and select a leaf node among them
                if (!leafNode.IsTerminal)
                {
                    tree.ExpandNode(leafNode, stateAtLeaf);
                    (leafNode, stateAtLeaf) = tree.SelectLeafNode(leafNode, stateAtLeaf, suppressStateCopy: true);
                }

                tree.MarkPendingPath(leafNode);

                leaves[i] = (leafNode, stateAtLeaf);

                // Featurize right away, on the same thread (ended matches don't need the value net, their row is left as it is)
                if (!stateAtLeaf.HasEnded)
                    _vectorizer.VectorizeInto(stateAtLeaf, featureBuffer.Span.Slice(i * featureDim, featureDim));
            }

            _selectionTime.Start();
            if (_parallelSelection)
                Parallel.For(0, LeavesPerBatch, SelectLeaf);
            else
                for (int i = 0; i < LeavesPerBatch; i++) SelectLeaf(i);
            _selectionTime.Stop();
            _distinctLeaves += leaves.Select(leaf => leaf.leafNode).Distinct().Count();

            return new LeafBatch(leaves, features);
        }

        /// <summary>
        /// Utilities of the leaves of a batch, one per player (flat: leaf * 4 + player). Only reads the batch, so it can run
        /// while the tree is in use.
        /// </summary>
        private async Task<double[]> EvaluateBatchAsync(LeafBatch batch)
        {
            // Simulation
            // Ended matches have a known result, everything else is estimated by the value net in one batched run
            _evaluationTime.Start();
            var leaves = batch.Leaves;
            double[] utilities = new double[leaves.Length * MCTSNode.PlayerCount];
            double placementWeight = _placementWeight;

            LeafValues? values = null;
            if (leaves.Any(leaf => !leaf.stateAtLeaf.HasEnded))
                values = await _evaluator.EvaluateAsync(batch.Features, placementWeight > 0).ConfigureAwait(false);

            for (int i = 0; i < leaves.Length; i++)
            {
                Span<double> destination = utilities.AsSpan(i * MCTSNode.PlayerCount, MCTSNode.PlayerCount);
                if (leaves[i].stateAtLeaf.HasEnded)
                    TerminalUtilities(leaves[i].stateAtLeaf, placementWeight, destination);
                else
                    LeafUtilities(values!.Value.WinProbabilities.AsSpan(i * 4, 4),
                        values.Value.Placements == null ? default : values.Value.Placements.AsSpan(i * 16, 16), placementWeight, destination);
            }
            _evaluationTime.Stop();

            return utilities;
        }

        private void BackpropagateBatch(MCTSTree tree, LeafBatch batch, double[] utilities)
        {
            var leaves = batch.Leaves;
            sbyte owner = tree.OwnerIndex;

            void Backpropagate(int i)
            {
                // Backpropagation
                // Update the total scores and visit counts for all nodes in the path from the simulated node back to the root;
                // the opponent models that use every player's values get all four utilities, Random only the owner's
                if (tree.Model == OpponentModel.Random)
                    tree.Backpropagate(leaves[i].leafNode, utilities[i * MCTSNode.PlayerCount + owner], pathAlreadyMarked: true);
                else
                    tree.Backpropagate(leaves[i].leafNode, utilities.AsSpan(i * MCTSNode.PlayerCount, MCTSNode.PlayerCount), pathAlreadyMarked: true);
            }

            _backpropagationTime.Start();
            if (_parallelSelection)
                Parallel.For(0, leaves.Length, Backpropagate);
            else
                for (int i = 0; i < leaves.Length; i++) Backpropagate(i);
            _backpropagationTime.Stop();
        }

        /// <summary>
        /// Every player's utility from the value net: the win probability, plus the placement weight times the expected
        /// placement score if the placements are given
        /// </summary>
        public static void LeafUtilities(ReadOnlySpan<float> winProbabilities, ReadOnlySpan<float> placements, double placementWeight, Span<double> destination)
        {
            float sum = 0;
            foreach (float value in winProbabilities)
                sum += value;

            for (int player = 0; player < MCTSNode.PlayerCount; player++)
            {
                destination[player] = winProbabilities[player] / sum;
                if (!placements.IsEmpty)
                    destination[player] += placementWeight * MCTSAgent.PlacementScore(placements.Slice(player * 4, 4));
            }
        }

        /// <summary>
        /// Every player's utility in an ended match: 1 for the winner, plus the placement weight times the actual placement score
        /// </summary>
        public static void TerminalUtilities(GameState state, double placementWeight, Span<double> destination)
        {
            // Players can only win on their own turn, so the turn player of an ended match is the winner
            for (int player = 0; player < MCTSNode.PlayerCount; player++)
            {
                destination[player] = state.Turn.PlayerIndex == player ? 1 : 0;
                if (placementWeight > 0)
                    destination[player] += placementWeight * MCTSAgent.TerminalPlacementScore(state, player);
            }
        }
    }
}
