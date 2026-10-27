using Agents.Inference;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using Action = Common.Actions.Action;

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
    /// With a value net that has a policy head, every evaluated leaf also gets the priors of its moves (PUCT, see
    /// MCTSTree.ApplyPriors); their legal moves are generated during selection, the priors computed with the evaluation and
    /// applied with the backpropagation. The root gets its priors from an evaluation of its own before the search.
    /// </summary>
    public sealed class MCTSSearch
    {
        // Leaves of a batch with their features, written by selection so the evaluation only runs the value net
        private sealed record LeafBatch((MCTSNode leafNode, GameState stateAtLeaf)[] Leaves, DenseTensor<float> Features, List<Action>?[] Moves);

        // Utilities per leaf and player (flat: leaf * 4 + player) and the priors of the leaves' moves (null without policy)
        private sealed record BatchResults(double[] Utilities, float[]?[] Priors);

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
        /// Dirichlet noise mixed into the root's priors at the start of each search (self-play exploration, needs a policy
        /// head): the noise's weight, 0 for none, and its total concentration (alpha per move = concentration / moves)
        /// </summary>
        public double RootNoiseWeight { get; set; }
        public double RootNoiseConcentration { get; set; } = 10;

        /// <summary>
        /// Random numbers for the root noise
        /// </summary>
        public Random NoiseRandom { get; set; } = Random.Shared;

        /// <summary>
        /// Search by PUCT with uniform priors instead of the policy head's (also for a value-only net): separates the effect
        /// of PUCT from that of the learned policy
        /// </summary>
        public bool UniformPriors { get; set; }

        // Priors from the policy head
        private bool UsesPolicy => _evaluator.Policy != null && !UniformPriors;

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

            tree.UsesPolicy = UsesPolicy || UniformPriors;
            if (UsesPolicy && !tree.RootState.HasEnded && (!tree.RootNode.HasPolicyPriors || RootNoiseWeight > 0))
                await EvaluateRootPriorsAsync(tree).ConfigureAwait(false);

            LeafBatch? evaluatedBatch = null;
            Task<BatchResults>? evaluation = null;

            long iterationCounter = 0;
            do
            {
                LeafBatch selectedBatch = SelectBatch(tree);

                BatchResults? previousResults = null;
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
            BatchResults lastResults = await evaluation.ConfigureAwait(false);
            _evaluationWaitTime.Stop();
            BackpropagateBatch(tree, evaluatedBatch, lastResults);

            LastSearch = new SearchStatistics(iterationCounter, _distinctLeaves, stopwatch.Elapsed.TotalSeconds,
                _selectionTime.Elapsed.TotalSeconds, _evaluationTime.Elapsed.TotalSeconds, _backpropagationTime.Elapsed.TotalSeconds,
                _evaluationWaitTime.Elapsed.TotalSeconds);
        }

        /// <summary>
        /// The root's priors from a policy evaluation of its own, with the root noise mixed in
        /// </summary>
        private async Task EvaluateRootPriorsAsync(MCTSTree tree)
        {
            var features = new DenseTensor<float>([1, _vectorizer.FeatureDim]);
            _vectorizer.VectorizeInto(tree.RootState, features.Buffer.Span);
            LeafValues values = await _evaluator.EvaluateAsync(features, false, withPolicy: true).ConfigureAwait(false);
            List<Action> moves = MCTSTree.GetValidActions(tree.RootState);
            float[] priors = Priors(values.PolicyQueries!, 0, moves);

            if (RootNoiseWeight > 0 && priors.Length > 1)
            {
                double[] noise = Dirichlet(RootNoiseConcentration / priors.Length, priors.Length, NoiseRandom);
                for (int i = 0; i < priors.Length; i++)
                    priors[i] = (float)((1 - RootNoiseWeight) * priors[i] + RootNoiseWeight * noise[i]);
            }
            tree.ApplyPriors(tree.RootNode, priors);
        }

        /// <summary>
        /// The policy's distribution over the moves from the queries of row `row`
        /// </summary>
        private float[] Priors(float[] queries, int row, IReadOnlyList<Action> moves)
        {
            PolicyHead policy = _evaluator.Policy!;
            float[] priors = new float[moves.Count];
            ActionEncoding.Logits(queries.AsSpan(row * policy.QueryLength, policy.QueryLength), policy.Embeddings, policy.Dim, moves, priors);
            ActionEncoding.Softmax(priors);
            return priors;
        }

        /// <summary>
        /// A sample of the symmetric Dirichlet distribution (normalized Gamma(alpha) variates)
        /// </summary>
        public static double[] Dirichlet(double alpha, int count, Random random)
        {
            double[] sample = new double[count];
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                sample[i] = Gamma(alpha, random);
                sum += sample[i];
            }
            for (int i = 0; i < count; i++)
                sample[i] = sum > 0 ? sample[i] / sum : 1.0 / count;
            return sample;
        }

        private static double Gamma(double alpha, Random random)
        {
            // Marsaglia and Tsang; for alpha < 1 boosted with U^(1/alpha)
            if (alpha < 1)
                return Gamma(alpha + 1, random) * Math.Pow(1 - random.NextDouble(), 1 / alpha);
            double d = alpha - 1.0 / 3, c = 1 / Math.Sqrt(9 * d);
            while (true)
            {
                double x, v;
                do
                {
                    // Standard normal by Box-Muller
                    x = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
                    v = 1 + c * x;
                }
                while (v <= 0);
                v = v * v * v;
                double u = 1 - random.NextDouble();
                if (Math.Log(u) < 0.5 * x * x + d - d * v + d * Math.Log(v))
                    return d * v;
            }
        }

        private LeafBatch SelectBatch(MCTSTree tree)
        {
            var leaves = new (MCTSNode leafNode, GameState stateAtLeaf)[LeavesPerBatch];
            var moves = new List<Action>?[LeavesPerBatch];
            bool usesPolicy = UsesPolicy;
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
                {
                    _vectorizer.VectorizeInto(stateAtLeaf, featureBuffer.Span.Slice(i * featureDim, featureDim));
                    // The leaf's moves for its priors, unless it has them already (a leaf can be selected again before)
                    if (usesPolicy && !leafNode.HasPolicyPriors)
                        moves[i] = MCTSTree.GetValidActions(stateAtLeaf);
                }
            }

            _selectionTime.Start();
            if (_parallelSelection)
                Parallel.For(0, LeavesPerBatch, SelectLeaf);
            else
                for (int i = 0; i < LeavesPerBatch; i++) SelectLeaf(i);
            _selectionTime.Stop();
            _distinctLeaves += leaves.Select(leaf => leaf.leafNode).Distinct().Count();

            return new LeafBatch(leaves, features, moves);
        }

        /// <summary>
        /// Utilities of the leaves of a batch, one per player (flat: leaf * 4 + player), and the priors of their moves. Only
        /// reads the batch, so it can run while the tree is in use.
        /// </summary>
        private async Task<BatchResults> EvaluateBatchAsync(LeafBatch batch)
        {
            // Simulation
            // Ended matches have a known result, everything else is estimated by the value net in one batched run
            _evaluationTime.Start();
            var leaves = batch.Leaves;
            double[] utilities = new double[leaves.Length * MCTSNode.PlayerCount];
            double placementWeight = _placementWeight;

            LeafValues? values = null;
            bool withPolicy = batch.Moves.Any(moves => moves != null);
            if (leaves.Any(leaf => !leaf.stateAtLeaf.HasEnded))
                values = await _evaluator.EvaluateAsync(batch.Features, placementWeight > 0, withPolicy).ConfigureAwait(false);

            var priors = new float[]?[leaves.Length];
            if (withPolicy)
                for (int i = 0; i < leaves.Length; i++)
                    if (batch.Moves[i] is List<Action> moves)
                        priors[i] = Priors(values!.Value.PolicyQueries!, i, moves);

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

            return new BatchResults(utilities, priors);
        }

        private void BackpropagateBatch(MCTSTree tree, LeafBatch batch, BatchResults results)
        {
            var leaves = batch.Leaves;
            sbyte owner = tree.OwnerIndex;
            double[] utilities = results.Utilities;

            void Backpropagate(int i)
            {
                if (results.Priors[i] is float[] priors)
                    tree.ApplyPriors(leaves[i].leafNode, priors);

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
