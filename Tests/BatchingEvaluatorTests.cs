using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Microsoft.ML.OnnxRuntime.Tensors;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The BatchingEvaluator, which collects the leaf batches of many searches into large value net batches, and searches
    /// sharing it. Uses the tiny untrained fixture net on the CPU.
    /// </summary>
    public class BatchingEvaluatorTests
    {
        private static string TinyModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        private static StateValueNet CpuNet() => new(TinyModelPath, intraOpThreads: 2, provider: InferenceProvider.Cpu);

        [Fact]
        public async Task ConcurrentRequestsGetTheirOwnRowsBack()
        {
            StateValueNet reference = CpuNet();
            using BatchingEvaluator evaluator = new([CpuNet(), CpuNet()], maxRows: 64);
            Random random = new(3);

            // Requests of different sizes, half with placements, sent at once from many threads
            var requests = Enumerable.Range(0, 200).Select(i =>
            {
                int rows = 1 + random.Next(40);
                var features = new DenseTensor<float>([rows, reference.InputDim]);
                for (int j = 0; j < features.Buffer.Length; j++) features.Buffer.Span[j] = (float)random.NextDouble();
                return (features, withPlacements: i % 2 == 0);
            }).ToArray();

            LeafValues[] results = await Task.WhenAll(requests.Select(request => Task.Run(() => evaluator.EvaluateAsync(request.features, request.withPlacements))));

            for (int i = 0; i < requests.Length; i++)
            {
                (float[] wins, float[] placements) = reference.RunWithPlacements(requests[i].features);
                Assert.Equal(wins.Length, results[i].WinProbabilities.Length);
                for (int j = 0; j < wins.Length; j++) Assert.Equal(wins[j], results[i].WinProbabilities[j], 1e-5);

                if (!requests[i].withPlacements)
                {
                    Assert.Null(results[i].Placements);
                    continue;
                }
                Assert.Equal(placements.Length, results[i].Placements!.Length);
                for (int j = 0; j < placements.Length; j++) Assert.Equal(placements[j], results[i].Placements![j], 1e-5);
            }

            Assert.Equal(requests.Sum(request => request.features.Dimensions[0]), evaluator.Rows);
            Assert.True(evaluator.Batches < requests.Length, "no request was batched with another one");
        }

        [Fact]
        public void RequestsLargerThanABatchAreRejected()
        {
            StateValueNet net = CpuNet();
            using BatchingEvaluator evaluator = new([net], maxRows: 16);
            // Rejected right away, not through the task
            Assert.Throws<ArgumentException>(() => { _ = evaluator.EvaluateAsync(new DenseTensor<float>([17, net.InputDim]), false); });
        }

        [Fact]
        public async Task SearchesSharingTheEvaluatorPlayValidMoves()
        {
            TestUtil.Seed(149);
            using BatchingEvaluator evaluator = new([CpuNet()], maxRows: 128);
            IStateVectorizer vectorizer = StateVectorizer.For(CpuNet());

            // Positions from the setup to the middle of the game, each searched on its own thread with small batches
            GameState[] positions = Enumerable.Range(0, 8).Select(i =>
            {
                GameState state = TestUtil.NewGame();
                TestUtil.PlayRandomGame(state, maxActions: 20 * i);
                return state;
            }).Where(state => !state.HasEnded).ToArray();

            const int Leaves = 16, Iterations = 400;
            var searches = positions.Select(async state =>
            {
                await Task.Yield();
                MCTSTree tree = new(new GameState(state), TestUtil.ActingPlayer(state), 1.414, model: OpponentModel.MaxN);
                MCTSSearch search = new(evaluator, vectorizer, Leaves, parallelSelection: false);
                await search.RunAsync(tree, double.PositiveInfinity, Iterations);
                return (state, tree, search.LastSearch!);
            }).ToArray();

            foreach ((GameState state, MCTSTree tree, SearchStatistics statistics) in await Task.WhenAll(searches))
            {
                Assert.InRange(statistics.Iterations, Iterations, Iterations + Leaves - 1);
                Assert.Equal((uint)statistics.Iterations, tree.RootNode.VisitCount);
                Assert.Equal(1.0, tree.RootValues().Sum(), 1e-6);

                Action move = tree.GetMostPromisingMove();
                Assert.True(move.IsValidFor(state));
            }
        }
    }
}
