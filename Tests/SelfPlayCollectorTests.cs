using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using SelfPlayCollector;
using System.Buffers.Binary;
using System.Globalization;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The self-play collector: command line, and a short run with the tiny untrained fixture net on the CPU whose output is
    /// checked against the games replayed from their seeds and save files
    /// </summary>
    public class SelfPlayCollectorTests
    {
        private static string TinyModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        [Fact]
        public void CommandLineSetsTheSearchBudgetsAndRejectsBadValues()
        {
            SelfPlaySettings settings = SelfPlayCollector.CommandLine.Parse(["--games", "20", "--full-iterations", "400", "--fast-iterations", "100",
                "--full-probability", "0.5", "--temperature-moves", "10", "--provider", "cpu", "--seed", "7"]).Settings;
            Assert.Equal(20, settings.Games);
            Assert.Equal(400, settings.FullIterations);
            Assert.Equal(100, settings.FastIterations);
            Assert.Equal(0.5, settings.FullSearchProbability);
            Assert.Equal(10, settings.TemperatureMoves);
            Assert.Equal(InferenceProvider.Cpu, settings.Provider);
            Assert.Equal(7UL, settings.Seed);

            Assert.Throws<ArgumentException>(() => SelfPlayCollector.CommandLine.Parse(["--full-probability", "1.5"]));
            Assert.Throws<ArgumentException>(() => SelfPlayCollector.CommandLine.Parse(["--leaves", "64", "--batch", "32"]));
            Assert.Throws<ArgumentException>(() => SelfPlayCollector.CommandLine.Parse(["--unknown"]));
        }

        [Theory]
        [InlineData("placements_tiny.onnx")]
        [InlineData("policy_tiny.onnx")]
        public void RunRecordsPositionsThatReplayFromTheSavedGames(string model)
        {
            string modelPath = TestUtil.FixturePath("ValueNet", model);
            string root = Path.Combine(Path.GetTempPath(), "catansim-selfplay-" + Guid.NewGuid().ToString("N"));
            try
            {
                SelfPlaySettings settings = new()
                {
                    Games = 3, ConcurrentGames = 3, LeavesPerBatch = 8, FullIterations = 32, FastIterations = 16, FullSearchProbability = 0.5,
                    TemperatureMoves = 10, MaxActions = 150, ModelPath = modelPath, Nets = 1, BatchRows = 64, Provider = InferenceProvider.Cpu,
                    Seed = 11, OutputDirectory = root, Name = "run",
                };
                SelfPlayCollector.SelfPlayCollector collector = new(settings, TextWriter.Null);
                SelfPlayResult result = collector.Collect();
                Assert.Equal(0, result.Errors);
                Assert.Equal(3, result.Games);

                string directory = collector.RunDirectory;
                IStateVectorizer vectorizer = StateVectorizer.For(new StateValueNet(modelPath, provider: InferenceProvider.Cpu));
                var games = ReadCsv(Path.Combine(directory, "games.csv"));
                var samples = ReadCsv(Path.Combine(directory, "samples.csv"));
                Assert.Equal(new long[] { 0, 1, 2 }, games.Select(game => long.Parse(game["game"])).Order());
                Assert.Equal(result.Positions, samples.Count);
                Assert.True(samples.Count > 0);

                byte[] states = File.ReadAllBytes(Path.Combine(directory, "states_c0000.f16"));
                Assert.Equal(samples.Count * vectorizer.FeatureDim * sizeof(ushort), states.Length);

                foreach (var game in games)
                {
                    long id = long.Parse(game["game"]);
                    ulong seed = ulong.Parse(game["seed"]);
                    Assert.Equal(Seeds.Derive(settings.Seed, (ulong)id), seed);

                    // The save file's actions replay from the seed's initial state to its final state
                    SaveFile saved = SelfPlayWriter.ReadRecord(directory, int.Parse(game["record_chunk"]), long.Parse(game["record_offset"]), int.Parse(game["record_length"]));
                    Assert.Equal(int.Parse(game["actions"]), saved.PlayedActions.Count);
                    Assert.Equal(game["finished"] == "1", saved.GameState.HasEnded);
                    Assert.Equal(saved.GameState.HasEnded ? saved.GameState.Turn.PlayerIndex : -1, int.Parse(game["winner"]));

                    GameState state = SelfPlayGame.InitialState(seed, settings.CenterDesert);
                    var positions = samples.Where(sample => sample["game"] == game["game"]).ToDictionary(sample => int.Parse(sample["action"]));
                    Assert.Equal(int.Parse(game["positions"]), positions.Count);

                    for (int i = 0; i <= saved.PlayedActions.Count; i++)
                    {
                        if (positions.TryGetValue(i, out var sample))
                            CheckPosition(sample, state, states, vectorizer, prunedVisits: model == "policy_tiny.onnx");
                        if (i == saved.PlayedActions.Count) break;

                        Action action = saved.PlayedActions[i];
                        Assert.True(action.IsValidFor(state), $"game {id}: action {i} doesn't replay");
                        action.Apply(state);
                    }
                    Assert.True(state.Equals(saved.GameState), $"game {id}: the replayed final state differs from the saved one");
                }

                // One entry of encoded moves per position, as many as the position's legal moves; writing them afterwards
                // by replaying the games gives the same files
                int[] counts = ReadInts(Path.Combine(directory, PolicyTargetWriter.CountsFile));
                Assert.Equal(samples.Count, counts.Length);
                Assert.Equal(counts.Sum() * ActionEncoding.MaxTokens * sizeof(short), new FileInfo(Path.Combine(directory, PolicyTargetWriter.TokensFile)).Length);
                string[] policyFiles = [PolicyTargetWriter.CountsFile, PolicyTargetWriter.TokensFile, PolicyTargetWriter.WeightsFile, PolicyTargetWriter.LayoutFile];
                var written = policyFiles.ToDictionary(name => name, name => File.ReadAllBytes(Path.Combine(directory, name)));
                foreach (string name in policyFiles) File.Delete(Path.Combine(directory, name));
                Assert.Equal(samples.Count, PolicyTargetWriter.ConvertRun(directory, TextWriter.Null));
                foreach (string name in policyFiles)
                    Assert.Equal(written[name], File.ReadAllBytes(Path.Combine(directory, name)));
                Assert.Throws<InvalidOperationException>(() => PolicyTargetWriter.ConvertRun(directory, TextWriter.Null));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            }
        }

        /// <summary>
        /// A recorded position matches the replayed state: its features, round and moves
        /// </summary>
        private static void CheckPosition(Dictionary<string, string> sample, GameState state, byte[] states, IStateVectorizer vectorizer, bool prunedVisits)
        {
            Assert.Equal(state.Turn.RoundCounter, int.Parse(sample["round"]));
            Assert.InRange(int.Parse(sample["actor"]), 0, 3);
            Assert.Equal(1.0, Enumerable.Range(0, 4).Sum(p => double.Parse(sample[$"q{p}"], CultureInfo.InvariantCulture)), 1e-6);

            float[] expected = new float[vectorizer.FeatureDim];
            vectorizer.VectorizeInto(state, expected);
            int row = int.Parse(sample["row"]);
            for (int f = 0; f < expected.Length; f++)
            {
                Half stored = BinaryPrimitives.ReadHalfLittleEndian(states.AsSpan((row * expected.Length + f) * sizeof(ushort)));
                Assert.Equal((Half)expected[f], stored);
            }

            // Every visited move is a valid action of the position, and the visits add up to the root's (minus its own first visit at most)
            int moves = MCTSTree.GetValidActions(state).Count;
            var visits = sample["visits"].Split(' ').Select(pair => pair.Split(':')).Select(pair => (move: int.Parse(pair[0]), visits: uint.Parse(pair[1]))).ToArray();
            Assert.All(visits, v => Assert.InRange(v.move, 0, moves - 1));
            // (with a policy head the visits only the root noise caused are left out, so they can add up to less)
            Assert.InRange((long)visits.Sum(v => v.visits), prunedVisits ? 1 : long.Parse(sample["root_visits"]) - 1, long.Parse(sample["root_visits"]));
        }

        private static int[] ReadInts(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            return Enumerable.Range(0, bytes.Length / sizeof(int)).Select(i => BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * sizeof(int)))).ToArray();
        }

        private static List<Dictionary<string, string>> ReadCsv(string path)
        {
            string[] lines = File.ReadAllLines(path);
            string[] header = lines[0].Split(',');
            return lines.Skip(1).Select(line => header.Zip(line.Split(',')).ToDictionary(pair => pair.First, pair => pair.Second)).ToList();
        }
    }
}
