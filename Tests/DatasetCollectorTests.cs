using Agents;
using Common;
using Common.Actions;
using Common.Serialization;
using System.Buffers.Binary;
using System.Globalization;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The dataset collector: labels and the samples it writes
    /// </summary>
    public class DatasetCollectorTests
    {
        [Fact]
        public void WinCountsAreNormalizedAndEmptyCountsGiveNoLabel()
        {
            Assert.Null(DatasetCollector.DatasetCollector.NormalizeWinCounts([0, 0, 0, 0]));
            Assert.Equal([0.25, 0.75, 0.0, 0.0], DatasetCollector.DatasetCollector.NormalizeWinCounts([1, 3, 0, 0]));
        }

        [Fact]
        public void CollectedSamplesAreConsistentGamePrefixesWithValidLabels()
        {
            TestUtil.Seed(150);
            string workingDirectory = Path.Combine(Path.GetTempPath(), "catansim-collector-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(workingDirectory);
            string previousDirectory = Environment.CurrentDirectory;

            try
            {
                // The collector writes relative to the working directory
                Environment.CurrentDirectory = workingDirectory;
                new DatasetCollector.DatasetCollector(threadCount: 2, matchCount: 2, samplesPerMatch: 4, playoutsPerSample: 6).Collect();
            }
            finally
            {
                Environment.CurrentDirectory = previousDirectory;
            }

            try
            {
                string datasetDirectory = Directory.GetDirectories(Path.Combine(workingDirectory, "datasets")).Single();
                string[] samples = Directory.GetFiles(Path.Combine(datasetDirectory, "input"), "*.yaml");
                Assert.Equal(8, samples.Length);

                foreach (int match in new[] { 0, 1 })
                {
                    List<(SaveFile save, double[] label)> matchSamples = samples
                        .Where(path => Path.GetFileName(path).StartsWith($"r{match}_"))
                        .OrderBy(path => path)
                        .Select(path => (SaveFileSerializer.Deserialize(File.ReadAllText(path)),
                            File.ReadAllLines(Path.Combine(datasetDirectory, "output", Path.GetFileNameWithoutExtension(path) + ".txt"))
                                .Where(line => line.Length > 0).Select(line => double.Parse(line, CultureInfo.InvariantCulture)).ToArray()))
                        .ToList();

                    // Sample 0 is the empty board, the last one the ended game
                    Assert.Empty(matchSamples[0].save.PlayedActions);
                    Assert.True(matchSamples[^1].save.GameState.HasEnded);

                    // Each sample stores exactly the actions that led to it: reverting them always reaches the same base state
                    GameState? baseState = null;
                    int previousCount = -1;
                    foreach ((SaveFile save, double[] label) in matchSamples)
                    {
                        Assert.True(save.PlayedActions.Count > previousCount);
                        previousCount = save.PlayedActions.Count;

                        GameState state = new GameState(save.GameState);
                        for (int i = save.PlayedActions.Count - 1; i >= 0; i--)
                            save.PlayedActions[i].Revert(state);

                        baseState ??= state;
                        Assert.True(state.Equals(baseState), "a sample's action log does not lead back to the match's base state");

                        Assert.Equal(TestUtil.PlayerCount, label.Length);
                        Assert.All(label, value => Assert.True(double.IsFinite(value) && value >= 0));
                        Assert.Equal(1.0, label.Sum(), 9);
                    }

                    // The ended game's label goes entirely to its winner
                    (SaveFile last, double[] lastLabel) = matchSamples[^1];
                    Assert.Equal(1.0, lastLabel[last.GameState.Turn.PlayerIndex], 9);
                }

                // Every sample has exactly one row in the vector chunks, equal to the vector of its state as float16
                Dictionary<string, Half[]> vectors = ReadVectorChunks(Path.Combine(datasetDirectory, "vectors"));
                Assert.Equal(samples.Select(Path.GetFileNameWithoutExtension).Order(), vectors.Keys.Order());

                StateVectorizer vectorizer = new();
                foreach (string path in samples)
                {
                    float[] expected = vectorizer.Vectorize(SaveFileSerializer.Deserialize(File.ReadAllText(path)).GameState).ToArray();
                    Half[] row = vectors[Path.GetFileNameWithoutExtension(path)!];
                    Assert.Equal(expected.Select(value => (Half)value), row);
                }
            }
            finally
            {
                Directory.Delete(workingDirectory, recursive: true);
            }
        }

        [Fact]
        public void AssembledSaveFilesAreIdenticalToSerializingEachSample()
        {
            TestUtil.Seed(152);
            HashSet<string> actionTypes = [];

            for (int game = 0; game < 6; game++)
            {
                GameState start = TestUtil.NewGame();
                List<Action> played = TestUtil.PlayRandomGame(new GameState(start));
                actionTypes.UnionWith(played.Select(action => action.GetType().Name));
                var saveFiles = new DatasetCollector.MatchSaveFiles(played);

                // Replay copies, so the recorded actions and their histories stay as the game left them
                GameState state = new GameState(start);
                for (int count = 0; count <= played.Count; count++)
                {
                    if (count > 0) played[count - 1].Clone().Apply(state);

                    // The whole game covers every action that occurred, the other lengths where the history is cut
                    if (count <= 2 || count % 50 == 0 || count == played.Count)
                    {
                        string expected = SaveFileSerializer.Serialize(new SaveFile(state, played.GetRange(0, count), []));
                        Assert.True(expected == saveFiles.Build(state, count), $"game {game}: assembled file with {count} actions differs");
                    }
                }

                // Every file was assembled, none serialized on its own
                Assert.Equal(0, saveFiles.FallbackCount);
            }

            // The random games included actions with nested sequences in their histories
            Assert.Contains(nameof(RollAction), actionTypes);
            Assert.Contains(nameof(DiscardAction), actionTypes);
            Assert.Contains(nameof(RobberAction), actionTypes);
        }

        [Fact]
        public void VectorChunksRollOverAfterTheirRowLimit()
        {
            TestUtil.Seed(151);
            string directory = Path.Combine(Path.GetTempPath(), "catansim-vectors-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);

            try
            {
                List<GameState> states = [];
                GameState state = TestUtil.NewGame();
                TestUtil.PlayRandomGame(state, afterApply: (s, _) => { if (states.Count < 7) states.Add(new GameState(s)); }, maxActions: 50);

                using (var writer = new DatasetCollector.VectorChunkWriter(directory, workerIdx: 3, rowsPerChunk: 3))
                {
                    for (int i = 0; i < states.Count; i++)
                        writer.Write($"r0_s{i}", states[i]);
                }

                // 7 rows in chunks of 3, 3 and 1, named after the worker
                Assert.Equal(["w03_c0000", "w03_c0001", "w03_c0002"],
                    Directory.GetFiles(directory, "*.f16").Select(Path.GetFileNameWithoutExtension).Order());
                Assert.Equal([3, 3, 1], Directory.GetFiles(directory, "*.txt").Order().Select(path => File.ReadAllLines(path).Length));

                Dictionary<string, Half[]> vectors = ReadVectorChunks(directory);
                StateVectorizer vectorizer = new();
                for (int i = 0; i < states.Count; i++)
                    Assert.Equal(vectorizer.Vectorize(states[i]).ToArray().Select(value => (Half)value), vectors[$"r0_s{i}"]);
            }
            finally
            {
                Directory.Delete(directory, recursive: true);
            }
        }

        /// <summary>
        /// The rows of all chunk files in the directory by sample name
        /// </summary>
        private static Dictionary<string, Half[]> ReadVectorChunks(string directory)
        {
            Dictionary<string, Half[]> vectors = [];
            foreach (string dataPath in Directory.GetFiles(directory, "*.f16"))
            {
                byte[] data = File.ReadAllBytes(dataPath);
                string[] names = File.ReadAllLines(Path.ChangeExtension(dataPath, ".txt"));
                int rowBytes = StateVectorizer.FeatureDim * sizeof(ushort);
                Assert.Equal(names.Length * rowBytes, data.Length);

                for (int row = 0; row < names.Length; row++)
                {
                    Half[] values = new Half[StateVectorizer.FeatureDim];
                    for (int i = 0; i < values.Length; i++)
                        values[i] = BinaryPrimitives.ReadHalfLittleEndian(data.AsSpan(row * rowBytes + i * sizeof(ushort)));
                    vectors.Add(names[row], values);
                }
            }
            return vectors;
        }
    }
}
