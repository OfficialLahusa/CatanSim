using Common;
using Common.Serialization;
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
            }
            finally
            {
                Directory.Delete(workingDirectory, recursive: true);
            }
        }
    }
}
