using Agents;
using Common;
using Common.Actions;
using Common.Serialization;
using DatasetCollector;
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
        public void EverySampleIsAGamePrefixWithMatchingLabelsAndRecords()
        {
            // More samples than positions: every position of the match is sampled, the empty board and the ended match included
            var settings = new CollectorSettings { Threads = 1, Matches = 1, SamplesPerMatch = 100_000, PlayoutsPerSample = 3, Seed = 150, SiblingGroupsPerMatch = 0 };
            string datasetDirectory = CollectIntoTemporaryDirectory(settings, out CollectionResult result);

            try
            {
                List<SampleRecord> records = ReadSampleRecords(datasetDirectory);
                string[] samples = Directory.GetFiles(Path.Combine(datasetDirectory, "input"), "*.yaml");
                Assert.Equal(records.Count, samples.Length);
                Assert.Equal(records.Count, result.Samples);
                Assert.Equal(records[0].MatchLength + 1, records.Count);
                Assert.Equal(Enumerable.Range(0, records.Count), records.Select(record => record.Position).Order());

                GameState? baseState = null;
                foreach (SampleRecord record in records)
                {
                    SaveFile save = SaveFileSerializer.Deserialize(File.ReadAllText(Path.Combine(datasetDirectory, "input", record.Name + ".yaml")));
                    double[] label = File.ReadAllLines(Path.Combine(datasetDirectory, "output", record.Name + ".txt"))
                        .Where(line => line.Length > 0).Select(line => double.Parse(line, CultureInfo.InvariantCulture)).ToArray();

                    // Each sample stores exactly the actions that led to it: reverting them always reaches the same base state
                    Assert.Equal(record.Position, save.PlayedActions.Count);
                    GameState state = new GameState(save.GameState);
                    for (int i = save.PlayedActions.Count - 1; i >= 0; i--)
                        save.PlayedActions[i].Revert(state);
                    baseState ??= state;
                    Assert.True(state.Equals(baseState), "a sample's action log does not lead back to the match's base state");

                    Assert.Equal(Seeds.Derive(settings.Seed, 0), record.Group);
                    Assert.Equal(save.GameState.Turn.RoundCounter, record.Round);
                    Assert.Equal(record.Position == record.MatchLength, record.Terminal);
                    Assert.Equal(record.Terminal, save.GameState.HasEnded);
                    Assert.Equal(TestUtil.PlayerCount, label.Length);
                    Assert.Equal(1.0, label.Sum(), 9);

                    if (record.Terminal)
                    {
                        // The ended match needs no playouts, its label goes entirely to its winner
                        Assert.Empty(record.Playouts);
                        Assert.Equal(save.GameState.Turn.PlayerIndex, record.MatchWinner);
                        Assert.Equal(1.0, label[record.MatchWinner], 9);
                        Assert.Equal(save.GameState.Players.Select(player => player.VictoryPoints.Total), record.MatchVictoryPoints);
                    }
                    else
                    {
                        // The label is the win rate over the recorded playouts, whose winners reached the victory points
                        Assert.Equal(settings.PlayoutsPerSample, record.Playouts.Count);
                        Assert.Equal(record.Playouts.Count(playout => playout.Winner >= 0), record.Finished);
                        for (int p = 0; p < TestUtil.PlayerCount; p++)
                            Assert.Equal((double)record.Playouts.Count(playout => playout.Winner == p) / record.Finished, label[p], 9);
                        Assert.All(record.Playouts.Where(playout => playout.Winner >= 0), playout =>
                        {
                            Assert.True(playout.VictoryPoints[playout.Winner] >= save.GameState.Settings.VictoryPoints);
                            Assert.True(playout.EndRound >= record.Round);
                            Assert.All(playout.Agents, kind => Assert.True(kind <= (int)AgentKind.Builder));
                        });
                    }
                }

                // Every sample has exactly one row in the vector chunks, equal to the vector of its state as float16
                Dictionary<string, Half[]> vectors = ReadVectorChunks(Path.Combine(datasetDirectory, "vectors"));
                Assert.Equal(records.Select(record => record.Name).Order(), vectors.Keys.Order());

                StateVectorizer vectorizer = new();
                foreach (string path in samples)
                {
                    float[] expected = vectorizer.Vectorize(SaveFileSerializer.Deserialize(File.ReadAllText(path)).GameState).ToArray();
                    Assert.Equal(expected.Select(value => (Half)value), vectors[Path.GetFileNameWithoutExtension(path)!]);
                }

                string metadata = File.ReadAllText(Path.Combine(datasetDirectory, "metadata.json"));
                Assert.Contains("\"status\": \"finished\"", metadata);
                Assert.Contains("\"seed\": \"150\"", metadata);
            }
            finally
            {
                DeleteTemporaryDirectory(datasetDirectory);
            }
        }

        [Fact]
        public void TheSameSeedGivesTheSameDatasetWithAnyThreadCount()
        {
            var settings = new CollectorSettings { Matches = 6, SamplesPerMatch = 3, PlayoutsPerSample = 4, Seed = 151 };
            string single = CollectIntoTemporaryDirectory(settings with { Threads = 1 }, out _);
            string multiple = CollectIntoTemporaryDirectory(settings with { Threads = 3 }, out _);

            try
            {
                // Save files and labels are identical, and so are the playouts of every sample (in whatever worker file)
                foreach (string kind in new[] { "input", "output" })
                {
                    string[] files = Directory.GetFiles(Path.Combine(single, kind)).Select(path => Path.GetFileName(path)).Order().ToArray();
                    Assert.Equal(files, Directory.GetFiles(Path.Combine(multiple, kind)).Select(path => Path.GetFileName(path)).Order());
                    foreach (string file in files)
                        Assert.Equal(File.ReadAllBytes(Path.Combine(single, kind, file)), File.ReadAllBytes(Path.Combine(multiple, kind, file)));
                }

                Dictionary<string, SampleRecord> a = ReadSampleRecords(single).ToDictionary(record => record.Name);
                Dictionary<string, SampleRecord> b = ReadSampleRecords(multiple).ToDictionary(record => record.Name);
                Assert.Equal(a.Keys.Order(), b.Keys.Order());
                foreach (string name in a.Keys)
                    Assert.Equal(a[name].Playouts, b[name].Playouts);

                // A match seed reproduces its match
                ulong matchSeed = Seeds.Derive(settings.Seed, 2);
                var first = DatasetCollector.DatasetCollector.PlayMatch(matchSeed, 2, settings.Agents, false);
                var second = DatasetCollector.DatasetCollector.PlayMatch(matchSeed, 2, settings.Agents, false);
                Assert.Equal(SaveFileSerializer.Serialize(first.playedActions), SaveFileSerializer.Serialize(second.playedActions));
                Assert.Equal(first.playedActions.Count, a.Values.First(record => record.Group == matchSeed).MatchLength);
            }
            finally
            {
                DeleteTemporaryDirectory(single);
                DeleteTemporaryDirectory(multiple);
            }
        }

        [Fact]
        public void SiblingGroupsBranchOffWithOneDistinctMoveEach()
        {
            // Every position is sampled and every one that can gets a group: the last state before the end then has the winning
            // move as a sibling, the ended match none, and states where only a roll is possible none
            var settings = new CollectorSettings
            {
                Threads = 1, Matches = 1, SamplesPerMatch = 100_000, PlayoutsPerSample = 1, Seed = 155,
                SiblingGroupsPerMatch = int.MaxValue, SiblingsPerGroup = 4,
            };
            string datasetDirectory = CollectIntoTemporaryDirectory(settings, out CollectionResult result);

            try
            {
                Dictionary<string, SampleRecord> records = ReadSampleRecords(datasetDirectory).ToDictionary(record => record.Name);
                SaveFile Load(string name) => SaveFileSerializer.Load(Path.Combine(datasetDirectory, "input", name + ".yaml"));
                string Text(string name) => File.ReadAllText(Path.Combine(datasetDirectory, "input", name + ".yaml"));

                List<SampleRecord> roots = records.Values.Where(record => record.Source == "root").ToList();
                Assert.NotEmpty(roots);
                int terminalSiblings = 0, matchSiblings = 0;

                foreach (SampleRecord record in records.Values.Where(record => record.Source == "" && !record.Name.Contains("_c")))
                {
                    // States without a group: ended, or without two distinct moves besides rolling
                    GameState state = Load(record.Name).GameState;
                    Assert.True(state.HasEnded || DatasetCollector.DatasetCollector.SiblingStates(state, null, 4, 1).Count == 0, $"{record.Name} has no group");
                }

                ILookup<string, SampleRecord> groupMembers = records.Values.Where(record => record.Source is not ("" or "root")).ToLookup(record => record.SiblingGroup);
                foreach (SampleRecord root in roots)
                {
                    GameState rootState = Load(root.Name).GameState;
                    Assert.False(rootState.HasEnded);
                    Assert.Equal(root.Name, root.SiblingGroup);
                    Assert.Equal(TestUtil.ActingPlayer(rootState), root.Actor);

                    List<SampleRecord> siblings = groupMembers[root.Name].OrderBy(record => record.Name).ToList();
                    Assert.InRange(siblings.Count, 2, settings.SiblingsPerGroup);
                    Assert.Equal(Enumerable.Range(0, siblings.Count).Select(i => $"{root.Name}_c{i}"), siblings.Select(sibling => sibling.Name));

                    List<GameState> siblingStates = [];
                    foreach (SampleRecord sibling in siblings)
                    {
                        // The file holds only the move and references the root's file, loading puts the history together
                        SaveFile stored = SaveFileSerializer.Deserialize(Text(sibling.Name));
                        Assert.Equal(root.Name + ".yaml", stored.HistoryReference);
                        Action move = Assert.Single(stored.PlayedActions);
                        SaveFile save = Load(sibling.Name);
                        Assert.Null(save.LoadWarning);
                        Assert.Null(save.HistoryReference);

                        // The root's actions plus one move of the acting player, never a roll, leading to the sibling's state
                        Assert.Equal(root.Position + 1, sibling.Position);
                        Assert.Equal(root.Position + 1, save.PlayedActions.Count);
                        Assert.IsNotType<RollAction>(move);
                        Assert.Equal(root.Actor, move.PlayerIndex);
                        Assert.Equal(move.GetType().Name, sibling.Move);
                        Assert.Equal(root.Actor, sibling.Actor);
                        if (sibling.Chance) Assert.True(move is BuyDevelopmentCardAction or RobberAction);

                        GameState replayed = new GameState(rootState);
                        move.Clone().Apply(replayed);
                        Assert.True(replayed.Equals(save.GameState), $"{sibling.Name} is not its root after its move");
                        Assert.DoesNotContain(siblingStates, other => other.Equals(save.GameState));
                        siblingStates.Add(save.GameState);

                        // The same playout seeds as the root: as many playouts, and an ended sibling none, labeled by its result
                        Assert.Equal(save.GameState.HasEnded, sibling.Terminal);
                        if (sibling.Terminal)
                        {
                            terminalSiblings++;
                            Assert.Empty(sibling.Playouts);
                            Assert.Equal(root.Actor, sibling.EndWinner);
                            Assert.Equal(save.GameState.Players.Select(player => player.VictoryPoints.Total), sibling.EndVictoryPoints!);
                        }
                        else
                        {
                            Assert.Equal(settings.PlayoutsPerSample, sibling.Playouts.Count);
                        }

                        // The match's own move leads to the match's next state, which was sampled too: loaded, it is that file
                        if (sibling.Source == "match")
                        {
                            matchSiblings++;
                            string next = $"r{root.Name[1..root.Name.IndexOf('_')]}_s{int.Parse(root.Name[(root.Name.IndexOf("_s") + 2)..]) + 1}";
                            Assert.Equal(sibling.Position, records[next].Position);
                            Assert.Equal(Text(next), SaveFileSerializer.Serialize(save));
                        }
                    }
                }

                // The match's winning move is the sibling of the state before the end that ends the match
                foreach (SampleRecord beforeEnd in roots.Where(root => root.Position == root.MatchLength - 1))
                    Assert.Contains(records.Values, record => record.SiblingGroup == beforeEnd.Name && record.Source == "match" && record.Terminal);
                Assert.True(terminalSiblings > 0);
                Assert.True(matchSiblings > roots.Count / 2);
                Assert.Equal(0, result.Errors);
            }
            finally
            {
                DeleteTemporaryDirectory(datasetDirectory);
            }
        }

        [Fact]
        public void PartialRunsEqualASingleRunAndResume()
        {
            string root = Path.Combine(Path.GetTempPath(), "catansim-parts-" + Guid.NewGuid().ToString("N"));
            var settings = new CollectorSettings { Threads = 2, Matches = 6, SamplesPerMatch = 2, PlayoutsPerSample = 3, Seed = 160, SiblingsPerGroup = 3, OutputDirectory = root };

            try
            {
                PartialRuns.Collect(settings with { Name = "single" }, TextWriter.Null);
                PartialRuns.Collect(settings with { Name = "parts", Parts = 3 }, TextWriter.Null);

                // Three parts of two matches each, numbered on from the previous part, together the same files as the single run
                string[] parts = Directory.GetDirectories(Path.Combine(root, "parts")).Select(path => Path.GetFileName(path)!).Order().ToArray();
                Assert.Equal(["part_00", "part_01", "part_02"], parts);
                Dictionary<string, string> Files(params string[] directories) => directories
                    .SelectMany(directory => Directory.GetFiles(Path.Combine(directory, "input")).Concat(Directory.GetFiles(Path.Combine(directory, "output"))))
                    .ToDictionary(path => Path.GetFileName(Path.GetDirectoryName(path)) + "/" + Path.GetFileName(path), File.ReadAllText);
                Dictionary<string, string> single = Files(Path.Combine(root, "single"));
                Dictionary<string, string> combined = Files(parts.Select(part => Path.Combine(root, "parts", part)).ToArray());
                Assert.Equal(single.Keys.Order(), combined.Keys.Order());
                Assert.All(single, file => Assert.Equal(file.Value, combined[file.Key]));
                Assert.All(Directory.GetFiles(Path.Combine(root, "parts", "part_01", "input")), path => Assert.Matches(@"^r[23]_", Path.GetFileName(path)));

                // Resuming: a finished part stays as it is, an unfinished one is collected again with the same result
                string part0Metadata = Path.Combine(root, "parts", "part_00", "metadata.json");
                DateTime part0Written = File.GetLastWriteTimeUtc(part0Metadata);
                File.Delete(Path.Combine(root, "parts", "part_01", "metadata.json"));
                File.Delete(Directory.GetFiles(Path.Combine(root, "parts", "part_01", "input"))[0]);
                PartialRuns.Collect(settings with { Name = "parts", Parts = 3 }, TextWriter.Null);
                Assert.Equal(part0Written, File.GetLastWriteTimeUtc(part0Metadata));
                Dictionary<string, string> resumed = Files(parts.Select(part => Path.Combine(root, "parts", part)).ToArray());
                Assert.Equal(combined.Keys.Order(), resumed.Keys.Order());
                Assert.All(combined, file => Assert.Equal(file.Value, resumed[file.Key]));

                // A part collected with other settings stops the run instead of being replaced
                File.WriteAllText(Path.Combine(root, "parts", "part_02", "metadata.json"),
                    File.ReadAllText(Path.Combine(root, "parts", "part_02", "metadata.json")).Replace("\"status\": \"finished\"", "\"status\": \"running\""));
                var error = Assert.Throws<InvalidOperationException>(() => PartialRuns.Collect(settings with { Name = "parts", Parts = 3, PlayoutsPerSample = 4 }, TextWriter.Null));
                Assert.Contains("other settings", error.Message);
                Assert.True(Directory.Exists(Path.Combine(root, "parts", "part_02", "input")));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
            }
        }

        [Fact]
        public void SampledPositionsAreDistinctAndUniform()
        {
            TestUtil.Seed(153);
            int[] counts = new int[10];
            const int draws = 20_000;
            for (int i = 0; i < draws; i++)
            {
                int[] positions = DatasetCollector.DatasetCollector.SamplePositions(lastPosition: 9, count: 4);
                Assert.Equal(positions.Order(), positions);
                Assert.Equal(4, positions.Distinct().Count());
                foreach (int position in positions) counts[position]++;
            }

            // Every position, the first and the last included, has the chance k/(N+1) = 0.4
            Assert.All(counts, count => Assert.InRange(count / (double)draws, 0.38, 0.42));

            // With fewer positions than samples, all are taken
            Assert.Equal([0, 1, 2], DatasetCollector.DatasetCollector.SamplePositions(lastPosition: 2, count: 5));
        }

        [Fact]
        public void TheCommandLineSetsEverySetting()
        {
            var commandLine = CommandLine.Parse(["--matches", "7", "--samples", "3", "--playouts", "50", "--threads", "5", "--seed", "18446744073709551615",
                "--agents", "builder=1", "--exploration", "0.05-0.1", "--center-desert", "--output", "out", "--name", "run", "--sibling-groups", "2", "--siblings", "8", "--parts", "7", "--first-match", "100"]);
            CollectorSettings settings = commandLine.Settings;

            Assert.Equal((7, 3, 50, 5, ulong.MaxValue, true, "out", "run"),
                (settings.Matches, settings.SamplesPerMatch, settings.PlayoutsPerSample, settings.Threads, settings.Seed, settings.CenterDesert, settings.OutputDirectory, settings.Name));
            Assert.Equal(new AgentMix(0, 0, 1, 0.05, 0.1), settings.Agents);
            Assert.Equal((2, 8, 7, 100), (settings.SiblingGroupsPerMatch, settings.SiblingsPerGroup, settings.Parts, settings.FirstMatch));
            Assert.False(commandLine.CalibrateThreads);

            Assert.True(CommandLine.Parse(["--threads", "auto"]).CalibrateThreads);
            Assert.True(CommandLine.Parse(["--calibrate"]).CalibrateOnly);
            Assert.Equal(AgentMix.Default, CommandLine.Parse([]).Settings.Agents);
            Assert.Throws<ArgumentException>(() => CommandLine.Parse(["--agents", "random=0"]));
            Assert.Throws<ArgumentException>(() => CommandLine.Parse(["--unknown"]));
            Assert.Throws<ArgumentException>(() => CommandLine.Parse(["--siblings", "1"]));
            Assert.Throws<ArgumentException>(() => CommandLine.Parse(["--matches", "3", "--parts", "4"]));

            Assert.Equal([32, 24, 16, 8], ThreadCalibration.Candidates(32));
            Assert.Equal([64, 48, 32, 24, 16], ThreadCalibration.Candidates(64));
            Assert.Equal([32, 24, 16, 12, 8, 4], ThreadCalibration.Candidates(16));
            Assert.Equal([32, 24, 1], ThreadCalibration.Candidates(1));
        }

        [Fact]
        public void ACancelledRunAbandonsItsMatches()
        {
            string directory = Path.Combine(Path.GetTempPath(), "catansim-collector-" + Guid.NewGuid().ToString("N"));
            try
            {
                var settings = new CollectorSettings { Threads = 2, Matches = 1000, PlayoutsPerSample = 1000, Seed = 154, OutputDirectory = directory, Name = "cancelled" };
                CollectionResult result = new DatasetCollector.DatasetCollector(settings, TextWriter.Null).Collect(new CancellationToken(canceled: true));

                Assert.True(result.Cancelled);
                Assert.Equal(0, result.Samples);
                Assert.Contains("\"status\": \"cancelled\"", File.ReadAllText(Path.Combine(directory, "cancelled", "metadata.json")));
            }
            finally
            {
                DeleteTemporaryDirectory(directory);
            }
        }

        [Fact]
        public void AnExistingDatasetIsNeverOverwritten()
        {
            string directory = Path.Combine(Path.GetTempPath(), "catansim-collector-" + Guid.NewGuid().ToString("N"));
            try
            {
                var settings = new CollectorSettings { Threads = 2, Matches = 1000, PlayoutsPerSample = 1000, Seed = 155, OutputDirectory = directory, Name = "existing" };
                new DatasetCollector.DatasetCollector(settings, TextWriter.Null).Collect(new CancellationToken(canceled: true));
                string metadata = File.ReadAllText(Path.Combine(directory, "existing", "metadata.json"));

                var error = Assert.Throws<InvalidOperationException>(() => new DatasetCollector.DatasetCollector(settings, TextWriter.Null).Collect());
                Assert.Contains("already holds a dataset (status cancelled)", error.Message);
                Assert.Equal(metadata, File.ReadAllText(Path.Combine(directory, "existing", "metadata.json")));
            }
            finally
            {
                DeleteTemporaryDirectory(directory);
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

                        // A sibling's file: the match's first actions and one more move off the match
                        if (!state.HasEnded)
                        {
                            Action move = TestUtil.RandomAgents()[TestUtil.ActingPlayer(state)].Act(state);
                            GameState sibling = new GameState(state);
                            move.Apply(sibling);
                            string expectedSibling = SaveFileSerializer.Serialize(new SaveFile(sibling, [.. played.GetRange(0, count), move], []));
                            Assert.True(expectedSibling == saveFiles.Build(sibling, count, move), $"game {game}: assembled sibling file after {count} actions differs");
                        }
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

        private record PlayoutRecord(int Winner, int[] VictoryPoints, int[] Agents, int EndRound)
        {
            public virtual bool Equals(PlayoutRecord? other) => other != null && Winner == other.Winner && EndRound == other.EndRound
                && VictoryPoints.SequenceEqual(other.VictoryPoints) && Agents.SequenceEqual(other.Agents);

            public override int GetHashCode() => HashCode.Combine(Winner, EndRound);
        }

        private record SampleRecord(string Name, ulong Group, int Position, int MatchLength, int Round, bool Terminal, int Finished,
            int MatchWinner, int[] MatchVictoryPoints, List<PlayoutRecord> Playouts, int EndWinner, int[]? EndVictoryPoints,
            string SiblingGroup, string Source, int Actor, string Move, bool Chance);

        /// <summary>
        /// The sample lines of all workers with their playout records
        /// </summary>
        private static List<SampleRecord> ReadSampleRecords(string datasetDirectory)
        {
            const int recordSize = SampleRecordWriter.PlayoutRecordSize;
            List<SampleRecord> records = [];

            foreach (string csvPath in Directory.GetFiles(Path.Combine(datasetDirectory, "samples"), "*.csv"))
            {
                byte[] playouts = File.ReadAllBytes(Path.Combine(datasetDirectory, "playouts", Path.GetFileNameWithoutExtension(csvPath) + ".bin"));
                string[] lines = File.ReadAllLines(csvPath);
                Assert.Equal(SampleRecordWriter.Header, lines[0]);
                string[] header = lines[0].Split(',');

                foreach (string line in lines.Skip(1))
                {
                    Dictionary<string, string> row = header.Zip(line.Split(',')).ToDictionary(pair => pair.First, pair => pair.Second);
                    int offset = int.Parse(row["playout_offset"]), count = int.Parse(row["playouts"]);
                    Assert.True((offset + count) * recordSize <= playouts.Length);

                    List<PlayoutRecord> samplePlayouts = Enumerable.Range(offset, count).Select(i =>
                    {
                        byte[] record = playouts[(i * recordSize)..((i + 1) * recordSize)];
                        return new PlayoutRecord((sbyte)record[0], record[1..5].Select(b => (int)b).ToArray(), record[5..9].Select(b => (int)b).ToArray(),
                            BinaryPrimitives.ReadUInt16LittleEndian(record.AsSpan(9)));
                    }).ToList();

                    records.Add(new SampleRecord(row["name"], ulong.Parse(row["group"]), int.Parse(row["position"]), int.Parse(row["match_length"]),
                        int.Parse(row["round"]), row["terminal"] == "1", int.Parse(row["finished"]), int.Parse(row["match_winner"]),
                        row["match_vp"].Split('/').Select(int.Parse).ToArray(), samplePlayouts,
                        row["end_winner"] == "" ? -1 : int.Parse(row["end_winner"]), row["end_vp"] == "" ? null : row["end_vp"].Split('/').Select(int.Parse).ToArray(),
                        row["sibling_group"], row["source"], row["actor"] == "" ? -1 : int.Parse(row["actor"]), row["move"], row["chance"] == "1"));
                }
            }
            return records;
        }

        private static string CollectIntoTemporaryDirectory(CollectorSettings settings, out CollectionResult result)
        {
            string directory = Path.Combine(Path.GetTempPath(), "catansim-collector-" + Guid.NewGuid().ToString("N"));
            var collector = new DatasetCollector.DatasetCollector(settings with { OutputDirectory = directory, Name = "dataset" }, TextWriter.Null);
            result = collector.Collect();
            return collector.DatasetDirectory;
        }

        private static void DeleteTemporaryDirectory(string directory)
        {
            // A collected dataset lies in its own temporary directory
            string root = Path.GetFileName(directory) == "dataset" ? Path.GetDirectoryName(directory)! : directory;
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
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
