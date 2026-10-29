using Agents;
using Common;
using Common.Actions;
using Common.Serialization;
using System.Globalization;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The C# state vectorizer: layout, equivalences and parity with the Python vectorizer via shared fixtures
    /// </summary>
    public class VectorizerTests
    {
        private const string UpdateFixturesVariable = "HEXFARER_UPDATE_FIXTURES";
        private static readonly StateVectorizer Vectorizer = new();

        [Fact]
        public void VectorsHaveTheDeclaredLengthThroughoutAGame()
        {
            TestUtil.Seed(90);
            GameState state = TestUtil.NewGame();
            Assert.Equal(StateVectorizer.FeatureDim, Vectorize(state).Length);

            TestUtil.PlayRandomGame(state, afterApply: (s, _) => Assert.Equal(StateVectorizer.FeatureDim, Vectorize(s).Length));
        }

        [Fact]
        public void TheLegacyLayoutIsTheStartOfTheCurrentOne()
        {
            TestUtil.Seed(95);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, afterApply: (s, _) =>
            {
                float[] legacy = StateVectorizer.Legacy.Vectorize(s).ToArray();
                Assert.Equal(StateVectorizer.LegacyFeatureDim, legacy.Length);
                Assert.Equal(legacy, Vectorize(s)[..StateVectorizer.LegacyFeatureDim]);
            }, maxActions: 300);

            Assert.Throws<ArgumentException>(() => new StateVectorizer(StateVectorizer.BaseFeatureDim));
        }

        [Fact]
        public void EveryPortMarksItsTwoCoastalIntersections()
        {
            TestUtil.Seed(96);
            for (int game = 0; game < 20; game++)
            {
                float[] ports = Vectorize(TestUtil.NewGame()).AsSpan(StateVectorizer.LegacyFeatureDim, StateVectorizer.PortFeatureDim).ToArray();
                int types = Enum.GetValues<Port.TradeType>().Length;

                // 9 ports with 2 intersections each, at most one type per intersection: 4 generic ports, one of each 2:1 port
                Assert.Equal(18, ports.Count(value => value == 1f));
                Assert.Equal(ports.Length, ports.Count(value => value is 0f or 1f));
                for (int intersection = 0; intersection < ports.Length / types; intersection++)
                    Assert.True(ports.AsSpan(intersection * types, types).ToArray().Sum() <= 1f);
                for (int type = 0; type < types; type++)
                    Assert.Equal(type == (int)Port.TradeType.Generic ? 8 : 2, Enumerable.Range(0, ports.Length / types).Count(i => ports[i * types + type] == 1f));
            }
        }

        [Fact]
        public void EquivalentRollsProduceIdenticalVectors()
        {
            TestUtil.Seed(91);
            GameState state = TestUtil.NewGame();
            TestUtil.PlayRandomGame(state, maxActions: 150);

            GameState a = new GameState(state);
            GameState b = new GameState(state);
            a.Turn.LastRoll = new RollResult { First = 1, Second = 6 };
            b.Turn.LastRoll = new RollResult { First = 6, Second = 1 };
            Assert.Equal(Vectorize(a), Vectorize(b));

            b.Turn.LastRoll = new RollResult { First = 2, Second = 6 };
            Assert.NotEqual(Vectorize(a), Vectorize(b));
        }

        [Fact]
        public void TheLastRollCountsOnlyOnceSomeoneRolled()
        {
            TestUtil.Seed(94);
            GameState state = TestUtil.NewGame();

            float[] WithRoll(GameState s, byte total)
            {
                GameState changed = new GameState(s);
                changed.Turn.LastRoll = new RollResult { First = (byte)(total - 1), Second = 1 };
                return Vectorize(changed);
            }

            // During setup the last roll is a random placeholder
            Assert.Equal(WithRoll(state, 5), WithRoll(state, 9));

            // So it is in the first normal turn before its roll
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.TypeOfRound == TurnState.RoundType.Normal));
            Assert.True(state.Turn.MustRoll);
            Assert.Equal(WithRoll(state, 5), WithRoll(state, 9));

            // After the roll it counts
            Assert.True(TestUtil.PlayUntil(state, s => !s.Turn.MustRoll));
            Assert.NotEqual(WithRoll(state, 5), WithRoll(state, 9));
        }

        [Fact]
        public void TileFeaturesFollowTheMapStorageOrder()
        {
            TestUtil.Seed(92);
            GameState state = TestUtil.NewGame();
            Board board = state.Board;

            // Layout after the tiles: intersections (3 building + 5 owner), edges (2 + 5), banks (6 + 6), players (35 each),
            // then the derived features (unaffected here: nobody has built yet)
            const int TileWidth = 10;
            int tilesStart = StateVectorizer.BaseFeatureDim
                - board.Map.Length * TileWidth - board.Intersections.Count * 8 - board.Edges.Count * 7 - 12 - state.Players.Length * 35;

            float[] baseline = Vectorize(state);

            for (int tileIdx = 0; tileIdx < board.Map.Length; tileIdx++)
            {
                Tile tile = board.Map.Values[tileIdx];
                if (board.HasRobber(tile)) continue;

                // Tiles are shared between copies, so change the tile on a separately loaded board
                GameState changed = SaveFileSerializer.Deserialize(SaveFileSerializer.Serialize(new SaveFile(state, [], []))).GameState;
                changed.Board.Map.SetTile(tile.X, tile.Y, new Tile(tile.X, tile.Y, tile.Type == Tile.TileType.Ore ? Tile.TileType.Wool : Tile.TileType.Ore, tile.Number));

                float[] vector = Vectorize(changed);
                int[] differing = Enumerable.Range(0, vector.Length).Where(i => vector[i] != baseline[i]).ToArray();

                Assert.NotEmpty(differing);
                Assert.All(differing, i => Assert.InRange(i, tilesStart + tileIdx * TileWidth, tilesStart + (tileIdx + 1) * TileWidth - 1));
            }
        }

        [Fact]
        public void VectorsMatchTheSharedParityFixtures()
        {
            string[] yamlFiles = Directory.GetFiles(TestUtil.FixturePath("Vectorizer"), "*.yaml");
            Assert.NotEmpty(yamlFiles);

            foreach (string yamlPath in yamlFiles)
            {
                SaveFile save = SaveFileSerializer.Deserialize(File.ReadAllText(yamlPath));
                float[] expected = File.ReadAllText(Path.ChangeExtension(yamlPath, ".csv"))
                    .Split(',').Select(value => float.Parse(value, CultureInfo.InvariantCulture)).ToArray();

                Assert.Equal(StateVectorizer.FeatureDim, expected.Length);
                Assert.Equal(expected, Vectorize(save.GameState));
            }
        }

        /// <summary>
        /// Regenerates the parity fixtures shared with Tests/Python after an intended change of the encoding.
        /// Runs only when HEXFARER_UPDATE_FIXTURES=1 is set.
        /// </summary>
        [SkippableFact]
        public void RegenerateParityFixtures()
        {
            Skip.IfNot(Environment.GetEnvironmentVariable(UpdateFixturesVariable) == "1", $"Set {UpdateFixturesVariable}=1 to regenerate the vectorizer fixtures.");

            string directory = Path.Combine(TestUtil.SourceFixtureDirectory(), "Vectorizer");
            Directory.CreateDirectory(directory);
            foreach (string file in Directory.GetFiles(directory))
                File.Delete(file);

            TestUtil.Seed(93);
            var snapshots = new List<(string name, GameState state, List<Action> played)>();

            GameState state = TestUtil.NewGame();
            List<Action> played = [];
            snapshots.Add(("initial", new GameState(state), []));

            void SnapshotWhen(string name, Func<GameState, bool> condition)
            {
                Assert.True(TestUtil.PlayUntil(state, condition, played));
                snapshots.Add((name, new GameState(state), [.. played]));
            }

            SnapshotWhen("setup", s => s.Turn.TypeOfRound == TurnState.RoundType.SecondInitial);
            SnapshotWhen("first_roll", s => s.Turn.TypeOfRound == TurnState.RoundType.Normal && s.Turn.MustRoll);
            SnapshotWhen("discard", s => s.Turn.MustDiscard);
            SnapshotWhen("robber", s => s.Turn.MustMoveRobber && !s.Turn.MustDiscard);
            SnapshotWhen("midgame", s => s.Players.Any(p => p.VictoryPoints.Total >= 6) && TestUtil.IsFreeTurn(s));
            SnapshotWhen("ended", s => s.HasEnded);

            foreach ((string name, GameState snapshot, List<Action> actions) in snapshots)
            {
                File.WriteAllText(Path.Combine(directory, $"{name}.yaml"), SaveFileSerializer.Serialize(new SaveFile(snapshot, actions, [])));
                File.WriteAllText(Path.Combine(directory, $"{name}.csv"), string.Join(",", Vectorize(snapshot).Select(value => value.ToString("R", CultureInfo.InvariantCulture))));
            }
        }

        private static float[] Vectorize(GameState state)
        {
            return Vectorizer.Vectorize(state).ToArray();
        }
    }
}
