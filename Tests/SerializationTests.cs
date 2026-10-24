using Common;
using Common.Actions;
using Common.Serialization;
using System.Text.RegularExpressions;
using YamlDotNet.RepresentationModel;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Save file round trips, versioning and the upgrade of older save files
    /// </summary>
    public class SerializationTests
    {
        [Fact]
        public void SaveFilesRoundTripAndReplayInBothDirections()
        {
            TestUtil.Seed(80);

            foreach (int prefix in new[] { 0, 5, 40, 200, int.MaxValue })
            {
                GameState state = TestUtil.NewGame();
                GameState initial = new GameState(state);
                List<Action> played = TestUtil.PlayRandomGame(state, maxActions: prefix);

                string yaml = SaveFileSerializer.Serialize(new SaveFile(state, played, []));
                Assert.StartsWith($"Version: {SaveFile.CurrentVersion}", yaml);

                SaveFile loaded = SaveFileSerializer.Deserialize(yaml);
                Assert.Equal(SaveFile.CurrentVersion, loaded.Version);
                Assert.True(loaded.GameState.Equals(state), $"prefix {prefix}: loaded state differs");
                Assert.Equal(played.Count, loaded.PlayedActions.Count);
                Assert.Equal(played.Select(a => a.GetType()), loaded.PlayedActions.Select(a => a.GetType()));

                // The loaded histories are complete: undo everything, then redo everything
                GameState replay = loaded.GameState;
                for (int i = loaded.PlayedActions.Count - 1; i >= 0; i--)
                    loaded.PlayedActions[i].Revert(replay);
                Assert.True(replay.Equals(initial), $"prefix {prefix}: reverting the loaded actions did not reach the initial state");

                foreach (Action action in loaded.PlayedActions)
                    action.Apply(replay);
                Assert.True(replay.Equals(state), $"prefix {prefix}: reapplying the loaded actions did not reach the saved state");
            }
        }

        [Fact]
        public void RobberIsAnchoredToTheMapAfterLoadingAndUndoWorks()
        {
            TestUtil.Seed(81);
            GameState state = TestUtil.NewGame();
            List<Action> played = [];
            Assert.True(TestUtil.PlayUntil(state, s => played.Count > 0 && played[^1] is RobberAction, played));
            GameState beforeRobber = new GameState(state);
            played[^1].Revert(beforeRobber);

            SaveFile loaded = SaveFileSerializer.Deserialize(SaveFileSerializer.Serialize(new SaveFile(state, played, [])));
            Board board = loaded.GameState.Board;

            Assert.Same(board.Map.GetTile(board.Robber!.X, board.Robber.Y), board.Robber);
            Assert.True(board.HasRobber(board.Map.GetTile(board.Robber.X, board.Robber.Y)));

            loaded.PlayedActions[^1].Revert(loaded.GameState);
            Assert.True(loaded.GameState.Equals(beforeRobber));
            Assert.Same(board.Map.GetTile(board.Robber!.X, board.Robber.Y), board.Robber);
        }

        [Fact]
        public void FilesOfOtherVersionsAreRejected()
        {
            TestUtil.Seed(82);
            GameState state = TestUtil.NewGame();
            List<Action> played = TestUtil.PlayRandomGame(state, maxActions: 30);
            string yaml = SaveFileSerializer.Serialize(new SaveFile(state, played, []));

            string withoutVersion = string.Join('\n', yaml.Split('\n').Where(line => !line.StartsWith("Version:")));
            var missing = Assert.Throws<UnsupportedSaveFileVersionException>(() => SaveFileSerializer.Deserialize(withoutVersion));
            Assert.Equal(0, missing.Version);
            Assert.True(SaveFileUpgrader.CanUpgrade(0));

            string future = yaml.Replace($"Version: {SaveFile.CurrentVersion}", $"Version: {SaveFile.CurrentVersion + 1}");
            var newer = Assert.Throws<UnsupportedSaveFileVersionException>(() => SaveFileSerializer.Deserialize(future));
            Assert.Equal(SaveFile.CurrentVersion + 1, newer.Version);
            Assert.False(SaveFileUpgrader.CanUpgrade(SaveFile.CurrentVersion + 1));
            Assert.Throws<UnsupportedSaveFileVersionException>(() => SaveFileUpgrader.Upgrade(future));
        }

        [Fact]
        public void UpgradingACurrentFileChangesNothing()
        {
            TestUtil.Seed(83);
            GameState state = TestUtil.NewGame();
            string yaml = SaveFileSerializer.Serialize(new SaveFile(state, TestUtil.PlayRandomGame(state, maxActions: 50), []));

            Assert.Equal(SaveFile.CurrentVersion, SaveFileUpgrader.ReadVersion(yaml));
            Assert.Equal(yaml, SaveFileUpgrader.Upgrade(yaml));
        }

        /// <summary>
        /// The fixture is a version 0 file written by the old DatasetCollector: the base state of a game
        /// followed by every action of that game, in the old RollAction and RobberAction layouts
        /// </summary>
        [Fact]
        public void VersionZeroGameUpgradesAndReplaysForwardAndBackward()
        {
            string original = File.ReadAllText(TestUtil.FixturePath("SaveFiles", "v0_full_game.yaml"));
            Assert.Equal(0, SaveFileUpgrader.ReadVersion(original));
            Assert.Throws<UnsupportedSaveFileVersionException>(() => SaveFileSerializer.Deserialize(original));

            string upgraded = SaveFileUpgrader.Upgrade(original);
            Assert.StartsWith($"Version: {SaveFile.CurrentVersion}", upgraded);
            Assert.Equal(upgraded, SaveFileUpgrader.Upgrade(upgraded));
            Assert.DoesNotContain("PrevRobber:", upgraded);

            SaveFile save = SaveFileSerializer.Deserialize(upgraded);
            GameState baseState = new GameState(save.GameState);
            List<Action> actions = save.PlayedActions;

            Assert.All(actions.OfType<RollAction>(), roll => Assert.InRange(roll.History!.RollResult.Total, (byte)2, (byte)12));
            Assert.All(actions.OfType<RobberAction>(), robber => Assert.NotNull(robber.History!.PrevRobberTileIndex));

            // Forward: the stored outcomes replay the whole game to its end
            GameState state = save.GameState;
            foreach (Action action in actions)
            {
                Assert.True(action.IsValidFor(state), $"{action} is not valid during the replay");
                action.Apply(state);
            }
            Assert.True(state.HasEnded);
            Assert.True(state.Players[state.Turn.PlayerIndex].VictoryPoints.Total >= 10);

            // Backward: reverting everything restores the base state, robber included
            for (int i = actions.Count - 1; i >= 0; i--)
                actions[i].Revert(state);
            Assert.True(state.Equals(baseState));
        }

        [Fact]
        public void VersionOneFilesOnlyChangeTheirVersionLine()
        {
            TestUtil.Seed(84);
            GameState state = TestUtil.NewGame();
            List<Action> played = TestUtil.PlayRandomGame(state, maxActions: 300);

            string current = SaveFileSerializer.Serialize(new SaveFile(state, played, []));
            string versionOne = new Regex($"^Version: {SaveFile.CurrentVersion}", RegexOptions.Multiline).Replace(current, "Version: 1", 1);

            Assert.Equal(current, SaveFileUpgrader.Upgrade(versionOne));
        }

        [Fact]
        public void VersionTwoFilesOnlyChangeTheirVersionLine()
        {
            TestUtil.Seed(85);
            GameState state = TestUtil.NewGame();
            List<Action> played = TestUtil.PlayRandomGame(state, maxActions: 300);

            string current = SaveFileSerializer.Serialize(new SaveFile(state, played, []));
            string versionTwo = new Regex($"^Version: {SaveFile.CurrentVersion}", RegexOptions.Multiline).Replace(current, "Version: 2", 1);

            Assert.Equal(current, SaveFileUpgrader.Upgrade(versionTwo));
            Assert.DoesNotContain("HistoryReference", current);
        }

        [Fact]
        public void ForfeitedFreeRoadsAreRestoredWhenUndoingLoadedActions()
        {
            // A Road Building card with only one of its roads placed: ending the turn gives the other one up
            (GameState start, _, _) = RoadBuildingTurn();
            GameState state = new GameState(start);
            int edge = state.Board.Adjacency.GetEdgeIndices(state.Board.Intersections.First(i => state.Board.GetBuilding(i.Index) != Intersection.BuildingType.None).Index)[0];
            Action roll = new RollAction(0).GetOutcomeVariants(state, 0).First(variant => ((RollAction)variant.Action).History!.RollResult.Total == 8).Action;
            List<Action> played = [roll, new RoadBuildingAction(0), new RoadAction(0, edge), new EndTurnAction(0)];
            foreach (Action action in played)
                action.Apply(state);
            Assert.Equal(1u, played[^1].ForfeitedFreeRoads);

            SaveFile loaded = SaveFileSerializer.Deserialize(SaveFileSerializer.Serialize(new SaveFile(state, played, [])));
            Assert.Equal(1u, loaded.PlayedActions[^1].ForfeitedFreeRoads);

            // Undoing the loaded turn end gives the road back
            loaded.PlayedActions[^1].Revert(loaded.GameState);
            Assert.Equal(1u, loaded.GameState.Players[0].BuildingStock.FreeRoads);
            for (int i = loaded.PlayedActions.Count - 2; i >= 0; i--)
                loaded.PlayedActions[i].Revert(loaded.GameState);
            Assert.True(loaded.GameState.Equals(start), "undoing the loaded actions did not reach the start of the turn");
        }

        [Fact]
        public void HistoryReferencesAreResolvedWhenLoading()
        {
            TestUtil.Seed(86);
            string directory = Path.Combine(Path.GetTempPath(), "catansim-references-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);

            try
            {
                GameState initial = TestUtil.NewGame();
                GameState state = new GameState(initial);
                List<Action> played = TestUtil.PlayRandomGame(state, maxActions: 120);
                File.WriteAllText(Path.Combine(directory, "r0_s1.yaml"), SaveFileSerializer.Serialize(new SaveFile(state, played, [])));

                // Two moves off the match, each file holding only its move and referencing the previous file
                List<Action> moves = [];
                string reference = "r0_s1.yaml";
                foreach (string name in new[] { "r0_s1_c0.yaml", "r0_s1_c0_c0.yaml" })
                {
                    Action move = TestUtil.RandomAgents()[TestUtil.ActingPlayer(state)].Act(state);
                    move.Apply(state);
                    moves.Add(move);
                    File.WriteAllText(Path.Combine(directory, name), SaveFileSerializer.Serialize(new SaveFile(state, [move], []) { HistoryReference = reference }));
                    reference = name;
                }

                SaveFile loaded = SaveFileSerializer.Load(Path.Combine(directory, "r0_s1_c0_c0.yaml"));
                Assert.Null(loaded.LoadWarning);
                Assert.Null(loaded.HistoryReference);
                Assert.Equal(played.Count + 2, loaded.PlayedActions.Count);

                // The whole history: saving the result writes a self-contained file, and undoing it reaches the start of the game
                Assert.Equal(SaveFileSerializer.Serialize(new SaveFile(state, [.. played, .. moves], [])), SaveFileSerializer.Serialize(loaded));
                GameState replay = new GameState(loaded.GameState);
                for (int i = loaded.PlayedActions.Count - 1; i >= 0; i--)
                    loaded.PlayedActions[i].Revert(replay);
                Assert.True(replay.Equals(initial), "the loaded history does not lead back to the start");

                // A missing referenced file: only the file's own move, and a warning that explains sibling files
                File.Delete(Path.Combine(directory, "r0_s1.yaml"));
                SaveFile orphan = SaveFileSerializer.Load(Path.Combine(directory, "r0_s1_c0.yaml"));
                Assert.Single(orphan.PlayedActions);
                Assert.Contains("\"r0_s1.yaml\"", orphan.LoadWarning);
                Assert.Contains("same directory", orphan.LoadWarning);
                Assert.Contains("can't be undone further", orphan.LoadWarning);
                Assert.Equal(orphan.LoadWarning, SaveFileSerializer.Load(Path.Combine(directory, "r0_s1_c0_c0.yaml")).LoadWarning);

                // A referenced file with another state (e.g. from another dataset) is not used either
                File.WriteAllText(Path.Combine(directory, "r0_s1.yaml"), SaveFileSerializer.Serialize(new SaveFile(initial, [], [])));
                SaveFile mismatched = SaveFileSerializer.Load(Path.Combine(directory, "r0_s1_c0.yaml"));
                Assert.Single(mismatched.PlayedActions);
                Assert.Contains("different state", mismatched.LoadWarning);
            }
            finally
            {
                Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void RoadBuildingRoadsPlacedAfterTheRollUnderTheOldRuleAreMovedBehindTheCard()
        {
            (GameState start, GameState end, List<Action> played) = RoadBuildingTurn();

            // Old rule: card, roll, both roads, end of turn
            string old = AsVersionOneWithOrder(end, played, [0, 3, 1, 2, 4]);
            SaveFile loaded = SaveFileSerializer.Deserialize(SaveFileUpgrader.Upgrade(old));

            Assert.Equal([typeof(RoadBuildingAction), typeof(RoadAction), typeof(RoadAction), typeof(RollAction), typeof(EndTurnAction)],
                loaded.PlayedActions.Select(action => action.GetType()));
            Assert.True(loaded.GameState.Equals(end));

            // The renewed histories undo and redo the whole turn
            GameState replay = loaded.GameState;
            for (int i = loaded.PlayedActions.Count - 1; i >= 0; i--)
                loaded.PlayedActions[i].Revert(replay);
            Assert.True(replay.Equals(start));

            foreach (Action action in loaded.PlayedActions)
                action.Apply(replay);
            Assert.True(replay.Equals(end));
        }

        [Fact]
        public void SaveFilesWhoseRoadBuildingRoadsCantBeMovedAreRejected()
        {
            (_, GameState end, List<Action> played) = RoadBuildingTurn();

            // The second road extends the first one, so it can't come first
            string old = AsVersionOneWithOrder(end, played, [0, 3, 2, 1, 4]);

            Assert.Throws<InvalidDataException>(() => SaveFileUpgrader.Upgrade(old));
        }

        /// <summary>
        /// Before the roll, player 0 plays Road Building and places two connected roads from a settlement, rolls an 8 and ends the turn
        /// </summary>
        private static (GameState start, GameState end, List<Action> played) RoadBuildingTurn()
        {
            TestUtil.Seed(85);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            Intersection site = state.Board.Intersections.First(i => state.Board.Adjacency.GetEdges(i).Count == 3 && state.Board.Adjacency.GetTiles(i).All(t => t.IsLandTile()));
            state.Board.SetBuilding(site.Index, Intersection.BuildingType.Settlement, 0);
            state.Players[0].BuildingStock.RemainingSettlements--;
            state.Players[0].VictoryPoints.SettlementPoints++;
            state.DevelopmentBank.Remove(DevelopmentCardType.RoadBuilding, 1);
            state.Players[0].DevelopmentCards.Add(DevelopmentCardType.RoadBuilding, 1);
            GameState start = new(state);

            int first = state.Board.Adjacency.GetEdgeIndices(site.Index)[0];
            (int top, int bottom) = state.Board.Adjacency.GetIntersectionIndices(first);
            int far = top == site.Index ? bottom : top;
            int second = state.Board.Adjacency.GetEdgeIndices(far).ToArray().First(edge => edge != first);

            Action roll = new RollAction(0).GetOutcomeVariants(state, 0).First(variant => ((RollAction)variant.Action).History!.RollResult.Total == 8).Action;
            List<Action> played = [new RoadBuildingAction(0), new RoadAction(0, first), new RoadAction(0, second), roll, new EndTurnAction(0)];
            foreach (Action action in played)
                action.Apply(state);

            return (start, state, played);
        }

        /// <summary>
        /// The save file of the state and actions as a version 1 file, with the played actions in the given order
        /// </summary>
        private static string AsVersionOneWithOrder(GameState state, List<Action> played, int[] order)
        {
            YamlStream stream = new();
            stream.Load(new StringReader(SaveFileSerializer.Serialize(new SaveFile(state, played, []))));
            YamlMappingNode root = (YamlMappingNode)stream.Documents[0].RootNode;

            YamlSequenceNode actions = (YamlSequenceNode)root.Children[new YamlScalarNode("PlayedActions")];
            List<YamlNode> recorded = actions.Children.ToList();
            actions.Children.Clear();
            foreach (int i in order)
                actions.Children.Add(recorded[i]);

            root.Children[new YamlScalarNode("Version")] = new YamlScalarNode("1");

            using StringWriter writer = new();
            stream.Save(writer, assignAnchors: false);
            return writer.ToString();
        }
    }
}
