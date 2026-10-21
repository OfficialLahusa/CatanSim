using Common;
using Common.Actions;
using Common.Serialization;
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
                Assert.StartsWith("Version: 1", yaml);

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

            string future = yaml.Replace("Version: 1", "Version: 2");
            var newer = Assert.Throws<UnsupportedSaveFileVersionException>(() => SaveFileSerializer.Deserialize(future));
            Assert.Equal(2, newer.Version);
            Assert.False(SaveFileUpgrader.CanUpgrade(2));
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
            Assert.StartsWith("Version: 1", upgraded);
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
    }
}
