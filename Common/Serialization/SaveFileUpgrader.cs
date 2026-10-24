using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Common.Actions;
using YamlDotNet.Core;
using YamlDotNet.RepresentationModel;
using Action = Common.Actions.Action;

namespace Common.Serialization
{
    /// <summary>
    /// Converts save files written by older builds to the current SaveFile.CurrentVersion.
    /// Layout changes work on the YAML document itself, so the old layouts never need to be deserializable.
    ///
    /// Version history:
    ///   0 (no Version field): RollAction stored its RollResult on the action instead of in its History
    ///                         and RobberActionHistory stored the previous robber tile as a nested Tile.
    ///   1: RollResult lives in RollActionHistory, RobberActionHistory stores PrevRobberTileIndex.
    ///   2: Road Building roads are placed right after the card (also before the roll) or given up by the next other action.
    ///      Same layout as 1, but games recorded under the old rule placed them later in the turn, so the upgrade moves them.
    ///   3: Optional HistoryReference: the played actions continue those of another file (sibling states of the dataset collector).
    ///      Older files have none, so the upgrade only relabels them; a version 2 reader would take such a file's actions for its whole history.
    /// </summary>
    public static class SaveFileUpgrader
    {
        private static readonly string[] ActionListKeys = ["PlayedActions", "UndoHistory"];

        /// <summary>
        /// Reads the format version of a save file document without deserializing it (missing field = 0)
        /// </summary>
        public static int ReadVersion(string yaml)
        {
            return ReadVersion(LoadRoot(yaml));
        }

        public static bool CanUpgrade(int version)
        {
            return version >= 0 && version < SaveFile.CurrentVersion;
        }

        /// <summary>
        /// Returns the document converted to the current version. Documents that already are current are returned unchanged.
        /// </summary>
        public static string Upgrade(string yaml)
        {
            YamlMappingNode root = LoadRoot(yaml);
            int version = ReadVersion(root);

            if (version == SaveFile.CurrentVersion) return yaml;
            if (!CanUpgrade(version)) throw new UnsupportedSaveFileVersionException(version);

            if (version < 1)
            {
                UpgradeV0ToV1(root);
                yaml = WriteWithVersion(root, 1);
            }

            if (version < 2)
                yaml = UpgradeV1ToV2(yaml);

            return UpgradeV2ToV3(yaml);
        }

        private static string UpgradeV2ToV3(string yaml)
        {
            // Files the previous step had to rewrite are already serialized with the current version
            if (ReadVersion(yaml) >= 3) return yaml;

            Regex versionLine = new(@"^Version: 2(?=?$)", RegexOptions.Multiline);
            return versionLine.IsMatch(yaml) ? versionLine.Replace(yaml, "Version: 3", 1) : WriteWithVersion(LoadRoot(yaml), 3);
        }

        private static string WriteWithVersion(YamlMappingNode root, int version)
        {
            // Version goes first, like in freshly written files
            YamlMappingNode upgradedRoot = new();
            upgradedRoot.Add("Version", version.ToString(CultureInfo.InvariantCulture));
            foreach (var (key, value) in root.Children)
            {
                if (key is YamlScalarNode { Value: "Version" }) continue;
                upgradedRoot.Add(key, value);
            }

            using StringWriter writer = new();
            new YamlStream(new YamlDocument(upgradedRoot)).Save(writer, assignAnchors: false);
            return writer.ToString();
        }

        private static string UpgradeV1ToV2(string yaml)
        {
            // Same layout: files without a Road Building card only get the new version line, byte for byte otherwise
            Regex versionLine = new(@"^Version: 1(?=\r?$)", RegexOptions.Multiline);
            string relabeled = versionLine.IsMatch(yaml) ? versionLine.Replace(yaml, "Version: 2", 1) : WriteWithVersion(LoadRoot(yaml), 2);
            if (!relabeled.Contains(nameof(RoadBuildingAction))) return relabeled;

            SaveFile save = SaveFileSerializer.Deserialize(relabeled, checkVersion: false);

            // One chronological sequence: played actions, then the undone ones from the oldest to the newest undo
            List<Action> sequence = [.. save.PlayedActions, .. Enumerable.Reverse(save.UndoHistory)];
            int playedCount = save.PlayedActions.Count;

            List<Action> reordered = MoveFreeRoadsBehindTheirCard(sequence);
            if (reordered.SequenceEqual(sequence)) return relabeled;

            if (!reordered.Take(playedCount).ToHashSet().SetEquals(sequence.Take(playedCount)))
                throw new InvalidDataException("A Road Building turn is split between the played and the undone actions, so the save file can't be upgraded.");

            // The moved actions are replayed under the current rules, which validates them and renews their histories.
            // Files written by the client store the current state, other files (e.g. complete recorded games) the starting state.
            List<Action> replayed = ReplayFrom(new GameState(save.GameState), reordered, out _)
                ?? ReplayFromCurrentState(save, sequence, reordered, playedCount)
                ?? throw new InvalidDataException("The recorded Road Building roads can't be placed right after the card, so the save file can't be upgraded.");

            SaveFile upgraded = new(save.GameState, replayed.Take(playedCount).ToList(), Enumerable.Reverse(replayed.Skip(playedCount)).ToList());
            return SaveFileSerializer.Serialize(upgraded);
        }

        /// <summary>
        /// Moves the free roads of every Road Building card to right after the card. Under the old rule they could be placed
        /// at any time later in the same turn and only Road Building gave free roads, so they are the player's free roads until the end of the turn.
        /// </summary>
        private static List<Action> MoveFreeRoadsBehindTheirCard(List<Action> sequence)
        {
            List<Action> result = new(sequence.Count);
            HashSet<int> moved = [];

            for (int i = 0; i < sequence.Count; i++)
            {
                if (moved.Contains(i)) continue;
                result.Add(sequence[i]);

                if (sequence[i] is not RoadBuildingAction card) continue;

                for (int j = i + 1; j < sequence.Count; j++)
                {
                    if (sequence[j] is EndTurnAction && sequence[j].PlayerIndex == card.PlayerIndex) break;

                    if (sequence[j] is RoadAction { History.WasFree: true } road && road.PlayerIndex == card.PlayerIndex)
                    {
                        result.Add(road);
                        moved.Add(j);
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Applies copies of the actions in order, or returns null if one of them is not valid
        /// </summary>
        private static List<Action>? ReplayFrom(GameState state, List<Action> actions, out GameState finalState)
        {
            finalState = state;
            List<Action> replayed = new(actions.Count);

            try
            {
                foreach (Action action in actions)
                {
                    Action copy = action.Clone();
                    if (!copy.IsValidFor(state)) return null;
                    copy.Apply(state);
                    replayed.Add(copy);
                }
            }
            catch (Exception e) when (e is InvalidOperationException or ArgumentException)
            {
                return null;
            }

            return replayed;
        }

        /// <summary>
        /// Reverts the played actions in their recorded order from the saved state, replays the reordered sequence
        /// and checks that it arrives at the saved state again. Returns null if it doesn't.
        /// </summary>
        private static List<Action>? ReplayFromCurrentState(SaveFile save, List<Action> recorded, List<Action> reordered, int playedCount)
        {
            GameState start = new(save.GameState);
            try
            {
                for (int i = playedCount - 1; i >= 0; i--)
                    recorded[i].Clone().Revert(start);
            }
            catch (Exception e) when (e is InvalidOperationException or ArgumentException)
            {
                return null;
            }

            List<Action>? played = ReplayFrom(start, reordered.Take(playedCount).ToList(), out GameState current);
            if (played == null || !current.Equals(save.GameState)) return null;

            List<Action>? undone = ReplayFrom(new GameState(current), reordered.Skip(playedCount).ToList(), out _);
            return undone == null ? null : [.. played, .. undone];
        }

        private static void UpgradeV0ToV1(YamlMappingNode root)
        {
            // Tile index = y * width + x, so the map width is needed to convert nested robber tiles
            int mapWidth = int.Parse(
                GetScalar(GetMapping(GetMapping(GetMapping(root, "GameState"), "Board"), "Map"), "Width"),
                CultureInfo.InvariantCulture);

            foreach (string listKey in ActionListKeys)
            {
                if (!TryGet(root, listKey, out YamlNode? listNode) || listNode is not YamlSequenceNode actions) continue;

                foreach (YamlNode actionNode in actions)
                {
                    if (actionNode is not YamlMappingNode action) continue;

                    switch (TryGet(action, "ActionType", out YamlNode? typeNode) ? (typeNode as YamlScalarNode)?.Value : null)
                    {
                        case "RollAction":
                            MoveRollResultIntoHistory(action);
                            break;
                        case "RobberAction":
                            ReplacePrevRobberTileWithIndex(action, mapWidth);
                            break;
                    }
                }
            }
        }

        private static void MoveRollResultIntoHistory(YamlMappingNode action)
        {
            if (!TryGet(action, "RollResult", out YamlNode? rollResult)) return;

            action.Children.Remove(new YamlScalarNode("RollResult"));

            // Actions that were never applied have no history; one holding just the roll is enough for Apply to replay it
            if (!TryGet(action, "History", out YamlNode? historyNode) || historyNode is not YamlMappingNode history)
            {
                history = new YamlMappingNode();
                action.Children[new YamlScalarNode("History")] = history;
            }

            if (!TryGet(history, "RollResult", out _))
                history.Add("RollResult", rollResult);
        }

        private static void ReplacePrevRobberTileWithIndex(YamlMappingNode action, int mapWidth)
        {
            if (!TryGet(action, "History", out YamlNode? historyNode) || historyNode is not YamlMappingNode history) return;
            if (!TryGet(history, "PrevRobber", out YamlNode? prevRobber)) return;

            history.Children.Remove(new YamlScalarNode("PrevRobber"));

            string index = "~";
            if (prevRobber is YamlMappingNode tile)
            {
                int x = int.Parse(GetScalar(tile, "X"), CultureInfo.InvariantCulture);
                int y = int.Parse(GetScalar(tile, "Y"), CultureInfo.InvariantCulture);
                index = (y * mapWidth + x).ToString(CultureInfo.InvariantCulture);
            }

            history.Add("PrevRobberTileIndex", index);
        }

        private static YamlMappingNode LoadRoot(string yaml)
        {
            YamlStream stream = new();
            stream.Load(new StringReader(yaml));

            if (stream.Documents.Count == 0 || stream.Documents[0].RootNode is not YamlMappingNode root)
                throw new InvalidDataException("Save file does not contain a YAML mapping document.");

            return root;
        }

        private static int ReadVersion(YamlMappingNode root)
        {
            if (TryGet(root, "Version", out YamlNode? node) && node is YamlScalarNode scalar
                && int.TryParse(scalar.Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int version))
                return version;

            return 0;
        }

        private static bool TryGet(YamlMappingNode mapping, string key, out YamlNode? value)
        {
            return mapping.Children.TryGetValue(new YamlScalarNode(key), out value);
        }

        private static YamlMappingNode GetMapping(YamlMappingNode mapping, string key)
        {
            if (TryGet(mapping, key, out YamlNode? node) && node is YamlMappingNode child) return child;
            throw new InvalidDataException($"Save file is missing the \"{key}\" section.");
        }

        private static string GetScalar(YamlMappingNode mapping, string key)
        {
            if (TryGet(mapping, key, out YamlNode? node) && node is YamlScalarNode { Value: not null } scalar) return scalar.Value;
            throw new InvalidDataException($"Save file is missing the \"{key}\" value.");
        }
    }
}
