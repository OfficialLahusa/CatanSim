using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using YamlDotNet.Core;
using YamlDotNet.RepresentationModel;

namespace Common.Serialization
{
    /// <summary>
    /// Converts save files written by older builds to the current SaveFile.CurrentVersion.
    /// Works on the YAML document itself, so the old layouts never need to be deserializable.
    ///
    /// Version history:
    ///   0 (no Version field): RollAction stored its RollResult on the action instead of in its History
    ///                         and RobberActionHistory stored the previous robber tile as a nested Tile.
    ///   1: RollResult lives in RollActionHistory, RobberActionHistory stores PrevRobberTileIndex.
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

            if (version < 1) UpgradeV0ToV1(root);

            // Version goes first, like in freshly written files
            YamlMappingNode upgradedRoot = new();
            upgradedRoot.Add("Version", SaveFile.CurrentVersion.ToString(CultureInfo.InvariantCulture));
            foreach (var (key, value) in root.Children)
            {
                if (key is YamlScalarNode { Value: "Version" }) continue;
                upgradedRoot.Add(key, value);
            }

            using StringWriter writer = new();
            new YamlStream(new YamlDocument(upgradedRoot)).Save(writer, assignAnchors: false);
            return writer.ToString();
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
