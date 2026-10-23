using Common;
using Common.Serialization;
using System.Text;
using Action = Common.Actions.Action;

namespace DatasetCollector
{
    /// <summary>
    /// Save files of the samples of one match. Every sample stores the actions that led to it, and these are prefixes of the
    /// match's actions, so the actions are serialized once and each sample's file is assembled from its state and the first
    /// actions. The text is the same as serializing the sample with SaveFileSerializer; if the serialized YAML ever doesn't
    /// have the expected shape, the sample is serialized on its own instead.
    /// </summary>
    internal sealed class MatchSaveFiles
    {
        private const string EmptyActionList = "PlayedActions: []";

        private readonly List<Action> _actions;

        // One serialized list item per action (each starting with "- " at the beginning of a line), null if the
        // serialized list couldn't be split into exactly one item per action
        private readonly string[]? _actionItems;

        /// <summary>
        /// How many files had to be serialized on their own because the YAML didn't have the expected shape
        /// </summary>
        internal int FallbackCount { get; private set; }

        public MatchSaveFiles(List<Action> actions)
        {
            _actions = actions;
            _actionItems = actions.Count > 0 ? SplitItems(SaveFileSerializer.Serialize(actions), actions.Count) : [];
        }

        /// <summary>
        /// The save file of the state reached after the first actionCount actions, with these actions
        /// </summary>
        public string Build(GameState state, int actionCount)
        {
            string stateFile = SaveFileSerializer.Serialize(new SaveFile(state, [], []));
            if (actionCount == 0) return stateFile;

            int at = stateFile.IndexOf(EmptyActionList, StringComparison.Ordinal);
            if (_actionItems == null || at < 0 || stateFile.IndexOf(EmptyActionList, at + 1, StringComparison.Ordinal) >= 0)
            {
                FallbackCount++;
                return SaveFileSerializer.Serialize(new SaveFile(state, _actions.GetRange(0, actionCount), []));
            }

            // "PlayedActions: []" + newline becomes "PlayedActions:" + newline + the items, keeping the serializer's line ending
            int afterList = at + EmptyActionList.Length;
            int newlineLength = stateFile.AsSpan(afterList).StartsWith("\r\n") ? 2 : 1;
            int rest = afterList + newlineLength;

            int itemLength = 0;
            for (int i = 0; i < actionCount; i++) itemLength += _actionItems[i].Length;

            StringBuilder file = new(stateFile.Length + itemLength);
            file.Append(stateFile, 0, at).Append("PlayedActions:").Append(stateFile, afterList, newlineLength);
            for (int i = 0; i < actionCount; i++) file.Append(_actionItems[i]);
            file.Append(stateFile, rest, stateFile.Length - rest);
            return file.ToString();
        }

        private static string[]? SplitItems(string list, int expectedCount)
        {
            List<string> items = new(expectedCount);
            int start = 0;
            for (int i = 1; i < list.Length - 1; i++)
            {
                // Nested sequences of an action are indented, so only the actions themselves start a line with "- "
                if (list[i] == '-' && list[i + 1] == ' ' && list[i - 1] == '\n')
                {
                    items.Add(list.Substring(start, i - start));
                    start = i;
                }
            }
            items.Add(list.Substring(start));

            return items.Count == expectedCount && items[0].StartsWith("- ", StringComparison.Ordinal) ? items.ToArray() : null;
        }
    }
}
