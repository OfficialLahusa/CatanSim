using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;

namespace Common
{
    using Action = Actions.Action;

    public class SaveFile
    {
        /// <summary>
        /// Format version written into new save files. Bump it whenever a change makes older files replay
        /// incorrectly (e.g. a changed action history layout). Files without the field read as version 0.
        /// </summary>
        public const int CurrentVersion = 3;

        public int Version { get; set; }
        public GameState GameState { get; set; }

        /// <summary>
        /// File name of another save file (in the same directory) whose played actions come before this file's PlayedActions,
        /// or null if PlayedActions is the whole history. Used by the dataset collector's sibling states, which store only their
        /// own move and take the history of the sampled state they branch off. SaveFileSerializer.Load resolves it.
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitNull)]
        public string? HistoryReference { get; set; }

        public List<Action> PlayedActions { get; set; }
        public List<Action> UndoHistory {  get; set; }

        /// <summary>
        /// Set by SaveFileSerializer.Load when the referenced history could not be loaded, with what happened and what to do
        /// </summary>
        [YamlIgnore]
        public string? LoadWarning { get; set; }

        public SaveFile(GameState gameState, List<Action> playedActions, List<Action> undoHistory)
        {
            Version = CurrentVersion;
            GameState = gameState;
            PlayedActions = playedActions;
            UndoHistory = undoHistory;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private SaveFile()
        {
            GameState = null!;
            PlayedActions = null!;
            UndoHistory = null!;
        }
    }
}
