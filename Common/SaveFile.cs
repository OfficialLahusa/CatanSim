using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    using Action = Actions.Action;

    public class SaveFile
    {
        /// <summary>
        /// Format version written into new save files. Bump it whenever a change makes older files replay
        /// incorrectly (e.g. a changed action history layout). Files without the field read as version 0.
        /// </summary>
        public const int CurrentVersion = 2;

        public int Version { get; set; }
        public GameState GameState { get; set; }
        public List<Action> PlayedActions { get; set; }
        public List<Action> UndoHistory {  get; set; }

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
