using Common;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// A played move or an edit, in the order they happened; undo goes back through them
    /// </summary>
    public sealed record HistoryEntry(Action? Action, BoardEdit? Edit)
    {
        public static HistoryEntry Of(Action action) => new(action, null);
        public static HistoryEntry Of(BoardEdit edit) => new(null, edit);
    }

    /// <summary>
    /// A position with the history that led to it and the undone entries that can be redone. The match and every client keep
    /// one; the client's follows the match by replaying its entries (moves carry their resolved randomness in their history,
    /// so replaying gives the same position).
    /// </summary>
    public class GameRecord
    {
        public GameState State { get; private set; }

        /// <summary>
        /// The position before the first history entry, kept by a record that follows a seat's view: its entries can't be
        /// taken back from the view, so the log is rebuilt forward from here (null for a full record)
        /// </summary>
        public GameState? Start { get; private set; }

        private readonly List<HistoryEntry> _history = [];
        private readonly List<HistoryEntry> _redo = [];

        /// <summary>
        /// Oldest first
        /// </summary>
        public IReadOnlyList<HistoryEntry> History => _history;

        /// <summary>
        /// The next entry to redo last
        /// </summary>
        public IReadOnlyList<HistoryEntry> Redo => _redo;

        /// <summary>
        /// Counts every change, so caches know when they are stale
        /// </summary>
        public int Version { get; private set; }

        public GameRecord(GameState state)
        {
            State = state;
        }

        /// <summary>
        /// Starts over from a position, with an optional history and redo list (oldest first, next to redo last)
        /// </summary>
        public void Reset(GameState state, IEnumerable<HistoryEntry>? history = null, IEnumerable<HistoryEntry>? redo = null)
        {
            State = state;
            Start = null;
            _history.Clear();
            _redo.Clear();
            if (history != null) _history.AddRange(history);
            if (redo != null) _redo.AddRange(redo);
            Version++;
        }

        /// <summary>
        /// Plays a move or makes an edit; a redone entry takes the next redo off the list, a new one clears it
        /// </summary>
        public void Apply(HistoryEntry entry, bool isRedo = false)
        {
            if (entry.Action is Action action) action.Apply(State);
            else State = entry.Edit!.After;

            _history.Add(entry);
            if (isRedo)
            {
                if (_redo.Count > 0) _redo.RemoveAt(_redo.Count - 1);
            }
            else
            {
                _redo.Clear();
            }
            Version++;
        }

        /// <summary>
        /// Starts over from a seat's view: the view's position, the history start and the redacted entries
        /// </summary>
        public void ResetView(GameState view, GameState start, IEnumerable<HistoryEntry> history, IEnumerable<HistoryEntry> redo)
        {
            Reset(view, history, redo);
            Start = start;
        }

        /// <summary>
        /// Adds a redacted entry as a move of a seat's view: the match sends the resulting view along, the entry itself isn't
        /// replayed (it can't be, without the hidden outcome)
        /// </summary>
        public void ApplyView(HistoryEntry entry, GameState view, bool isRedo = false)
        {
            State = view;
            _history.Add(entry);
            if (isRedo)
            {
                if (_redo.Count > 0) _redo.RemoveAt(_redo.Count - 1);
            }
            else
            {
                _redo.Clear();
            }
            Version++;
        }

        /// <summary>
        /// Takes back the last entry of a seat's view, which the match sends along
        /// </summary>
        public void UndoView(GameState view)
        {
            if (_history.Count == 0) return;
            _redo.Add(_history[^1]);
            _history.RemoveAt(_history.Count - 1);
            State = view;
            Version++;
        }

        /// <summary>
        /// Takes back the last entry; it becomes the next to redo
        /// </summary>
        public HistoryEntry? Undo()
        {
            if (_history.Count == 0) return null;
            HistoryEntry entry = _history[^1];
            _history.RemoveAt(_history.Count - 1);
            if (entry.Action is Action action) action.Revert(State);
            else State = entry.Edit!.Before;
            _redo.Add(entry);
            Version++;
            return entry;
        }

        /// <summary>
        /// The played moves since the last edit, oldest first: what a save file can replay
        /// </summary>
        public List<Action> ActionsSinceLastEdit()
        {
            int start = _history.FindLastIndex(e => e.Edit != null) + 1;
            return _history.Skip(start).Select(e => e.Action!).ToList();
        }

        /// <summary>
        /// The undone moves up to the first undone edit, in save file order (the next to redo last)
        /// </summary>
        public List<Action> RedoActionsUntilEdit()
        {
            int start = _redo.FindLastIndex(e => e.Edit != null) + 1;
            return _redo.Skip(start).Select(e => e.Action!).ToList();
        }
    }
}
