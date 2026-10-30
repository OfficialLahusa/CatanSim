using Common;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// The belief trackers of every seat and of the public observer along a record's history: one set per entry, so undo
    /// goes back exactly and redo observes the same entry again. Lets the match send each seat its view, and spectators and
    /// admins look at the game as any seat sees it.
    /// </summary>
    public sealed class RecordBeliefs
    {
        private readonly GameState _start;

        // After i entries: the trackers of BeliefTracker.Public and of each seat, at index observer + 1
        private readonly List<BeliefTracker[]> _steps = [];

        /// <summary>
        /// Trackers that start at the given position, whose hands count as known
        /// </summary>
        public RecordBeliefs(GameState start, int players)
        {
            _start = new GameState(start);
            _steps.Add(Enumerable.Range(BeliefTracker.Public, players + 1).Select(observer => new BeliefTracker(observer, start)).ToArray());
        }

        /// <summary>
        /// The trackers along a record's history: its start is found by taking back every entry on a copy
        /// </summary>
        public static RecordBeliefs Of(GameRecord record)
        {
            GameState state = new GameState(record.State);
            for (int i = record.History.Count - 1; i >= 0; i--)
            {
                HistoryEntry entry = record.History[i];
                if (entry.Action is Action action) action.Clone().Revert(state);
                else state = new GameState(entry.Edit!.Before);
            }

            RecordBeliefs beliefs = new RecordBeliefs(state, state.Players.Length);
            foreach (HistoryEntry entry in record.History)
            {
                GameState before = new GameState(state);
                if (entry.Action is Action action) action.Clone().Apply(state);
                else state = new GameState(entry.Edit!.After);
                beliefs.Apply(before, entry, state);
            }
            return beliefs;
        }

        /// <summary>
        /// The position before the first entry
        /// </summary>
        public GameState Start => _start;

        /// <summary>
        /// How many entries the trackers have taken in
        /// </summary>
        public int Count => _steps.Count - 1;

        /// <summary>
        /// Takes in an entry, given the true positions before and after it
        /// </summary>
        public void Apply(GameState before, HistoryEntry entry, GameState after)
        {
            BeliefTracker[] next = _steps[^1].Select(tracker => new BeliefTracker(tracker)).ToArray();
            foreach (BeliefTracker tracker in next) tracker.Observe(before, entry.Action, after);
            _steps.Add(next);
        }

        /// <summary>
        /// Takes back the last entry
        /// </summary>
        public void Undo()
        {
            if (_steps.Count > 1) _steps.RemoveAt(_steps.Count - 1);
        }

        /// <summary>
        /// The current tracker of a seat, or of BeliefTracker.Public
        /// </summary>
        public BeliefTracker For(int observer) => _steps[^1][observer + 1];

        /// <summary>
        /// A seat's view of the true position (see BeliefTracker.View)
        /// </summary>
        public GameState View(GameState truth, int observer) => For(observer).View(truth);

        /// <summary>
        /// A seat's view of the history's start
        /// </summary>
        public GameState StartView(int observer) => _steps[0][observer + 1].View(_start);

        /// <summary>
        /// A record's history as a seat sees it, entry by entry (see Redact); the trackers must have followed that history
        /// </summary>
        public List<HistoryEntry> Redact(IReadOnlyList<HistoryEntry> history, int observer)
        {
            List<HistoryEntry> redacted = new List<HistoryEntry>(history.Count);
            for (int i = 0; i < history.Count; i++)
            {
                HistoryEntry entry = history[i];
                redacted.Add(entry.Edit is BoardEdit edit
                    ? Redact(entry, observer, _steps[i][observer + 1].View(edit.Before), _steps[i + 1][observer + 1].View(edit.After))
                    : Redact(entry, observer, null!, null!));
            }
            return redacted;
        }

        /// <summary>
        /// An entry as a seat sees it: a move redacted (see Action.RedactedFor), an edit with the seat's views before and after
        /// </summary>
        public static HistoryEntry Redact(HistoryEntry entry, int observer, GameState viewBefore, GameState viewAfter) => entry.Action is Action action
            ? HistoryEntry.Of(action.RedactedFor(observer))
            : HistoryEntry.Of(new BoardEdit(entry.Edit!.Description, entry.Edit.Player, viewBefore, viewAfter));
    }
}
