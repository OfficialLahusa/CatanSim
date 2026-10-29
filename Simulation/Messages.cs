using Common;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// What a client asks the match to do. Every command is checked against the sender's role; a refused one comes back as
    /// an ErrorEvent to the sender only.
    /// </summary>
    public abstract record Command;

    /// <summary>
    /// A move for a seat the sender may play
    /// </summary>
    public sealed record PlayMove(Action Action) : Command;
    public sealed record UndoCommand : Command;
    public sealed record RedoCommand : Command;

    /// <summary>
    /// An edit made on the sender's copy of the position; refused if the position changed meanwhile
    /// </summary>
    public sealed record ApplyEdit(BoardEdit Edit) : Command;

    /// <summary>
    /// Starts over on a new map (or a cleared board) with the given seats
    /// </summary>
    public sealed record NewGame(SeatKind[] Seats, bool CenterDesert, bool ClearBoard = false) : Command;

    /// <summary>
    /// Replaces the match with a loaded game: its position and the moves before and after it (oldest first, next to redo last)
    /// </summary>
    public sealed record LoadGame(GameState State, List<Action> Played, List<Action> Redo) : Command;
    public sealed record SetSeat(int Seat, SeatKind Kind) : Command;
    public sealed record SetPlayback(PlaybackState Playback) : Command;

    /// <summary>
    /// One agent move while playback is paused (the next redo if there is one)
    /// </summary>
    public sealed record StepCommand : Command;
    public sealed record ConfigureAgents(AgentSettings Settings) : Command;

    /// <summary>
    /// Closes another connection (admins only)
    /// </summary>
    public sealed record KickSession(string SessionId) : Command;

    /// <summary>
    /// What the match tells its clients, in order. Each client's GameRecord follows the match by applying them.
    /// </summary>
    public abstract record MatchEvent;

    /// <summary>
    /// The connection is accepted with this role
    /// </summary>
    public sealed record WelcomeEvent(SessionInfo Session) : MatchEvent;

    /// <summary>
    /// The whole match: after connecting, a new game or a loaded one
    /// </summary>
    public sealed record SnapshotEvent(GameState State, List<HistoryEntry> History, List<HistoryEntry> Redo, SeatKind[] Seats, PlaybackState Playback, AgentSettings Agents) : MatchEvent;

    /// <summary>
    /// A move was played or an edit made (a redo takes the next entry off the redo list)
    /// </summary>
    public sealed record AppliedEvent(HistoryEntry Entry, bool IsRedo) : MatchEvent;
    public sealed record UndoneEvent : MatchEvent;
    public sealed record SeatsEvent(SeatKind[] Seats) : MatchEvent;
    public sealed record PlaybackEvent(PlaybackState Playback) : MatchEvent;
    public sealed record AgentsEvent(AgentSettings Settings) : MatchEvent;

    /// <summary>
    /// An agent started or stopped thinking about its move
    /// </summary>
    public sealed record ThinkingEvent(int Seat, bool Thinking) : MatchEvent;

    /// <summary>
    /// The connections to the match (sent to admins only)
    /// </summary>
    public sealed record SessionsEvent(List<SessionInfo> Sessions) : MatchEvent;

    /// <summary>
    /// A refused command or a failure, to the sender only
    /// </summary>
    public sealed record ErrorEvent(string Message) : MatchEvent;

    /// <summary>
    /// The connection was closed by the server
    /// </summary>
    public sealed record ClosedEvent(string Reason) : MatchEvent;
}
