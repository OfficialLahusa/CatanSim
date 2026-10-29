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
    /// Starts over on a new map (or a cleared board) with the given seats; with free trade the players trade with each other
    /// </summary>
    public sealed record NewGame(SeatKind[] Seats, bool CenterDesert, bool ClearBoard = false, bool FreeTrade = true) : Command;

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
    /// A new token for a role (admins only); the token goes back to the sender alone, the others see the new list entry
    /// </summary>
    public sealed record CreateToken(Role Role, int? Seat, string Label) : Command;

    /// <summary>
    /// Removes a token and closes the connections that logged in with it (admins only)
    /// </summary>
    public sealed record RevokeToken(string Id) : Command;

    /// <summary>
    /// Server options admins can change while it runs
    /// </summary>
    public sealed record SetServerOptions(bool AllowAnonymousSpectators) : Command;

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
    /// What the match's agents make of the current position (spectators and admins only): the win chances and the policy's
    /// priors from their value net, the priors in the order of MCTSTree.GetValidActions (empty without a policy head)
    /// </summary>
    public sealed record EvaluationEvent(float[] WinChances, float[] Priors) : MatchEvent;

    /// <summary>
    /// The search of the agent thinking about the current position (spectators and admins only), a few times a second: per
    /// move in the order of MCTSTree.GetValidActions its visits, prior and mean value for each player
    /// </summary>
    public sealed record SearchEvent(int Seat, long Iterations, uint[] Visits, float[] Priors, float[][] MeanValues) : MatchEvent;

    /// <summary>
    /// The connections to the match (sent to admins only)
    /// </summary>
    public sealed record SessionsEvent(List<SessionInfo> Sessions) : MatchEvent;

    /// <summary>
    /// A token just created, to the admin who asked only: shown once, the server keeps only its hash
    /// </summary>
    public sealed record TokenCreatedEvent(TokenInfo Info, string Token) : MatchEvent;

    /// <summary>
    /// The server's tokens without their secrets (admins only)
    /// </summary>
    public sealed record TokensEvent(List<TokenInfo> Tokens) : MatchEvent;

    /// <summary>
    /// The server the match is hosted on (admins only), or null once it stopped
    /// </summary>
    public sealed record ServerInfoEvent(ServerInfo? Info) : MatchEvent;

    /// <summary>
    /// A refused command or a failure, to the sender only
    /// </summary>
    public sealed record ErrorEvent(string Message) : MatchEvent;

    /// <summary>
    /// The connection was closed by the server
    /// </summary>
    public sealed record ClosedEvent(string Reason) : MatchEvent;
}
