using Common;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// A client's copy of a match, kept up to date by applying the match's events in order
    /// </summary>
    public class ClientGame
    {
        public GameRecord Record { get; private set; } = new GameRecord(new GameState(MapGenerator.GenerateRandomClassic(), 4));
        public SeatKind[] Seats { get; private set; } = [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human];
        public PlaybackState Playback { get; private set; } = PlaybackState.Default;
        public AgentSettings AgentSettings { get; private set; } = AgentSettings.Default;

        /// <summary>
        /// The seat whose agent is searching, if any
        /// </summary>
        public int? ThinkingSeat { get; private set; }

        /// <summary>
        /// What the match's agents make of the current position and the search of the agent thinking about it, for spectators
        /// and admins; null until the match sends them for this position
        /// </summary>
        public EvaluationEvent? Evaluation { get; private set; }
        public SearchEvent? Search { get; private set; }

        /// <summary>
        /// The connections to the match, if this client is an admin of a hosted match
        /// </summary>
        public List<SessionInfo> Sessions { get; private set; } = [];

        /// <summary>
        /// The server's tokens (without secrets) and the server itself, for admins of a hosted match
        /// </summary>
        public List<TokenInfo> Tokens { get; private set; } = [];
        public ServerInfo? Server { get; private set; }

        /// <summary>
        /// The token this admin created last, shown once; null once it was dismissed
        /// </summary>
        public TokenCreatedEvent? CreatedToken { get; set; }

        public void Apply(MatchEvent e)
        {
            switch (e)
            {
                case SnapshotEvent snapshot:
                    // Copies, so the record's moves are its own; a seat's view comes with the view of its history's start
                    if (snapshot.Start is GameState start)
                        Record.ResetView(new GameState(snapshot.State), new GameState(start), snapshot.History.Select(Copy), snapshot.Redo.Select(Copy));
                    else
                        Record.Reset(new GameState(snapshot.State), snapshot.History.Select(Copy), snapshot.Redo.Select(Copy));
                    Seats = (SeatKind[])snapshot.Seats.Clone();
                    Playback = snapshot.Playback;
                    AgentSettings = snapshot.Agents;
                    ThinkingSeat = null;
                    (Evaluation, Search) = (null, null);
                    break;
                case AppliedEvent applied:
                    if (applied.View is GameState view) Record.ApplyView(Copy(applied.Entry), new GameState(view), applied.IsRedo);
                    else Record.Apply(Copy(applied.Entry), applied.IsRedo);
                    (Evaluation, Search) = (null, null);
                    break;
                case UndoneEvent undone:
                    if (undone.View is GameState undoneView) Record.UndoView(new GameState(undoneView));
                    else Record.Undo();
                    (Evaluation, Search) = (null, null);
                    break;
                case EvaluationEvent evaluation:
                    Evaluation = evaluation;
                    break;
                case SearchEvent search:
                    Search = search;
                    break;
                case SeatsEvent seats:
                    Seats = (SeatKind[])seats.Seats.Clone();
                    break;
                case PlaybackEvent playback:
                    Playback = playback.Playback;
                    break;
                case AgentsEvent agents:
                    AgentSettings = agents.Settings;
                    break;
                case ThinkingEvent thinking:
                    ThinkingSeat = thinking.Thinking ? thinking.Seat : ThinkingSeat == thinking.Seat ? null : ThinkingSeat;
                    break;
                case SessionsEvent sessions:
                    Sessions = sessions.Sessions;
                    break;
                case TokensEvent tokens:
                    Tokens = tokens.Tokens;
                    break;
                case TokenCreatedEvent created:
                    CreatedToken = created;
                    break;
                case ServerInfoEvent server:
                    Server = server.Info;
                    if (server.Info == null)
                    {
                        Tokens = [];
                        Sessions = [];
                        CreatedToken = null;
                    }
                    break;
            }
        }

        private static HistoryEntry Copy(HistoryEntry entry) => entry.Action is Action action ? HistoryEntry.Of(action.Clone()) : entry;
    }
}
