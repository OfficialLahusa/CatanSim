using Agents;
using Agents.MCTS;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// The authoritative game: its record (position, history, redo), the seats and their agents. Every change goes through
    /// Handle, which checks the sender's role, applies the command and publishes what happened, in order, to everyone (or a
    /// refusal to the sender only). Agent seats move on their own once Start is called (see AgentRunner).
    /// </summary>
    public sealed class Match : IDisposable
    {
        private readonly object _lock = new();
        private readonly bool _valueNetAvailable;
        private SeatKind[] _seats;
        private Agent[] _agents;
        private AgentRunner? _runner;
        private AnalysisRunner? _analysis;

        // The analysis of the current position, for listeners that join later
        private readonly List<MatchEvent> _analysisEvents = [];
        private int _analysisVersion = -1;

        // The agent searching on the runner thread: the match leaves it alone and resets it once the search is over
        private Agent? _busyAgent;
        private bool _busyAgentStale;

        public GameRecord Record { get; }

        // What each seat knows along the history, for the views the seats get
        private RecordBeliefs _beliefs;
        public PlaybackState Playback { get; private set; } = PlaybackState.Default;
        public AgentSettings AgentSettings { get; private set; } = AgentSettings.Default;

        /// <summary>
        /// Raised for every event with its recipient: a session ID, or null for everyone. Raised while the match is locked, so
        /// handlers see the events in order; they must not call back into the match.
        /// </summary>
        public event Action<MatchEvent, string?>? Published;

        /// <summary>
        /// Raised for a command the match can't handle itself (kicking a session, tokens, server options), after the role check.
        /// Without a handler (a match nobody hosts) such commands are refused.
        /// </summary>
        public event Action<SessionInfo, Command>? HostCommand;

        /// <summary>
        /// The recipient for events only admins get (connections, tokens, the server)
        /// </summary>
        public const string Admins = "@admins";

        /// <summary>
        /// The recipient for the agents' analysis: spectators and admins (for a player it would show what the bots plan)
        /// </summary>
        public const string Watchers = "@watchers";

        /// <summary>
        /// The recipient of the full game (positions with every hand): spectators and admins
        /// </summary>
        public const string Full = "@full";

        /// <summary>
        /// The recipient of a seat's view (see BeliefTracker): the player or agent bridge of that seat
        /// </summary>
        public static string SeatView(int seat) => $"@seat{seat}";

        /// <summary>
        /// Whether a listener of the given session gets an event for this recipient
        /// </summary>
        public static bool IsFor(string? to, SessionInfo session) => to switch
        {
            null => true,
            Admins => session.Role == Role.Admin,
            Watchers => session.Role is Role.Admin or Role.Spectator,
            Full => !SeesView(session),
            _ when to.StartsWith("@seat") => SeesView(session) && to == SeatView(session.Seat!.Value),
            _ => to == session.Id
        };

        /// <summary>
        /// Whether a session follows its seat's view instead of the full game: players and agent bridges
        /// </summary>
        public static bool SeesView(SessionInfo session) => session.Role is Role.Player or Role.Agent && session.Seat != null;

        public Match(GameState state, SeatKind[] seats, bool? valueNetAvailable = null)
        {
            if (seats.Length != state.Players.Length) throw new ArgumentException("One seat kind per player is needed.", nameof(seats));
            _valueNetAvailable = valueNetAvailable ?? Simulation.Seats.ValueNetAvailable;
            Record = new GameRecord(state);
            _beliefs = new RecordBeliefs(state, state.Players.Length);
            _seats = (SeatKind[])seats.Clone();
            _agents = CreateAgents();
        }

        public SeatKind[] Seats
        {
            get { lock (_lock) return (SeatKind[])_seats.Clone(); }
        }

        public bool ValueNetAvailable => _valueNetAvailable;

        /// <summary>
        /// The player who has to act now, or -1 once the game has ended
        /// </summary>
        public int ActingPlayer
        {
            get { lock (_lock) return ActingPlayerOf(Record.State); }
        }

        internal static int ActingPlayerOf(GameState state)
        {
            if (state.HasEnded) return -1;
            for (int player = 0; player < state.Players.Length; player++)
                if (state.CanPlayerAct(player)) return player;
            return -1;
        }

        public bool IsAgentThinking => _runner?.IsThinking ?? false;

        /// <summary>
        /// Lets the agents play their seats from now on, and analyzes the positions for spectators and admins
        /// </summary>
        public void Start()
        {
            lock (_lock)
            {
                _runner ??= new AgentRunner(this);
                _analysis ??= new AnalysisRunner(this, _valueNetAvailable);
            }
            _runner.Wake();
            _analysis.Wake();
        }

        public void Dispose()
        {
            _runner?.Dispose();
            _analysis?.Dispose();
        }

        /// <summary>
        /// Gives a new listener the snapshot and then every event from that point on, with nothing missed or doubled
        /// </summary>
        public void Subscribe(Action<MatchEvent, string?> listener, Action<SnapshotEvent> onSnapshot)
        {
            Subscribe(new SessionInfo("full", "Full", Role.Admin, null), listener, onSnapshot);
        }

        /// <summary>
        /// Like Subscribe, with the snapshot the session may see: its seat's view for a player or agent bridge
        /// </summary>
        public void Subscribe(SessionInfo session, Action<MatchEvent, string?> listener, Action<SnapshotEvent> onSnapshot)
        {
            lock (_lock)
            {
                onSnapshot(SnapshotForLocked(session));
                if (_analysisVersion == Record.Version)
                {
                    foreach (MatchEvent e in _analysisEvents) listener(e, Watchers);
                }
                Published += listener;
            }
        }

        /// <summary>
        /// Everything a new client needs to follow the match
        /// </summary>
        public SnapshotEvent Snapshot()
        {
            lock (_lock) return SnapshotLocked();
        }

        /// <summary>
        /// The snapshot a session may see: the full game, or for a player or agent bridge its seat's view with redacted
        /// entries and the view of the history's start
        /// </summary>
        public SnapshotEvent SnapshotFor(SessionInfo session)
        {
            lock (_lock) return SnapshotForLocked(session);
        }

        private SnapshotEvent SnapshotForLocked(SessionInfo session) => SeesView(session) ? SeatSnapshotLocked(session.Seat!.Value) : SnapshotLocked();

        // A seat gets no redo list: the undone moves may be moves of a loaded game it hasn't seen yet
        private SnapshotEvent SeatSnapshotLocked(int seat) => new(_beliefs.View(Record.State, seat), _beliefs.Redact(Record.History, seat), [],
            (SeatKind[])_seats.Clone(), Playback, AgentSettings, _beliefs.StartView(seat));

        /// <summary>
        /// What a seat (or BeliefTracker.Public) knows along the history, kept with every move and taken back with undo
        /// </summary>
        public RecordBeliefs Beliefs
        {
            get { lock (_lock) return _beliefs; }
        }

        /// <summary>
        /// A new position with its history: the snapshot for spectators and admins, each seat's view for its players
        /// </summary>
        private void PublishSnapshotsLocked()
        {
            _beliefs = RecordBeliefs.Of(Record);
            Publish(SnapshotLocked(), Full);
            for (int seat = 0; seat < _seats.Length; seat++) Publish(SeatSnapshotLocked(seat), SeatView(seat));
        }

        /// <summary>
        /// Plays a move or makes an edit and tells everyone: the entry to spectators and admins, the redacted entry with the
        /// view after it to each seat
        /// </summary>
        private void ApplyLocked(HistoryEntry entry, bool isRedo)
        {
            GameState before = new GameState(Record.State);
            GameState[] viewsBefore = entry.Edit != null ? Enumerable.Range(0, _seats.Length).Select(seat => _beliefs.View(before, seat)).ToArray() : [];
            Record.Apply(entry, isRedo);
            _beliefs.Apply(before, entry, Record.State);

            // A copy taken after applying, with the resolved outcome, so clients replay the same dice and cards
            Publish(new AppliedEvent(Copy(entry), isRedo), Full);
            for (int seat = 0; seat < _seats.Length; seat++)
            {
                GameState view = _beliefs.View(Record.State, seat);
                HistoryEntry redacted = RecordBeliefs.Redact(entry, seat, entry.Edit != null ? viewsBefore[seat] : null!, view);
                Publish(new AppliedEvent(redacted, isRedo, view), SeatView(seat));
            }
        }

        private void UndoLocked()
        {
            Record.Undo();
            _beliefs.Undo();
            Publish(new UndoneEvent(), Full);
            for (int seat = 0; seat < _seats.Length; seat++) Publish(new UndoneEvent(_beliefs.View(Record.State, seat)), SeatView(seat));
        }

        private SnapshotEvent SnapshotLocked() => new(new GameState(Record.State), Record.History.Select(Copy).ToList(), Record.Redo.Select(Copy).ToList(),
            (SeatKind[])_seats.Clone(), Playback, AgentSettings);

        /// <summary>
        /// A copy of an entry for clients, so their replays never touch the match's actions
        /// </summary>
        private static HistoryEntry Copy(HistoryEntry entry) => entry.Action is Action action ? HistoryEntry.Of(action.Clone()) : entry;

        private void Publish(MatchEvent e, string? to = null) => Published?.Invoke(e, to);

        /// <summary>
        /// Sends an event from the host (the server) to the match's listeners, in order with the match's own events
        /// </summary>
        public void Notify(MatchEvent e, string? to)
        {
            lock (_lock) Publish(e, to);
        }

        /// <summary>
        /// Checks and applies a command from a session
        /// </summary>
        public void Handle(SessionInfo sender, Command command)
        {
            bool positionChanged = false;
            lock (_lock)
            {
                string? refusal = Permissions.Check(sender, command, Record.State, _seats);
                if (refusal != null)
                {
                    Publish(new ErrorEvent(refusal), sender.Id);
                    return;
                }

                switch (command)
                {
                    case PlayMove move:
                        positionChanged = PlayLocked(move.Action, sender);
                        break;

                    case UndoCommand:
                        if (Record.History.Count == 0) { Publish(new ErrorEvent("There is nothing to undo."), sender.Id); return; }
                        UndoLocked();
                        ResetAgentsLocked();
                        positionChanged = true;
                        break;

                    case RedoCommand:
                        positionChanged = RedoLocked(sender);
                        break;

                    case ApplyEdit edit:
                        if (!edit.Edit.Before.Equals(Record.State))
                        {
                            Publish(new ErrorEvent("The position changed before the edit arrived."), sender.Id);
                            return;
                        }
                        ApplyLocked(HistoryEntry.Of(edit.Edit), false);
                        ResetAgentsLocked();
                        positionChanged = true;
                        break;

                    case NewGame newGame:
                        {
                            if (newGame.Seats.Length != _seats.Length || newGame.Seats.Any(k => !Enum.IsDefined(k)))
                            {
                                Publish(new ErrorEvent($"A new game needs {_seats.Length} known seat kinds."), sender.Id);
                                return;
                            }
                            GameState state = newGame.ClearBoard ? new GameState(Record.State) : new GameState(MapGenerator.GenerateRandomClassic(newGame.CenterDesert), (uint)Record.State.Players.Length);
                            if (newGame.ClearBoard)
                            {
                                state.Board.Clear();
                                state.Reset();
                            }
                            state.Settings.FreeTrade = newGame.FreeTrade;
                            state.Settings.ShowTrackedCards = newGame.ShowTrackedCards;
                            _seats = (SeatKind[])newGame.Seats.Clone();
                            Record.Reset(state);
                            _agents = CreateAgents();
                            PublishSnapshotsLocked();
                            positionChanged = true;
                            break;
                        }

                    case LoadGame load:
                        Record.Reset(new GameState(load.State), load.Played.Select(HistoryEntry.Of), load.Redo.Select(HistoryEntry.Of));
                        ResetAgentsLocked();
                        PublishSnapshotsLocked();
                        positionChanged = true;
                        break;

                    case SetSeat seat:
                        if (seat.Seat is < 0 or >= 4 || !Enum.IsDefined(seat.Kind))
                        {
                            Publish(new ErrorEvent($"There is no seat {seat.Seat + 1} or seat kind {seat.Kind}."), sender.Id);
                            return;
                        }
                        _seats[seat.Seat] = seat.Kind;
                        _agents[seat.Seat] = Simulation.Seats.CreateAgent(seat.Kind, (sbyte)seat.Seat, Record.State, AgentSettings, _valueNetAvailable);
                        Publish(new SeatsEvent((SeatKind[])_seats.Clone()));
                        positionChanged = true;
                        break;

                    case SetPlayback playback:
                        Playback = playback.Playback;
                        Publish(new PlaybackEvent(Playback));
                        positionChanged = true;
                        break;

                    case ConfigureAgents configure:
                        AgentSettings = configure.Settings;
                        foreach (Agent agent in _agents)
                        {
                            if (agent is MCTSAgent mcts)
                            {
                                mcts.TimeLimitSeconds = AgentSettings.SecondsForAll ?? Simulation.Seats.PresetSeconds(_seats[mcts.PlayerIndex]);
                                mcts.IterationLimit = AgentSettings.IterationsForAll;
                                mcts.OpponentModel = AgentSettings.OpponentModel;
                            }
                        }
                        Publish(new AgentsEvent(AgentSettings));
                        break;

                    case StepCommand:
                        if (Record.Redo.Count > 0) positionChanged = RedoLocked(sender);
                        else _runner?.RequestStep();
                        break;

                    case KickSession or CreateToken or RevokeToken or SetServerOptions:
                        if (HostCommand == null)
                        {
                            Publish(new ErrorEvent("This match isn't hosted: start hosting it first."), sender.Id);
                            return;
                        }
                        HostCommand.Invoke(sender, command);
                        break;
                }
            }

            // Searches of the old position are dropped; the runner looks for the next agent move, the analysis at the position
            if (positionChanged)
            {
                _runner?.PositionChanged();
                _analysis?.Wake();
            }
        }

        private bool PlayLocked(Action action, SessionInfo sender)
        {
            // The match decides the random outcomes (dice, drawn and stolen cards) and what undo restores: a history that came
            // with the move is dropped, or a client could pick its own dice
            Action played = action.Clone();
            if (played is IReplayAction replay) replay.ClearHistory();
            if (played.PlayerIndex is < 0 or >= 4 || Record.State.HasEnded || !Record.State.CanPlayerAct(played.PlayerIndex) || !played.IsValidFor(Record.State))
            {
                Publish(new ErrorEvent($"{action.GetType().Name} is not a legal move now."), sender.Id);
                return false;
            }

            ApplyLocked(HistoryEntry.Of(played), false);
            NotifyAgentsLocked(played);
            return true;
        }

        private bool RedoLocked(SessionInfo sender)
        {
            if (Record.Redo.Count == 0)
            {
                Publish(new ErrorEvent("There is nothing to redo."), sender.Id);
                return false;
            }

            HistoryEntry entry = Record.Redo[^1];
            if (entry.Action is Action action && (Record.State.HasEnded || !action.IsValidFor(Record.State)))
            {
                Publish(new ErrorEvent("The next move can't be redone in this position."), sender.Id);
                return false;
            }
            ApplyLocked(entry, isRedo: true);
            if (entry.Action is Action replayed) NotifyAgentsLocked(replayed);
            else ResetAgentsLocked();
            return true;
        }

        private Agent[] CreateAgents() => Enumerable.Range(0, _seats.Length)
            .Select(seat => Simulation.Seats.CreateAgent(_seats[seat], (sbyte)seat, Record.State, AgentSettings, _valueNetAvailable)).ToArray();

        private void ResetAgentsLocked()
        {
            foreach (Agent agent in _agents)
            {
                if (ReferenceEquals(agent, _busyAgent)) _busyAgentStale = true;
                else agent.ResetToState(Record.State);
            }
        }

        private void NotifyAgentsLocked(Action played)
        {
            foreach (Agent agent in _agents)
            {
                if (ReferenceEquals(agent, _busyAgent)) _busyAgentStale = true;
                else agent.HandleAction(played);
            }
        }

        #region For the agent runner
        /// <summary>
        /// Whether an agent should move now: its turn (or a human's turn with stand-ins), not paused unless it's a step, and the
        /// game not over
        /// </summary>
        internal bool WantsAgentMove(bool step)
        {
            lock (_lock) return WantsAgentMoveLocked(step);
        }

        private bool WantsAgentMoveLocked(bool step)
        {
            int acting = ActingPlayerOf(Record.State);
            if (acting < 0 || _busyAgent != null) return false;
            return step || (!Playback.Paused && (_seats[acting] != SeatKind.Human || Playback.HumanStandIns));
        }

        /// <summary>
        /// The agent that moves now, with a copy of the position and its version; the agent is busy until ReleaseAgent
        /// </summary>
        internal (Agent agent, GameState state, int version)? TakeAgentMove(bool step)
        {
            lock (_lock)
            {
                if (!WantsAgentMoveLocked(step)) return null;
                int acting = ActingPlayerOf(Record.State);
                _busyAgent = _agents[acting];
                _busyAgentStale = false;
                Publish(new ThinkingEvent(acting, true));
                _analysis?.Wake();
                return (_busyAgent, new GameState(Record.State), Record.Version);
            }
        }

        /// <summary>
        /// The search of the busy agent is over: it is the match's again, reset if the position changed meanwhile
        /// </summary>
        internal void ReleaseAgent(Agent agent)
        {
            lock (_lock)
            {
                if (!ReferenceEquals(agent, _busyAgent)) return;
                if (_busyAgentStale) agent.ResetToState(Record.State);
                _busyAgent = null;
                _busyAgentStale = false;
                Publish(new ThinkingEvent(agent.PlayerIndex, false));
            }
        }

        /// <summary>
        /// Plays an agent's move if the position is still the one it searched
        /// </summary>
        internal bool PlayAgentMove(Agent agent, Action action, int version)
        {
            lock (_lock)
            {
                if (Record.Version != version || !ReferenceEquals(_agents[agent.PlayerIndex], agent)) return false;
                if (!PlayLocked(action, new SessionInfo("agents", "Agents", Role.Admin, null))) return false;
            }
            _runner?.PositionChanged();
            _analysis?.Wake();
            return true;
        }

        internal void PublishError(string message)
        {
            lock (_lock) Publish(new ErrorEvent(message));
        }
        #endregion

        #region For the analysis runner
        /// <summary>
        /// The position's version, a copy of the position if its version isn't the known one, and the MCTS agent searching it
        /// </summary>
        internal (int version, GameState? state, MCTSAgent? thinking) AnalysisSnapshot(int knownVersion)
        {
            lock (_lock)
            {
                return (Record.Version, Record.Version != knownVersion ? new GameState(Record.State) : null, _busyAgent as MCTSAgent);
            }
        }

        /// <summary>
        /// Sends an analysis to spectators and admins if the position is still the one it is for, and keeps it for later
        /// listeners (one evaluation and the latest search)
        /// </summary>
        internal void PublishAnalysis(int version, MatchEvent analysis)
        {
            lock (_lock)
            {
                if (Record.Version != version) return;
                if (_analysisVersion != version)
                {
                    _analysisEvents.Clear();
                    _analysisVersion = version;
                }
                _analysisEvents.RemoveAll(e => e.GetType() == analysis.GetType());
                _analysisEvents.Add(analysis);
                Publish(analysis, Watchers);
            }
        }
        #endregion
    }
}
