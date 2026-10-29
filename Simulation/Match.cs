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

        // The agent searching on the runner thread: the match leaves it alone and resets it once the search is over
        private Agent? _busyAgent;
        private bool _busyAgentStale;

        public GameRecord Record { get; }
        public PlaybackState Playback { get; private set; } = PlaybackState.Default;
        public AgentSettings AgentSettings { get; private set; } = AgentSettings.Default;

        /// <summary>
        /// Raised for every event with its recipient: a session ID, or null for everyone. Raised while the match is locked, so
        /// handlers see the events in order; they must not call back into the match.
        /// </summary>
        public event Action<MatchEvent, string?>? Published;

        /// <summary>
        /// Raised for a command the match can't handle itself (kicking a session), after the role check
        /// </summary>
        public event Action<SessionInfo, Command>? HostCommand;

        public Match(GameState state, SeatKind[] seats, bool? valueNetAvailable = null)
        {
            if (seats.Length != state.Players.Length) throw new ArgumentException("One seat kind per player is needed.", nameof(seats));
            _valueNetAvailable = valueNetAvailable ?? Simulation.Seats.ValueNetAvailable;
            Record = new GameRecord(state);
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
        /// The agent searching right now, if it is an MCTS agent (for showing its search; local use only)
        /// </summary>
        public MCTSAgent? ThinkingAgent => _runner?.ThinkingAgent as MCTSAgent;

        /// <summary>
        /// Lets the agents play their seats from now on
        /// </summary>
        public void Start()
        {
            lock (_lock)
            {
                _runner ??= new AgentRunner(this);
            }
            _runner.Wake();
        }

        public void Dispose()
        {
            _runner?.Dispose();
        }

        /// <summary>
        /// Gives a new listener the snapshot and then every event from that point on, with nothing missed or doubled
        /// </summary>
        public void Subscribe(Action<MatchEvent, string?> listener, Action<SnapshotEvent> onSnapshot)
        {
            lock (_lock)
            {
                onSnapshot(SnapshotLocked());
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

        private SnapshotEvent SnapshotLocked() => new(new GameState(Record.State), Record.History.Select(Copy).ToList(), Record.Redo.Select(Copy).ToList(),
            (SeatKind[])_seats.Clone(), Playback, AgentSettings);

        /// <summary>
        /// A copy of an entry for clients, so their replays never touch the match's actions
        /// </summary>
        private static HistoryEntry Copy(HistoryEntry entry) => entry.Action is Action action ? HistoryEntry.Of(action.Clone()) : entry;

        private void Publish(MatchEvent e, string? to = null) => Published?.Invoke(e, to);

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
                        if (Record.Undo() == null) { Publish(new ErrorEvent("There is nothing to undo."), sender.Id); return; }
                        ResetAgentsLocked();
                        Publish(new UndoneEvent());
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
                        Record.Apply(HistoryEntry.Of(edit.Edit));
                        ResetAgentsLocked();
                        Publish(new AppliedEvent(HistoryEntry.Of(edit.Edit), false));
                        positionChanged = true;
                        break;

                    case NewGame newGame:
                        {
                            GameState state = newGame.ClearBoard ? new GameState(Record.State) : new GameState(MapGenerator.GenerateRandomClassic(newGame.CenterDesert), (uint)Record.State.Players.Length);
                            if (newGame.ClearBoard)
                            {
                                state.Board.Clear();
                                state.Reset();
                            }
                            _seats = (SeatKind[])newGame.Seats.Clone();
                            Record.Reset(state);
                            _agents = CreateAgents();
                            Publish(SnapshotLocked());
                            positionChanged = true;
                            break;
                        }

                    case LoadGame load:
                        Record.Reset(new GameState(load.State), load.Played.Select(HistoryEntry.Of), load.Redo.Select(HistoryEntry.Of));
                        ResetAgentsLocked();
                        Publish(SnapshotLocked());
                        positionChanged = true;
                        break;

                    case SetSeat seat:
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

                    case KickSession:
                        HostCommand?.Invoke(sender, command);
                        break;
                }
            }

            // Searches of the old position are dropped; the runner looks for the next agent move
            if (positionChanged) _runner?.PositionChanged();
        }

        private bool PlayLocked(Action action, SessionInfo sender)
        {
            if (Record.State.HasEnded || !Record.State.CanPlayerAct(action.PlayerIndex) || !action.IsValidFor(Record.State))
            {
                Publish(new ErrorEvent($"{action.GetType().Name} is not a legal move now."), sender.Id);
                return false;
            }

            Action played = action.Clone();
            Record.Apply(HistoryEntry.Of(played));
            NotifyAgentsLocked(played);
            Publish(new AppliedEvent(HistoryEntry.Of(played.Clone()), false));
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
            Record.Apply(entry, isRedo: true);
            if (entry.Action is Action replayed) NotifyAgentsLocked(replayed);
            else ResetAgentsLocked();
            Publish(new AppliedEvent(Copy(entry), true));
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
            return true;
        }

        internal void PublishError(string message)
        {
            lock (_lock) Publish(new ErrorEvent(message));
        }
        #endregion
    }
}
