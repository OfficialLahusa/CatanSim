using Client.Logging;
using Client.UI;
using Common;
using Common.Actions;
using Common.Serialization;
using ImGuiNET;
using Agents;
using Agents.Inference;
using SFML.Audio;
using SFML.Graphics;
using SFML.System;
using SFML.Window;
using Simulation;
using System.Collections.Concurrent;
using Action = Common.Actions.Action;
using Agents.MCTS;

namespace Client
{
    /// <summary>
    /// How the user takes part: playing seats, watching agents play, or editing the position freely
    /// </summary>
    public enum InteractionMode { Play, Spectate, Edit }

    /// <summary>
    /// The game screen. The game itself runs in a match (Simulation): in this process, or on a server the client hosts or
    /// joins. The screen sends commands to it and shows its copy of the match, which follows the match's events.
    /// </summary>
    public partial class GameScreen : Screen
    {
        private RenderWindow _window;
        private View _mapView;
        private View _uiView;

        // The match this client plays through; a local one unless connected to a server
        private ISimulation _sim;

        private GameRecord Record => _sim.Game.Record;
        private GameState State => _sim.Game.Record.State;
        private IReadOnlyList<HistoryEntry> History => _sim.Game.Record.History;
        private IReadOnlyList<HistoryEntry> Redo => _sim.Game.Record.Redo;
        private SeatKind[] Seats => _sim.Game.Seats;
        private PlaybackState Playback => _sim.Game.Playback;

        // Counts every change of the position, so caches (legal moves, priors) know when they are stale
        private int _stateVersion;

        private BoardRenderer _renderer;
        private BoardOverlay _overlay;
        private DiceWidget _diceWidget;
        private Particles _particles = new();

        private EventLog _eventLog;
        private ActionLogger _actionLogger;
        private float[] _rollDistribution;

        public static Font Font;
        public static Font BoldFont;

        // View movement
        private float _viewZoom = 1f;
        private float _viewZoomBase = 1f;
        private const float _moveSpeed = 300f;
        private Vector2i? _panStart;

        // FPS Counter
        private float _frameTimeSum = 0.0f;
        private int _frameTimeCount = 0;
        private float _latestAvgFrameTime = 0.0f;

        // Map generator options
        private bool _centerDesert = false;

        private InteractionMode _mode = InteractionMode.Play;
        public const int PLAYER_COUNT = 4;

        // Sounds
        private bool _muteQuickPlayouts = false;
        private Sound _placeSound;
        private Sound _diceRollSound;

        // UI Modals
        private readonly string saveFileModalId = "save-file";
        private readonly string loadFileModalId = "load-file";
        private readonly string upgradeSaveModalId = "upgrade-save";

        // Older save file waiting for the user's decision in the upgrade modal
        private string? _pendingUpgradePath;
        private int _pendingUpgradeVersion;
        private bool _openUpgradeModal;

        // Search budget of the MCTS seats (developer tools): the seat presets, one thinking time, or a number of iterations
        // (~250k iterations per second on the GPU)
        private float _mctsThinkingSeconds = 0.5f;
        private readonly string[] _mctsBudgetTypes = ["Seat presets", "Time", "Iterations"];
        private int _selectedMctsBudget = 0;
        private int _mctsIterations = 125_000;
        private readonly string[] _opponentModels = Enum.GetNames<OpponentModel>();
        private int _selectedOpponentModel = (int)OpponentModel.MaxN;

        // How long the agent on turn has been thinking
        private readonly System.Diagnostics.Stopwatch _thinkingClock = new();

        private int _randomPlayoutCount = 1000;
        private int _randomPlayoutThreadCount = Environment.ProcessorCount / 2; // Usually optimal, since the playouts are not CPU-bound

        // State value network for win rate evaluation
        // Null when the model file is missing (it is not tracked in git), the net evaluation is unavailable then
        private StateValueNet? _stateValueNet = StateValueNet.IsDefaultModelAvailable ? new StateValueNet(StateValueNet.DefaultModelFile) : null;
        private float[] _valuation = new float[PLAYER_COUNT];
        // Win chances after each history entry, for the eval bar's history
        private List<float[]> _valuationHistory = [];

        // Developer window (the old menu)
        private bool _showDeveloperWindow;

        static GameScreen()
        {
            Font = new Font(@"..\..\..\res\Open_Sans\static\OpenSans-Regular.ttf");
            BoldFont = new Font(@"..\..\..\res\Open_Sans\static\OpenSans-Bold.ttf");
        }

        public GameScreen(RenderWindow window)
        {
            _window = window;

            if (_stateValueNet == null)
                Console.WriteLine($"Value net not found at \"{StateValueNet.DefaultModelPath}\": net evaluation, GreedyAgent and MCTSAgent are unavailable.");

            // A local match: three agents and the human on seat 4
            SeatKind bot = _stateValueNet != null ? SeatKind.MctsMedium : SeatKind.Builder;
            _sim = StartLocal(new GameState(MapGenerator.GenerateRandomClassic(), PLAYER_COUNT), [bot, bot, bot, SeatKind.Human]);

            _renderer = new BoardRenderer(State.Board, 120, 20);
            _overlay = new BoardOverlay(_renderer);
            _renderer.BeforeBuildings = (target, states) => _overlay.DrawUnderBuildings(target, State.Board);
            _diceWidget = new DiceWidget(window);
            _diceWidget.Active = true;

            _eventLog = new EventLog();
            _actionLogger = new ActionLogger(_eventLog);
            _actionLogger.Init();

            // Bucket indices are the roll totals
            // Padded for symmetric plotting with correct indices
            _rollDistribution = new float[15];

            _placeSound = new Sound(Sounds.Place);
            _placeSound.Volume = 40f;

            _diceRollSound = new Sound(Sounds.DiceRolling);
            _diceRollSound.Volume = 50f;

            _mapView = new View(ClientUtils.RoundVec2f(_renderer.GetTileCenter(3, 3)), new Vector2f(window.Size.X, window.Size.Y));
            _uiView = new View(new Vector2f(0, 0), _mapView.Size);

            ProcessSimulation();

            _window.SetView(_mapView);
            FitBoard();

            _window.MouseWheelScrolled += Window_MouseWheelScrolled;
            _window.Resized += Window_Resized;
            _window.KeyPressed += Window_KeyPressed;
            _window.MouseButtonPressed += Window_MouseButtonPressed;
            _window.MouseButtonReleased += Window_MouseButtonReleased;
        }

        #region Seats
        private bool IsHuman(int seat) => Seats[seat] == SeatKind.Human;

        /// <summary>
        /// Whether this client may move for a seat: an admin for any human seat, a player or agent bridge for its own seat
        /// </summary>
        private bool ControlsSeat(int seat) => _sim.Session is SessionInfo session && session.Role switch
        {
            Role.Admin => IsHuman(seat),
            Role.Player or Role.Agent => session.Seat == seat && IsHuman(seat),
            _ => false
        };

        private bool IsAdmin => _sim.Session?.Role == Role.Admin;

        /// <summary>
        /// The player who has to act now, or -1 once the game has ended
        /// </summary>
        private int ActingPlayer
        {
            get
            {
                if (State.HasEnded) return -1;
                for (int playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                {
                    if (State.CanPlayerAct(playerIdx))
                        return playerIdx;
                }
                return -1;
            }
        }

        /// <summary>
        /// Whose hidden information the screen shows: in Play mode the human who acts (or the first human seat this client
        /// plays), nobody's (-1) for a spectator and while the screen is passed between humans, everything (null) in Spectate and
        /// Edit mode for an admin and once the game is over
        /// </summary>
        private int? Viewer
        {
            get
            {
                if (State.HasEnded) return null;
                if (_sim.Session?.Role == Role.Spectator) return -1;
                if (_sim.Session?.Role is Role.Player or Role.Agent) return _sim.Session.Seat;
                if (_mode != InteractionMode.Play) return null;
                if (_curtainFor != null) return -1;
                int acting = ActingPlayer;
                if (acting >= 0 && ControlsSeat(acting)) return acting;
                if (_lastHumanViewer is int last && ControlsSeat(last)) return last;
                for (int seat = 0; seat < PLAYER_COUNT; seat++)
                    if (ControlsSeat(seat)) return seat;
                return null;
            }
        }

        private int? _lastHumanViewer;

        // The human the screen is being passed to; until they confirm, no hidden information is shown (Viewer is -1)
        private int? _curtainFor;

        /// <summary>
        /// The agent settings of the developer tools
        /// </summary>
        private AgentSettings AgentSettingsFromTools => new(_selectedMctsBudget == 1 ? _mctsThinkingSeconds : null,
            _selectedMctsBudget == 2 ? _mctsIterations : 0, (OpponentModel)_selectedOpponentModel);
        #endregion

        public void Draw(Time deltaTime)
        {
            _window.Clear(new Color(8, 25, 75));

            // Draw map
            _window.SetView(_mapView);
            PrepareOverlay();
            _renderer.Draw(_window, RenderStates.Default, State, (sbyte)Math.Max(0, ActingPlayer));
            _overlay.Draw(_window, State.Board);

            // Draw UI
            _window.SetView(_uiView);
            _window.Draw(_diceWidget);

            DrawHud();

            GuiImpl.Render(_window);
            CaptureScreenshot();

            _window.Display();
        }

        public void HandleInput(Time deltaTime)
        {
            // Block if ImGui is capturing input
            if (ImGui.GetIO().WantCaptureKeyboard)
                return;

            Vector2f moveDelta = new Vector2f();
            if (Keyboard.IsKeyPressed(Keyboard.Key.W)) moveDelta.Y -= 1;
            if (Keyboard.IsKeyPressed(Keyboard.Key.S)) moveDelta.Y += 1;
            if (Keyboard.IsKeyPressed(Keyboard.Key.A)) moveDelta.X -= 1;
            if (Keyboard.IsKeyPressed(Keyboard.Key.D)) moveDelta.X += 1;
            moveDelta *= deltaTime.AsSeconds() * _moveSpeed * _viewZoom;
            _mapView.Move(moveDelta);

            // Right mouse button drags the board
            if (_panStart is Vector2i start)
            {
                Vector2i now = Mouse.GetPosition(_window);
                _mapView.Move(new Vector2f(start.X - now.X, start.Y - now.Y) * _viewZoom);
                _panStart = now;
            }
        }

        public void Update(Time deltaTime)
        {
            float seconds = deltaTime.AsSeconds();

            // Update FPS counter every 0.5s. Is initialized on the first frame
            _frameTimeCount++;
            _frameTimeSum += seconds;
            if (_frameTimeSum > 0.5f || _latestAvgFrameTime <= 0.000001f)
            {
                _latestAvgFrameTime = _frameTimeSum / _frameTimeCount;
                _frameTimeSum = 0.0f;
                _frameTimeCount = 0;
            }

            _window.DispatchEvents();
            GuiImpl.Update(_window, deltaTime);

            _overlay.Update(seconds);
            _particles.Update(seconds);

            ProcessSimulation();
            UpdateNetwork();
            StepSelfTest();
            UpdateAnalysis();
        }

        #region Simulation
        /// <summary>
        /// A match in this process, with its agents running
        /// </summary>
        private LocalSimulation StartLocal(GameState state, SeatKind[] seats, IEnumerable<HistoryEntry>? history = null, IEnumerable<HistoryEntry>? redo = null)
        {
            Match match = new Match(state, seats, _stateValueNet != null);
            if (history != null || redo != null)
            {
                // Continue a game from another match (e.g. after leaving a server) with its moves
                match.Handle(new SessionInfo("local", "Local", Role.Admin, null), new LoadGame(new GameState(state),
                    history?.Where(e => e.Action != null).Select(e => e.Action!).ToList() ?? [], redo?.Where(e => e.Action != null).Select(e => e.Action!).ToList() ?? []));
            }
            match.Handle(new SessionInfo("local", "Local", Role.Admin, null), new SetPlayback(PlaybackState.Default));
            match.Start();
            return new LocalSimulation(match);
        }

        private void Send(Command command)
        {
            _sim.Send(command);
            ProcessSimulation();
        }

        /// <summary>
        /// Applies what the match did since the last frame and lets the screen follow: log, effects, sounds, win chances
        /// </summary>
        private void ProcessSimulation()
        {
            _sim.ProcessEvents(OnSimulationEvent);
        }

        private void OnSimulationEvent(MatchEvent e)
        {
            switch (e)
            {
                case SnapshotEvent:
                    RebuildLogAndRollDistribution();
                    _overlay.ClearRoll();
                    _particles.Clear();
                    ClearAnalysis();
                    _endScreenDismissed = false;
                    _lastHumanViewer = null;
                    _curtainFor = null;
                    AfterPositionChanged();
                    _valuationHistory = [(float[])_valuation.Clone()];
                    break;

                case AppliedEvent applied:
                    if (applied.Entry.Action is Action action)
                    {
                        if (action is RollAction rollAction)
                        {
                            _rollDistribution[rollAction.History!.RollResult.Total]++;
                            ShowRollEffects(rollAction);
                        }
                        else if (action is EndTurnAction)
                        {
                            _overlay.ClearRoll();
                        }
                        _actionLogger.Log(action, State);
                        if (TargetOf(action) is BoardTarget placed)
                            _overlay.ShowPlacement(placed, action is RobberAction ? Color.White : Theme.PlayerColor(action.PlayerIndex));
                        if (!_muteQuickPlayouts) PlaySoundForAction(action);
                    }
                    else if (applied.Entry.Edit is BoardEdit edit)
                    {
                        _eventLog.WriteLine(new SentenceEntry([("Edit: ", Theme.TextMuted), (char.ToUpper(edit.Description[0]) + edit.Description[1..], Theme.TextSoft)]));
                        _eventLog.PushSection();
                        _overlay.ClearRoll();
                    }
                    AfterPositionChanged();
                    _valuationHistory.Add((float[])_valuation.Clone());
                    break;

                case UndoneEvent:
                    if (Redo.Count > 0 && Redo[^1].Action is RollAction undoneRoll)
                        _rollDistribution[undoneRoll.History!.RollResult.Total]--;
                    _eventLog.PopSection();
                    _actionLogger.UpdateTracking(State);
                    _overlay.ClearRoll();
                    _particles.Clear();
                    if (_valuationHistory.Count > 1) _valuationHistory.RemoveAt(_valuationHistory.Count - 1);
                    AfterPositionChanged();
                    break;

                case SeatsEvent:
                    ClearHumanInput();
                    _stateVersion++;
                    break;

                case ThinkingEvent thinking:
                    if (thinking.Thinking) _thinkingClock.Restart();
                    break;

                case ErrorEvent error:
                    _eventLog.WriteLine(new ColoredStrEntry(error.Message, Color.Red));
                    break;

                case ClosedEvent closed:
                    OnConnectionClosed(closed.Reason);
                    break;
            }
        }

        /// <summary>
        /// Everything that follows from a changed position: board geometry, win chances, dice, cached legal moves
        /// </summary>
        private void AfterPositionChanged()
        {
            _stateVersion++;
            _renderer.Board = State.Board;
            _renderer.Update();
            UpdateValuation();

            _diceWidget.Active = State.Turn.MustRoll;
            _diceWidget.RollResult = State.Turn.LastRoll;
            _diceWidget.UpdateSprites();

            // Between two humans on one screen a curtain hides the hands until the next one takes over
            int acting = ActingPlayer;
            if (acting >= 0 && ControlsSeat(acting) && _sim.Session?.Role == Role.Admin)
            {
                if (_mode == InteractionMode.Play && HumanCount() > 1 && _lastHumanViewer is int last && last != acting) _curtainFor = acting;
                _lastHumanViewer = acting;
            }
            ClearHumanInput();
        }

        private void UndoAction() => Send(new UndoCommand());

        private void RedoAction(bool playSound = true) => Send(new RedoCommand());

        /// <summary>
        /// Starts over on a new map (or a cleared board) with the current seats
        /// </summary>
        private void NewGame(bool clearBoard = false) => Send(new Simulation.NewGame(Seats, _centerDesert, clearBoard));

        private void SetSeat(int seat, SeatKind kind)
        {
            if (Simulation.Seats.NeedsValueNet(kind) && kind != SeatKind.Human && _stateValueNet == null)
            {
                _eventLog.WriteLine(new ColoredStrEntry($"{Simulation.Seats.Name(kind)} needs the value net {StateValueNet.DefaultModelFile}, which is missing", Color.Red));
                return;
            }
            Send(new Simulation.SetSeat(seat, kind));
        }

        /// <summary>
        /// How the agents move in each mode: in Play mode the agent seats, in Spectate mode all seats while playback runs, in
        /// Edit mode none
        /// </summary>
        private void ApplyPlaybackForMode(bool playing)
        {
            if (!IsAdmin) return;
            PlaybackState playback = _mode switch
            {
                InteractionMode.Play => Playback with { Paused = false, HumanStandIns = false },
                InteractionMode.Spectate => Playback with { Paused = !playing, HumanStandIns = true },
                _ => Playback with { Paused = true, HumanStandIns = false }
            };
            if (playback != Playback) Send(new SetPlayback(playback));
        }
        #endregion

        #region Roll effects
        /// <summary>
        /// Rings on the producing tiles and a card flying to each receiving player per yield
        /// </summary>
        private void ShowRollEffects(RollAction roll)
        {
            RollAction.RollActionHistory outcome = roll.History!;
            int total = outcome.RollResult.Total;
            Board board = State.Board;

            List<int> tiles = [];
            for (int tileIdx = 0; tileIdx < board.Map.Length; tileIdx++)
            {
                Tile tile = board.Map.Values[tileIdx];
                if (tile.HasYield() && tile.Number == total && !board.HasRobber(tile)) tiles.Add(tileIdx);
            }
            _overlay.ShowRoll(tiles);

            if (outcome.TriggeredRobber || outcome.AwardedYields == null) return;

            // One card per yield from its tile to the player, as far as the bank paid it out
            uint[,] awarded = outcome.AwardedYields;
            uint[,] sent = new uint[PLAYER_COUNT, CardSet<ResourceCardType>.Values.Count];
            float delay = 0;
            foreach (int tileIdx in tiles)
            {
                Tile tile = board.Map.Values[tileIdx];
                ResourceCardType resource = tile.Type.ToCardType();
                int resourceIdx = CardSet<ResourceCardType>.ToInt(resource);
                Vector2f tileCenter = _renderer.GetTileCenter(tile);
                foreach (Intersection intersection in board.Adjacency.GetIntersections(tile))
                {
                    if (!board.HasBuilding(intersection.Index)) continue;
                    int owner = board.GetBuildingOwner(intersection.Index);
                    int amount = board.GetBuilding(intersection.Index) == Intersection.BuildingType.City ? 2 : 1;
                    for (int i = 0; i < amount && sent[owner, resourceIdx] < awarded[owner, resourceIdx]; i++)
                    {
                        sent[owner, resourceIdx]++;
                        Vector2i from = _window.MapCoordsToPixel(tileCenter, _mapView);
                        _particles.Add(resource, new System.Numerics.Vector2(from.X, from.Y) / GuiImpl.UiScale, CardDestination(owner, resource), delay);
                        delay += 0.12f;
                    }
                }
            }
        }
        #endregion

        #region Developer tools
        /// <summary>
        /// Plays a random game to its end, then goes back through it undoing and redoing every move and reports moves whose
        /// redo doesn't give the same state hash (local games only)
        /// </summary>
        private void FindInconsistentHashes()
        {
            if (_sim is not LocalSimulation) return;
            Agent[] agents = Enumerable.Range(0, PLAYER_COUNT).Select(i => (Agent)new RandomAgent((sbyte)i)).ToArray();
            Send(new SetPlayback(Playback with { Paused = true }));
            while (!State.HasEnded)
                Send(new PlayMove(agents[ActingPlayer].Act(State)));

            int checkedMoves = 0, mismatches = 0;
            while (History.Count > 0)
            {
                int prevHash = State.GetHashCode();
                int prevPlayerHash = HashCode.Combine(State.Players[0], State.Players[1], State.Players[2], State.Players[3]);
                int prevTurnHash = State.Turn.GetHashCode();
                int prevBoardHash = State.Board.GetHashCode();
                int prevResHash = State.ResourceBank.GetHashCode();
                string name = History[^1].Action?.GetType().Name ?? "Edit";

                UndoAction();
                RedoAction();

                if (prevHash != State.GetHashCode())
                {
                    mismatches++;
                    Console.WriteLine($"{name} mismatched: pre {prevHash:X}, post {State.GetHashCode():X}");
                    if (prevPlayerHash != HashCode.Combine(State.Players[0], State.Players[1], State.Players[2], State.Players[3])) Console.WriteLine("\t- PlayerState");
                    if (prevTurnHash != State.Turn.GetHashCode()) Console.WriteLine("\t- TurnState");
                    if (prevBoardHash != State.Board.GetHashCode()) Console.WriteLine("\t- Board");
                    if (prevResHash != State.ResourceBank.GetHashCode()) Console.WriteLine("\t- ResourceBank");
                }
                checkedMoves++;
                UndoAction();
            }
            _eventLog.WriteLine(new ColoredStrEntry($"Checked {checkedMoves} moves: {mismatches} hash mismatches", mismatches == 0 ? Color.Green : Color.Red));
        }

        private void BenchmarkPlayouts(int matches)
        {
            if (State.HasEnded) return;

            Agent[] agents = new Agent[PLAYER_COUNT];
            for (sbyte i = 0; i < PLAYER_COUNT; i++) agents[i] = new RandomAgent(i);

            Clock playoutClock = new Clock();
            int playedActions = 0;
            int playedRounds = 0;

            for (int it = 0; it < matches; it++)
            {
                GameState state = new GameState(MapGenerator.GenerateRandomClassic(_centerDesert), PLAYER_COUNT);
                while (!state.HasEnded)
                {
                    int acting = 0;
                    while (!state.CanPlayerAct(acting)) acting++;
                    agents[acting].Act(state).Apply(state);
                    playedActions++;
                }
                playedRounds += state.Turn.RoundCounter;
            }

            float ms = playoutClock.ElapsedTime.AsSeconds() * 1000f;
            Console.WriteLine($"\nRandom playout of {matches:n0} matches ({playedRounds:n0} rounds, {playedActions:n0} actions) took {ms:n} ms");
            Console.WriteLine($"Avg. {ms / matches} ms/match, {ms / playedRounds} ms/round, {ms / playedActions} ms/action\n");
        }

        private async Task SimulateRandomRollouts(GameState startState, int playoutCount, int threadCount)
        {
            Console.WriteLine($"Simulating {playoutCount:n0} random playouts...");

            Clock playoutClock = new Clock();

            // Create shared RandomAgents to run the playouts
            // Simulator agent configuration is irrelevant
            Agent[] randomAgents = new Agent[4];
            for (sbyte i = 0; i < 4; i++)
                randomAgents[i] = new RandomAgent(i);

            ConcurrentQueue<int> runs = new ConcurrentQueue<int>(Enumerable.Range(0, playoutCount));
            CountdownEvent countdown = new CountdownEvent(playoutCount);

            object lockObj = new object();
            List<int> timesWon = new List<int>();
            for (int i = 0; i < PLAYER_COUNT; i++)
                timesWon.Add(0);

            System.Action doPlayout = () =>
            {
                while (runs.TryDequeue(out int runIdx))
                {
                    GameState simState = new GameState(startState);
                    while (!simState.HasEnded)
                    {
                        // Find first player that is allowed to act on state
                        sbyte? actingPlayerIdx = null;
                        for (sbyte playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                        {
                            if (simState.CanPlayerAct(playerIdx))
                            {
                                actingPlayerIdx = playerIdx;
                                break;
                            }
                        }

                        if (!actingPlayerIdx.HasValue) throw new InvalidOperationException();

                        // Pick action
                        Action playedAction = randomAgents[actingPlayerIdx.Value].Act(simState);

                        // Apply action to state
                        playedAction.Apply(simState);

                        // No agent state update necessary, since RandomAgents are stateless
                    }

                    // Since players can only win on their own turn, the winner is the current turn player
                    int winnerIdx = simState.Turn.PlayerIndex;

                    // Collect match result thread-safely
                    lock (lockObj)
                    {
                        timesWon[winnerIdx]++;
                    }

                    countdown.Signal();
                }
            };

            Console.WriteLine($"Starting {threadCount} threads...");

            List<Task> tasks = new List<Task>();
            for (int t = 0; t < threadCount; t++)
            {
                tasks.Add(Task.Run(doPlayout));
            }

            countdown.Wait();

            // Not strictly necessary, but ensures all tasks are complete
            await Task.WhenAll(tasks);

            float totalSeconds = playoutClock.ElapsedTime.AsSeconds();
            Console.WriteLine($"All playouts completed in {totalSeconds} seconds ({playoutCount / totalSeconds:n1}/s, {playoutCount / totalSeconds / threadCount:n1}/s per thread).");

            for (int i = 0; i < PLAYER_COUNT; i++)
            {
                Console.WriteLine($"Player {i} won {timesWon[i]:n0} times ({(float)timesWon[i] / playoutCount:P2})");
            }
        }
        #endregion

        private void ResetLog()
        {
            _eventLog.Clear();
            _actionLogger.Init();
            _rollDistribution = new float[_rollDistribution.Length];
        }

        private int HumanCount() => Seats.Count(s => s == SeatKind.Human);

        private void UpdateValuation()
        {
            _valuation = _stateValueNet != null
                ? _stateValueNet.Run(StateVectorizer.For(_stateValueNet).Vectorize(State))
                : SimpleAgent.StateValueFunc(State);

            // The net rates a finished game by its position; the result itself is certain
            if (State.HasEnded)
            {
                for (int i = 0; i < _valuation.Length; i++)
                    _valuation[i] = i == State.Turn.PlayerIndex ? 1 : 0;
            }
        }

        private void Window_MouseWheelScrolled(object? sender, MouseWheelScrollEventArgs e)
        {
            if (ImGui.GetIO().WantCaptureMouse)
                return;

            _mapView.Zoom(1 / _viewZoom);
            _viewZoomBase = (float)Math.Max(0.001, _viewZoomBase - e.Delta);
            _viewZoom = (float)Math.Pow(1.3, _viewZoomBase) / 1.3f;
            _mapView.Zoom(_viewZoom);

            _mapView.Size = ClientUtils.RoundVec2f(_mapView.Size);
        }

        private void Window_Resized(object? sender, SizeEventArgs e)
        {
            _uiView.Size = new Vector2f(e.Width, e.Height);
            FitBoard();
        }

        /// <summary>
        /// Zooms and centers the board so it fits between the panel columns
        /// </summary>
        private void FitBoard()
        {
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            foreach (Tile tile in State.Board.Map.Values)
            {
                if (tile.Type == Tile.TileType.NonPlayable) continue;
                Vector2f c = _renderer.GetTileCenter(tile);
                minX = Math.Min(minX, c.X - _renderer.FlatSideLength);
                maxX = Math.Max(maxX, c.X + _renderer.FlatSideLength);
                minY = Math.Min(minY, c.Y - _renderer.SideLength);
                maxY = Math.Max(maxY, c.Y + _renderer.SideLength);
            }
            if (minX > maxX) return;

            // The space between the seat panels and the right column, in pixels
            float scale = GuiImpl.UiScale;
            float columns = 2 * (Margin + ColumnWidth + 8) * scale;
            float width = Math.Max(200, _window.Size.X - columns);
            float height = Math.Max(200, _window.Size.Y - 16 * scale);
            _viewZoom = Math.Max((maxX - minX) / width, (maxY - minY) / height);
            _viewZoomBase = (float)(Math.Log(_viewZoom * 1.3) / Math.Log(1.3));
            _mapView.Size = new Vector2f(_window.Size.X, _window.Size.Y) * _viewZoom;
            _mapView.Center = new Vector2f((minX + maxX) / 2, (minY + maxY) / 2);
        }

        public void PlaySoundForAction(Action action)
        {
            if (action is RollAction && _diceRollSound.Status != SoundStatus.Playing)
            {
                _diceRollSound.Play();
            }
            else if ((action is SettlementAction
                || action is CityAction
                || action is RoadAction
                || action is FirstInitialRoadAction
                || action is FirstInitialSettlementAction
                || action is SecondInitialRoadAction
                || action is SecondInitialSettlementAction
                || action is RobberAction)
                && _placeSound.Status != SoundStatus.Playing)
            {
                _placeSound.Play();
            }
        }

        #region Save files
        public void ReadSaveFile(string filepath)
        {
            SaveFile saveFile;

            // Unreadable, malformed or outdated files are reported instead of taking the client down.
            // Load also takes the history of sibling states from the file they reference.
            try
            {
                saveFile = SaveFileSerializer.Load(filepath);
            }
            catch (UnsupportedSaveFileVersionException e) when (SaveFileUpgrader.CanUpgrade(e.Version))
            {
                // Older file: the upgrade modal in the menu asks before converting it
                _pendingUpgradePath = filepath;
                _pendingUpgradeVersion = e.Version;
                _openUpgradeModal = true;
                return;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not load save file \"{filepath}\": {e.Message}");
                _eventLog.WriteLine(new ColoredStrEntry($"Could not load save file: {e.Message}", Color.Red));
                return;
            }

            LoadSaveFile(saveFile);

            if (saveFile.LoadWarning != null)
            {
                Console.WriteLine($"Warning: {saveFile.LoadWarning}");
                _eventLog.WriteLine(new ColoredStrEntry(saveFile.LoadWarning, Color.Yellow));
            }
        }

        private void UpgradeAndLoadSaveFile(string filepath)
        {
            try
            {
                string upgraded = SaveFileUpgrader.Upgrade(File.ReadAllText(filepath));
                LoadSaveFile(SaveFileSerializer.Deserialize(upgraded));
                _eventLog.WriteLine(new ColoredStrEntry($"Upgraded \"{Path.GetFileName(filepath)}\" to save file version {SaveFile.CurrentVersion} and loaded it", Color.Green));
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not upgrade save file \"{filepath}\": {e.Message}");
                _eventLog.WriteLine(new ColoredStrEntry($"Could not upgrade save file: {e.Message}", Color.Red));
            }
        }

        private void LoadSaveFile(SaveFile saveFile)
        {
            if (!IsAdmin)
            {
                _eventLog.WriteLine(new ColoredStrEntry("Only an admin can load a game into this match", Color.Red));
                return;
            }
            Send(new LoadGame(saveFile.GameState, saveFile.PlayedActions, saveFile.UndoHistory));
        }

        /// <summary>
        /// Rebuilds the event log (one section per history entry, as undo expects) and the roll histogram from the history, by
        /// going back from the current position to the start and forward again
        /// </summary>
        private void RebuildLogAndRollDistribution()
        {
            ResetLog();

            try
            {
                GameState replay = new GameState(State);
                for (int i = History.Count - 1; i >= 0; i--)
                {
                    if (History[i].Action is Action action) action.Clone().Revert(replay);
                    else replay = History[i].Edit!.Before;
                }

                foreach (HistoryEntry entry in History)
                {
                    if (entry.Action is Action action)
                    {
                        Action copy = action.Clone();
                        copy.Apply(replay);
                        _actionLogger.Log(copy, replay);
                        if (copy is RollAction roll && roll.History != null) _rollDistribution[roll.History.RollResult.Total]++;
                    }
                    else
                    {
                        replay = entry.Edit!.After;
                        _eventLog.WriteLine(new SentenceEntry([("Edit: ", Theme.TextMuted), (char.ToUpper(entry.Edit.Description[0]) + entry.Edit.Description[1..], Theme.TextSoft)]));
                        _eventLog.PushSection();
                    }
                }
            }
            catch (Exception e)
            {
                ResetLog();
                _eventLog.WriteLine(new ColoredStrEntry($"Could not rebuild the log of this game: {e.Message}", Color.Red));
            }

            _actionLogger.UpdateTracking(State);
        }

        public void WriteSaveFile(string filepath)
        {
            // Edits change the position outside the rules, so only the moves since the last edit can be replayed from the file
            if (History.Any(entry => entry.Edit != null))
                _eventLog.WriteLine(new ColoredStrEntry("The save keeps the edited position and the moves since the last edit", Color.Yellow));

            SaveFile saveFile = new SaveFile(State, Record.ActionsSinceLastEdit(), Record.RedoActionsUntilEdit());
            string saveData = SaveFileSerializer.Serialize(saveFile);
            File.WriteAllText(filepath, saveData);
        }

        /// <summary>
        /// Saves the game into memory and loads it again (a check of the save file format)
        /// </summary>
        public void SerializeLoop()
        {
            SaveFile saveFile = new SaveFile(State, Record.ActionsSinceLastEdit(), Record.RedoActionsUntilEdit());
            LoadSaveFile(SaveFileSerializer.Deserialize(SaveFileSerializer.Serialize(saveFile)));
        }
        #endregion
    }
}
