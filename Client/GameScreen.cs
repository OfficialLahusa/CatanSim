using Client.Editing;
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
    /// Who plays a seat
    /// </summary>
    public enum SeatKind { Human, MctsEasy, MctsMedium, MctsHard, Greedy, Builder, Simple, Random }

    /// <summary>
    /// A played move or an edit, in the order they happened; undo goes back through them
    /// </summary>
    public sealed record HistoryEntry(Action? Action, BoardEdit? Edit);

    public partial class GameScreen : Screen
    {
        private RenderWindow _window;
        private View _mapView;
        private View _uiView;

        private GameState _state;
        private Stack<HistoryEntry> _history;
        private Stack<HistoryEntry> _redo;

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

        // Seats and their agents; a human seat keeps a stand-in agent that plays it in Spectate mode
        private SeatKind[] _seats = [SeatKind.MctsMedium, SeatKind.MctsMedium, SeatKind.MctsMedium, SeatKind.Human];
        private Agent[] _agents;
        private float _mctsThinkingSeconds = 0.5f;
        // Search budget per decision: thinking time or a number of iterations (~250k iterations per second on the GPU)
        private readonly string[] _mctsBudgetTypes = ["Seat presets", "Time", "Iterations"];
        private int _selectedMctsBudget = 0;
        private int _mctsIterations = 125_000;
        private readonly string[] _opponentModels = Enum.GetNames<OpponentModel>();
        private int _selectedOpponentModel = (int)OpponentModel.MaxN;

        // Agent move computed in the background on a copy of the state, so the window stays responsive while an agent thinks.
        // Only the UI thread touches _state; it applies the move once it is done and the state is still the one it was computed for.
        private Task<Action>? _agentMove;
        private CancellationTokenSource? _agentMoveCancellation;
        private Agent? _movingAgent;
        private GameState? _agentMoveStartState;
        private bool _agentMovePlaySound;
        private readonly System.Diagnostics.Stopwatch _agentMoveClock = new();

        // Agents move on their own in Play mode and while playback runs in Spectate mode, with a pause between moves so the
        // game can be followed
        private bool _spectatePlaying = true;
        private float _botPaceSeconds = 0.6f;
        private readonly System.Diagnostics.Stopwatch _sinceLastMove = System.Diagnostics.Stopwatch.StartNew();

        private int _randomPlayoutCount = 1000;
        private int _randomPlayoutThreadCount = Environment.ProcessorCount / 2; // Usually optimal, since the playouts are not CPU-bound

        // State value network for win rate evaluation
        // Null when the model file is missing (it is not tracked in git), the net evaluation is unavailable then
        private StateValueNet? _stateValueNet = StateValueNet.IsDefaultModelAvailable ? new StateValueNet(StateValueNet.DefaultModelFile) : null;
        private float[] _valuation = new float[PLAYER_COUNT];
        // Win chances after each move, for the eval bar's history
        private List<float[]> _valuationHistory = [];

        // Developer window (the old menu) and the window size it was opened at
        private bool _showDeveloperWindow;

        static GameScreen()
        {
            Font = new Font(@"..\..\..\res\Open_Sans\static\OpenSans-Regular.ttf");
            BoldFont = new Font(@"..\..\..\res\Open_Sans\static\OpenSans-Bold.ttf");
        }

        public GameScreen(RenderWindow window)
        {
            _window = window;

            _state = new GameState(MapGenerator.GenerateRandomClassic(), PLAYER_COUNT);
            _history = [];
            _redo = [];

            _renderer = new BoardRenderer(_state.Board, 120, 20);
            _overlay = new BoardOverlay(_renderer);
            _renderer.BeforeBuildings = (target, states) => _overlay.DrawUnderBuildings(target, _state.Board);
            _diceWidget = new DiceWidget(window);
            _diceWidget.Active = true;

            if (_stateValueNet == null)
            {
                Console.WriteLine($"Value net not found at \"{StateValueNet.DefaultModelPath}\": net evaluation, GreedyAgent and MCTSAgent are unavailable.");
                for (int i = 0; i < PLAYER_COUNT; i++)
                    if (_seats[i] != SeatKind.Human) _seats[i] = SeatKind.Builder;
            }

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

            _agents = new Agent[PLAYER_COUNT];
            for (sbyte i = 0; i < PLAYER_COUNT; i++)
                _agents[i] = CreateAgent(_seats[i], i);

            UpdateValuation();
            _valuationHistory.Add((float[])_valuation.Clone());

            _mapView = new View(ClientUtils.RoundVec2f(_renderer.GetTileCenter(3, 3)), new Vector2f(window.Size.X, window.Size.Y));
            _uiView = new View(new Vector2f(0, 0), _mapView.Size);

            _window.SetView(_mapView);
            FitBoard();

            _window.MouseWheelScrolled += Window_MouseWheelScrolled;
            _window.Resized += Window_Resized;
            _window.KeyPressed += Window_KeyPressed;
            _window.MouseButtonPressed += Window_MouseButtonPressed;
            _window.MouseButtonReleased += Window_MouseButtonReleased;
        }

        #region Seats and agents
        public static string SeatName(SeatKind kind) => kind switch
        {
            SeatKind.Human => "Human",
            SeatKind.MctsEasy => "MCTS · Easy",
            SeatKind.MctsMedium => "MCTS · Medium",
            SeatKind.MctsHard => "MCTS · Hard",
            SeatKind.Greedy => "Greedy",
            SeatKind.Builder => "Builder",
            SeatKind.Simple => "Simple",
            SeatKind.Random => "Random",
            _ => kind.ToString()
        };

        private static bool NeedsValueNet(SeatKind kind) => kind is SeatKind.MctsEasy or SeatKind.MctsMedium or SeatKind.MctsHard or SeatKind.Greedy or SeatKind.Human;

        private static float PresetSeconds(SeatKind kind) => kind switch
        {
            SeatKind.MctsEasy => 0.1f,
            SeatKind.MctsHard => 2f,
            _ => 0.5f
        };

        private Agent CreateAgent(SeatKind kind, sbyte seat)
        {
            // The stand-in for a human seat plays like the medium preset; without the net it falls back to the builder
            if (kind == SeatKind.Human) kind = _stateValueNet != null ? SeatKind.MctsMedium : SeatKind.Builder;

            return kind switch
            {
                SeatKind.Random => new RandomAgent(seat),
                SeatKind.Simple => new SimpleAgent(seat),
                SeatKind.Builder => new BuilderAgent(seat),
                SeatKind.Greedy => new GreedyAgent(seat),
                _ => new MCTSAgent(seat, _state, timeLimitSeconds: MctsSeconds(kind), opponentModel: (OpponentModel)_selectedOpponentModel,
                    iterationLimit: MctsIterationLimit)
            };
        }

        private double MctsSeconds(SeatKind kind) => _selectedMctsBudget == 1 ? _mctsThinkingSeconds : PresetSeconds(kind);

        // 0 searches for the thinking time
        private long MctsIterationLimit => _selectedMctsBudget == 2 ? _mctsIterations : 0;

        private void ApplyMctsBudget()
        {
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                if (_agents[seat] is MCTSAgent mctsAgent)
                {
                    mctsAgent.TimeLimitSeconds = MctsSeconds(_seats[seat]);
                    mctsAgent.IterationLimit = MctsIterationLimit;
                }
            }
        }

        private void SetSeat(int seat, SeatKind kind)
        {
            if (NeedsValueNet(kind) && kind != SeatKind.Human && _stateValueNet == null)
            {
                _eventLog.WriteLine(new ColoredStrEntry($"{SeatName(kind)} needs the value net {StateValueNet.DefaultModelFile}, which is missing", Color.Red));
                return;
            }

            CancelAgentMove();
            _seats[seat] = kind;
            _agents[seat] = CreateAgent(kind, (sbyte)seat);
            ClearHumanInput();
        }

        private bool IsHuman(int seat) => _seats[seat] == SeatKind.Human;

        /// <summary>
        /// The player who has to act now, or -1 once the game has ended
        /// </summary>
        private int ActingPlayer
        {
            get
            {
                if (_state.HasEnded) return -1;
                for (int playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                {
                    if (_state.CanPlayerAct(playerIdx))
                        return playerIdx;
                }
                return -1;
            }
        }

        /// <summary>
        /// Whose hidden information the screen shows: in Play mode the human who acts (or the first human seat), everything
        /// (null) in Spectate and Edit mode and once the game is over
        /// </summary>
        private int? Viewer
        {
            get
            {
                if (_mode != InteractionMode.Play || _state.HasEnded) return null;
                int acting = ActingPlayer;
                if (acting >= 0 && IsHuman(acting)) return acting;
                if (_lastHumanViewer is int last && IsHuman(last)) return last;
                for (int seat = 0; seat < PLAYER_COUNT; seat++)
                    if (IsHuman(seat)) return seat;
                return null;
            }
        }

        private int? _lastHumanViewer;
        #endregion

        public void Draw(Time deltaTime)
        {
            _window.Clear(new Color(8, 25, 75));

            // Draw map
            _window.SetView(_mapView);
            PrepareOverlay();
            _renderer.Draw(_window, RenderStates.Default, _state, (sbyte)Math.Max(0, ActingPlayer));
            _overlay.Draw(_window, _state.Board);

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

            PollAgentMove();
            AdvanceAgents();
            StepSelfTest();
            UpdateAnalysis();
        }

        /// <summary>
        /// Starts the next agent move when it is an agent's turn: in Play mode for agent seats, in Spectate mode for all seats
        /// while playback runs; after the pause between moves
        /// </summary>
        private void AdvanceAgents()
        {
            if (_agentMove != null || _state.HasEnded) return;
            if (_sinceLastMove.Elapsed.TotalSeconds < _botPaceSeconds) return;

            int acting = ActingPlayer;
            if (acting < 0) return;

            bool agentTurn = _mode switch
            {
                InteractionMode.Play => !IsHuman(acting),
                InteractionMode.Spectate => _spectatePlaying,
                _ => false
            };
            if (agentTurn) StartAgentMove(!_muteQuickPlayouts);
        }

        #region Moves and history
        /// <summary>
        /// Plays a move on the current position: agents, history, log, sounds, roll effects and the visuals follow
        /// </summary>
        private void ApplyMove(Action action, bool playSound, bool keepRedo = false)
        {
            action.Apply(_state);

            // Agents are only used by one thread at a time; a move applied by an agent itself is already done
            if (_agentMove != null) CancelAgentMove();
            foreach (Agent agent in _agents)
                agent.HandleAction(action);

            _history.Push(new HistoryEntry(action, null));
            if (!keepRedo) _redo.Clear();

            if (action is RollAction rollAction)
            {
                _rollDistribution[rollAction.History!.RollResult.Total]++;
                ShowRollEffects(rollAction);
            }
            else if (action is EndTurnAction)
            {
                _overlay.ClearRoll();
            }

            _actionLogger.Log(action, _state);
            if (TargetOf(action) is BoardTarget placed)
                _overlay.ShowPlacement(placed, action is RobberAction ? Color.White : Theme.PlayerColor(action.PlayerIndex));

            if (playSound)
                PlaySoundForAction(action);

            AfterPositionChanged();
            _valuationHistory.Add((float[])_valuation.Clone());
            _sinceLastMove.Restart();
        }

        /// <summary>
        /// Goes one entry back: reverts a move, or restores the position before an edit
        /// </summary>
        private void UndoAction()
        {
            if (_history.Count == 0) return;
            CancelAgentMove();

            HistoryEntry entry = _history.Pop();
            if (entry.Action is Action action)
            {
                action.Revert(_state);
                if (action is RollAction rollAction)
                    _rollDistribution[rollAction.History!.RollResult.Total]--;
            }
            else if (entry.Edit is BoardEdit edit)
            {
                ReplaceState(edit.Before);
            }

            ResetAgents();
            _redo.Push(entry);

            // Remove latest section from log
            _eventLog.PopSection();
            _actionLogger.UpdateTracking(_state);

            _overlay.ClearRoll();
            _particles.Clear();
            if (_valuationHistory.Count > 1) _valuationHistory.RemoveAt(_valuationHistory.Count - 1);
            AfterPositionChanged();
        }

        private void RedoAction(bool playSound = true)
        {
            if (_redo.Count == 0) return;

            HistoryEntry entry = _redo.Peek();
            if (entry.Action is Action action)
            {
                if (_state.HasEnded || !action.IsValidFor(_state)) return;
                _redo.Pop();
                ApplyMove(action, playSound, keepRedo: true);
            }
            else if (entry.Edit is BoardEdit edit)
            {
                _redo.Pop();
                ApplyEdit(edit, keepRedo: true);
            }
        }

        /// <summary>
        /// Replaces the position (load, edits), keeping the renderer and agents in step
        /// </summary>
        private void ReplaceState(GameState state)
        {
            _state = state;
            _renderer.Board = _state.Board;
            ResetAgents();
        }

        /// <summary>
        /// Everything that follows from a changed position: board geometry, win chances, dice, cached legal moves
        /// </summary>
        private void AfterPositionChanged()
        {
            _stateVersion++;
            _renderer.Update();
            UpdateValuation();

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();

            int acting = ActingPlayer;
            if (acting >= 0 && IsHuman(acting)) _lastHumanViewer = acting;
            ClearHumanInput();
        }

        /// <summary>
        /// The played moves since the last edit, oldest first: what a save file can replay
        /// </summary>
        private List<Action> ActionsSinceLastEdit()
        {
            List<Action> actions = [];
            foreach (HistoryEntry entry in _history)
            {
                if (entry.Edit != null) break;
                actions.Add(entry.Action!);
            }
            actions.Reverse();
            return actions;
        }

        private List<Action> RedoActions()
        {
            List<Action> actions = [];
            foreach (HistoryEntry entry in _redo)
            {
                if (entry.Edit != null) break;
                actions.Add(entry.Action!);
            }
            return actions;
        }
        #endregion

        #region Agents
        private void ResetAgents()
        {
            // Agents are only used by one thread at a time
            CancelAgentMove();

            foreach (Agent agent in _agents)
            {
                agent.ResetToState(_state);
            }
        }

        /// <summary>
        /// Lets the acting player's agent compute its move in the background (see PollAgentMove); nothing if one is already thinking
        /// </summary>
        private void StartAgentMove(bool playSound = true)
        {
            if (_agentMove != null || _state.HasEnded) return;

            int acting = ActingPlayer;
            if (acting < 0) return;

            Agent agent = _agents[acting];
            GameState snapshot = new GameState(_state);
            CancellationTokenSource cancellation = new();

            _movingAgent = agent;
            _agentMoveStartState = new GameState(_state);
            _agentMovePlaySound = playSound;
            _agentMoveCancellation = cancellation;
            _agentMoveClock.Restart();
            _agentMove = Task.Run(() => agent.Act(snapshot, cancellation.Token));
        }

        /// <summary>
        /// Plays the background move once it is done, unless it was cancelled or the game changed meanwhile
        /// </summary>
        private void PollAgentMove()
        {
            if (_agentMove == null || !_agentMove.IsCompleted) return;

            Task<Action> move = _agentMove;
            Agent agent = _movingAgent!;
            GameState startState = _agentMoveStartState!;
            bool cancelled = _agentMoveCancellation!.IsCancellationRequested;
            bool playSound = _agentMovePlaySound;
            ClearAgentMove();

            if (move.IsFaulted)
            {
                _spectatePlaying = false;
                string message = move.Exception!.GetBaseException().Message;
                Console.WriteLine($"Agent of player {agent.PlayerIndex} failed: {move.Exception}");
                _eventLog.WriteLine(new ColoredStrEntry($"Agent of {Theme.PlayerName(agent.PlayerIndex)} failed: {message}", Color.Red));
                return;
            }

            // A move for another state or by a replaced agent is dropped
            if (cancelled || !ReferenceEquals(_agents[agent.PlayerIndex], agent) || !_state.Equals(startState))
                return;

            Action playedAction = move.Result;
            if (!playedAction.IsValidFor(_state))
            {
                _spectatePlaying = false;
                _eventLog.WriteLine(new ColoredStrEntry($"Agent of {Theme.PlayerName(agent.PlayerIndex)} returned an invalid move: {playedAction}", Color.Red));
                return;
            }

            // The search behind the move, for the heat map of the position it was made in
            if (agent is MCTSAgent mcts) KeepSearch(mcts, startState);

            ApplyMove(playedAction, playSound);
        }

        /// <summary>
        /// Stops the agent move in progress (agents stop searching early if they support it) and waits for it, so the agents
        /// are free again
        /// </summary>
        private void CancelAgentMove()
        {
            if (_agentMove == null) return;

            _agentMoveCancellation!.Cancel();
            try
            {
                _agentMove.Wait();
            }
            catch (AggregateException)
            {
                // A failing move is dropped like a cancelled one
            }
            ClearAgentMove();
        }

        private void ClearAgentMove()
        {
            _agentMoveCancellation?.Dispose();
            _agentMove = null;
            _agentMoveCancellation = null;
            _movingAgent = null;
            _agentMoveStartState = null;
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
            Board board = _state.Board;

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
        private void FindInconsistentHashes()
        {
            PlayFullAgentPlayout(_agents, false);

            while (_history.Count > 0)
            {
                int prevHash = _state.GetHashCode();
                int prevPlayerHash = HashCode.Combine(_state.Players[0], _state.Players[1], _state.Players[2], _state.Players[3]);
                int prevSettingsHash = _state.Settings.GetHashCode();
                int prevTurnHash = _state.Turn.GetHashCode();
                int prevBoardHash = _state.Board.GetHashCode();
                int prevResHash = _state.ResourceBank.GetHashCode();
                int prevDevHash = _state.ResourceBank.GetHashCode();

                UndoAction();

                RedoAction();

                int postHash = _state.GetHashCode();
                int postPlayerHash = HashCode.Combine(_state.Players[0], _state.Players[1], _state.Players[2], _state.Players[3]);
                int postSettingsHash = _state.Settings.GetHashCode();
                int postTurnHash = _state.Turn.GetHashCode();
                int postBoardHash = _state.Board.GetHashCode();
                int postResHash = _state.ResourceBank.GetHashCode();
                int postDevHash = _state.ResourceBank.GetHashCode();

                if (prevHash != postHash)
                {
                    Console.WriteLine($"{_history.Peek().Action?.GetType().Name} mismatched: pre {prevHash.ToString("X")}, post {postHash.ToString("X")}");

                    if (_redo.Count > 0)
                        Console.WriteLine("\t- Previous: " + _redo.Peek().Action?.GetType().Name);

                    if (prevPlayerHash != postPlayerHash)
                        Console.WriteLine("\t- PlayerState");
                    if (prevSettingsHash != postSettingsHash)
                        Console.WriteLine("\t- Settings");
                    if (prevTurnHash != postTurnHash)
                        Console.WriteLine("\t- TurnState");
                    if (prevBoardHash != postBoardHash)
                        Console.WriteLine("\t- Board");
                    if (prevResHash != postResHash)
                        Console.WriteLine("\t- ResourceBank");
                    if (prevDevHash != postDevHash)
                        Console.WriteLine("\t- DevelopmentBank");
                }

                UndoAction();
            }
        }

        /// <summary>
        /// Plays the game to its end at once, blocking (for FindInconsistentHashes)
        /// </summary>
        private void PlayFullAgentPlayout(Agent[] agents, bool verbose = true)
        {
            if (_state.HasEnded) return;

            ResetAgents();
            _redo.Clear();

            Clock playoutClock = new Clock();
            int moves = 0;

            while (!_state.HasEnded)
            {
                Action playedAction = agents[ActingPlayer].Act(_state);
                if (!playedAction.IsValidFor(_state)) throw new InvalidOperationException();
                ApplyMove(playedAction, false);
                moves++;
            }

            float ms = playoutClock.ElapsedTime.AsSeconds() * 1000f;

            if (verbose)
                Console.WriteLine($"Full playout of {_state.Turn.RoundCounter:n0} rounds ({moves:n0} actions) took {ms:n} ms ({ms / _state.Turn.RoundCounter} ms/round, {ms / moves} ms/action)");
        }

        private void BenchmarkPlayouts(int matches)
        {
            if (_state.HasEnded) return;

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

        #region New game and map
        /// <summary>
        /// Starts over on a new map (or a cleared board) with the current seats
        /// </summary>
        private void NewGame(bool clearBoard = false)
        {
            CancelAgentMove();
            if (clearBoard)
            {
                _state.Board.Clear();
                _state.Reset();
            }
            else
            {
                _state = new GameState(MapGenerator.GenerateRandomClassic(_centerDesert), PLAYER_COUNT);
            }

            ResetHistory();
            _renderer.Board = _state.Board;
            ResetAgents();
            AfterPositionChanged();
            _valuationHistory = [(float[])_valuation.Clone()];
            _lastHumanViewer = null;
            _sinceLastMove.Restart();
        }

        private void ResetHistory()
        {
            _history.Clear();
            _redo.Clear();
            _eventLog.Clear();
            _actionLogger.Init();
            _rollDistribution = new float[_rollDistribution.Length];
            _overlay.ClearRoll();
            _particles.Clear();
            ClearAnalysis();
        }
        #endregion

        private void UpdateValuation()
        {
            _valuation = _stateValueNet != null
                ? _stateValueNet.Run(StateVectorizer.For(_stateValueNet).Vectorize(_state))
                : SimpleAgent.StateValueFunc(_state);

            // The net rates a finished game by its position; the result itself is certain
            if (_state.HasEnded)
            {
                for (int i = 0; i < _valuation.Length; i++)
                    _valuation[i] = i == _state.Turn.PlayerIndex ? 1 : 0;
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
            foreach (Tile tile in _state.Board.Map.Values)
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

            ApplySaveFile(saveFile);

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
                ApplySaveFile(SaveFileSerializer.Deserialize(upgraded));
                _eventLog.WriteLine(new ColoredStrEntry($"Upgraded \"{Path.GetFileName(filepath)}\" to save file version {SaveFile.CurrentVersion} and loaded it", Color.Green));
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not upgrade save file \"{filepath}\": {e.Message}");
                _eventLog.WriteLine(new ColoredStrEntry($"Could not upgrade save file: {e.Message}", Color.Red));
            }
        }

        private void ApplySaveFile(SaveFile saveFile)
        {
            CancelAgentMove();
            ReplaceState(saveFile.GameState);
            _history = new(saveFile.PlayedActions.Select(action => new HistoryEntry(action, null)));
            _redo = new(saveFile.UndoHistory.Select(action => new HistoryEntry(action, null)));

            // Log and roll histogram of the loaded game instead of the previous one's
            _overlay.ClearRoll();
            _particles.Clear();
            ClearAnalysis();
            RebuildLogAndRollDistribution();

            AfterPositionChanged();
            _valuationHistory = [(float[])_valuation.Clone()];
        }

        /// <summary>
        /// Rebuilds the event log (one section per played action, as undo expects) and the roll histogram from the played actions,
        /// by reverting a copy of the state to the start and replaying them
        /// </summary>
        private void RebuildLogAndRollDistribution()
        {
            _eventLog.Clear();
            _actionLogger.Init();
            _rollDistribution = new float[_rollDistribution.Length];

            Action[] actions = ActionsSinceLastEdit().ToArray();
            foreach (Action action in actions)
            {
                if (action is RollAction rollAction && rollAction.History != null)
                    _rollDistribution[rollAction.History.RollResult.Total]++;
            }

            try
            {
                GameState replay = new GameState(_state);
                for (int i = actions.Length - 1; i >= 0; i--)
                    actions[i].Revert(replay);

                foreach (Action action in actions)
                {
                    action.Apply(replay);
                    _actionLogger.Log(action, replay);
                }
            }
            catch (Exception e)
            {
                _eventLog.Clear();
                _actionLogger.Init();
                _eventLog.WriteLine(new ColoredStrEntry($"Could not rebuild the log of the loaded game: {e.Message}", Color.Red));
            }

            _actionLogger.UpdateTracking(_state);
        }

        public void WriteSaveFile(string filepath)
        {
            // Edits change the position outside the rules, so only the moves since the last edit can be replayed from the file
            if (_history.Any(entry => entry.Edit != null))
                _eventLog.WriteLine(new ColoredStrEntry("The save keeps the edited position and the moves since the last edit", Color.Yellow));

            SaveFile saveFile = new SaveFile(_state, ActionsSinceLastEdit(), RedoActions().AsEnumerable().Reverse().ToList());
            string saveData = SaveFileSerializer.Serialize(saveFile);
            File.WriteAllText(filepath, saveData);
        }

        public void SerializeLoop()
        {
            SaveFile saveFile = new SaveFile(_state, ActionsSinceLastEdit(), RedoActions().AsEnumerable().Reverse().ToList());
            string saveData = SaveFileSerializer.Serialize(saveFile);
            ApplySaveFile(SaveFileSerializer.Deserialize(saveData));
        }
        #endregion
    }
}
