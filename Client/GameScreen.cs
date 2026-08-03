using Client.Logging;
using Common;
using Common.Actions;
using Common.Agents;
using Common.Serialization;
using ImGuiNET;
using Models;
using SFML.Audio;
using SFML.Graphics;
using SFML.System;
using SFML.Window;
using System.Collections.Concurrent;
using Action = Common.Actions.Action;
using Edge = Common.Edge;

namespace Client
{
    public class GameScreen : Screen
    {
        private RenderWindow _window;
        private View _mapView;
        private View _uiView;

        private GameState _state;
        private Stack<Action> _playedActions;
        private Stack<Action> _undoHistory;

        private BoardRenderer _renderer;
        private CardWidget _cardWidget;
        private DiceWidget _diceWidget;

        private EventLog _eventLog;
        private ActionLogger _actionLogger;
        private float[] _rollDistribution;
        private bool _showEvaluationBar = true;

        public static Font Font;

        // View movement
        private float _viewZoom = 1f;
        private float _viewZoomBase = 1f;
        private const float _moveSpeed = 300f;

        // FPS Counter
        private float _frameTimeSum = 0.0f;
        private int _frameTimeCount = 0;
        private float _latestAvgFrameTime = 0.0f;

        // Map generator options
        private bool _centerDesert = false;

        // Gameplay options
        private bool _spectatorMode = true;
        private bool _editMode = false;
        private int _playerIndex = 0;
        public const int PLAYER_COUNT = 4;

        // Point-and-click building
        private CircleShape _intersectionHitbox;
        private RectangleShape _edgeHitbox;
        private CircleShape _centerHitbox;

        // Sounds
        private bool _muteQuickPlayouts = false;
        private Sound _placeSound;
        private Sound _diceRollSound;

        // UI Modals
        private readonly string saveFileModalId = "save-file";
        private readonly string loadFileModalId = "load-file";

        // Player Agents
        private Agent[] _agents;
        private readonly string[] _agentTypes = ["RandomAgent", "SimpleAgent"];
        private int[] _selectedAgentTypes;
        private int _randomPlayoutCount = 1000;
        private int _randomPlayoutThreadCount = Environment.ProcessorCount / 2; // Usually optimal, since the playouts are not CPU-bound

        // TODO: Remove this once ONNX Evaluation is fully implemented
        private StateVectorizer _stateVectorizer = new StateVectorizer();

        static GameScreen()
        {
            Font = new Font(@"..\..\..\res\Open_Sans\static\OpenSans-Regular.ttf");
        }

        public GameScreen(RenderWindow window)
        {
            _window = window;

            _state = new GameState(MapGenerator.GenerateRandomClassic(), PLAYER_COUNT);
            _playedActions = [];
            _undoHistory = [];

            _renderer = new BoardRenderer(_state.Board, 120, 20);
            _cardWidget = new CardWidget(window, _state.Players[_playerIndex]);
            _diceWidget = new DiceWidget(window);
            _diceWidget.Active = true;

            _eventLog = new EventLog();
            _actionLogger = new ActionLogger(_eventLog);
            _actionLogger.Init();

            // Bucket indices are the roll totals
            // Padded for symmetric plotting with correct indices
            _rollDistribution = new float[15];

            _intersectionHitbox = new CircleShape(_renderer.SideLength / 4, 32);
            _intersectionHitbox.Origin = new Vector2f(_intersectionHitbox.Radius, _intersectionHitbox.Radius);

            _edgeHitbox = new RectangleShape(new Vector2f(_renderer.SideLength * 0.35f, _renderer.SideLength));
            _edgeHitbox.Origin = _edgeHitbox.Size / 2;

            _centerHitbox = new CircleShape(_renderer.SideLength * 0.85f, 32);
            _centerHitbox.Origin = new Vector2f(_centerHitbox.Radius, _centerHitbox.Radius);

            _placeSound = new Sound(Sounds.Place);
            _placeSound.Volume = 40f;

            _diceRollSound = new Sound(Sounds.DiceRolling);
            _diceRollSound.Volume = 50f;

            _selectedAgentTypes = new int[PLAYER_COUNT];
            _agents = new Agent[PLAYER_COUNT];

            for(sbyte i = 0; i < PLAYER_COUNT; i++)
            {
                _selectedAgentTypes[i] = 1;
                _agents[i] = new SimpleAgent(i);
            }

            _mapView = new View(ClientUtils.RoundVec2f(_renderer.GetTileCenter(3, 3)), new Vector2f(window.Size.X, window.Size.Y)); ;
            _uiView = new View(new Vector2f(0, 0), _mapView.Size);

            _window.SetView(_mapView);

            _window.MouseWheelScrolled += Window_MouseWheelScrolled;
            _window.Resized += Window_Resized;
            _window.KeyPressed += Window_KeyPressed;
            _window.MouseButtonPressed += Window_MouseButtonPressed;
        }

        public void Draw(Time deltaTime)
        {
            _window.Clear(new Color(8, 25, 75));

            // Draw map
            _window.SetView(_mapView);

            //_window.Draw(_renderer);
            _renderer.Draw(_window, RenderStates.Default, _state, (sbyte)_playerIndex);

            // Draw UI
            _window.SetView(_uiView);

            _window.Draw(_cardWidget);
            _window.Draw(_diceWidget);

            // Evaluation bar
            if(_showEvaluationBar)
            {
                float[] valuation = SimpleAgent.StateValueFunc(_state);
                var sortedValuation = valuation.Select((val, idx) => (val, idx)).OrderByDescending(x => (x.val, x.idx));

                // Draw outline
                RectangleShape valuationBar = new RectangleShape();
                valuationBar.Position = new Vector2f(_window.Size.X / 2 - 220, -_window.Size.Y / 2 + 20);
                valuationBar.OutlineColor = new Color(10, 10, 10);
                valuationBar.OutlineThickness = 6;
                valuationBar.Size = new Vector2f(200, 25);

                _window.Draw(valuationBar);

                valuationBar.OutlineThickness = 0;

                // Player bars
                foreach ((float val, int playerIdx) in sortedValuation)
                {
                    valuationBar.Size = new Vector2f(200 * val, 25);
                    valuationBar.FillColor = ColorPalette.GetPlayerColor(playerIdx);

                    _window.Draw(valuationBar);
                    valuationBar.Position += new Vector2f(200 * val, 0);
                }
            }

            // ImGui Windows
            ImGui.Begin("Menu", ImGuiWindowFlags.AlwaysAutoResize);

            ImGui.Text($"FPS: {Math.Floor(1 / _latestAvgFrameTime)}");
            ImGui.Text($"Frametime: {_latestAvgFrameTime * 1000.0f:0.000}ms");

            ImGui.Separator();

            using (new ImGuiTextColor(ColorPalette.GetPlayerColor(_playerIndex)))
                if (ImGui.InputInt("Player", ref _playerIndex))
                {
                    if (_playerIndex < 0)
                    {
                        _playerIndex = PLAYER_COUNT - 1;
                    }
                    else if (_playerIndex > PLAYER_COUNT - 1)
                    {
                        _playerIndex = 0;
                    }

                    _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

                    _eventLog.WriteLine(new SeparatorEntry());
                    _eventLog.WriteLine(new StrEntry("Switching to"), new PlayerEntry(_playerIndex));
                }

            ImGui.Checkbox("Spectator Mode", ref _spectatorMode);
            ImGui.Checkbox("Edit Mode", ref _editMode);

            ImGui.Separator();

            ImGui.Text("Map Generation:");
            ImGui.Checkbox("Center Desert", ref _centerDesert);

            if (ImGui.Button("Generate [Space]"))
            {
                RegenerateMap();
            }

            ImGui.SameLine();

            if (ImGui.Button("Clear [C]"))
            {
                ClearMap();
            }

            ImGui.Separator();

            if (ImGui.TreeNode("Save Files"))
            {
                if (ImGui.Button("Save Game"))
                {
                    ImGui.OpenPopup(saveFileModalId);
                }

                ImGui.SameLine();

                if (ImGui.Button("Load Game"))
                {
                    ImGui.OpenPopup(loadFileModalId);
                }

                bool isSaveModalOpen = true;
                if (ImGui.BeginPopupModal(saveFileModalId, ref isSaveModalOpen, ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.AlwaysAutoResize))
                {
                    string defaultFileName = string.Format("gamestate_dump_{0}.yaml", DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
                    var picker = FilePicker.GetFilePicker(this, Path.Combine(Environment.CurrentDirectory, ""), true, ".yaml", false, defaultFileName);
                    if (picker.Draw())
                    {
                        Console.WriteLine(picker.GetSaveFilePath());
                        WriteSaveFile(picker.GetSaveFilePath());
                        FilePicker.RemoveFilePicker(this);
                    }
                    ImGui.EndPopup();
                }

                bool isLoadModalOpen = true;
                if (ImGui.BeginPopupModal(loadFileModalId, ref isLoadModalOpen, ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.AlwaysAutoResize))
                {
                    var picker = FilePicker.GetFilePicker(this, Path.Combine(Environment.CurrentDirectory, ""), false, ".yaml");
                    if (picker.Draw())
                    {
                        Console.WriteLine(picker.SelectedFile);
                        ReadSaveFile(picker.SelectedFile);
                        FilePicker.RemoveFilePicker(this);
                    }
                    ImGui.EndPopup();
                }

                if (ImGui.Button("Quick Reload"))
                {
                    SerializeLoop();
                }

                ImGui.TreePop();
            }

            if (ImGui.TreeNode("Debug"))
            {
                ImGui.Text($"State Hash: {_state.GetHashCode().ToString("X")}");

                if (ImGui.Button("Find Inconsistent Hashes [O]"))
                {
                    FindInconsistentHashes();
                }

                ImGui.Checkbox("Silent Quick Playouts", ref _muteQuickPlayouts);
                ImGui.Checkbox("Show Yield Points", ref _renderer.DrawYieldPoints);
                ImGui.Checkbox("Show Shadows", ref _renderer.DrawTokenShadows);
                ImGui.Checkbox("Show Intersections", ref _renderer.DrawIntersectionMarkers);
                ImGui.Checkbox("Show Edges", ref _renderer.DrawEdgeMarkers);
                ImGui.Checkbox("Show Centers", ref _renderer.DrawCenterMarkers);

                ImGui.TreePop();
            }

            if (ImGui.TreeNode("Agents"))
            {
                for (sbyte agentIdx = 0; agentIdx < _agents.Length; agentIdx++)
                {
                    using (new ImGuiTextColor(ColorPalette.GetPlayerColor(agentIdx)))
                        ImGui.Text($"Player {agentIdx}:");

                    if (ImGui.Combo($"##Agent {agentIdx}", ref _selectedAgentTypes[agentIdx], _agentTypes, _agentTypes.Length))
                    {
                        _agents[agentIdx] = _selectedAgentTypes[agentIdx] switch
                        {
                            0 => new RandomAgent(agentIdx),
                            1 => new SimpleAgent(agentIdx),
                            _ => throw new InvalidOperationException()
                        };
                    }
                }

                ImGui.Separator();

                ImGui.InputInt("Playouts", ref _randomPlayoutCount);
                ImGui.InputInt("Threads", ref _randomPlayoutThreadCount);

                if (ImGui.Button("Simulate Random Playouts"))
                {
                    Task.Run(() => SimulateRandomRollouts(_randomPlayoutCount, _randomPlayoutThreadCount));
                }

                ImGui.TreePop();
            }

            if (ImGui.TreeNode("Analytics"))
            {
                ImGui.Checkbox("Show Evaluation Bar", ref _showEvaluationBar);
                ImGui.TextUnformatted("Roll Result Distribution:");
                ImGui.PlotHistogram(string.Empty, ref _rollDistribution[0], _rollDistribution.Length, 0, string.Empty, 0, _rollDistribution.Max(), new System.Numerics.Vector2(225, 100));

                ImGui.TreePop();
            }

            if (ImGui.TreeNode("Event Log"))
            {
                ImGui.TreePop();
                _eventLog.Draw();
            }

            ImGui.End();

            // Legal action window
            ImGui.Begin("Scoreboard", ImGuiWindowFlags.AlwaysAutoResize | ImGuiWindowFlags.AlwaysVerticalScrollbar);

            ImGui.Text($"Round: {_state.Turn.RoundCounter}");
            using (new ImGuiTextColor(ColorPalette.GetPlayerColor(_state.Turn.PlayerIndex)))
                ImGui.Text($"Player: {_state.Turn.PlayerIndex}'s Turn");
            ImGui.Text($"RoundType: {_state.Turn.TypeOfRound}");
            ImGui.Text($"MustRoll: {_state.Turn.MustRoll}");
            ImGui.Text($"MustDiscard: {_state.Turn.MustDiscard} ({_state.Turn.AwaitedPlayerDiscards.Count(x => x)} players)");
            ImGui.Text($"MustMoveRobber: {_state.Turn.MustMoveRobber}");

            ImGui.Separator();

            for(int playerIdx = 0; playerIdx < _state.Players.Length; playerIdx++)
            {
                using (new ImGuiTextColor(ColorPalette.GetPlayerColor(playerIdx)))
                    ImGui.Text($"Player {playerIdx}:");

                ImGui.SameLine();

                ImGui.Text($"{_state.Players[playerIdx].VictoryPoints.Total}/{_state.Settings.VictoryPoints} VPs");

                bool hasLongestRoad = _state.Players[playerIdx].VictoryPoints.LongestRoadPoints > 0;
                bool hasLargestArmy = _state.Players[playerIdx].VictoryPoints.LargestArmyPoints > 0;

                using (new ImGuiTextColor(Color.Yellow, hasLongestRoad))
                    ImGui.Text($"Road: {_state.Players[playerIdx].LongestRoadLength}");

                ImGui.SameLine();

                using (new ImGuiTextColor(Color.Yellow, hasLargestArmy))
                    ImGui.Text($"Knights: {_state.Players[playerIdx].PlayedKnights}");
            }

            ImGui.Separator();

            /*if(ImGui.Button("Recalculate"))
            {
                _legalActions = LegalActionProvider.GetActionsForState(_state);
            }*/

            if (ImGui.Button("Play Action [R]"))
            {
                PlayAgentAction();
            }

            ImGui.SameLine();

            if (ImGui.Button("Full Playout [E]"))
            {
                PlayFullAgentPlayout();
            }

            if (ImGui.Button("Undo Action [U]"))
            {
                UndoAction();
            }

            ImGui.SameLine();

            bool redoPossible = _undoHistory.Count > 0;
            if (!redoPossible)
                ImGui.PushStyleVar(ImGuiStyleVar.Alpha, 0.5f);
            if (ImGui.Button("Redo Action [I]") && redoPossible)
            {
                RedoAction();
            }
            if (!redoPossible)
                ImGui.PopStyleVar();

            /*ImGui.Text($"{_legalActions.Count} Legal Actions");

            foreach (Action action in _legalActions)
            {
                if (ImGui.Button(action.ToString()))
                {
                    action.Apply(_state);
                    _legalActions = LegalActionProvider.GetActionsForState(_state);
                    _renderer.Update();
                }
            }*/

            ImGui.End();

            GuiImpl.Render(_window);

            _window.Display();
        }

        public void HandleInput(Time deltaTime)
        {
            // Block if ImGui is capturing input
            if (ImGui.GetIO().WantCaptureKeyboard)
                return;

            Vector2f moveDelta = new Vector2f();
            if (Keyboard.IsKeyPressed(Keyboard.Key.W))
            {
                moveDelta.Y -= 1;
            }
            if (Keyboard.IsKeyPressed(Keyboard.Key.S))
            {
                moveDelta.Y += 1;
            }
            if (Keyboard.IsKeyPressed(Keyboard.Key.A))
            {
                moveDelta.X -= 1;
            }
            if (Keyboard.IsKeyPressed(Keyboard.Key.D))
            {
                moveDelta.X += 1;
            }
            moveDelta *= deltaTime.AsSeconds() * _moveSpeed * _viewZoom;
            _mapView.Move(moveDelta);

            if (Keyboard.IsKeyPressed(Keyboard.Key.Space))
            {
                RegenerateMap();
            }
            
            if (Keyboard.IsKeyPressed(Keyboard.Key.C))
            {
                ClearMap();
            }

            if (Keyboard.IsKeyPressed(Keyboard.Key.R))
            {
                PlayAgentAction(!_muteQuickPlayouts);
            }

            if (Keyboard.IsKeyPressed(Keyboard.Key.U))
            {
                UndoAction();
            }

            if (Keyboard.IsKeyPressed(Keyboard.Key.I))
            {
                RedoAction(!_muteQuickPlayouts);
            }

            if (Keyboard.IsKeyPressed(Keyboard.Key.E))
            {
                PlayFullAgentPlayout();
            }

            if (Keyboard.IsKeyPressed(Keyboard.Key.B))
            {
                BenchmarkPlayouts(10000);
            }

            if (Keyboard.IsKeyPressed(Keyboard.Key.O))
            {
                FindInconsistentHashes();
            }
            // Profile GameState Memory Usage
            if (Keyboard.IsKeyPressed(Keyboard.Key.Add))
            {
                long pre = GC.GetTotalMemory(true);
                Console.WriteLine("Pre: " + pre);
                List<GameState> storedStates = new();
                for (int i = 0; i < 1_000_000; i++)
                {
                    storedStates.Add(new GameState(_state));
                }
                long post = GC.GetTotalMemory(true);
                Console.WriteLine("Post: " + post);
                Console.WriteLine("Diff: " + (post - pre));
            }
            // YAML Serialization Test
            if (Keyboard.IsKeyPressed(Keyboard.Key.Subtract))
            {
                string yaml = SaveFileSerializer.Serialize(_state);

                // Write YAML to file for inspection
                string filename = string.Format("gamestate_dump_{0}.yaml", DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
                File.WriteAllText(filename, yaml);

                GameState? loadedState = SaveFileSerializer.Deserialize<GameState>(yaml);
                if (loadedState == null || !_state.Equals(loadedState))
                {
                    Console.WriteLine("YAML Serialization Test Failed!");

                    string diffFilename = string.Format("gamestate_diff_{0}.txt", DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
                    string diffYaml = SaveFileSerializer.Serialize(loadedState);
                    File.WriteAllText(diffFilename, diffYaml);
                }
                else
                {
                    Console.WriteLine("YAML Serialization Test Succeeded!");
                }
            }
            // GameState Vectorization Test
            if (Keyboard.IsKeyPressed(Keyboard.Key.Enter))
            {
                _stateVectorizer.Vectorize(_state, (uint)_playedActions.Count);
            }
        }

        public void Update(Time deltaTime)
        {
            // Update FPS counter every 0.5s. Is initialized on the first frame
            _frameTimeCount++;
            _frameTimeSum += deltaTime.AsSeconds();
            if (_frameTimeSum > 0.5f || _latestAvgFrameTime <= 0.000001f)
            {
                _latestAvgFrameTime = _frameTimeSum / _frameTimeCount;
                _frameTimeSum = 0.0f;
                _frameTimeCount = 0;
            }

            Vector2f uiMousePos = _window.MapPixelToCoords(Mouse.GetPosition(_window), _uiView);
            _cardWidget.Update(deltaTime.AsSeconds(), uiMousePos);

            _window.DispatchEvents();
            GuiImpl.Update(_window, deltaTime);
        }

        private void BenchmarkPlayouts(int matches, bool verbose = true)
        {
            if (_state.HasEnded) return;

            _undoHistory.Clear();

            Clock playoutClock = new Clock();
            int playedActions = 0;
            int playedRounds = 0;
            float ms = 0;

            for (int it = 0; it < matches; it++)
            {
                playoutClock.Restart();

                while (!_state.HasEnded)
                {
                    // Find first player that is allowed to act on state
                    int? actingPlayerIdx = null;
                    for (int playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                    {
                        if (_state.CanPlayerAct(playerIdx))
                        {
                            actingPlayerIdx = playerIdx;
                            break;
                        }
                    }

                    if (!actingPlayerIdx.HasValue) throw new InvalidOperationException();

                    // Query agent for current state
                    Action playedAction = _agents[actingPlayerIdx.Value].Act(_state);

                    if (!playedAction.IsValidFor(_state)) throw new InvalidOperationException();

                    // Apply action to state
                    playedAction.Apply(_state);

                    // Push to action stack
                    _playedActions.Push(playedAction);

                    // Track distribution of rolls
                    if (playedAction is RollAction rollAction)
                    {
                        _rollDistribution[rollAction.RollResult.Total]++;
                    }

                    // Write entry to event log
                    _actionLogger.Log(playedAction, _state);

                    playedActions++;
                }

                // Track played rounds and elapsed time
                playedRounds += _state.Turn.RoundCounter;
                ms += playoutClock.ElapsedTime.AsSeconds() * 1000f;

                // Generate new map
                _state.Board = MapGenerator.GenerateRandomClassic(_centerDesert);
                _state.Reset();

                _playedActions.Clear();

                _eventLog.Clear();
                _actionLogger.Init();

                _rollDistribution = new float[_rollDistribution.Length];
            }

            if (verbose)
            {
                Console.WriteLine($"\nFull playout of {matches:n0} matches ({playedRounds:n0} rounds, {playedActions:n0} actions) took {ms:n} ms");
                Console.WriteLine($"Avg. {ms / matches} ms/match, {ms / playedRounds} ms/round, {ms / playedActions} ms/action");
                Console.WriteLine($"Avg. {playedRounds / matches} rounds/match, {playedActions / matches} actions/match, {playedActions / playedRounds} actions/round\n");
            }

            // Update visuals
            _renderer.Board = _state.Board;
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        private void PlayFullAgentPlayout(bool verbose = true)
        {
            if (_state.HasEnded) return;

            _undoHistory.Clear();

            Clock playoutClock = new Clock();

            while (!_state.HasEnded)
            {
                // Find first player that is allowed to act on state
                int? actingPlayerIdx = null;
                for (int playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
                {
                    if (_state.CanPlayerAct(playerIdx))
                    {
                        actingPlayerIdx = playerIdx;
                        break;
                    }
                }

                if (!actingPlayerIdx.HasValue) throw new InvalidOperationException();

                // Query agent for current state
                Action playedAction = _agents[actingPlayerIdx.Value].Act(_state);

                if (!playedAction.IsValidFor(_state)) throw new InvalidOperationException();

                // Apply action to state
                playedAction.Apply(_state);

                // Push to action stack
                _playedActions.Push(playedAction);

                // Track distribution of rolls
                if (playedAction is RollAction rollAction)
                {
                    _rollDistribution[rollAction.RollResult.Total]++;
                }

                // Write entry to event log
                _actionLogger.Log(playedAction, _state);
            }

            float ms = playoutClock.ElapsedTime.AsSeconds() * 1000f;

            if (verbose)
                Console.WriteLine($"Full playout of {_state.Turn.RoundCounter:n0} rounds ({_playedActions.Count:n0} actions) took {ms:n} ms ({ms / _state.Turn.RoundCounter} ms/round, {ms / _playedActions.Count} ms/action)");

            // Update visuals
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        private void PlayAgentAction(bool playSound = true)
        {
            if (_state.HasEnded) return;

            _undoHistory.Clear();

            // Find first player that is allowed to act on state
            int? actingPlayerIdx = null;
            for (int playerIdx = 0; playerIdx < PLAYER_COUNT; playerIdx++)
            {
                if (_state.CanPlayerAct(playerIdx))
                {
                    actingPlayerIdx = playerIdx;
                    break;
                }
            }

            if (!actingPlayerIdx.HasValue) throw new InvalidOperationException();

            // Query agent for current state
            Action playedAction = _agents[actingPlayerIdx.Value].Act(_state);

            if (!playedAction.IsValidFor(_state)) throw new InvalidOperationException();

            // Apply action to state
            playedAction.Apply(_state);

            // Push to action stack
            _playedActions.Push(playedAction);

            // Track distribution of rolls
            if (playedAction is RollAction rollAction)
            {
                _rollDistribution[rollAction.RollResult.Total]++;
            }

            // Write entry to event log
            _actionLogger.Log(playedAction, _state);

            // Play associated sound
            if (playSound)
                PlaySoundForAction(playedAction);

            // Update visuals
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        private void UndoAction()
        {
            if (_playedActions.Count == 0) return;

            // Get most recent action
            Action playedAction = _playedActions.Pop();

            // Revert action
            playedAction.Revert(_state);

            // Add to undo history
            _undoHistory.Push(playedAction);

            // Remove from roll distribution tracking
            if (playedAction is RollAction rollAction)
            {
                _rollDistribution[rollAction.RollResult.Total]--;
            }

            // Remove latest section from log
            _eventLog.PopSection();
            _actionLogger.UpdateTracking(_state);

            // Update visuals
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        private void RedoAction(bool playSound = true)
        {
            if (_state.HasEnded || _undoHistory.Count == 0) return;

            // Get latest undone action
            Action playedAction = _undoHistory.Pop();

            if (!playedAction.IsValidFor(_state)) throw new InvalidOperationException();

            // Apply action to state
            playedAction.Apply(_state);

            // Push to action stack
            _playedActions.Push(playedAction);

            // Track distribution of rolls
            if (playedAction is RollAction rollAction)
            {
                _rollDistribution[rollAction.RollResult.Total]++;
            }

            // Write entry to event log
            _actionLogger.Log(playedAction, _state);

            // Play associated sound
            if (playSound)
                PlaySoundForAction(playedAction);

            // Update visuals
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        private void FindInconsistentHashes()
        {
            PlayFullAgentPlayout(false);

            while (_playedActions.Count > 0)
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
                    Console.WriteLine($"{_playedActions.Peek().GetType().Name} mismatched: pre {prevHash.ToString("X")}, post {postHash.ToString("X")}");

                    if (_undoHistory.Count > 0)
                        Console.WriteLine("\t- Previous: " + _undoHistory.Peek().GetType().Name);

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

        private async Task SimulateRandomRollouts(int playoutCount, int threadCount)
        {
            Console.WriteLine($"Simulating {playoutCount:n0} random playouts...");

            Clock playoutClock = new Clock();

            // Precompute Adjacency Cache
            _state.Board.Adjacency.PrecomputeAll();

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
                    GameState simState = new GameState(_state);
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
                        Action playedAction = _agents[actingPlayerIdx.Value].Act(simState);

                        // Apply action to state
                        playedAction.Apply(simState);
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

        private void RegenerateMap()
        {
            _state.Board = MapGenerator.GenerateRandomClassic(_centerDesert);
            _state.Reset();

            _playedActions.Clear();
            _undoHistory.Clear();

            _eventLog.Clear();
            _actionLogger.Init();

            _rollDistribution = new float[_rollDistribution.Length];

            _renderer.Board = _state.Board;
            _renderer.Update();

            _playerIndex = 0;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        private void ClearMap()
        {
            _state.Board.Clear();
            _state.Reset();

            _playedActions.Clear();
            _undoHistory.Clear();

            _eventLog.Clear();
            _actionLogger.Init();

            _rollDistribution = new float[_rollDistribution.Length];

            _renderer.Board = _state.Board;
            _renderer.Update();

            _playerIndex = 0;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
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
            _mapView.Size = new Vector2f(e.Width, e.Height);
            _uiView.Size = new Vector2f(e.Width, e.Height);
        }

        private void Window_KeyPressed(object? sender, KeyEventArgs e)
        {
            /*if (e.Code == Keyboard.Key.Enter)
            {
                if (_spectatorMode) return;

                RollDice();
            }*/
        }

        private void Window_MouseButtonPressed(object? sender, MouseButtonEventArgs e)
        {
            // Mouse pos in UI view
            Vector2f uiMousePos = _window.MapPixelToCoords(new Vector2i(e.X, e.Y), _uiView);

            // Dice widget click to roll interaction
            /*if (!_spectatorMode && e.Button == Mouse.Button.Left && _diceWidget.Contains(uiMousePos.X, uiMousePos.Y))
            {
                RollDice();
                return;
            }*/


            // >> Edit mode click interactions <<
            if (!_editMode) return;

            // Mouse pos in map view
            Vector2f mapMousePos = _window.MapPixelToCoords(new Vector2i(e.X, e.Y), _mapView);

            // Intersection clicking
            foreach (Intersection intersection in _state.Board.Intersections)
            {
                _intersectionHitbox.Position = _renderer.GetIntersectionCenter(intersection, _state.Board);

                float dist = MathF.Sqrt(
                    MathF.Pow(mapMousePos.X - _intersectionHitbox.Position.X, 2)
                    + MathF.Pow(mapMousePos.Y - _intersectionHitbox.Position.Y, 2)
                );

                // Click was on intersection
                if (e.Button == Mouse.Button.Left && dist < _intersectionHitbox.Radius)
                {
                    // Clear previous buildings
                    if(intersection.Owner != _playerIndex && intersection.Building != Intersection.BuildingType.None)
                    {
                        intersection.Building = Intersection.BuildingType.None;
                    }
                    // Cycle own buildings
                    else
                    {
                        intersection.Building = intersection.Building switch
                        {
                            Intersection.BuildingType.None => Intersection.BuildingType.Settlement,
                            Intersection.BuildingType.Settlement => Intersection.BuildingType.City,
                            Intersection.BuildingType.City => Intersection.BuildingType.None,
                            _ => throw new InvalidOperationException()
                        };

                        if (intersection.Building == Intersection.BuildingType.Settlement)
                        {
                            _eventLog.WriteLine(new PlayerEntry(_playerIndex), new StrEntry("placed a settlement"));
                        }
                        else if(intersection.Building == Intersection.BuildingType.City)
                        {
                            _eventLog.WriteLine(new PlayerEntry(_playerIndex), new StrEntry("promoted a settlement to a city"));
                        }
                    }

                    // Claim intersection for current player
                    intersection.Owner = (sbyte)_playerIndex;

                    // Play place sound
                    _placeSound.Play();

                    // Rebuild geometry
                    _renderer.Update();

                    return;
                }
            }

            // Edge clicking
            foreach (Edge edge in _state.Board.Edges)
            {
                // Transform hitbox
                (Vector2f left, Vector2f right) = _renderer.GetEdgePoints(edge);

                _edgeHitbox.Position = (left + right) / 2f;

                _edgeHitbox.Rotation = edge.Direction switch
                {
                    Direction.Edge.UpDown => 0f,
                    Direction.Edge.LeftTop => -60f,
                    Direction.Edge.RightTop => 60f,
                    _ => throw new InvalidOperationException()
                };

                // Apply inverse transform to mouse pos for local AABB alignment
                Vector2f transformedMousePos = _edgeHitbox.InverseTransform.TransformPoint(mapMousePos);

                // Click was on edge
                if (e.Button == Mouse.Button.Left && _edgeHitbox.GetLocalBounds().Contains(transformedMousePos.X, transformedMousePos.Y))
                {
                    // Clear previous roads
                    if(edge.Owner != _playerIndex && edge.Building == Edge.BuildingType.Road)
                    {
                        edge.Building = Edge.BuildingType.None;
                    }
                    // Toggle own roads
                    else
                    {
                        edge.Building = edge.Building switch
                        {
                            Edge.BuildingType.None => Edge.BuildingType.Road,
                            Edge.BuildingType.Road => Edge.BuildingType.None,
                            _ => throw new InvalidOperationException(),
                        };

                        if(edge.Building == Edge.BuildingType.Road)
                        {
                            _eventLog.WriteLine(new PlayerEntry(_playerIndex), new StrEntry("placed a road"));
                        }
                    }

                    // Claim edge for current player
                    edge.Owner = (sbyte)_playerIndex;

                    // Play place sound
                    _placeSound.Play();

                    // Rebuild geometry
                    _renderer.Update();

                    return;
                }
            }

            // Center clicking
            foreach(Tile tile in _state.Board.Map.Where(x => x.IsLandTile()))
            {
                _centerHitbox.Position = _renderer.GetTileCenter(tile);

                float dist = MathF.Sqrt(
                    MathF.Pow(mapMousePos.X - _centerHitbox.Position.X, 2)
                    + MathF.Pow(mapMousePos.Y - _centerHitbox.Position.Y, 2)
                );

                // Click was on intersection
                if (e.Button == Mouse.Button.Left && dist < _centerHitbox.Radius)
                {
                    // Move robber
                    _state.Board.Robber = tile;

                    // Play place sound
                    _placeSound.Play();

                    _eventLog.WriteLine(new PlayerEntry(_playerIndex), new StrEntry("moved the robber"));

                    return;
                }
            }
        }

        public void PlaySoundForAction(Action action)
        {
            if(action is RollAction && _diceRollSound.Status != SoundStatus.Playing)
            {
                _diceRollSound.Play();
            }
            else if((action is SettlementAction 
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

        public void ReadSaveFile(string filepath)
        {
            string saveData = File.ReadAllText(filepath);
            SaveFile saveFile = SaveFileSerializer.Deserialize(saveData);
            _state = saveFile.GameState;
            _playedActions = new(saveFile.PlayedActions);
            _undoHistory = new(saveFile.UndoHistory);

            // Update visuals
            _renderer.Board = _state.Board;
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }

        public void WriteSaveFile(string filepath)
        {
            SaveFile saveFile = new SaveFile(_state, _playedActions.Reverse().ToList(), _undoHistory.Reverse().ToList());
            string saveData = SaveFileSerializer.Serialize(saveFile);
            File.WriteAllText(filepath, saveData);
        }

        public void SerializeLoop()
        {
            SaveFile saveFile = new SaveFile(_state, _playedActions.Reverse().ToList(), _undoHistory.Reverse().ToList());
            string saveData = SaveFileSerializer.Serialize(saveFile);
            saveFile = SaveFileSerializer.Deserialize(saveData);
            _state = saveFile.GameState;
            _playedActions = new(saveFile.PlayedActions);
            _undoHistory = new(saveFile.UndoHistory);

            // Update visuals
            _renderer.Board = _state.Board;
            _renderer.Update();

            _playerIndex = _state.Turn.PlayerIndex;
            _cardWidget.SetPlayerState(_state.Players[_playerIndex]);

            _diceWidget.Active = _state.Turn.MustRoll;
            _diceWidget.RollResult = _state.Turn.LastRoll;
            _diceWidget.UpdateSprites();
        }
    }
}
