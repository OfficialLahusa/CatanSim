using Client.UI;
using Common;
using Common.Actions;
using ImGuiNET;
using SFML.Graphics;
using SFML.System;
using SFML.Window;
using Action = Common.Actions.Action;

namespace Client
{
    /// <summary>
    /// Human moves in Play mode. Nothing is validated here: every click picks one of the legal moves of the LegalActionProvider,
    /// so the rules live in one place.
    /// </summary>
    public partial class GameScreen
    {
        private enum BuildMode { None, Road, Settlement, City }

        private enum Dialog { None, Discard, Victim, Trade, DevelopmentCard, YearOfPlenty, Monopoly, NewGame }

        private BuildMode _buildMode;
        private Dialog _dialog;

        // Legal moves of the acting human, cached per position
        private List<Action> _legal = [];
        private int _legalVersion = -1;
        private int _legalPlayer = -1;

        // Dialog selections
        private readonly int[] _discardSelection = new int[5];
        private int _victimTile = -1;
        private ResourceCardType? _tradeGive, _tradeGet;
        private ResourceCardType? _plentyFirst, _plentySecond;

        /// <summary>
        /// The human who acts now in Play mode, if any (none while an agent move is still being computed)
        /// </summary>
        private int? HumanToAct
        {
            get
            {
                if (_mode != InteractionMode.Play || _agentMove != null) return null;
                int acting = ActingPlayer;
                return acting >= 0 && IsHuman(acting) ? acting : null;
            }
        }

        private List<Action> Legal(int player)
        {
            if (_legalVersion != _stateVersion || _legalPlayer != player)
            {
                _legal = _state.HasEnded ? [] : LegalActionProvider.GetActionsForState(_state, (sbyte)player);
                _legalVersion = _stateVersion;
                _legalPlayer = player;
            }
            return _legal;
        }

        private bool HasLegal<T>(int player) where T : Action => Legal(player).Any(a => a is T);

        private void ClearHumanInput()
        {
            _buildMode = BuildMode.None;
            if (_dialog != Dialog.NewGame) _dialog = Dialog.None;
            Array.Clear(_discardSelection);
            _victimTile = -1;
            _tradeGive = _tradeGet = null;
            _plentyFirst = _plentySecond = null;
        }

        /// <summary>
        /// Where a move happens on the board, if it is placed there
        /// </summary>
        private static BoardTarget? TargetOf(Action action) => action switch
        {
            FirstInitialSettlementAction a => new BoardTarget(TargetKind.Intersection, a.IntersectionIndex),
            SecondInitialSettlementAction a => new BoardTarget(TargetKind.Intersection, a.IntersectionIndex),
            SettlementAction a => new BoardTarget(TargetKind.Intersection, a.IntersectionIndex),
            CityAction a => new BoardTarget(TargetKind.Intersection, a.IntersectionIndex),
            FirstInitialRoadAction a => new BoardTarget(TargetKind.Edge, a.EdgeIndex),
            SecondInitialRoadAction a => new BoardTarget(TargetKind.Edge, a.EdgeIndex),
            RoadAction a => new BoardTarget(TargetKind.Edge, a.EdgeIndex),
            RobberAction a => new BoardTarget(TargetKind.Tile, a.TargetTileIndex),
            _ => null
        };

        private static TextureAtlas.Sprite SpriteOf(Action action) => action switch
        {
            CityAction => TextureAtlas.Sprite.City,
            RobberAction => TextureAtlas.Sprite.Robber,
            RoadAction or FirstInitialRoadAction or SecondInitialRoadAction => TextureAtlas.Sprite.Road,
            _ => TextureAtlas.Sprite.Settlement
        };

        private bool PlacingFreeRoads(int player) => _state.Players[player].BuildingStock.FreeRoads > 0;

        /// <summary>
        /// The board moves the human can click now: placements of the setup and the robber always, building moves in their
        /// build mode, roads also while Road Building's free roads are placed
        /// </summary>
        private IEnumerable<Action> ClickableMoves(int player)
        {
            foreach (Action action in Legal(player))
            {
                bool clickable = action switch
                {
                    FirstInitialSettlementAction or SecondInitialSettlementAction or FirstInitialRoadAction or SecondInitialRoadAction => true,
                    RobberAction => true,
                    RoadAction => _buildMode == BuildMode.Road || PlacingFreeRoads(player),
                    SettlementAction => _buildMode == BuildMode.Settlement,
                    CityAction => _buildMode == BuildMode.City,
                    _ => false
                };
                if (clickable) yield return action;
            }
        }

        /// <summary>
        /// The board spot under the mouse, if the mouse is over the board and not over a panel
        /// </summary>
        private BoardTarget? HoveredTarget(IEnumerable<BoardTarget> candidates)
        {
            if (ImGui.GetIO().WantCaptureMouse) return null;
            Vector2i mouse = Mouse.GetPosition(_window);
            if (mouse.X < 0 || mouse.Y < 0 || mouse.X >= _window.Size.X || mouse.Y >= _window.Size.Y) return null;
            Vector2f point = _window.MapPixelToCoords(mouse, _mapView);
            return _overlay.HitTest(_state.Board, point, candidates);
        }

        /// <summary>
        /// Fills the overlay for the current frame: ghosts and hover of the human's build mode, the edit preview, or the heat map
        /// </summary>
        private void PrepareOverlay()
        {
            _overlay.Ghosts.Clear();
            _overlay.Heat.Clear();
            _overlay.Hovered = null;

            switch (_mode)
            {
                case InteractionMode.Play:
                    if (HumanToAct is int player && _dialog == Dialog.None)
                    {
                        Color color = Theme.PlayerColor(player);
                        List<Action> moves = ClickableMoves(player).ToList();
                        foreach (Action move in moves)
                        {
                            BoardTarget target = TargetOf(move)!.Value;
                            if (_overlay.Ghosts.Any(g => g.Target == target)) continue;
                            _overlay.Ghosts.Add(new BoardOverlay.Ghost(target, SpriteOf(move), move is RobberAction ? new Color(25, 25, 25) : color));
                        }
                        _overlay.Hovered = HoveredTarget(_overlay.Ghosts.Select(g => g.Target));
                    }
                    break;

                case InteractionMode.Edit:
                    PrepareEditOverlay();
                    break;

                case InteractionMode.Spectate:
                    PrepareHeatmap();
                    break;
            }
        }

        /// <summary>
        /// A click on the board in Play mode: plays the move on the clicked spot, or asks for the victim of the robber
        /// </summary>
        private void HandlePlayClick()
        {
            if (HumanToAct is not int player || _dialog != Dialog.None) return;

            List<Action> moves = ClickableMoves(player).ToList();
            BoardTarget? clicked = HoveredTarget(moves.Select(m => TargetOf(m)!.Value).Distinct());
            if (clicked == null) return;

            List<Action> atTarget = moves.Where(m => TargetOf(m) == clicked).ToList();
            if (atTarget.Count == 1)
            {
                PlayHumanMove(atTarget[0]);
            }
            else if (atTarget.Count > 1 && atTarget.All(m => m is RobberAction))
            {
                // Several players to steal from on this tile
                _victimTile = clicked.Value.Index;
                _dialog = Dialog.Victim;
            }
        }

        private void PlayHumanMove(Action action)
        {
            if (!action.IsValidFor(_state)) return;
            ApplyMove(action, true);
        }

        private void ToggleBuildMode(BuildMode mode)
        {
            if (HumanToAct is not int player) return;
            bool available = mode switch
            {
                BuildMode.Road => HasLegal<RoadAction>(player),
                BuildMode.Settlement => HasLegal<SettlementAction>(player),
                BuildMode.City => HasLegal<CityAction>(player),
                _ => true
            };
            _buildMode = _buildMode == mode || !available ? BuildMode.None : mode;
        }

        private void PlayFirst<T>(int player, Func<T, bool>? match = null) where T : Action
        {
            Action? action = Legal(player).OfType<T>().FirstOrDefault(a => match == null || match(a));
            if (action != null) PlayHumanMove(action);
        }

        /// <summary>
        /// Space: rolls if the dice are due, otherwise ends the turn
        /// </summary>
        private void RollOrEndTurn()
        {
            if (HumanToAct is not int player || _dialog != Dialog.None) return;
            if (HasLegal<RollAction>(player)) PlayFirst<RollAction>(player);
            else if (HasLegal<EndTurnAction>(player)) PlayFirst<EndTurnAction>(player);
        }

        /// <summary>
        /// What the human has to do now, for the banner
        /// </summary>
        private string HumanPrompt(int player)
        {
            List<Action> legal = Legal(player);
            if (legal.Any(a => a is FirstInitialSettlementAction)) return "Place your first settlement";
            if (legal.Any(a => a is SecondInitialSettlementAction)) return "Place your second settlement";
            if (legal.Any(a => a is FirstInitialRoadAction or SecondInitialRoadAction)) return "Place a road next to it";
            if (legal.Any(a => a is DiscardAction)) return $"Discard {_state.Players[player].ResourceCards.Count() / 2} cards";
            if (legal.Any(a => a is RobberAction)) return _dialog == Dialog.Victim ? "Choose whom to rob" : "Move the robber: pick a tile";
            if (PlacingFreeRoads(player)) return $"Place {_state.Players[player].BuildingStock.FreeRoads} free road" + (_state.Players[player].BuildingStock.FreeRoads > 1 ? "s" : "");
            if (legal.Any(a => a is RollAction)) return "Roll the dice";
            return _buildMode switch
            {
                BuildMode.Road => "Build a road: pick a highlighted edge",
                BuildMode.Settlement => "Build a settlement: pick a highlighted spot",
                BuildMode.City => "Build a city: pick one of your settlements",
                _ => "Build, trade or end your turn"
            };
        }

        /// <summary>
        /// Opens the dialog the rules ask for, like discarding after a seven
        /// </summary>
        private void OpenRequiredDialog()
        {
            if (HumanToAct is not int player) return;
            if (_dialog == Dialog.None && HasLegal<DiscardAction>(player))
            {
                Array.Clear(_discardSelection);
                _dialog = Dialog.Discard;
            }
        }

        private void Window_KeyPressed(object? sender, KeyEventArgs e)
        {
            if (ImGui.GetIO().WantCaptureKeyboard) return;

            // Mode switch and screenshots work everywhere
            switch (e.Code)
            {
                case Keyboard.Key.F1: SetMode(InteractionMode.Play); return;
                case Keyboard.Key.F2: SetMode(InteractionMode.Spectate); return;
                case Keyboard.Key.F3: SetMode(InteractionMode.Edit); return;
                case Keyboard.Key.F12: SaveScreenshot(); return;
            }

            if (e.Control && e.Code == Keyboard.Key.Z && _mode != InteractionMode.Play) { UndoAction(); return; }
            if (e.Control && e.Code == Keyboard.Key.Y && _mode != InteractionMode.Play) { RedoAction(); return; }

            switch (_mode)
            {
                case InteractionMode.Play:
                    switch (e.Code)
                    {
                        case Keyboard.Key.Space: RollOrEndTurn(); break;
                        case Keyboard.Key.Num1: ToggleBuildMode(BuildMode.Road); break;
                        case Keyboard.Key.Num2: ToggleBuildMode(BuildMode.Settlement); break;
                        case Keyboard.Key.Num3: ToggleBuildMode(BuildMode.City); break;
                        case Keyboard.Key.Num4: if (HumanToAct is int p) PlayFirst<BuyDevelopmentCardAction>(p); break;
                        case Keyboard.Key.T: if (HumanToAct is int t && HasTrades(t)) _dialog = Dialog.Trade; break;
                        case Keyboard.Key.Escape: if (_dialog is not (Dialog.Discard or Dialog.Victim)) ClearHumanInput(); break;
                    }
                    break;

                case InteractionMode.Spectate:
                    switch (e.Code)
                    {
                        case Keyboard.Key.Space: ToggleSpectatePlaying(); break;
                        case Keyboard.Key.Right: StepSpectate(); break;
                        case Keyboard.Key.Left: _spectatePlaying = false; UndoAction(); break;
                    }
                    break;

                case InteractionMode.Edit:
                    switch (e.Code)
                    {
                        case Keyboard.Key.Num1: _editTool = EditTool.Settlement; break;
                        case Keyboard.Key.Num2: _editTool = EditTool.City; break;
                        case Keyboard.Key.Num3: _editTool = EditTool.Road; break;
                        case Keyboard.Key.Num4: _editTool = EditTool.Robber; break;
                        case Keyboard.Key.Num5: _editTool = EditTool.Erase; break;
                    }
                    break;
            }
        }

        private void Window_MouseButtonPressed(object? sender, MouseButtonEventArgs e)
        {
            if (e.Button == Mouse.Button.Right && !ImGui.GetIO().WantCaptureMouse)
            {
                _panStart = new Vector2i(e.X, e.Y);
                return;
            }

            if (e.Button != Mouse.Button.Left || ImGui.GetIO().WantCaptureMouse) return;

            // The dice roll for a human who has to roll
            Vector2f uiMousePos = _window.MapPixelToCoords(new Vector2i(e.X, e.Y), _uiView);
            if (_mode == InteractionMode.Play && _diceWidget.Contains(uiMousePos.X, uiMousePos.Y))
            {
                if (HumanToAct is int player && HasLegal<RollAction>(player)) PlayFirst<RollAction>(player);
                return;
            }

            switch (_mode)
            {
                case InteractionMode.Play: HandlePlayClick(); break;
                case InteractionMode.Edit: HandleEditClick(); break;
            }
        }

        private void Window_MouseButtonReleased(object? sender, MouseButtonEventArgs e)
        {
            if (e.Button == Mouse.Button.Right) _panStart = null;
        }

        private void SetMode(InteractionMode mode)
        {
            if (_mode == mode) return;
            CancelAgentMove();
            _mode = mode;
            ClearHumanInput();
            _sinceLastMove.Restart();
            if (mode == InteractionMode.Spectate) _spectatePlaying = false;
        }

        #region Trades and development cards
        private IEnumerable<Action> Trades(int player) =>
            Legal(player).Where(a => a is FourToOneTradeAction or ThreeToOneTradeAction or TwoToOneTradeAction);

        private bool HasTrades(int player) => Trades(player).Any();

        private static (ResourceCardType give, ResourceCardType get, int ratio)? TradeOf(Action action) => action switch
        {
            FourToOneTradeAction t => (t.InputType, t.OutputType, 4),
            ThreeToOneTradeAction t => (t.InputType, t.OutputType, 3),
            TwoToOneTradeAction t => (t.InputType, t.OutputType, 2),
            _ => null
        };

        /// <summary>
        /// The best rate at which the player can give a resource away now, if any
        /// </summary>
        private int? BestRatio(int player, ResourceCardType give) =>
            Trades(player).Select(TradeOf).Where(t => t!.Value.give == give).Select(t => (int?)t!.Value.ratio).Min();

        private Action? BestTrade(int player, ResourceCardType give, ResourceCardType get) =>
            Trades(player).Where(a => TradeOf(a) is var t && t!.Value.give == give && t.Value.get == get).OrderBy(a => TradeOf(a)!.Value.ratio).FirstOrDefault();

        /// <summary>
        /// The legal discard that matches the cards picked in the discard dialog
        /// </summary>
        private DiscardAction? SelectedDiscard(int player) => Legal(player).OfType<DiscardAction>().FirstOrDefault(a =>
            CardTypeExtensions.ResourceTypes.Select((type, idx) => a.SelectedCards.Get(type) == _discardSelection[idx]).All(x => x));

        /// <summary>
        /// The legal Year of Plenty move for the two cards picked in its dialog, in either order
        /// </summary>
        private YearOfPlentyAction? SelectedPlenty(int player) => Legal(player).OfType<YearOfPlentyAction>().FirstOrDefault(a =>
            (a.FirstChoice == _plentyFirst && a.SecondChoice == _plentySecond) || (a.FirstChoice == _plentySecond && a.SecondChoice == _plentyFirst)
            || (_plentySecond == null && a.FirstChoice == _plentyFirst && a.SecondChoice == null));

        private bool HasDevelopmentMoves(int player) =>
            Legal(player).Any(a => a is KnightAction or RoadBuildingAction or YearOfPlentyAction or MonopolyAction);
        #endregion
    }
}
