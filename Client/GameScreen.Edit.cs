using Client.Editing;
using Client.Logging;
using Client.UI;
using Common;
using SFML.Graphics;
using Edge = Common.Edge;

namespace Client
{
    /// <summary>
    /// Edit mode: pieces, the robber and hand cards change freely. Every change is an edit in the history, so it can be undone
    /// and redone like a move and shows up in the log.
    /// </summary>
    public partial class GameScreen
    {
        private enum EditTool { Settlement, City, Road, Robber, Erase }

        private EditTool _editTool = EditTool.Settlement;
        private int _editPlayer = 0;

        private IEnumerable<BoardTarget> EditCandidates()
        {
            Board board = _state.Board;
            switch (_editTool)
            {
                case EditTool.Settlement:
                case EditTool.City:
                    return Enumerable.Range(0, board.Intersections.Count).Select(i => new BoardTarget(TargetKind.Intersection, i));
                case EditTool.Road:
                    return Enumerable.Range(0, board.Edges.Count).Select(i => new BoardTarget(TargetKind.Edge, i));
                case EditTool.Robber:
                    return Enumerable.Range(0, board.Map.Length).Where(i => board.Map.Values[i].IsLandTile()).Select(i => new BoardTarget(TargetKind.Tile, i));
                default:
                    return Enumerable.Range(0, board.Intersections.Count).Where(board.HasBuilding).Select(i => new BoardTarget(TargetKind.Intersection, i))
                        .Concat(Enumerable.Range(0, board.Edges.Count).Where(i => board.GetRoad(i) != Edge.BuildingType.None).Select(i => new BoardTarget(TargetKind.Edge, i)));
            }
        }

        /// <summary>
        /// The edit tool's preview on the hovered spot: the piece it would place, or a red marker for erasing
        /// </summary>
        private void PrepareEditOverlay()
        {
            BoardTarget? hovered = HoveredTarget(EditCandidates());
            _overlay.Hovered = hovered;
            if (hovered is not BoardTarget target) return;

            Color color = Theme.PlayerColor(_editPlayer);
            switch (_editTool)
            {
                case EditTool.Settlement: _overlay.Ghosts.Add(new BoardOverlay.Ghost(target, TextureAtlas.Sprite.Settlement, color)); break;
                case EditTool.City: _overlay.Ghosts.Add(new BoardOverlay.Ghost(target, TextureAtlas.Sprite.City, color)); break;
                case EditTool.Road: _overlay.Ghosts.Add(new BoardOverlay.Ghost(target, TextureAtlas.Sprite.Road, color)); break;
                case EditTool.Robber: _overlay.Ghosts.Add(new BoardOverlay.Ghost(target, TextureAtlas.Sprite.Robber, new Color(25, 25, 25))); break;
                case EditTool.Erase: _overlay.HoverColor = new Color(255, 80, 80); break;
            }
            if (_editTool != EditTool.Erase) _overlay.HoverColor = Color.White;
        }

        private void HandleEditClick()
        {
            if (HoveredTarget(EditCandidates()) is BoardTarget target) EditAt(target);
        }

        /// <summary>
        /// What a click with the edit tool does on a spot
        /// </summary>
        private void EditAt(BoardTarget target)
        {
            GameState after = new GameState(_state);
            Board board = after.Board;
            PlayerState player = after.Players[_editPlayer];
            string name = Theme.PlayerName(_editPlayer);
            string? description = null;

            switch (_editTool)
            {
                case EditTool.Settlement:
                case EditTool.City:
                    {
                        Intersection.BuildingType wanted = _editTool == EditTool.City ? Intersection.BuildingType.City : Intersection.BuildingType.Settlement;
                        bool same = board.HasBuildingOf(target.Index, _editPlayer) && board.GetBuilding(target.Index) == wanted;
                        if (same)
                        {
                            board.SetBuilding(target.Index, Intersection.BuildingType.None, -1);
                            description = $"removed {name}'s {(wanted == Intersection.BuildingType.City ? "city" : "settlement")}";
                            break;
                        }

                        // Within the stock: a replaced own building frees its piece
                        bool ownSettlement = board.HasBuildingOf(target.Index, _editPlayer) && board.GetBuilding(target.Index) == Intersection.BuildingType.Settlement;
                        bool ownCity = board.HasBuildingOf(target.Index, _editPlayer) && board.GetBuilding(target.Index) == Intersection.BuildingType.City;
                        bool inStock = wanted == Intersection.BuildingType.City ? player.BuildingStock.RemainingCities > 0 || ownCity : player.BuildingStock.RemainingSettlements > 0 || ownSettlement;
                        if (!inStock)
                        {
                            _eventLog.WriteLine(new ColoredStrEntry($"{name} has no {(wanted == Intersection.BuildingType.City ? "cities" : "settlements")} left", Color.Yellow));
                            return;
                        }

                        board.SetBuilding(target.Index, wanted, (sbyte)_editPlayer);
                        description = $"placed a {(wanted == Intersection.BuildingType.City ? "city" : "settlement")} for {name}";
                        break;
                    }

                case EditTool.Road:
                    if (board.HasRoadOf(target.Index, _editPlayer))
                    {
                        board.SetRoad(target.Index, Edge.BuildingType.None, -1);
                        description = $"removed {name}'s road";
                    }
                    else
                    {
                        if (player.BuildingStock.RemainingRoads == 0)
                        {
                            _eventLog.WriteLine(new ColoredStrEntry($"{name} has no roads left", Color.Yellow));
                            return;
                        }
                        board.SetRoad(target.Index, Edge.BuildingType.Road, (sbyte)_editPlayer);
                        description = $"placed a road for {name}";
                    }
                    break;

                case EditTool.Robber:
                    if (board.GetRobberTileIndex() == target.Index) return;
                    board.Robber = board.Map.Values[target.Index];
                    description = "moved the robber";
                    break;

                case EditTool.Erase:
                    if (target.Kind == TargetKind.Intersection)
                    {
                        description = $"removed {Theme.PlayerName(board.GetBuildingOwner(target.Index))}'s {(board.GetBuilding(target.Index) == Intersection.BuildingType.City ? "city" : "settlement")}";
                        board.SetBuilding(target.Index, Intersection.BuildingType.None, -1);
                    }
                    else
                    {
                        description = $"removed {Theme.PlayerName(board.GetRoadOwner(target.Index))}'s road";
                        board.SetRoad(target.Index, Edge.BuildingType.None, -1);
                    }
                    break;
            }

            if (description == null) return;
            BoardEdit.RecomputeFromBoard(after);
            ApplyEdit(new BoardEdit(description, _editPlayer, _state, after));
            _placeSound.Play();
        }

        /// <summary>
        /// Gives a player a card from the bank or puts one back
        /// </summary>
        private void EditResource(int playerIdx, ResourceCardType type, int delta)
        {
            GameState after = new GameState(_state);
            PlayerState player = after.Players[playerIdx];
            string name = Theme.PlayerName(playerIdx);
            if (delta > 0)
            {
                if (!after.ResourceBank.Contains(type)) return;
                after.ResourceBank.Remove(type, 1);
                player.ResourceCards.Add(type, 1);
                ApplyEdit(new BoardEdit($"gave {name} a {type.GetName().ToLower()} card", playerIdx, _state, after));
            }
            else
            {
                if (!player.ResourceCards.Contains(type)) return;
                player.ResourceCards.Remove(type, 1);
                after.ResourceBank.Add(type, 1);
                ApplyEdit(new BoardEdit($"took a {type.GetName().ToLower()} card from {name}", playerIdx, _state, after));
            }
        }

        /// <summary>
        /// Makes an edit the current position and puts it into the history and the log
        /// </summary>
        private void ApplyEdit(BoardEdit edit, bool keepRedo = false)
        {
            ReplaceState(edit.After);
            _history.Push(new HistoryEntry(null, edit));
            if (!keepRedo) _redo.Clear();

            _eventLog.WriteLine(new SentenceEntry([("Edit: ", Theme.TextMuted), (char.ToUpper(edit.Description[0]) + edit.Description[1..], Theme.TextSoft)]));
            _eventLog.PushSection();

            _overlay.ClearRoll();
            AfterPositionChanged();
            _valuationHistory.Add((float[])_valuation.Clone());
        }
    }
}
