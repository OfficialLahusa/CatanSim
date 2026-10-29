using Client.UI;
using Common;
using Common.Actions;
using Agents;
using Agents.Inference;
using Agents.MCTS;
using ImGuiNET;
using SFML.Graphics;
using System.Numerics;
using Action = Common.Actions.Action;
using Color = SFML.Graphics.Color;

using Simulation;

namespace Client
{
    /// <summary>
    /// Spectate mode: what the agents think. The win chances of the value net or the search, a heat map of the acting
    /// player's policy over all its legal moves (the search's visit shares while it thinks, otherwise the net's priors), and
    /// playback controls. Edit mode reuses the win chances and has its own tool panels.
    /// </summary>
    public partial class GameScreen
    {
        private enum HeatSource { Visits, Prior, Off }
        private enum EvalSource { ValueNet, Search }

        private HeatSource _heatSource = HeatSource.Visits;
        private EvalSource _evalSource = EvalSource.ValueNet;
        private bool _showHands = true;
        private bool _showEditEval = true;

        /// <summary>
        /// A legal move of the acting player with the net's prior and the search's visit share (if there is a search)
        /// </summary>
        private sealed record MoveAnalysis(Action Move, float Prior, float? Visits, double[]? MeanValues);

        // The net's priors for the current position
        private List<(Action move, float prior)> _priors = [];
        private int _priorsVersion = -1;

        // The search for the current position: live while an agent thinks, or kept from the last decision
        private GameState? _searchState;
        private IReadOnlyList<MCTSAgent.RootMove> _searchMoves = [];
        private readonly System.Diagnostics.Stopwatch _sinceSearchRead = System.Diagnostics.Stopwatch.StartNew();
        private long _liveIterations;

        private void ClearAnalysis()
        {
            _priorsVersion = -1;
            _priors = [];
            _searchState = null;
            _searchMoves = [];
            _liveIterations = 0;
        }

        /// <summary>
        /// Keeps the search an agent made its move with, for the position it was made in (seen again after an undo)
        /// </summary>
        private void KeepSearch(MCTSAgent agent, GameState state)
        {
            (GameState? root, IReadOnlyList<MCTSAgent.RootMove> moves) = agent.GetRootMoves();
            if (root == null || !root.Equals(state)) return;
            _searchState = new GameState(state);
            _searchMoves = moves;
        }

        private void UpdateAnalysis()
        {
            bool wanted = _mode == InteractionMode.Spectate || (_mode == InteractionMode.Edit && _showEditEval);
            if (!wanted) return;

            // The net's priors once per position, for the acting player
            if (_priorsVersion != _stateVersion)
            {
                _priorsVersion = _stateVersion;
                _priors = ComputePriors();
            }

            // The running search a few times per second
            if (_sim.ThinkingAgent is MCTSAgent mcts && _sinceSearchRead.Elapsed.TotalSeconds > 0.2)
            {
                _sinceSearchRead.Restart();
                (GameState? root, IReadOnlyList<MCTSAgent.RootMove> moves) = mcts.GetRootMoves();
                if (root != null && root.Equals(State) && moves.Count > 0)
                {
                    _searchState = root;
                    _searchMoves = moves;
                    _liveIterations = moves.Sum(m => (long)m.Visits);
                }
            }
            else if (_sim.Game.ThinkingSeat == null)
            {
                _liveIterations = 0;
            }
        }

        private List<(Action move, float prior)> ComputePriors()
        {
            int acting = ActingPlayer;
            if (acting < 0 || _stateValueNet?.Policy is not PolicyHead policy) return [];

            List<Action> moves = MCTSTree.GetValidActions(State);
            if (moves.Count == 0) return [];
            if (moves.Count == 1) return [(moves[0], 1f)];

            LeafValues values = _stateValueNet.Evaluate(StateVectorizer.For(_stateValueNet).Vectorize(State), false, withPolicy: true);
            float[] priors = new float[moves.Count];
            ActionEncoding.Logits(values.PolicyQueries!.AsSpan(0, policy.QueryLength), policy.Embeddings, policy.Dim, moves, priors);
            ActionEncoding.Softmax(priors);
            return moves.Select((move, i) => (move, priors[i])).ToList();
        }

        /// <summary>
        /// The acting player's moves with priors and visit shares, the most chosen first
        /// </summary>
        private List<MoveAnalysis> Analysis()
        {
            bool hasSearch = _searchState != null && _searchState.Equals(State) && _searchMoves.Count > 0;
            if (!hasSearch)
                return _priors.Select(p => new MoveAnalysis(p.move, p.prior, null, null)).OrderByDescending(m => m.Prior).ToList();

            double total = Math.Max(1, _searchMoves.Sum(m => (double)m.Visits));
            List<MoveAnalysis> result = [];
            foreach (MCTSAgent.RootMove root in _searchMoves)
            {
                float prior = root.Prior;
                if (prior == 0)
                {
                    int i = _priors.FindIndex(p => p.move.Equals(root.Move));
                    if (i >= 0) prior = _priors[i].prior;
                }
                result.Add(new MoveAnalysis(root.Move, prior, (float)(root.Visits / total), root.MeanValues));
            }
            return result.OrderByDescending(m => m.Visits).ThenByDescending(m => m.Prior).ToList();
        }

        /// <summary>
        /// The heat map on the board: the share of each spot (moves on the same spot, like robbing different players on one
        /// tile, add up), colored on a square-root scale so small shares stay visible
        /// </summary>
        private void PrepareHeatmap()
        {
            if (_heatSource == HeatSource.Off || State.HasEnded) return;

            List<MoveAnalysis> analysis = Analysis();
            bool visits = _heatSource == HeatSource.Visits && analysis.Any(m => m.Visits != null);
            Dictionary<BoardTarget, float> shares = [];
            foreach (MoveAnalysis move in analysis)
            {
                if (TargetOf(move.Move) is not BoardTarget target) continue;
                float share = visits ? move.Visits ?? 0 : move.Prior;
                shares[target] = shares.GetValueOrDefault(target) + share;
            }
            if (shares.Count == 0) return;

            float max = Math.Max(1e-6f, shares.Values.Max());
            foreach ((BoardTarget target, float share) in shares)
            {
                if (share < 0.005f) continue;
                _overlay.Heat.Add(new BoardOverlay.HeatSpot(target, share, MathF.Sqrt(share / max)));
            }
        }

        #region Panels
        private void DrawEvalPanel()
        {
            if (_mode == InteractionMode.Edit && !_showEditEval) return;

            float x = Display.X - Margin - ColumnWidth;
            Vector2 min = new Vector2(x, Margin);
            Vector2 size = new Vector2(ColumnWidth, 96);
            float[] values = EvalValues();

            ImDrawListPtr dl = Theme.BeginHud("##eval", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size);
            Theme.DrawText(dl, Theme.Bold17, min + new Vector2(20, 12), Theme.Text, "Win chances");
            string source = _evalSource == EvalSource.Search && _searchMoves.Count > 0 && _mode == InteractionMode.Spectate ? "search" : "value net";
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 20, 16), Theme.TextMuted, $"{source}, after move {History.Count}");

            // The bar: one segment per player in seat order
            Vector2 bar = min + new Vector2(16, 44);
            float width = size.X - 32, offset = 0;
            float total = Math.Max(1e-6f, values.Sum());
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                float w = width * values[seat] / total;
                ImDrawCornerFlags corners = seat == 0 ? ImDrawCornerFlags.Left : seat == PLAYER_COUNT - 1 ? ImDrawCornerFlags.Right : ImDrawCornerFlags.None;
                dl.RoundRectFilled(bar + new Vector2(offset, 0), bar + new Vector2(offset + w, 26), Theme.U32(Theme.PlayerColor(seat)), 8, corners);
                if (w > 34) Theme.TextCentered(dl, Theme.Bold15, bar + new Vector2(offset + w / 2, 13), Theme.TextOnPlayer(seat), $"{values[seat] / total * 100:0}%");
                offset += w;
            }

            // History of the win chances, one line per player
            if (_valuationHistory.Count > 1)
            {
                Vector2 chart = min + new Vector2(16, 76);
                int count = _valuationHistory.Count;
                for (int seat = 0; seat < PLAYER_COUNT; seat++)
                {
                    Vector2[] points = new Vector2[count];
                    for (int i = 0; i < count; i++)
                        points[i] = chart + new Vector2(width * i / (count - 1), 14 - 14 * _valuationHistory[i][seat]);
                    for (int i = 1; i < count; i++)
                        dl.AddLine(points[i - 1], points[i], Theme.U32(Theme.WithAlpha(Theme.PlayerTextColor(seat), 0.8f)), 1.5f);
                }
            }
            ImGui.End();
        }

        /// <summary>
        /// Win chances per player from the selected source: the value net, or the mean values of the search's root moves
        /// weighted by their visits
        /// </summary>
        private float[] EvalValues()
        {
            if (_evalSource == EvalSource.Search && _mode == InteractionMode.Spectate && _searchState != null && _searchState.Equals(State) && _searchMoves.Count > 0)
            {
                double totalVisits = _searchMoves.Sum(m => (double)m.Visits);
                if (totalVisits > 0)
                {
                    float[] values = new float[PLAYER_COUNT];
                    foreach (MCTSAgent.RootMove move in _searchMoves)
                        for (int seat = 0; seat < PLAYER_COUNT; seat++)
                            values[seat] += (float)(move.MeanValues[seat] * move.Visits / totalVisits);
                    return values;
                }
            }
            return _valuation;
        }

        private void DrawPlaybackPanel()
        {
            float x = Display.X - Margin - ColumnWidth;
            float height = 312;
            float top = Display.Y - Margin - DiceWidget.Size - 40 - height;
            Vector2 min = new Vector2(x, top);
            Vector2 size = new Vector2(ColumnWidth, height);

            ImDrawListPtr dl = Theme.BeginHud("##playback", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size);
            Theme.DrawText(dl, Theme.Bold17, min + new Vector2(20, 14), Theme.Text, "Playback");

            float half = (size.X - 32 - 10) / 2;
            if (DrawButton(dl, "playpause", min + new Vector2(16, 46), new Vector2(half, 54), SpectatePlaying ? "Pause" : "Play", IsAdmin && !State.HasEnded, Theme.Accent, Theme.AccentBorder, Theme.Bold17, "Space"))
                ToggleSpectatePlaying();
            if (DrawButton(dl, "step", min + new Vector2(26 + half, 46), new Vector2(half, 54), "Step", IsAdmin && !State.HasEnded && _sim.Game.ThinkingSeat == null, Theme.PanelRaised, Theme.BorderLight, Theme.Bold17, "Right"))
                StepSpectate();

            // Move history: back and forward through the played moves
            Vector2 nav = min + new Vector2(16, 112);
            dl.RoundRectFilled(nav, nav + new Vector2(size.X - 32, 44), Theme.U32(Theme.PanelRaised), 12);
            dl.RoundRect(nav, nav + new Vector2(size.X - 32, 44), Theme.U32(Theme.BorderLight), 12);
            int total = History.Count + Redo.Count;
            Theme.TextCentered(dl, Theme.Bold17, nav + new Vector2((size.X - 32) / 2, 22), Theme.Text, $"Move {History.Count} / {total}");
            // One command per step: over the network the history only changes once the server's events are back
            bool control = IsAdmin;
            if (NavButton(dl, "first", nav + new Vector2(8, 4), "|<", control && History.Count > 0)) { SetSpectatePlaying(false); for (int i = History.Count; i > 0; i--) UndoAction(); }
            if (NavButton(dl, "back", nav + new Vector2(48, 4), "<", control && History.Count > 0)) { SetSpectatePlaying(false); UndoAction(); }
            if (NavButton(dl, "forward", nav + new Vector2(size.X - 32 - 84, 4), ">", control && Redo.Count > 0)) { SetSpectatePlaying(false); RedoAction(false); }
            if (NavButton(dl, "last", nav + new Vector2(size.X - 32 - 44, 4), ">|", control && Redo.Count > 0)) { SetSpectatePlaying(false); for (int i = Redo.Count; i > 0; i--) RedoAction(false); }

            // Thinking time of the agents and the pause between moves
            ImGui.PushStyleVar(ImGuiStyleVar.FramePadding, new Vector2(6, 3));
            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(20, 172), Theme.TextMuted, "Pause between moves");
            ImGui.SetCursorScreenPos(min + new Vector2(180, 168));
            ImGui.SetNextItemWidth(size.X - 200);
            float pace = Playback.PaceSeconds;
            if (ImGui.SliderFloat("##pace", ref pace, 0f, 3f, "%.1f s") && control) Send(new SetPlayback(Playback with { PaceSeconds = pace }));

            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(20, 214), Theme.TextMuted, "Heat map");
            DrawSegmented(dl, "heat", min + new Vector2(110, 206), new Vector2(size.X - 126, 30), ["Visits", "Prior", "Off"], (int)_heatSource, i => _heatSource = (HeatSource)i);
            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(20, 260), Theme.TextMuted, "Win chances");
            DrawSegmented(dl, "eval", min + new Vector2(110, 252), new Vector2(size.X - 126, 30), ["Value net", "Search"], (int)_evalSource, i => _evalSource = (EvalSource)i);
            ImGui.PopStyleVar();
            ImGui.End();
        }

        private static bool NavButton(ImDrawListPtr dl, string id, Vector2 min, string label, bool enabled)
        {
            bool clicked = Theme.HitArea(id, min, new Vector2(36, 36), out bool hovered);
            if (hovered && enabled) dl.RoundRectFilled(min, min + new Vector2(36, 36), Theme.U32(Theme.PanelActive), 8);
            Theme.TextCentered(dl, Theme.Bold17, min + new Vector2(18, 18), enabled ? Theme.TextSoft : Theme.TextFaint, label);
            return clicked && enabled;
        }

        private static void DrawSegmented(ImDrawListPtr dl, string id, Vector2 min, Vector2 size, string[] labels, int selected, Action<int> select)
        {
            dl.RoundRectFilled(min, min + size, Theme.U32(new Color(11, 22, 64)), size.Y / 2);
            dl.RoundRect(min, min + size, Theme.U32(Theme.Border), size.Y / 2);
            float w = (size.X - 6) / labels.Length;
            for (int i = 0; i < labels.Length; i++)
            {
                Vector2 a = min + new Vector2(3 + i * w, 3);
                bool clicked = Theme.HitArea($"{id}{i}", a, new Vector2(w, size.Y - 6), out bool hovered);
                if (i == selected) dl.RoundRectFilled(a, a + new Vector2(w, size.Y - 6), Theme.U32(Theme.PlayerColor(3)), (size.Y - 6) / 2);
                else if (hovered) dl.RoundRectFilled(a, a + new Vector2(w, size.Y - 6), Theme.U32(Theme.PanelRaised), (size.Y - 6) / 2);
                Theme.TextCentered(dl, Theme.Bold13, a + new Vector2(w / 2, (size.Y - 6) / 2), i == selected ? new Color(58, 38, 0) : Theme.TextMuted, labels[i]);
                if (clicked) select(i);
            }
        }

        /// <summary>
        /// Whether playback runs in Spectate mode: the agents (stand-ins for the human seats too) play on
        /// </summary>
        private bool SpectatePlaying => !Playback.Paused;

        private void SetSpectatePlaying(bool playing)
        {
            if (_mode == InteractionMode.Spectate) ApplyPlaybackForMode(playing && !State.HasEnded);
        }

        private void ToggleSpectatePlaying() => SetSpectatePlaying(!SpectatePlaying);

        /// <summary>
        /// One agent move while playback is paused: the next one from the history if there is one, otherwise a new one
        /// </summary>
        private void StepSpectate()
        {
            SetSpectatePlaying(false);
            Send(new StepCommand());
        }

        /// <summary>
        /// The acting player's policy: its most chosen moves with the net's prior and the search's visits side by side
        /// </summary>
        private void DrawPolicyPanel()
        {
            Vector2 min = new Vector2(Margin, Display.Y - Margin - 194);
            Vector2 size = new Vector2(590, 194);
            int acting = ActingPlayer;
            if (acting < 0) return;

            List<MoveAnalysis> analysis = Analysis();
            bool hasVisits = analysis.Any(m => m.Visits != null);
            ImDrawListPtr dl = Theme.BeginHud("##policy", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size, Theme.PlayerColor(acting), 16, null, 2);
            dl.AddCircleFilled(min + new Vector2(26, 28), 10, Theme.U32(Theme.PlayerColor(acting)), 24);
            Theme.DrawText(dl, Theme.Bold20, min + new Vector2(44, 15), Theme.Text, Theme.PlayerName(acting));
            Theme.DrawText(dl, Theme.Regular17, min + new Vector2(44 + Theme.TextSize(Theme.Bold20, Theme.PlayerName(acting)).X, 17), Theme.TextMuted, " (policy)");
            string info = hasVisits ? $"{analysis.Count} moves, {(long)_searchMoves.Sum(m => (double)m.Visits) / 1000.0:0.#}k visits" : $"{analysis.Count} moves, net only";
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 16, 21), Theme.TextMuted, info);

            if (_stateValueNet?.Policy == null && !hasVisits)
            {
                Theme.DrawText(dl, Theme.Regular15, min + new Vector2(24, 70), Theme.TextMuted, "The value net has no policy head.");
                ImGui.End();
                return;
            }

            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(300, 48), Theme.TextMuted, "Prior (net)");
            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(450, 48), Theme.TextMuted, hasVisits ? "Visits (search)" : "Visits (no search)");
            float maxShare = Math.Max(1e-6f, analysis.Max(m => Math.Max(m.Prior, m.Visits ?? 0)));
            int rows = Math.Min(7, analysis.Count);
            for (int i = 0; i < rows; i++)
            {
                MoveAnalysis move = analysis[i];
                float y = min.Y + 72 + i * 16;
                bool top = i == 0;
                string label = DescribeMove(move.Move);
                if (i == rows - 1 && analysis.Count > rows)
                {
                    float restPrior = analysis.Skip(rows - 1).Sum(m => m.Prior);
                    float? restVisits = hasVisits ? analysis.Skip(rows - 1).Sum(m => m.Visits ?? 0) : null;
                    move = new MoveAnalysis(move.Move, restPrior, restVisits, null);
                    label = $"{analysis.Count - rows + 1} more moves";
                }
                Theme.DrawText(dl, top ? Theme.Bold13 : Theme.Regular13, new Vector2(min.X + 24, y - 8), top ? Theme.Text : Theme.TextSoft, label);
                float pw = 100 * move.Prior / maxShare;
                dl.RoundRectFilled(new Vector2(min.X + 300, y - 5), new Vector2(min.X + 300 + pw, y + 5), Theme.U32(Theme.TextFaint), 3);
                Theme.DrawText(dl, Theme.Regular13, new Vector2(min.X + 306 + pw, y - 9), Theme.TextMuted, $"{move.Prior * 100:0}%");
                if (move.Visits is float v)
                {
                    float vw = 100 * v / maxShare;
                    dl.RoundRectFilled(new Vector2(min.X + 450, y - 5), new Vector2(min.X + 450 + vw, y + 5), Theme.U32(Theme.Heat(MathF.Sqrt(v / maxShare))), 3);
                    Theme.DrawText(dl, Theme.Regular13, new Vector2(min.X + 456 + vw, y - 9), Theme.TextSoft, $"{v * 100:0}%");
                }
            }
            ImGui.End();
        }

        /// <summary>
        /// A move in words, with board spots named by the numbers of their tiles
        /// </summary>
        private string DescribeMove(Action action)
        {
            Board board = State.Board;
            string Tiles(IEnumerable<Tile?> tiles)
            {
                List<string> names = tiles.Where(t => t != null && t.IsLandTile()).Select(t => t!.HasYield() ? t.Number.ToString() : t!.Type.ToString().ToLower()).ToList();
                return names.Count == 0 ? "coast" : string.Join(" | ", names) + (names.Count < 2 && TileCountOnEdge(tiles) < 2 ? " | coast" : "");
            }
            static int TileCountOnEdge(IEnumerable<Tile?> tiles) => tiles.Count(t => t != null && t.IsLandTile());
            string EdgeName(int edge) => Tiles([board.Adjacency.GetWestTile(board.Edges[edge]), board.Adjacency.GetEastTile(board.Edges[edge])]);
            string CornerName(int intersection) => string.Join(" | ", board.Adjacency.GetTiles(board.Intersections[intersection]).Where(t => t.IsLandTile())
                .Select(t => t.HasYield() ? t.Number.ToString() : t.Type.ToString().ToLower()).DefaultIfEmpty("coast"));
            string Card(ResourceCardType type) => type.GetName().ToLower();

            return action switch
            {
                RoadAction a => $"Road {EdgeName(a.EdgeIndex)}",
                FirstInitialRoadAction a => $"Road {EdgeName(a.EdgeIndex)}",
                SecondInitialRoadAction a => $"Road {EdgeName(a.EdgeIndex)}",
                SettlementAction a => $"Settlement {CornerName(a.IntersectionIndex)}",
                FirstInitialSettlementAction a => $"Settlement {CornerName(a.IntersectionIndex)}",
                SecondInitialSettlementAction a => $"Settlement {CornerName(a.IntersectionIndex)}",
                CityAction a => $"City {CornerName(a.IntersectionIndex)}",
                RobberAction a => $"Robber to {DescribeTile(board.Map.Values[a.TargetTileIndex])}" + (a.TargetPlayerIndex is sbyte v ? $", rob {Theme.PlayerName(v)}" : ""),
                EndTurnAction => "End turn",
                RollAction => "Roll",
                BuyDevelopmentCardAction => "Buy a development card",
                KnightAction => "Play a knight",
                RoadBuildingAction => "Play road building",
                MonopolyAction a => $"Monopoly on {Card(a.ChosenType)}",
                YearOfPlentyAction a => $"Year of plenty: {Card(a.FirstChoice)}" + (a.SecondChoice is ResourceCardType s ? $", {Card(s)}" : ""),
                FourToOneTradeAction a => $"Trade 4 {Card(a.InputType)} for {Card(a.OutputType)}",
                ThreeToOneTradeAction a => $"Trade 3 {Card(a.InputType)} for {Card(a.OutputType)}",
                TwoToOneTradeAction a => $"Trade 2 {Card(a.InputType)} for {Card(a.OutputType)}",
                DiscardAction a => "Discard " + string.Join(", ", CardTypeExtensions.ResourceTypes.Where(t => a.SelectedCards.Get(t) > 0).Select(t => $"{a.SelectedCards.Get(t)} {Card(t)}")),
                RespondToTradeAction a => a.Accept ? "Accept the offer" : "Reject the offer",
                ConfirmTradeAction a => $"Trade with {Theme.PlayerName(a.Partner)}",
                CancelTradeAction => "Withdraw the offer",
                _ => action.GetType().Name
            };
        }

        private static string DescribeTile(Tile tile) => tile.HasYield() ? $"{tile.Number} ({tile.Type.ToString().ToLower()})" : tile.Type.ToString().ToLower();

        private static string EditToolName(EditTool tool) => tool switch
        {
            EditTool.Settlement => "Settlement",
            EditTool.City => "City",
            EditTool.Road => "Road",
            EditTool.Robber => "Robber",
            _ => "Erase"
        };

        /// <summary>
        /// The edit tools: which piece a click places, undo and redo, and the map
        /// </summary>
        private void DrawEditPanel()
        {
            float x = Display.X - Margin - ColumnWidth;
            float height = 312;
            float top = Display.Y - Margin - DiceWidget.Size - 40 - height;
            Vector2 min = new Vector2(x, top);
            Vector2 size = new Vector2(ColumnWidth, height);

            ImDrawListPtr dl = Theme.BeginHud("##edittools", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size);
            Theme.DrawText(dl, Theme.Bold17, min + new Vector2(20, 14), Theme.Text, "Edit");
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 20, 18), Theme.TextMuted, $"for {Theme.PlayerName(_editPlayer)} (click a seat to switch)");

            (EditTool tool, TextureAtlas.Sprite sprite, string label)[] tools =
            [
                (EditTool.Settlement, TextureAtlas.Sprite.Settlement, "Settlement"),
                (EditTool.City, TextureAtlas.Sprite.City, "City"),
                (EditTool.Road, TextureAtlas.Sprite.Road, "Road"),
                (EditTool.Robber, TextureAtlas.Sprite.Robber, "Robber"),
                (EditTool.Erase, TextureAtlas.Sprite.Crossmark, "Erase"),
            ];
            float w = (size.X - 32 - 4 * 8) / 5;
            for (int i = 0; i < tools.Length; i++)
            {
                Vector2 pos = min + new Vector2(16 + i * (w + 8), 48);
                bool active = _editTool == tools[i].tool;
                bool clicked = Theme.HitArea($"tool{i}", pos, new Vector2(w, 92), out bool hovered);
                Theme.DrawPanel(dl, pos, pos + new Vector2(w, 92), active ? Theme.PlayerColor(_editPlayer) : Theme.BorderLight, 10,
                    active ? Theme.PanelActive : hovered ? new Color(32, 52, 112) : Theme.PanelRaised, active ? 3 : 1);
                Color iconColor = tools[i].tool switch
                {
                    EditTool.Robber => new Color(20, 20, 20),
                    EditTool.Erase => new Color(255, 96, 96),
                    _ => Theme.PlayerColor(_editPlayer)
                };
                if (tools[i].tool == EditTool.Robber) dl.AddCircleFilled(pos + new Vector2(w / 2, 34), 22, Theme.U32(new Color(220, 220, 220)), 24);
                Theme.IconCentered(dl, tools[i].sprite, pos + new Vector2(w / 2, 34), 38, iconColor);
                Theme.TextCentered(dl, Theme.Bold13, pos + new Vector2(w / 2, 74), Theme.Text, tools[i].label);
                Theme.TextCentered(dl, Theme.Regular13, pos + new Vector2(w / 2, 102), Theme.TextFaint, (i + 1).ToString());
                if (clicked) _editTool = tools[i].tool;
            }

            float half = (size.X - 32 - 10) / 2;
            if (DrawButton(dl, "undo", min + new Vector2(16, 132), new Vector2(half, 46), "Undo", History.Count > 0, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15, "Ctrl+Z"))
                UndoAction();
            if (DrawButton(dl, "redo", min + new Vector2(26 + half, 132), new Vector2(half, 46), "Redo", Redo.Count > 0, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15, "Ctrl+Y"))
                RedoAction(false);

            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(20, 196), Theme.TextMuted, "Map (starts a new game, not undoable)");
            ImGui.SetCursorScreenPos(min + new Vector2(20, 216));
            ImGui.Checkbox("Desert in the center", ref _centerDesert);
            if (DrawButton(dl, "newmap", min + new Vector2(16, 250), new Vector2(half, 44), "New map", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                NewGame();
            if (DrawButton(dl, "clearboard", min + new Vector2(26 + half, 250), new Vector2(half, 44), "Clear board", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                NewGame(clearBoard: true);
            ImGui.End();
        }

        /// <summary>
        /// The selected player's hand with buttons to give or take cards
        /// </summary>
        private void DrawEditCardsPanel()
        {
            Vector2 min = new Vector2(Margin, Display.Y - Margin - 194);
            Vector2 size = new Vector2(590, 194);
            PlayerState player = State.Players[_editPlayer];

            ImDrawListPtr dl = Theme.BeginHud("##editcards", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size, Theme.PlayerColor(_editPlayer), 16, null, 2);
            dl.AddCircleFilled(min + new Vector2(26, 28), 10, Theme.U32(Theme.PlayerColor(_editPlayer)), 24);
            Theme.DrawText(dl, Theme.Bold20, min + new Vector2(44, 15), Theme.Text, $"{Theme.PlayerName(_editPlayer)}'s hand");
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 16, 21), Theme.TextMuted, "+ takes a card from the bank, - puts it back");

            int i = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Vector2 pos = min + new Vector2(24 + i * 112, 56);
                uint count = player.ResourceCards.Get(type);
                Theme.ResourceCard(dl, type, pos, new Vector2(60, 90), true, count > 0 ? 1 : 0.35f);
                Theme.Pill(dl, pos + new Vector2(42, -6), pos + new Vector2(72, 18), new Color(11, 22, 64), Theme.Bold15, Theme.Text, count.ToString(), Color.White);
                if (DrawButton(dl, $"minus{type}", pos + new Vector2(0, 98), new Vector2(28, 28), "-", count > 0, Theme.PanelRaised, Theme.BorderLight, Theme.Bold17))
                    EditResource(_editPlayer, type, -1);
                if (DrawButton(dl, $"plus{type}", pos + new Vector2(32, 98), new Vector2(28, 28), "+", State.ResourceBank.Contains(type), Theme.PanelRaised, Theme.BorderLight, Theme.Bold17))
                    EditResource(_editPlayer, type, 1);
                i++;
            }
            ImGui.End();
        }
        #endregion
    }
}
