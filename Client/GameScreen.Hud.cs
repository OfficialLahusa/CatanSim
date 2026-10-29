using Client.Logging;
using Client.UI;
using Common;
using Common.Actions;
using Agents;
using Agents.MCTS;
using Common.Serialization;
using ImGuiNET;
using SFML.Graphics;
using SFML.System;
using System.Numerics;
using Action = Common.Actions.Action;
using Color = SFML.Graphics.Color;

using Simulation;

namespace Client
{
    /// <summary>
    /// The HUD around the board: seat panels in turn order on the left, the banner on top, log and mode-specific controls on the
    /// right, the hand (Play), the policy (Spectate) or the card editor (Edit) at the bottom left, dialogs over the board
    /// </summary>
    public partial class GameScreen
    {
        private const float Margin = 20, ColumnWidth = 380, SeatHeight = 112, SeatGap = 12, SeatTop = 92;

        // Where cards fly to: the card icon of each seat panel and the stacks of the own hand, from the last frame
        private readonly Vector2[] _seatCardAnchor = new Vector2[PLAYER_COUNT];
        private readonly Dictionary<ResourceCardType, Vector2> _handStackCenter = [];
        private readonly Dictionary<DevelopmentCardType, Vector2> _handDevelopmentStackCenter = [];
        private bool _rollChartPinned;
        private readonly Dictionary<ResourceCardType, Vector2> _bankSlots = [];
        private Vector2 _bankDevelopmentSlot;
        private Vector2 _handAnchor;

        // Hover animation of the hand's card stacks
        private readonly float[] _stackHover = new float[CardSet<ResourceCardType>.Values.Count + CardSet<DevelopmentCardType>.Values.Count];

        private bool _openSaveModal, _openLoadModal, _endScreenDismissed;
        private readonly SeatKind[] _newGameSeats = new SeatKind[PLAYER_COUNT];

        private Vector2 Display => ImGui.GetIO().DisplaySize;

        private void DrawHud()
        {
            // The menu covers the board; loading a game from it still needs the file dialog
            if (_menuOpen)
            {
                DrawMenu();
                DrawModals();
                return;
            }

            EventLog.Viewer = Viewer;
            OpenRequiredDialog();

            DrawTopBar();
            DrawSeatPanels();
            DrawBanner();

            switch (_mode)
            {
                case InteractionMode.Play:
                    DrawLogPanel(Margin, Display.Y - 532);
                    DrawBank(Display.Y - 520);
                    DrawActionPanel();
                    DrawHandPanel();
                    break;
                case InteractionMode.Spectate:
                    DrawEvalPanel();
                    DrawLogPanel(116, Display.Y - 556);
                    DrawBank(Display.Y - 544);
                    DrawPlaybackPanel();
                    DrawPolicyPanel();
                    break;
                case InteractionMode.Edit:
                    DrawEvalPanel();
                    DrawLogPanel(116, Display.Y - 556);
                    DrawBank(Display.Y - 544);
                    DrawEditPanel();
                    DrawEditCardsPanel();
                    break;
            }

            DrawRollLabel();
            DrawRollChart();
            DrawGhostTooltip();
            DrawDialogs();
            DrawEndScreen();
            DrawModals();
            if (_showDeveloperWindow) DrawDeveloperWindow();
            DrawNetworkWindow();
            DrawAdminWindow();

            _particles.Draw(ImGui.GetForegroundDrawList());
        }

        #region Top bar
        private void DrawTopBar()
        {
            ImDrawListPtr dl = Theme.BeginHud("##topbar", new Vector2(Margin, Margin), new Vector2(860, 50));
            Vector2 o = new Vector2(Margin, Margin);

            // Menu button
            bool menuClicked = Theme.HitArea("menu", o, new Vector2(46, 46), out bool menuHovered);
            Theme.DrawPanel(dl, o, o + new Vector2(46, 46), menuHovered ? Theme.BorderLight : Theme.Border, 12);
            for (int i = 0; i < 3; i++)
                dl.AddLine(o + new Vector2(12, 13 + 10 * i), o + new Vector2(34, 13 + 10 * i), Theme.U32(new Color(223, 231, 255)), 3.5f);
            if (menuClicked) ImGui.OpenPopup("main-menu");

            // Mode switch
            Vector2 s = o + new Vector2(58, 2);
            dl.RoundRectFilled(s, s + new Vector2(270, 42), Theme.U32(Theme.Panel), 21);
            dl.RoundRect(s, s + new Vector2(270, 42), Theme.U32(Theme.Border), 21);
            string[] labels = ["Play", "Spectate", "Edit"];
            for (int i = 0; i < 3; i++)
            {
                Vector2 min = s + new Vector2(4 + 88 * i, 4);
                bool clicked = Theme.HitArea($"mode{i}", min, new Vector2(86, 34), out bool hovered);
                bool active = (int)_mode == i;
                if (active) dl.RoundRectFilled(min, min + new Vector2(86, 34), Theme.U32(Theme.PlayerColor(3)), 17);
                else if (hovered) dl.RoundRectFilled(min, min + new Vector2(86, 34), Theme.U32(Theme.PanelRaised), 17);
                Theme.TextCentered(dl, Theme.Bold15, min + new Vector2(43, 17), active ? new Color(58, 38, 0) : Theme.TextMuted, labels[i]);
                if (clicked) SetMode((InteractionMode)i);
            }

            // A switch for the mode, then where the match runs
            float x = 340;
            if (_mode == InteractionMode.Spectate)
                x += 12 + DrawToggle(dl, "hands", o + new Vector2(x, 7), "Hands shown", ref _showHands);
            else if (_mode == InteractionMode.Edit)
                x += 12 + DrawToggle(dl, "evalbar", o + new Vector2(x, 7), "Win chances", ref _showEditEval);
            DrawNetworkChip(dl, o + new Vector2(x, 7));

            DrawMainMenu();
            ImGui.End();
        }

        private float DrawToggle(ImDrawListPtr dl, string id, Vector2 min, string label, ref bool value)
        {
            // The label starts at 46 and keeps some room before the border
            float width = 54 + Theme.TextSize(Theme.Regular15, label).X;
            if (Theme.HitArea(id, min, new Vector2(width, 32), out _)) value = !value;
            dl.RoundRectFilled(min, min + new Vector2(width, 32), Theme.U32(Theme.Panel), 16);
            dl.RoundRect(min, min + new Vector2(width, 32), Theme.U32(Theme.Border), 16);
            Vector2 track = min + new Vector2(9, 8);
            dl.RoundRectFilled(track, track + new Vector2(30, 16), Theme.U32(value ? new Color(138, 92, 0) : new Color(58, 74, 116)), 8);
            dl.AddCircleFilled(track + new Vector2(value ? 22 : 8, 8), 6, Theme.U32(value ? Theme.PlayerColor(3) : Theme.TextMuted));
            Theme.DrawText(dl, Theme.Regular15, min + new Vector2(46, 6), value ? Theme.TextSoft : Theme.TextMuted, label);
            return width;
        }

        private void DrawMainMenu()
        {
            if (!ImGui.BeginPopup("main-menu")) return;

            if (ImGui.MenuItem("Main menu", "Esc")) OpenMenu();
            ImGui.Separator();
            if (ImGui.MenuItem("New game...", "", false, IsAdmin))
            {
                Array.Copy(Seats, _newGameSeats, PLAYER_COUNT);
                _dialog = Dialog.NewGame;
            }
            if (ImGui.MenuItem("Save game...")) _openSaveModal = true;
            if (ImGui.MenuItem("Load game...", "", false, IsAdmin)) _openLoadModal = true;
            ImGui.Separator();
            if (ImGui.MenuItem("Host or join a game...")) _showNetworkWindow = true;
            if (ImGui.MenuItem("Admin panel...", "", _showAdminWindow, CanAdminister)) _showAdminWindow = !_showAdminWindow;
            ImGui.Separator();
            if (ImGui.MenuItem("Developer tools", "", _showDeveloperWindow)) _showDeveloperWindow = !_showDeveloperWindow;
            if (ImGui.MenuItem("Settings...")) OpenMenu(MenuPage.Settings);
            ImGui.Separator();
            if (ImGui.MenuItem("Quit")) _window.Close();
            ImGui.EndPopup();
        }
        #endregion

        #region Seat panels
        private void DrawSeatPanels()
        {
            int acting = ActingPlayer;
            int? viewer = Viewer;
            float height = SeatTop + PLAYER_COUNT * (SeatHeight + SeatGap);
            ImDrawListPtr dl = Theme.BeginHud("##seats", new Vector2(Margin, SeatTop - 8), new Vector2(ColumnWidth + 16, height));

            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                Vector2 min = new Vector2(Margin, SeatTop + seat * (SeatHeight + SeatGap));
                Vector2 max = min + new Vector2(ColumnWidth, SeatHeight);
                PlayerState player = State.Players[seat];
                Color color = Theme.PlayerColor(seat);
                bool isActing = seat == acting;
                bool selected = _mode == InteractionMode.Edit && seat == _editPlayer;

                bool clicked = Theme.HitArea($"seat{seat}", min, max - min, out bool hovered);
                if (clicked && _mode == InteractionMode.Edit) _editPlayer = seat;

                if (isActing) Theme.Glow(dl, min, max, color, Theme.Rounding, 1.2f);
                Color border = isActing ? color : selected ? Color.White : hovered && _mode == InteractionMode.Edit ? Theme.BorderLight : Theme.Border;
                Theme.DrawPanel(dl, min, max, border, Theme.Rounding, isActing ? Theme.PanelRaised : null, isActing || selected ? 2.5f : 1f);
                // The color tab: the panel's own shape cut to its left edge, so its curves are the border's
                dl.PushClipRect(min - new Vector2(4), new Vector2(min.X + 9, max.Y + 4), true);
                dl.RoundRectFilled(min, max, Theme.U32(color), Theme.Rounding);
                dl.PopClipRect();

                // Avatar with the seat number
                Vector2 avatar = min + new Vector2(48, 40);
                dl.AddCircleFilled(avatar, 20, Theme.U32(color), 32);
                dl.AddCircle(avatar, 20, Theme.U32(Color.White), 32, 2);
                Theme.TextCentered(dl, Theme.Bold20, avatar, Theme.TextOnPlayer(seat), (seat + 1).ToString());

                // Name, who plays it and what it is doing
                string name = viewer == seat ? "You" : Theme.PlayerName(seat);
                Theme.DrawText(dl, Theme.Bold22, min + new Vector2(80, 10), Theme.Text, name);
                string who = viewer == seat ? $"{Theme.PlayerName(seat)}, human" : Simulation.Seats.Name(Seats[seat]);
                if (_mode == InteractionMode.Spectate && IsHuman(seat)) who = "Human (stand-in agent)";
                Theme.DrawText(dl, Theme.Regular15, min + new Vector2(80, 41), Theme.TextMuted, who);

                string? tag = SeatTag(seat, isActing);
                if (tag != null)
                {
                    float x = 80 + Theme.TextSize(Theme.Bold22, name).X + 10;
                    float w = Theme.TextSize(Theme.Bold13, tag).X + 20;
                    // Centered on the name's capitals, not on its line box
                    Theme.Pill(dl, min + new Vector2(x, 10), min + new Vector2(x + w, 34), color, Theme.Bold13, Theme.TextOnPlayer(seat), tag);
                }

                // Victory points: the hidden ones only where they may be seen
                bool hiddenVisible = viewer == null || viewer == seat;
                int vp = player.VictoryPoints.Total - (hiddenVisible ? 0 : player.VictoryPoints.DevelopmentCardPoints);
                Theme.TextRight(dl, Theme.ExtraBold36, max + new Vector2(-24, -SeatHeight + 4), Theme.Text, vp.ToString());
                Theme.TextRight(dl, Theme.Regular13, max + new Vector2(-24, -SeatHeight + 50), Theme.TextMuted, $"/ {State.Settings.VictoryPoints} VP");

                DrawSeatStats(dl, seat, min + new Vector2(28, 78), viewer);

                if (_mode != InteractionMode.Play && WinChances() is float[] chances)
                {
                    float share = chances[seat];
                    Theme.TextRight(dl, Theme.Bold15, max + new Vector2(-16, -26), Theme.PlayerTextColor(seat), $"{share * 100:0} %");
                }

                // Turn order: from one seat to the next
                if (seat < PLAYER_COUNT - 1)
                {
                    Vector2 a = new Vector2(min.X + ColumnWidth / 2, max.Y + 2);
                    dl.AddTriangleFilled(a + new Vector2(-8, 0), a + new Vector2(8, 0), a + new Vector2(0, 8), Theme.U32(Theme.TextFaint));
                }
            }

            ImGui.End();
        }

        private string? SeatTag(int seat, bool isActing)
        {
            if (State.HasEnded) return seat == State.Turn.PlayerIndex ? "WINNER" : null;
            if (!isActing) return null;
            if (State.Turn.MustDiscard) return "DISCARDING";
            if (_sim.Game.ThinkingSeat == seat) return "THINKING";
            return "ON TURN";
        }

        private void DrawSeatStats(ImDrawListPtr dl, int seat, Vector2 pos, int? viewer)
        {
            PlayerState player = State.Players[seat];
            bool handVisible = _mode switch
            {
                InteractionMode.Play => viewer == null || viewer == seat,
                InteractionMode.Spectate => _showHands,
                _ => true
            };

            // Hand: mini cards when visible, otherwise a card back with the count
            uint cards = player.ResourceCards.Count();
            if (handVisible && cards > 0)
            {
                int i = 0;
                // Up to 6 cards fanned 9 apart, more squeezed into the same width, so the count after them stays clear of the
                // development cards
                float step = cards > 6 ? 45f / (cards - 1) : 9;
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                {
                    for (uint n = 0; n < player.ResourceCards.Get(type); n++, i++)
                        Theme.ResourceCard(dl, type, pos + new Vector2(i * step, 1), new Vector2(15, 22), false);
                }
                Theme.DrawText(dl, Theme.Regular15, pos + new Vector2((cards - 1) * step + 20, 2), Theme.TextSoft, cards.ToString());
            }
            else
            {
                Theme.CardBack(dl, pos + new Vector2(0, 2), new Vector2(15, 21));
                Theme.CardBack(dl, pos + new Vector2(6, 0), new Vector2(15, 21));
                Theme.DrawText(dl, Theme.Regular15, pos + new Vector2(28, 2), Theme.TextSoft, cards.ToString());
            }
            _seatCardAnchor[seat] = pos + new Vector2(14, 12);

            // Development cards, knights and road length; the holders of the bonuses in yellow
            uint devCards = player.DevelopmentCards.Count();
            float x = 92;
            Theme.Card(dl, ColorPalette.DevelopmentCardPurple, ColorPalette.DevelopmentCardDarkPurple, TextureAtlas.Sprite.Development, pos + new Vector2(x, 0), new Vector2(17, 24), true, 1);
            Theme.DrawText(dl, Theme.Regular15, pos + new Vector2(x + 24, 2), Theme.TextSoft, devCards.ToString());

            x += 56;
            bool army = player.VictoryPoints.LargestArmyPoints > 0;
            Theme.Icon(dl, TextureAtlas.Sprite.Knight, pos + new Vector2(x, -1), pos + new Vector2(x + 26, 25), army ? Theme.Highlight : Theme.TextSoft);
            Theme.DrawText(dl, army ? Theme.Bold15 : Theme.Regular15, pos + new Vector2(x + 30, 2), army ? Theme.Highlight : Theme.TextSoft, player.PlayedKnights.ToString());

            x += 60;
            bool road = player.VictoryPoints.LongestRoadPoints > 0;
            Theme.Icon(dl, TextureAtlas.Sprite.Road, pos + new Vector2(x, -1), pos + new Vector2(x + 26, 25), road ? Theme.Highlight : Theme.TextSoft);
            Theme.DrawText(dl, road ? Theme.Bold15 : Theme.Regular15, pos + new Vector2(x + 30, 2), road ? Theme.Highlight : Theme.TextSoft, player.LongestRoadLength.ToString());
        }
        #endregion

        #region Bank
        /// <summary>
        /// The bank in the right column under the log (above the build menu in Play mode): each resource and the development card deck with how many cards are left, low counts in
        /// red; payments to the bank and trades with it fly into these slots
        /// </summary>
        private void DrawBank(float top)
        {
            // Cards as large as the build menu's costs
            const float card = 15;
            int slots = CardTypeExtensions.ResourceTypes.Count() + 1;
            float labelWidth = Theme.TextSize(Theme.SemiBold15, "Bank").X + 14;
            Vector2 size = new Vector2(ColumnWidth, 40);
            float slot = (size.X - 22 - labelWidth) / slots;
            Vector2 min = new Vector2(Display.X - Margin - ColumnWidth, top);

            ImDrawListPtr dl = Theme.BeginHud("##bank", min, size);
            Theme.DrawPanel(dl, min, min + size, Theme.Border, 10, new Color(15, 33, 86, 235));
            float midY = min.Y + size.Y / 2;
            Theme.DrawText(dl, Theme.SemiBold15, new Vector2(min.X + 14, midY - Theme.TextSize(Theme.SemiBold15, "Bank").Y / 2), Theme.TextMuted, "Bank");

            float x = min.X + 14 + labelWidth;
            Vector2 cardSize = new Vector2(card, card * 4 / 3);
            void Slot(FlyingCard what, uint count, string name, out Vector2 center)
            {
                Vector2 cardMin = new Vector2(x, midY - cardSize.Y / 2);
                center = cardMin + cardSize / 2;
                Theme.HitArea($"bank{name}", new Vector2(x - 4, min.Y), new Vector2(slot - 4, size.Y), out bool hovered);
                what.Draw(dl, cardMin, cardSize, count == 0 ? 0.35f : 1f);
                Color color = count == 0 ? Theme.TextFaint : count <= 3 ? new Color(255, 110, 110) : Theme.Text;
                Theme.DrawText(dl, Theme.Bold15, new Vector2(x + card + 7, midY - Theme.TextSize(Theme.Bold15, count.ToString()).Y / 2), color, count.ToString());
                if (hovered) DrawTooltip(ImGui.GetForegroundDrawList(), new Vector2(x + card / 2, min.Y - 6), $"{count} {name} left in the bank");
                x += slot;
            }

            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Slot(type, State.ResourceBank.Get(type), type.GetName().ToLower(), out Vector2 center);
                _bankSlots[type] = center;
            }
            Slot(FlyingCard.HiddenDevelopment, State.DevelopmentBank.Count(), "development cards", out _bankDevelopmentSlot);
            ImGui.End();
        }
        #endregion

        #region Banner
        private void DrawBanner()
        {
            int acting = ActingPlayer;
            string headline, detail = "";
            int colorSeat = Math.Max(0, acting);
            string? chip = null;

            if (State.HasEnded)
            {
                colorSeat = State.Turn.PlayerIndex;
                headline = $"{Theme.PlayerName(colorSeat)} won the game";
                detail = $" with {State.Players[colorSeat].VictoryPoints.Total} victory points";
            }
            else if (_mode == InteractionMode.Edit)
            {
                colorSeat = _editPlayer;
                headline = "Edit mode";
                detail = $" | {EditToolName(_editTool)} for {Theme.PlayerName(_editPlayer)} (Ctrl+Z undoes)";
            }
            else if (_mode == InteractionMode.Play && _curtainFor is int next)
            {
                colorSeat = next;
                headline = $"{Theme.PlayerName(next)}'s turn";
                detail = " | Pass the screen, then show the hand";
            }
            else if (HumanToAct is int human)
            {
                headline = HumanCount() > 1 ? $"{Theme.PlayerName(human)}'s turn" : "Your turn";
                detail = " | " + HumanPrompt(human);
                if (_buildMode != BuildMode.None) chip = "Esc";
            }
            else if (_sim.Game.ThinkingSeat is int thinkingSeat)
            {
                headline = $"{Theme.PlayerName(thinkingSeat)} is thinking";
                string agent = Simulation.Seats.Name(Seats[thinkingSeat] == SeatKind.Human ? SeatKind.MctsMedium : Seats[thinkingSeat]);
                long iterations = CurrentSearch is SearchEvent search && search.Seat == thinkingSeat ? search.Iterations : 0;
                detail = _mode == InteractionMode.Spectate && iterations > 0 ? $" ({agent}, {iterations / 1000.0:0.#}k iterations)" : $" ({agent})";
                chip = $"{_thinkingClock.Elapsed.TotalSeconds:0.0} s";
            }
            else if (_mode == InteractionMode.Spectate && !SpectatePlaying)
            {
                headline = "Paused";
                detail = acting >= 0 ? $", {Theme.PlayerName(acting)} to move" : "";
            }
            else
            {
                headline = acting >= 0 ? $"{Theme.PlayerName(acting)}'s turn" : "";
            }

            Vector2 headSize = Theme.TextSize(Theme.Bold20, headline);
            Vector2 detailSize = Theme.TextSize(Theme.Regular17, detail);
            float chipWidth = chip != null ? Theme.TextSize(Theme.Regular13, chip).X + 22 : 0;
            float width = 70 + headSize.X + detailSize.X + (chip != null ? chipWidth + 14 : 0);
            Vector2 min = new Vector2(Display.X / 2 - width / 2, 8);
            Vector2 size = new Vector2(width, 50);

            ImDrawListPtr dl = Theme.BeginHud("##banner", min - new Vector2(10, 4), size + new Vector2(20, 16));
            Color color = Theme.PlayerColor(colorSeat);
            Theme.DrawPanel(dl, min, min + size, color, 25, null, 2.5f);
            Vector2 dot = min + new Vector2(30, 25);
            dl.AddCircleFilled(dot, 11, Theme.U32(color), 24);
            dl.AddCircle(dot, 11, Theme.U32(Color.White), 24, 2);
            if (_sim.Game.ThinkingSeat != null)
            {
                // A spinner around the dot while an agent thinks
                float start = (float)(_thinkingClock.Elapsed.TotalSeconds * 6 % (2 * Math.PI));
                dl.PathArcTo(dot, 17, start, start + 2.2f, 20);
                dl.PathStroke(Theme.U32(color), false, 3);
            }
            Theme.DrawText(dl, Theme.Bold20, min + new Vector2(54, 25 - headSize.Y / 2), Theme.Text, headline);
            Theme.DrawText(dl, Theme.Regular17, min + new Vector2(54 + headSize.X, 25 - detailSize.Y / 2), Theme.TextMuted, detail);
            if (chip != null)
            {
                Vector2 c = new Vector2(min.X + width - 16 - chipWidth, min.Y + 12);
                dl.RoundRectFilled(c, c + new Vector2(chipWidth, 26), Theme.U32(new Color(34, 51, 106)), 6);
                dl.RoundRect(c, c + new Vector2(chipWidth, 26), Theme.U32(Theme.BorderLight), 6);
                Theme.TextCentered(dl, Theme.Regular13, c + new Vector2(chipWidth / 2, 13), Theme.TextSoft, chip);
            }
            ImGui.End();
        }

        #endregion

        #region Log
        private void DrawLogPanel(float top, float bottom)
        {
            float x = Display.X - Margin - ColumnWidth;
            bottom = Math.Max(top + 160, bottom);
            Vector2 min = new Vector2(x, top);
            Vector2 size = new Vector2(ColumnWidth, bottom - top);

            ImDrawListPtr dl = Theme.BeginHud("##log", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size);
            Theme.DrawText(dl, Theme.Bold17, min + new Vector2(20, 14), Theme.Text, "Game log");
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 20, 18), Theme.TextMuted, $"Round {State.Turn.RoundCounter}");
            dl.AddLine(min + new Vector2(16, 46), min + new Vector2(size.X - 16, 46), Theme.U32(Theme.Border));

            ImGui.SetCursorScreenPos(min + new Vector2(16, 54));
            ImGui.PushStyleColor(ImGuiCol.ChildBg, Vector4.Zero);
            ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(4, 5));
            _eventLog.Draw(size - new Vector2(24, 64), false);
            ImGui.PopStyleVar();
            ImGui.PopStyleColor();
            ImGui.End();
        }
        #endregion

        #region Play: actions and hand
        private void DrawActionPanel()
        {
            float x = Display.X - Margin - ColumnWidth;
            float diceTop = Display.Y - Margin - DiceWidget.Size;
            float endTop = diceTop - 40 - 64;
            float tradeTop = endTop - 12 - 48;
            float buildTop = tradeTop - 12 - 132;
            int? human = HumanToAct;
            int player = human ?? (Viewer is int viewing && viewing >= 0 ? viewing : 0);

            ImDrawListPtr dl = Theme.BeginHud("##actions", new Vector2(x - 8, buildTop - 26), new Vector2(ColumnWidth + 16, endTop + 72 - (buildTop - 26)));
            Theme.TextCentered(dl, Theme.Regular13, new Vector2(x + ColumnWidth / 2, buildTop - 12), Theme.TextFaint, "Build (keys 1-4)");

            (string label, TextureAtlas.Sprite sprite, ResourceCardType[] cost, bool available, bool active, System.Action onClick)[] buttons =
            [
                ("Road", TextureAtlas.Sprite.Road, [ResourceCardType.Lumber, ResourceCardType.Brick],
                    human is int a && HasLegal<RoadAction>(a) && !PlacingFreeRoads(a), _buildMode == BuildMode.Road, () => ToggleBuildMode(BuildMode.Road)),
                ("Settlement", TextureAtlas.Sprite.Settlement, [ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain],
                    human is int b && HasLegal<SettlementAction>(b), _buildMode == BuildMode.Settlement, () => ToggleBuildMode(BuildMode.Settlement)),
                ("City", TextureAtlas.Sprite.City, [ResourceCardType.Grain, ResourceCardType.Grain, ResourceCardType.Ore, ResourceCardType.Ore, ResourceCardType.Ore],
                    human is int c && HasLegal<CityAction>(c), _buildMode == BuildMode.City, () => ToggleBuildMode(BuildMode.City)),
                ("Dev card", TextureAtlas.Sprite.Development, [ResourceCardType.Wool, ResourceCardType.Grain, ResourceCardType.Ore],
                    human is int d && HasLegal<BuyDevelopmentCardAction>(d), false, () => { if (HumanToAct is int p) PlayFirst<BuyDevelopmentCardAction>(p); }),
            ];

            float w = (ColumnWidth - 3 * 9) / 4;
            for (int i = 0; i < buttons.Length; i++)
            {
                var button = buttons[i];
                Vector2 min = new Vector2(x + i * (w + 9), buildTop);
                Vector2 size = new Vector2(w, 132);
                bool clicked = Theme.HitArea($"build{i}", min, size, out bool hovered);
                float alpha = button.available || button.active ? 1f : 0.42f;
                Color fill = button.active ? Theme.PanelActive : hovered && button.available ? new Color(32, 52, 112) : Theme.PanelRaised;
                Color dimmed = Theme.Lerp(fill, new Color(15, 33, 86), 0.6f);
                Theme.DrawPanel(dl, min, min + size, button.active ? Theme.PlayerColor(player) : alpha < 1 ? Theme.Border : Theme.BorderLight, 12, alpha < 1 ? dimmed : fill, button.active ? 3 : 1);
                Color iconColor = Theme.WithAlpha(i == 3 ? new Color(200, 160, 230) : Theme.PlayerColor(player), alpha);
                Theme.IconCentered(dl, button.sprite, min + new Vector2(w / 2, 34), 42, iconColor);
                Theme.TextCentered(dl, Theme.Bold15, min + new Vector2(w / 2, 82), Theme.WithAlpha(Theme.Text, alpha), button.label);
                Theme.CostRow(dl, button.cost, min + new Vector2(w / 2, 108), 15, alpha);
                if (clicked && (button.available || button.active)) button.onClick();
            }

            // Trade, development cards, end turn
            bool canTrade = human is int t && (HasTrades(t) || CanOffer(t));
            bool canPlayDev = human is int dv && HasDevelopmentMoves(dv);
            bool canEnd = human is int e && HasLegal<EndTurnAction>(e);
            bool mustRoll = human is int r && HasLegal<RollAction>(r);
            float half = (ColumnWidth - 10) / 2;
            if (DrawButton(dl, "trade", new Vector2(x, tradeTop), new Vector2(half, 48), "Trade", canTrade, Theme.PanelRaised, Theme.BorderLight, Theme.Bold17))
                OpenTradeDialog(human!.Value);
            if (DrawButton(dl, "devcard", new Vector2(x + half + 10, tradeTop), new Vector2(half, 48), "Play dev card", canPlayDev, Theme.PanelRaised, Theme.BorderLight, Theme.Bold17))
                _dialog = Dialog.DevelopmentCard;

            string mainLabel = mustRoll ? "Roll dice" : "End turn";
            if (DrawButton(dl, "endturn", new Vector2(x, endTop), new Vector2(ColumnWidth, 64), mainLabel, canEnd || mustRoll, Theme.Accent, Theme.AccentBorder, Theme.Bold22, "Space"))
                RollOrEndTurn();
            ImGui.End();
        }

        /// <summary>
        /// A button drawn like the rest of the HUD; whether it was clicked
        /// </summary>
        private static bool DrawButton(ImDrawListPtr dl, string id, Vector2 min, Vector2 size, string label, bool enabled, Color fill, Color border, ImFontPtr font, string? key = null)
        {
            bool clicked = Theme.HitArea(id, min, size, out bool hovered);
            float alpha = enabled ? 1 : 0.42f;
            Color f = hovered && enabled ? Theme.Lerp(fill, Color.White, 0.08f) : fill;
            // Disabled: dimmed but opaque, so the board doesn't show through
            Theme.DrawPanel(dl, min, min + size, enabled ? border : Theme.Lerp(border, Theme.Panel, 0.6f), 12, enabled ? f : Theme.Lerp(f, new Color(15, 33, 86), 0.6f));
            Theme.TextCentered(dl, font, min + size / 2, Theme.WithAlpha(Theme.Text, alpha), label);
            if (key != null)
            {
                float kw = Theme.TextSize(Theme.Regular13, key).X + 18;
                Vector2 k = new Vector2(min.X + size.X - kw - 14, min.Y + size.Y / 2 - 12);
                dl.RoundRectFilled(k, k + new Vector2(kw, 24), Theme.U32(Theme.WithAlpha(new Color(30, 70, 173), alpha)), 6);
                Theme.TextCentered(dl, Theme.Regular13, k + new Vector2(kw / 2, 12), Theme.WithAlpha(new Color(207, 220, 255), alpha), key);
            }
            return clicked && enabled;
        }

        private void DrawRollLabel()
        {
            if (State.Turn.RoundCounter == 0 && !History.Any(h => h.Action is RollAction)) return;
            if (State.Turn.MustRoll && !History.Any(h => h.Action is RollAction)) return;

            int roller = History.Select(h => h.Action).OfType<RollAction>().LastOrDefault()?.PlayerIndex ?? -1;
            if (roller < 0) return;
            string who = Viewer == roller ? "You" : Theme.PlayerName(roller);
            // Centered over the dice, which are centered in the column
            float x = Display.X - Margin - ColumnWidth / 2;
            Vector2 center = new Vector2(x, Display.Y - Margin - DiceWidget.Size - 18);

            ImDrawListPtr dl = Theme.BeginHud("##rolllabel", center - new Vector2(140, 14), new Vector2(280, 28));
            string text = $"{who} rolled ";
            string number = State.Turn.LastRoll.Total.ToString();
            float tw = Theme.TextSize(Theme.Regular17, text).X, nw = Theme.TextSize(Theme.Bold17, number).X;
            Theme.DrawText(dl, Theme.Regular17, center - new Vector2((tw + nw) / 2, 11), new Color(223, 231, 255), text);
            Theme.DrawText(dl, Theme.Bold17, center - new Vector2((tw + nw) / 2 - tw, 11), Theme.Highlight, number);
            ImGui.End();
        }

        /// <summary>
        /// The rolls so far as one bar per total with a line at the count expected after as many rolls, above the dice while
        /// they or the roll label are hovered, or until the dice are clicked again after a click that didn't roll pinned it
        /// </summary>
        private void DrawRollChart()
        {
            float diceTop = Display.Y - Margin - DiceWidget.Size;
            Vector2 area = new Vector2(Display.X - Margin - ColumnWidth / 2 - DiceWidget.Size - DiceWidget.Gap / 2, diceTop - 32);
            Vector2 mouse = ImGui.GetIO().MousePos;
            bool hovered = mouse.X >= area.X && mouse.Y >= area.Y && mouse.X <= Display.X - Margin - ColumnWidth / 2 + DiceWidget.Size + DiceWidget.Gap / 2
                && mouse.Y <= Display.Y - Margin;
            if (!hovered && !_rollChartPinned) return;

            int rolls = (int)_rollDistribution.Sum();
            Vector2 size = new Vector2(ColumnWidth, 220);
            Vector2 min = new Vector2(Display.X - Margin - ColumnWidth, area.Y - 10 - size.Y);
            ImDrawListPtr dl = ImGui.GetForegroundDrawList();
            Theme.DrawPanel(dl, min, min + size, Theme.BorderLight, 12, new Color(15, 33, 86));
            Theme.DrawText(dl, Theme.Bold15, min + new Vector2(16, 12), Theme.Text, "Rolls");
            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(22 + Theme.TextSize(Theme.Bold15, "Rolls").X, 14), Theme.TextMuted, $"({rolls} so far)");

            // Legend for the expected counts
            float legendX = min.X + size.X - 16 - Theme.TextSize(Theme.Regular13, "expected").X;
            Theme.DrawText(dl, Theme.Regular13, new Vector2(legendX, min.Y + 14), Theme.TextMuted, "expected");
            dl.AddLine(new Vector2(legendX - 26, min.Y + 23), new Vector2(legendX - 6, min.Y + 23), Theme.U32(Theme.WithAlpha(Color.White, 0.75f)), 2);

            float left = min.X + 16, baseline = min.Y + size.Y - 34, top = min.Y + 60;
            float slot = (size.X - 32) / 11;
            if (rolls == 0) Theme.TextCentered(dl, Theme.Regular15, new Vector2(min.X + size.X / 2, (top + baseline) / 2), Theme.TextMuted, "No rolls yet");

            static float Expected(int total, int rolls) => rolls * (6 - Math.Abs(total - 7)) / 36f;
            float scale = Math.Max(1, Enumerable.Range(2, 11).Max(t => Math.Max(_rollDistribution[t], Expected(t, rolls))));
            Color red = new Color(232, 64, 64);
            for (int total = 2; total <= 12; total++)
            {
                float x = left + (total - 2) * slot;
                int count = (int)_rollDistribution[total];
                // The 7 in gray: those rolls moved the robber instead of paying out
                Color color = total == 7 ? new Color(150, 158, 180) : total is 6 or 8 ? red : new Color(88, 140, 255);
                float height = (baseline - top) * count / scale;
                float expected = MathF.Round(baseline - (baseline - top) * Expected(total, rolls) / scale);
                if (count > 0) dl.RoundRectFilled(new Vector2(x + 6, baseline - height), new Vector2(x + slot - 6, baseline), Theme.U32(color), 3, ImDrawCornerFlags.Top);
                if (rolls > 0) dl.AddLine(new Vector2(x + 2, expected), new Vector2(x + slot - 2, expected), Theme.U32(Theme.WithAlpha(Color.White, 0.75f)), 2);
                // The count above the bar, and above the expected line where that is close to the bar's top
                float labelY = baseline - height - 10;
                if (rolls > 0 && labelY + 8 > expected - 1) labelY = Math.Min(labelY, expected - 10);
                if (count > 0) Theme.TextCentered(dl, Theme.Bold13, new Vector2(x + slot / 2, labelY), Theme.Text, count.ToString());
                Theme.TextCentered(dl, Theme.Bold15, new Vector2(x + slot / 2, baseline + 16), total is 6 or 8 ? red : Theme.TextSoft, total.ToString());
            }
            dl.AddLine(new Vector2(left, baseline + 0.5f), new Vector2(left + 11 * slot, baseline + 0.5f), Theme.U32(Theme.BorderLight), 1);
        }

        /// <summary>
        /// Where a card flies to: the stack in the own hand when its owner is the viewer, the seat panel otherwise
        /// </summary>
        private Vector2 CardPosition(int player, FlyingCard card)
        {
            if (_mode != InteractionMode.Play || Viewer != player) return _seatCardAnchor[player];
            if (card.Development is DevelopmentCardType d && _handDevelopmentStackCenter.TryGetValue(d, out Vector2 devStack)) return devStack;
            if (card.Resource is ResourceCardType r && _handStackCenter.TryGetValue(r, out Vector2 stack)) return stack;
            return _handAnchor;
        }

        private void DrawHandPanel()
        {
            int? viewer = Viewer;
            Vector2 min = new Vector2(Margin, Display.Y - Margin - 194);
            Vector2 size = new Vector2(590, 194);
            _handAnchor = min + new Vector2(120, 110);
            if (_curtainFor is int next)
            {
                DrawCurtain(min, size, next);
                return;
            }
            if (viewer is not int seat || seat < 0) return;

            PlayerState player = State.Players[seat];

            // Stacks as in the client's card widget: each further card of a type behind it, to the left. The panel grows for
            // big hands up to the right column, beyond that the cards in a stack move closer together
            Vector2 cardSize = new Vector2(80, 120);
            const float gap = 12;
            uint[] counts = [.. CardTypeExtensions.ResourceTypes.Select(t => player.ResourceCards.Get(t)),
                .. CardSet<DevelopmentCardType>.Values.Select(t => player.DevelopmentCards.Get(t))];
            bool anyDev = player.DevelopmentCards.Count() > 0;
            int stacks = counts.Count(c => c > 0);
            float extra = counts.Where(c => c > 0).Sum(c => c - 1f);
            float fixedWidth = 32 + stacks * (cardSize.X + gap) + (anyDev ? 0 : 88);
            float maxWidth = Math.Max(size.X, Display.X - 3 * Margin - ColumnWidth - 12);
            float shift = extra == 0 ? 16 : Math.Clamp((maxWidth - fixedWidth) / extra, 4, 16);
            size.X = Math.Clamp(fixedWidth + extra * shift, size.X, maxWidth);

            ImDrawListPtr dl = Theme.BeginHud("##hand", min - new Vector2(0, 30), size + new Vector2(0, 38));
            Theme.DrawPanel(dl, min, min + size, Theme.PlayerColor(seat), 16, null, 2);
            dl.AddCircleFilled(min + new Vector2(26, 28), 10, Theme.U32(Theme.PlayerColor(seat)), 24);
            dl.AddCircle(min + new Vector2(26, 28), 10, Theme.U32(Color.White), 24, 2);
            Theme.DrawText(dl, Theme.Bold20, min + new Vector2(44, 15), Theme.Text, "Your hand");
            Theme.DrawText(dl, Theme.Regular17, min + new Vector2(44 + Theme.TextSize(Theme.Bold20, "Your hand").X, 17), Theme.TextMuted, player.ResourceCards.Count() == 1 ? " (1 card)" : $" ({player.ResourceCards.Count()} cards)");
            BuildingStock stock = player.BuildingStock;
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 16, 21), Theme.TextMuted,
                $"{stock.RemainingRoads} roads, {stock.RemainingSettlements} settlements, {stock.RemainingCities} cities left");

            float x = min.X + 16;
            float top = min.Y + 56;
            float dt = ImGui.GetIO().DeltaTime;
            _handStackCenter.Clear();
            _handDevelopmentStackCenter.Clear();

            int stackIdx = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                uint count = player.ResourceCards.Get(type);
                int hoverIdx = stackIdx++;
                if (count == 0) { _stackHover[hoverIdx] = 0; continue; }

                float width = cardSize.X + (count - 1) * shift;
                Theme.HitArea($"stack{type}", new Vector2(x, top - 10), new Vector2(width, cardSize.Y + 10), out bool hovered);
                _stackHover[hoverIdx] = Math.Clamp(_stackHover[hoverIdx] + (hovered ? 5f : -5f) * dt, 0, 1);
                float lift = Easing.ExpOut(_stackHover[hoverIdx]) * 12;

                for (uint i = 0; i < count; i++)
                    Theme.ResourceCard(dl, type, new Vector2(x + i * shift, top - lift), cardSize, i == count - 1);
                if (hovered) DrawTooltip(dl, new Vector2(x + width / 2, top - lift - 16), count > 1 ? $"{count} {type.GetName()}" : type.GetName());

                _handStackCenter[type] = new Vector2(x + width - cardSize.X / 2, top + cardSize.Y / 2);
                x += width + gap;
            }

            // Development cards: the ones bought this turn (also counted in DevelopmentCards) are muted, they can't be played
            // yet; a click on a playable one plays it as in the development card dialog
            foreach (DevelopmentCardType type in CardSet<DevelopmentCardType>.Values)
            {
                uint count = player.DevelopmentCards.Get(type), fresh = player.NewDevelopmentCards.Get(type);
                uint playable = count - fresh;
                int hoverIdx = stackIdx++;
                if (count == 0) continue;

                bool legal = HumanToAct == seat && CanPlayDevelopmentCard(seat, type);
                float width = cardSize.X + (count - 1) * shift;
                bool clicked = Theme.HitArea($"dev{type}", new Vector2(x, top - 10), new Vector2(width, cardSize.Y + 10), out bool hovered);
                _stackHover[hoverIdx] = Math.Clamp(_stackHover[hoverIdx] + (hovered ? 5f : -5f) * dt, 0, 1);
                float lift = Easing.ExpOut(_stackHover[hoverIdx]) * 12;
                Vector2 front = new Vector2(x + (count - 1) * shift, top - lift);
                if (hovered && legal) Theme.Glow(dl, front, front + cardSize, Color.White, 6, 1);
                for (uint i = 0; i < count; i++)
                    Theme.DevelopmentCard(dl, type, new Vector2(x + i * shift, top - lift), cardSize, i == count - 1, 1, i >= playable);
                if (hovered)
                {
                    string[] notes = [.. new[] { fresh > 0 ? $"{fresh} new" : null, legal ? "click to play" : null }.OfType<string>()];
                    DrawTooltip(dl, new Vector2(x + width / 2, top - lift - 16), notes.Length > 0 ? $"{type.GetName()} ({string.Join(", ", notes)})" : type.GetName());
                }
                if (clicked && legal) PlayDevelopmentCard(seat, type);
                _handDevelopmentStackCenter[type] = front + cardSize / 2;
                x += width + gap;
            }

            if (!anyDev && x + 88 < min.X + size.X)
            {
                Vector2 slot = new Vector2(x + 8, top);
                dl.RoundRect(slot, slot + cardSize, Theme.U32(new Color(91, 74, 143)), 2, ImDrawCornerFlags.All, 2);
                Theme.Card(dl, ColorPalette.DevelopmentCardPurple, ColorPalette.DevelopmentCardDarkPurple, TextureAtlas.Sprite.Development, slot + new Vector2(24, 22), new Vector2(32, 44), true, 0.5f);
                Theme.TextCentered(dl, Theme.Regular13, slot + new Vector2(40, 84), new Color(159, 143, 208), "No dev");
                Theme.TextCentered(dl, Theme.Regular13, slot + new Vector2(40, 101), new Color(159, 143, 208), "cards");
            }
            ImGui.End();
        }

        /// <summary>
        /// Instead of a hand while the screen goes to the next human: nothing hidden is shown until they confirm
        /// </summary>
        private void DrawCurtain(Vector2 min, Vector2 size, int next)
        {
            ImDrawListPtr dl = Theme.BeginHud("##curtain", min, size + new Vector2(0, 8));
            Theme.DrawPanel(dl, min, min + size, Theme.PlayerColor(next), 16, null, 2);
            Theme.DrawText(dl, Theme.Bold22, min + new Vector2(24, 22), Theme.Text, $"Pass the screen to {Theme.PlayerName(next)}");
            Theme.DrawText(dl, Theme.Regular15, min + new Vector2(24, 56), Theme.TextMuted, "The hands stay hidden until the next player is ready.");
            if (DrawButton(dl, "curtain", min + new Vector2(24, 100), new Vector2(260, 56), $"I'm {Theme.PlayerName(next)}, show my hand", true, Theme.Accent, Theme.AccentBorder, Theme.Bold15))
                _curtainFor = null;
            ImGui.End();
        }

        private static void DrawTooltip(ImDrawListPtr dl, Vector2 bottomCenter, string text)
        {
            Vector2 size = Theme.TextSize(Theme.SemiBold15, text) + new Vector2(18, 8);
            Vector2 min = bottomCenter - new Vector2(size.X / 2, size.Y);
            dl.RoundRectFilled(min, min + size, Theme.U32(new Color(11, 22, 64, 240)), 6);
            dl.RoundRect(min, min + size, Theme.U32(Theme.BorderLight), 6);
            Theme.TextCentered(dl, Theme.SemiBold15, min + size / 2, Theme.Text, text);
        }
        #endregion

        /// <summary>
        /// Next to the hovered ghost: what a click does there and what it costs
        /// </summary>
        private void DrawGhostTooltip()
        {
            if (_mode != InteractionMode.Play || HumanToAct is not int player || _dialog != Dialog.None || _overlay.Hovered is not BoardTarget hovered) return;
            List<Action> moves = ClickableMoves(player).Where(m => TargetOf(m) == hovered).ToList();
            if (moves.Count == 0) return;

            Action move = moves[0];
            bool free = PlacingFreeRoads(player);
            (string title, ResourceCardType[] cost) = move switch
            {
                RoadAction when free => ("Place a free road", Array.Empty<ResourceCardType>()),
                RoadAction => ("Build a road here", new[] { ResourceCardType.Lumber, ResourceCardType.Brick }),
                SettlementAction => ("Build a settlement here", new[] { ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain }),
                CityAction => ("Upgrade to a city", new[] { ResourceCardType.Grain, ResourceCardType.Grain, ResourceCardType.Ore, ResourceCardType.Ore, ResourceCardType.Ore }),
                RobberAction => ("Move the robber here", Array.Empty<ResourceCardType>()),
                FirstInitialSettlementAction or SecondInitialSettlementAction => ("Place your settlement here", Array.Empty<ResourceCardType>()),
                _ => ("Place your road here", Array.Empty<ResourceCardType>())
            };
            string detail = move is RobberAction
                ? moves.All(m => ((RobberAction)m).TargetPlayerIndex == null) ? "nobody to rob" : "rob " + string.Join(" or ", moves.Select(m => ((RobberAction)m).TargetPlayerIndex).OfType<sbyte>().Select(v => Theme.PlayerName(v)))
                : cost.Length > 0 ? "click to pay" : "click";

            Vector2f mapPos = _overlay.Position(State.Board, hovered);
            Vector2i pixel = _window.MapCoordsToPixel(mapPos, _mapView);
            Vector2 anchor = new Vector2(pixel.X, pixel.Y) / GuiImpl.UiScale;
            float costWidth = cost.Length * 19;
            float width = Math.Max(Theme.TextSize(Theme.Bold15, title).X, Theme.TextSize(Theme.Regular13, detail).X + (cost.Length > 0 ? costWidth + 6 : 0)) + 28;
            Vector2 min = anchor + new Vector2(22, -48);
            if (min.X + width > Display.X - Margin - ColumnWidth - 8) min.X = anchor.X - 22 - width;

            ImDrawListPtr dl = ImGui.GetForegroundDrawList();
            Theme.DrawPanel(dl, min, min + new Vector2(width, 52), Theme.BorderLight, 8, new Color(15, 33, 86, 245));
            Theme.DrawText(dl, Theme.Bold15, min + new Vector2(14, 6), Theme.Text, title);
            Theme.DrawText(dl, Theme.Regular13, min + new Vector2(14, 28), Theme.TextMuted, detail);
            if (cost.Length > 0)
            {
                float x = min.X + 14 + Theme.TextSize(Theme.Regular13, detail).X + 6 + costWidth / 2;
                Theme.CostRow(dl, cost, new Vector2(x, min.Y + 37), 16);
            }
        }

        #region Dialogs
        private void DrawDialogs()
        {
            // An open offer comes before anything else: nobody moves until it is answered and closed
            if (State.Turn.PendingTrade is TradeOffer offer && _mode != InteractionMode.Edit)
            {
                DrawPendingTrade(offer);
                return;
            }

            switch (_dialog)
            {
                case Dialog.Discard: DrawDiscardDialog(); break;
                case Dialog.Victim: DrawVictimDialog(); break;
                case Dialog.Trade: DrawTradeDialog(); break;
                case Dialog.Offer: DrawOfferDialog(); break;
                case Dialog.DevelopmentCard: DrawDevelopmentDialog(); break;
                case Dialog.YearOfPlenty: DrawPlentyDialog(); break;
                case Dialog.Monopoly: DrawMonopolyDialog(); break;
                case Dialog.NewGame: DrawNewGameDialog(); break;
            }
        }

        /// <summary>
        /// A dialog panel centered over the board; returns the draw list and its top left corner
        /// </summary>
        private ImDrawListPtr BeginDialog(string id, Vector2 size, string title, string? subtitle, int? colorSeat, out Vector2 min)
        {
            min = new Vector2(Display.X / 2 - size.X / 2, Display.Y / 2 - size.Y / 2 - 20);
            ImDrawListPtr dl = Theme.BeginHud(id, min - new Vector2(16), size + new Vector2(32));
            // Stay in front of the other HUD windows, but not of a combo's list opened in the dialog
            if (!ImGui.IsPopupOpen("", ImGuiPopupFlags.AnyPopupId)) ImGui.SetWindowFocus();
            Color border = colorSeat is int s ? Theme.PlayerColor(s) : Theme.BorderLight;
            Theme.Glow(dl, min, min + size, border, 16, 0.8f);
            Theme.DrawPanel(dl, min, min + size, border, 16, new Color(15, 33, 86, 250), 2);
            Theme.DrawText(dl, Theme.Bold22, min + new Vector2(24, 18), Theme.Text, title);
            if (subtitle != null) Theme.DrawText(dl, Theme.Regular15, min + new Vector2(24, 50), Theme.TextMuted, subtitle);
            return dl;
        }

        /// <summary>
        /// A resource card that can be clicked; whether it was
        /// </summary>
        private static bool CardButton(ImDrawListPtr dl, string id, ResourceCardType type, Vector2 min, Vector2 size, bool enabled, bool selected)
        {
            bool clicked = Theme.HitArea(id, min, size, out bool hovered);
            float lift = hovered && enabled ? 4 : 0;
            if (selected) Theme.Glow(dl, min - new Vector2(0, lift), min + size - new Vector2(0, lift), Color.White, 4, 1.5f);
            Theme.ResourceCard(dl, type, min - new Vector2(0, lift), size, true, enabled ? 1f : 0.3f);
            if (selected) dl.RoundRect(min - new Vector2(2, 2 + lift), min + size + new Vector2(2, 2 - lift), Theme.U32(Color.White), 3, ImDrawCornerFlags.All, 3);
            return clicked && enabled;
        }

        private void DrawDiscardDialog()
        {
            if (HumanToAct is not int player || !HasLegal<DiscardAction>(player)) { _dialog = Dialog.None; return; }
            PlayerState state = State.Players[player];
            int required = (int)(state.ResourceCards.Count() / 2);
            int chosen = _discardSelection.Sum();

            ImDrawListPtr dl = BeginDialog("##discard", new Vector2(560, 330), $"Discard {required} cards", "A seven was rolled. Click cards to discard them, click a minus to take one back.", player, out Vector2 min);
            int i = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Vector2 pos = min + new Vector2(32 + i * 102, 92);
                int have = (int)state.ResourceCards.Get(type);
                int selected = _discardSelection[i];
                if (CardButton(dl, $"discard{type}", type, pos, new Vector2(80, 120), have - selected > 0 && chosen < required, false))
                    _discardSelection[i]++;
                Theme.TextCentered(dl, Theme.Regular15, pos + new Vector2(40, 136), Theme.TextMuted, $"{have - selected} left");
                if (selected > 0)
                {
                    Theme.Pill(dl, pos + new Vector2(14, 152), pos + new Vector2(66, 176), Theme.PlayerColor(player), Theme.Bold15, Theme.TextOnPlayer(player), $"-{selected}");
                    if (Theme.HitArea($"undiscard{type}", pos + new Vector2(14, 152), new Vector2(52, 24), out _)) _discardSelection[i]--;
                }
                i++;
            }

            bool ready = chosen == required;
            if (DrawButton(dl, "discardok", min + new Vector2(560 - 24 - 180, 330 - 24 - 48), new Vector2(180, 48), ready ? "Discard" : $"{chosen} / {required}", ready, Theme.Accent, Theme.AccentBorder, Theme.Bold17))
            {
                if (SelectedDiscard(player) is DiscardAction match) PlayHumanMove(match);
            }
            if (DrawButton(dl, "discardclear", min + new Vector2(24, 330 - 24 - 48), new Vector2(120, 48), "Reset", chosen > 0, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                Array.Clear(_discardSelection);
            ImGui.End();
        }

        private void DrawVictimDialog()
        {
            if (HumanToAct is not int player) { _dialog = Dialog.None; return; }
            List<RobberAction> options = Legal(player).OfType<RobberAction>().Where(a => a.TargetTileIndex == _victimTile).ToList();
            if (options.Count == 0) { _dialog = Dialog.None; return; }

            float width = 48 + options.Count * 150;
            ImDrawListPtr dl = BeginDialog("##victim", new Vector2(Math.Max(width, 380), 220), "Rob whom?", "Steal a random card from one player on the tile.", player, out Vector2 min);
            for (int i = 0; i < options.Count; i++)
            {
                if (options[i].TargetPlayerIndex is not sbyte victim) continue;
                Vector2 pos = min + new Vector2(24 + i * 150, 88);
                bool clicked = Theme.HitArea($"victim{victim}", pos, new Vector2(138, 100), out bool hovered);
                Theme.DrawPanel(dl, pos, pos + new Vector2(138, 100), hovered ? Theme.PlayerColor(victim) : Theme.BorderLight, 12, Theme.PanelRaised, hovered ? 2.5f : 1);
                dl.AddCircleFilled(pos + new Vector2(69, 34), 18, Theme.U32(Theme.PlayerColor(victim)), 24);
                Theme.TextCentered(dl, Theme.Bold17, pos + new Vector2(69, 70), Theme.Text, Theme.PlayerName(victim));
                Theme.TextCentered(dl, Theme.Regular13, pos + new Vector2(69, 88), Theme.TextMuted, $"{State.Players[victim].ResourceCards.Count()} cards");
                if (clicked) PlayHumanMove(options[i]);
            }
            ImGui.End();
        }

        private void DrawTradeDialog()
        {
            if (HumanToAct is not int player || !HasTrades(player)) { _dialog = Dialog.None; return; }

            ImDrawListPtr dl = BeginDialog("##trade", new Vector2(600, 430), "Trade with the bank", "Give a resource at your best rate (ports count), get any other one.", player, out Vector2 min);
            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, 84), Theme.TextSoft, "Give");
            int i = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Vector2 pos = min + new Vector2(28 + i * 110, 110);
                int? ratio = BestRatio(player, type);
                if (CardButton(dl, $"give{type}", type, pos, new Vector2(66, 99), ratio != null, _tradeGive == type))
                {
                    _tradeGive = type;
                    if (_tradeGet == type) _tradeGet = null;
                }
                Theme.TextCentered(dl, Theme.Bold15, pos + new Vector2(33, 116), ratio != null ? Theme.TextSoft : Theme.TextFaint, ratio != null ? $"{ratio}:1" : "-");
                i++;
            }

            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, 248), Theme.TextSoft, "Get");
            i = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Vector2 pos = min + new Vector2(28 + i * 110, 274);
                bool possible = _tradeGive == null ? Trades(player).Any(a => TradeOf(a)!.Value.get == type) : BestTrade(player, _tradeGive.Value, type) != null;
                if (CardButton(dl, $"get{type}", type, pos, new Vector2(66, 99), possible && type != _tradeGive, _tradeGet == type))
                    _tradeGet = type;
                i++;
            }

            Action? trade = _tradeGive is ResourceCardType g && _tradeGet is ResourceCardType r ? BestTrade(player, g, r) : null;
            string label = trade != null ? $"Trade {TradeOf(trade)!.Value.ratio} for 1" : "Trade";
            if (DrawButton(dl, "tradeok", min + new Vector2(600 - 24 - 180, 430 - 20 - 44), new Vector2(180, 44), label, trade != null, Theme.Accent, Theme.AccentBorder, Theme.Bold17))
                PlayHumanMove(trade!);
            if (DrawButton(dl, "tradecancel", min + new Vector2(24, 430 - 20 - 44), new Vector2(120, 44), "Cancel", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                ClearHumanInput();
            if (CanOffer(player) && DrawButton(dl, "tradeplayers", min + new Vector2(600 / 2 - 90, 430 - 20 - 44), new Vector2(180, 44), "Offer to players", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                OpenOfferDialog(player);
            ImGui.End();
        }

        private bool CanPlayDevelopmentCard(int player, DevelopmentCardType type) => type switch
        {
            DevelopmentCardType.Knight => HasLegal<KnightAction>(player),
            DevelopmentCardType.RoadBuilding => HasLegal<RoadBuildingAction>(player),
            DevelopmentCardType.YearOfPlenty => HasLegal<YearOfPlentyAction>(player),
            DevelopmentCardType.Monopoly => HasLegal<MonopolyAction>(player),
            _ => false,
        };

        /// <summary>
        /// Plays a development card, or opens the dialog for its choice (year of plenty, monopoly)
        /// </summary>
        private void PlayDevelopmentCard(int player, DevelopmentCardType type)
        {
            switch (type)
            {
                case DevelopmentCardType.Knight: PlayFirst<KnightAction>(player); break;
                case DevelopmentCardType.RoadBuilding: PlayFirst<RoadBuildingAction>(player); break;
                case DevelopmentCardType.YearOfPlenty: _dialog = Dialog.YearOfPlenty; break;
                case DevelopmentCardType.Monopoly: _dialog = Dialog.Monopoly; break;
            }
        }

        private void DrawDevelopmentDialog()
        {
            if (HumanToAct is not int player || !HasDevelopmentMoves(player)) { _dialog = Dialog.None; return; }

            (DevelopmentCardType type, bool legal, string what)[] cards =
            [
                (DevelopmentCardType.Knight, CanPlayDevelopmentCard(player, DevelopmentCardType.Knight), "Move the robber and steal a card"),
                (DevelopmentCardType.RoadBuilding, CanPlayDevelopmentCard(player, DevelopmentCardType.RoadBuilding), "Place two free roads"),
                (DevelopmentCardType.YearOfPlenty, CanPlayDevelopmentCard(player, DevelopmentCardType.YearOfPlenty), "Take two cards from the bank"),
                (DevelopmentCardType.Monopoly, CanPlayDevelopmentCard(player, DevelopmentCardType.Monopoly), "Take all cards of one resource"),
            ];

            ImDrawListPtr dl = BeginDialog("##devcards", new Vector2(600, 360), "Play a development card", "One per turn; cards bought this turn have to wait.", player, out Vector2 min);
            PlayerState hand = State.Players[player];
            for (int i = 0; i < cards.Length; i++)
            {
                Vector2 pos = min + new Vector2(24 + i * 140, 86);
                var card = cards[i];
                bool clicked = Theme.HitArea($"play{card.type}", pos, new Vector2(128, 206), out bool hovered);
                uint owned = hand.DevelopmentCards.Get(card.type), fresh = hand.NewDevelopmentCards.Get(card.type);
                if (hovered && card.legal) Theme.Glow(dl, pos, pos + new Vector2(128, 206), Color.White, 10, 1);
                Theme.DrawPanel(dl, pos, pos + new Vector2(128, 206), card.legal ? Theme.BorderLight : Theme.Border, 10, card.legal ? Theme.PanelRaised : Theme.Panel);

                Theme.DevelopmentCard(dl, card.type, pos + new Vector2(34, 12), new Vector2(60, 90), true, 1, !card.legal);
                if (owned > 1) Theme.Pill(dl, pos + new Vector2(86, 6), pos + new Vector2(118, 28), Theme.PlayerColor(player), Theme.Bold13, Theme.Background, $"x{owned}");

                Theme.TextCentered(dl, Theme.Bold15, pos + new Vector2(64, 118), card.legal ? Theme.Text : Theme.TextSoft, card.type.GetName());
                WrapCentered(dl, Theme.Regular13, pos + new Vector2(64, 140), 112, card.legal ? Theme.TextMuted : Theme.TextFaint, card.what);
                string? status = card.legal ? null
                    : owned == 0 ? "Not in hand"
                    : owned <= fresh ? "Bought this turn"
                    : State.Turn.HasPlayedDevelopmentCard ? "Already played one"
                    : "Not playable now";
                if (status != null) Theme.TextCentered(dl, Theme.Bold13, pos + new Vector2(64, 186), Theme.TextMuted, status);
                if (clicked && card.legal) PlayDevelopmentCard(player, card.type);
            }
            if (DrawButton(dl, "devcancel", min + new Vector2(600 - 24 - 120, 360 - 20 - 34), new Vector2(120, 34), "Cancel", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                ClearHumanInput();
            ImGui.End();
        }

        private static void WrapCentered(ImDrawListPtr dl, ImFontPtr font, Vector2 topCenter, float width, Color color, string text)
        {
            List<string> lines = [];
            string line = "";
            foreach (string word in text.Split(' '))
            {
                string candidate = line.Length == 0 ? word : line + " " + word;
                if (Theme.TextSize(font, candidate).X > width && line.Length > 0)
                {
                    lines.Add(line);
                    line = word;
                }
                else line = candidate;
            }
            if (line.Length > 0) lines.Add(line);
            for (int i = 0; i < lines.Count; i++)
            {
                Vector2 size = Theme.TextSize(font, lines[i]);
                Theme.DrawText(dl, font, topCenter + new Vector2(-size.X / 2, i * (size.Y + 1)), color, lines[i]);
            }
        }

        private void DrawPlentyDialog()
        {
            if (HumanToAct is not int player || !HasLegal<YearOfPlentyAction>(player)) { _dialog = Dialog.None; return; }

            ImDrawListPtr dl = BeginDialog("##plenty", new Vector2(600, 300), "Year of plenty", "Pick two cards (the same one twice is fine).", player, out Vector2 min);
            int i = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Vector2 pos = min + new Vector2(28 + i * 110, 90);
                int picked = (_plentyFirst == type ? 1 : 0) + (_plentySecond == type ? 1 : 0);
                if (CardButton(dl, $"plenty{type}", type, pos, new Vector2(66, 99), State.ResourceBank.Contains(type), picked > 0))
                {
                    if (_plentyFirst == null) _plentyFirst = type;
                    else if (_plentySecond == null) _plentySecond = type;
                    else { _plentyFirst = type; _plentySecond = null; }
                }
                if (picked > 0) Theme.Pill(dl, pos + new Vector2(13, 108), pos + new Vector2(53, 130), Theme.PlayerColor(player), Theme.Bold15, Theme.TextOnPlayer(player), $"+{picked}");
                i++;
            }

            YearOfPlentyAction? match = SelectedPlenty(player);
            if (DrawButton(dl, "plentyok", min + new Vector2(600 - 24 - 180, 300 - 20 - 44), new Vector2(180, 44), "Take", match != null, Theme.Accent, Theme.AccentBorder, Theme.Bold17))
                PlayHumanMove(match!);
            if (DrawButton(dl, "plentycancel", min + new Vector2(24, 300 - 20 - 44), new Vector2(120, 44), "Back", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                _dialog = Dialog.DevelopmentCard;
            ImGui.End();
        }

        private void DrawMonopolyDialog()
        {
            if (HumanToAct is not int player || !HasLegal<MonopolyAction>(player)) { _dialog = Dialog.None; return; }

            ImDrawListPtr dl = BeginDialog("##monopoly", new Vector2(600, 280), "Monopoly", "Every other player gives you all their cards of this resource.", player, out Vector2 min);
            int i = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                Vector2 pos = min + new Vector2(28 + i * 110, 90);
                MonopolyAction? action = Legal(player).OfType<MonopolyAction>().FirstOrDefault(a => a.ChosenType == type);
                if (CardButton(dl, $"monopoly{type}", type, pos, new Vector2(66, 99), action != null, false))
                    PlayHumanMove(action!);
                i++;
            }
            if (DrawButton(dl, "monopolycancel", min + new Vector2(24, 280 - 20 - 44), new Vector2(120, 44), "Back", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                _dialog = Dialog.DevelopmentCard;
            ImGui.End();
        }

        private void DrawNewGameDialog()
        {
            ImDrawListPtr dl = BeginDialog("##newgame", new Vector2(520, 400), "New game", "Who plays each seat (seat 1 starts).", null, out Vector2 min);
            string[] kinds = Enum.GetValues<SeatKind>().Select(Simulation.Seats.Name).ToArray();
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                Vector2 pos = min + new Vector2(24, 92 + seat * 52);
                dl.AddCircleFilled(pos + new Vector2(16, 16), 14, Theme.U32(Theme.PlayerColor(seat)), 24);
                Theme.TextCentered(dl, Theme.Bold15, pos + new Vector2(16, 16), Theme.TextOnPlayer(seat), (seat + 1).ToString());
                Theme.DrawText(dl, Theme.Bold17, pos + new Vector2(42, 5), Theme.Text, Theme.PlayerName(seat));
                ImGui.SetCursorScreenPos(pos + new Vector2(150, 2));
                ImGui.SetNextItemWidth(300);
                int kind = (int)_newGameSeats[seat];
                if (ImGui.Combo($"##seatkind{seat}", ref kind, kinds, kinds.Length))
                {
                    SeatKind chosen = (SeatKind)kind;
                    if (chosen == SeatKind.Human || !Simulation.Seats.NeedsValueNet(chosen) || _valueNetAvailable) _newGameSeats[seat] = chosen;
                }
            }
            ImGui.SetCursorScreenPos(min + new Vector2(24, 304));
            ImGui.Checkbox("Desert in the center", ref _centerDesert);
            ImGui.SetCursorScreenPos(min + new Vector2(260, 304));
            ImGui.Checkbox("Trading between players", ref _freeTrade);

            if (DrawButton(dl, "newgameok", min + new Vector2(520 - 24 - 180, 400 - 20 - 44), new Vector2(180, 44), "Start", true, Theme.Accent, Theme.AccentBorder, Theme.Bold17))
            {
                _dialog = Dialog.None;
                SetMode(InteractionMode.Play);
                Send(new Simulation.NewGame((SeatKind[])_newGameSeats.Clone(), _centerDesert, FreeTrade: _freeTrade));
                _endScreenDismissed = false;
            }
            if (DrawButton(dl, "newgamecancel", min + new Vector2(24, 400 - 20 - 44), new Vector2(120, 44), "Cancel", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                _dialog = Dialog.None;
            ImGui.End();
        }
        #endregion

        #region End screen
        private void DrawEndScreen()
        {
            if (!State.HasEnded || _endScreenDismissed || _mode == InteractionMode.Edit || _dialog == Dialog.NewGame) return;

            int winner = State.Turn.PlayerIndex;
            ImDrawListPtr dl = BeginDialog("##endscreen", new Vector2(620, 380), $"{Theme.PlayerName(winner)} wins!",
                $"after {State.Turn.RoundCounter} rounds, played by {(IsHuman(winner) ? "a human" : Simulation.Seats.Name(Seats[winner]))}", winner, out Vector2 min);

            string[] headers = ["", "VP", "Settl.", "Cities", "VP cards", "Road", "Army"];
            float[] columns = [24, 190, 250, 320, 390, 480, 545];
            for (int c = 0; c < headers.Length; c++)
                Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(columns[c], 88), Theme.TextMuted, headers[c]);

            foreach (int seat in Enumerable.Range(0, PLAYER_COUNT).OrderByDescending(s => State.Players[s].VictoryPoints.Total))
            {
                int row = Enumerable.Range(0, PLAYER_COUNT).OrderByDescending(s => State.Players[s].VictoryPoints.Total).ToList().IndexOf(seat);
                Vector2 pos = min + new Vector2(0, 118 + row * 44);
                VictoryPointTracker vp = State.Players[seat].VictoryPoints;
                if (seat == winner) dl.RoundRectFilled(pos + new Vector2(14, -6), pos + new Vector2(606, 32), Theme.U32(Theme.WithAlpha(Theme.PlayerColor(seat), 0.18f)), 8);
                dl.AddCircleFilled(pos + new Vector2(columns[0] + 12, 13), 10, Theme.U32(Theme.PlayerColor(seat)), 20);
                Theme.DrawText(dl, Theme.Bold17, pos + new Vector2(columns[0] + 32, 2), Theme.Text, Theme.PlayerName(seat));
                string[] values = [vp.Total.ToString(), (vp.SettlementPoints).ToString(), (vp.CityPoints / 2).ToString(), vp.DevelopmentCardPoints.ToString(),
                    vp.LongestRoadPoints > 0 ? "+2" : "", vp.LargestArmyPoints > 0 ? "+2" : ""];
                for (int c = 0; c < values.Length; c++)
                    Theme.DrawText(dl, c == 0 ? Theme.Bold17 : Theme.Regular17, pos + new Vector2(columns[c + 1], 2), c == 0 ? Theme.Text : Theme.TextSoft, values[c]);
            }

            if (DrawButton(dl, "rematch", min + new Vector2(620 - 24 - 200, 380 - 20 - 44), new Vector2(200, 44), "Rematch (new map)", IsAdmin, Theme.Accent, Theme.AccentBorder, Theme.Bold15))
            {
                NewGame();
                _endScreenDismissed = false;
            }
            if (DrawButton(dl, "endclose", min + new Vector2(24, 380 - 20 - 44), new Vector2(160, 44), "Look at the board", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                _endScreenDismissed = true;
            ImGui.End();
        }
        #endregion

        #region Save, load and developer tools
        private void DrawModals()
        {
            if (_openSaveModal) { ImGui.OpenPopup(saveFileModalId); _openSaveModal = false; }
            if (_openLoadModal) { ImGui.OpenPopup(loadFileModalId); _openLoadModal = false; }
            if (_openUpgradeModal) { ImGui.OpenPopup(upgradeSaveModalId); _openUpgradeModal = false; }

            bool isSaveModalOpen = true;
            if (ImGui.BeginPopupModal(saveFileModalId, ref isSaveModalOpen, ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.AlwaysAutoResize))
            {
                string defaultFileName = string.Format("gamestate_dump_{0}.yaml", DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
                var picker = FilePicker.GetFilePicker(this, Path.Combine(Environment.CurrentDirectory, ""), true, ".yaml", false, defaultFileName);
                if (picker.Draw())
                {
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
                    ReadSaveFile(picker.SelectedFile);
                    FilePicker.RemoveFilePicker(this);
                    // Loaded from the main menu: the game is on (an older file asks about the upgrade first)
                    if (_menuOpen && !_openUpgradeModal) CloseMenu();
                }
                ImGui.EndPopup();
            }

            bool isUpgradeModalOpen = true;
            if (ImGui.BeginPopupModal(upgradeSaveModalId, ref isUpgradeModalOpen, ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.Text($"\"{Path.GetFileName(_pendingUpgradePath)}\" is a version {_pendingUpgradeVersion} save file.");
                ImGui.Text($"This build reads version {SaveFile.CurrentVersion}. Upgrade and load it?");
                ImGui.TextDisabled("The file on disk stays unchanged. Use Save Game afterwards to keep the upgraded copy.");
                ImGui.Separator();
                if (ImGui.Button("Upgrade and load"))
                {
                    UpgradeAndLoadSaveFile(_pendingUpgradePath!);
                    ImGui.CloseCurrentPopup();
                    if (_menuOpen) CloseMenu();
                }
                ImGui.SameLine();
                if (ImGui.Button("Cancel")) ImGui.CloseCurrentPopup();
                ImGui.EndPopup();
            }
        }

        /// <summary>
        /// The tools of the old menu: map generation, rendering switches, agent budgets, playouts and statistics
        /// </summary>
        private void DrawDeveloperWindow()
        {
            ImGui.SetNextWindowPos(new Vector2(Display.X / 2 - 200, 80), ImGuiCond.FirstUseEver);
            ImGui.Begin("Developer tools", ref _showDeveloperWindow, ImGuiWindowFlags.AlwaysAutoResize);

            ImGui.Text($"FPS: {Math.Floor(1 / _latestAvgFrameTime)} ({_latestAvgFrameTime * 1000.0f:0.000} ms)");
            ImGui.Text($"State hash: {State.GetHashCode():X}");
            ImGui.Separator();

            ImGui.Checkbox("Center desert", ref _centerDesert);
            if (ImGui.Button("New map")) NewGame();
            ImGui.SameLine();
            if (ImGui.Button("Clear board")) NewGame(clearBoard: true);
            ImGui.SameLine();
            if (ImGui.Button("Quick reload")) SerializeLoop();

            if (ImGui.CollapsingHeader("Rendering"))
            {
                ImGui.Checkbox("Show yield points", ref _renderer.DrawYieldPoints);
                ImGui.Checkbox("Show shadows", ref _renderer.DrawTokenShadows);
                ImGui.Checkbox("Show intersections", ref _renderer.DrawIntersectionMarkers);
                ImGui.Checkbox("Show edges", ref _renderer.DrawEdgeMarkers);
                ImGui.Checkbox("Show centers", ref _renderer.DrawCenterMarkers);
            }

            if (ImGui.CollapsingHeader("Agents"))
            {
                string[] kinds = Enum.GetValues<SeatKind>().Select(Simulation.Seats.Name).ToArray();
                for (int seat = 0; seat < PLAYER_COUNT; seat++)
                {
                    using (new ImGuiTextColor(Theme.PlayerTextColor(seat)))
                        ImGui.Text($"{Theme.PlayerName(seat)}:");
                    ImGui.SameLine(90);
                    ImGui.SetNextItemWidth(200);
                    int kind = (int)Seats[seat];
                    if (ImGui.Combo($"##devseat{seat}", ref kind, kinds, kinds.Length)) SetSeat(seat, (SeatKind)kind);
                }

                ImGui.Separator();
                ImGui.SetNextItemWidth(140);
                if (ImGui.Combo("MCTS budget", ref _selectedMctsBudget, _mctsBudgetTypes, _mctsBudgetTypes.Length)) Send(new ConfigureAgents(AgentSettingsFromTools));
                ImGui.SetNextItemWidth(140);
                if (_selectedMctsBudget == 1 && ImGui.InputFloat("Thinking time (s)", ref _mctsThinkingSeconds, 0.5f, 5f, "%.2f"))
                {
                    _mctsThinkingSeconds = Math.Clamp(_mctsThinkingSeconds, 0.05f, 600f);
                    Send(new ConfigureAgents(AgentSettingsFromTools));
                }
                else if (_selectedMctsBudget == 2 && ImGui.InputInt("Iterations", ref _mctsIterations, 10_000, 100_000))
                {
                    _mctsIterations = Math.Clamp(_mctsIterations, 256, 100_000_000);
                    Send(new ConfigureAgents(AgentSettingsFromTools));
                }
                ImGui.SetNextItemWidth(140);
                if (ImGui.Combo("Opponent model", ref _selectedOpponentModel, _opponentModels, _opponentModels.Length))
                    Send(new ConfigureAgents(AgentSettingsFromTools));
                ImGui.SetNextItemWidth(140);
                float pace = Playback.PaceSeconds;
                if (ImGui.SliderFloat("Pause between moves (s)", ref pace, 0, 3, "%.1f")) Send(new SetPlayback(Playback with { PaceSeconds = pace }));
                ImGui.Checkbox("Silent agent moves", ref _muteQuickPlayouts);
            }

            if (ImGui.CollapsingHeader("Playouts and tests"))
            {
                ImGui.SetNextItemWidth(120);
                ImGui.InputInt("Playouts", ref _randomPlayoutCount);
                ImGui.SetNextItemWidth(120);
                ImGui.InputInt("Threads", ref _randomPlayoutThreadCount);
                if (ImGui.Button("Simulate random playouts"))
                {
                    // The playouts copy a snapshot: State keeps changing on the UI thread while they run
                    GameState startState = new GameState(State);
                    Task.Run(() => SimulateRandomRollouts(startState, _randomPlayoutCount, _randomPlayoutThreadCount));
                }
                if (ImGui.Button("Benchmark 10,000 random games")) BenchmarkPlayouts(10_000);
                if (ImGui.Button("Find inconsistent hashes")) FindInconsistentHashes();
            }

            if (ImGui.CollapsingHeader("Statistics"))
            {
                ImGui.TextUnformatted("Roll distribution:");
                ImGui.PlotHistogram(string.Empty, ref _rollDistribution[0], _rollDistribution.Length, 0, string.Empty, 0, Math.Max(1, _rollDistribution.Max()), new Vector2(260, 100));
            }

            ImGui.End();
        }
        #endregion
    }
}
