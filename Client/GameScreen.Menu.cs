using Client.UI;
using ImGuiNET;
using SFML.Graphics;
using Simulation;
using System.Numerics;
using Color = SFML.Graphics.Color;

namespace Client
{
    /// <summary>
    /// The main menu over the board: a game on this computer (a human or an agent on each seat; several humans take turns at
    /// one screen), hosting, joining, loading, and the settings. It opens at the start and from the menu button or Escape; a
    /// local game waits while it is open.
    /// </summary>
    public partial class GameScreen
    {
        private enum MenuPage { Main, Local, Host, Join, Settings }

        private bool _menuOpen;
        private MenuPage _menuPage;

        // At the start there is no game to go back to: the board behind the menu is only a backdrop
        private bool _menuIsStartScreen;

        // How the agents moved before the menu paused a local game
        private PlaybackState? _playbackBeforeMenu;

        private readonly ClientSettings _settings = ClientSettings.Load();
        private string? _settingsError;

        // A new game: who plays each seat on this computer (several humans take turns) and when hosting (the humans join over
        // the network), and whether the seats are shuffled
        private readonly SeatKind[] _menuLocalSeats;
        private readonly SeatKind[] _menuHostSeats;
        private bool _menuRandomSeats = true;

        private const float MenuWidth = 520, MenuLabelX = 28, MenuFieldX = 230, MenuFieldWidth = 262, MenuSeatRow = 44;

        // The match can wait: a local game without connected players
        private bool CanPauseForMenu => IsAdmin && _sim is LocalSimulation && !_sim.Game.Sessions.Any(s => s.Role is Role.Player or Role.Agent);

        private void OpenMenu(MenuPage page = MenuPage.Main)
        {
            _menuPage = page;
            if (_menuOpen) return;
            _menuOpen = true;
            _networkError = null;
            if (CanPauseForMenu)
            {
                _playbackBeforeMenu = Playback;
                if (!Playback.Paused) Send(new SetPlayback(Playback with { Paused = true }));
            }
        }

        /// <summary>
        /// Opens the menu on a page by name (main, local, host, join, settings), for --menu
        /// </summary>
        public void ShowMenuPage(string? page) => OpenMenu(Enum.TryParse(page, true, out MenuPage parsed) ? parsed : MenuPage.Main);

        private void CloseMenu()
        {
            _menuOpen = false;
            _menuIsStartScreen = false;
            if (_playbackBeforeMenu is PlaybackState before && CanPauseForMenu && before != Playback) Send(new SetPlayback(before));
            _playbackBeforeMenu = null;
        }

        /// <summary>
        /// Escape in the menu: from a page back to the main page, from there back to the game
        /// </summary>
        private void MenuBack()
        {
            if (_menuPage == MenuPage.Settings) _settingsError = _settings.Save();
            if (_menuPage != MenuPage.Main) _menuPage = MenuPage.Main;
            else if (!_menuIsStartScreen) CloseMenu();
        }

        /// <summary>
        /// The settings take effect at once: volume, particles, and the agents' pace in a local game
        /// </summary>
        private void ApplySettings()
        {
            _particles.Enabled = _settings.Particles;
            _placeSound.Volume = 40f * _settings.Volume;
            _diceRollSound.Volume = 50f * _settings.Volume;
            if (_playbackBeforeMenu != null) _playbackBeforeMenu = _playbackBeforeMenu with { PaceSeconds = _settings.BotPace };
            else if (CanPauseForMenu && Playback.PaceSeconds != _settings.BotPace) Send(new SetPlayback(Playback with { PaceSeconds = _settings.BotPace }));
        }

        /// <summary>
        /// The configured seats, shuffled if asked (otherwise as listed: seat 1 starts)
        /// </summary>
        private SeatKind[] MenuSeats(SeatKind[] configured) =>
            _menuRandomSeats ? configured.OrderBy(_ => Random.Shared.Next()).ToArray() : (SeatKind[])configured.Clone();

        /// <summary>
        /// A new game in this process, hosted for others if asked; a server joined before is left, an old hosting stopped
        /// </summary>
        private void StartGameFromMenu(SeatKind[] seats, bool host)
        {
            if (_sim is RemoteSimulation) Disconnect();
            StopHosting();
            _menuOpen = false;
            _menuIsStartScreen = false;
            _playbackBeforeMenu = null;
            // Not SetMode: the new game gets its playback below
            _mode = InteractionMode.Play;
            ClearHumanInput();
            Send(new Simulation.NewGame(seats, _centerDesert, FreeTrade: _freeTrade, ShowTrackedCards: _showTrackedCards));
            Send(new SetPlayback(PlaybackState.Default with { PaceSeconds = _settings.BotPace }));
            _endScreenDismissed = false;
            if (host)
            {
                StartHosting();
                _showNetworkWindow = true;
            }
        }

        private void DrawMenu()
        {
            // The whole window, so the board behind takes no clicks
            ImDrawListPtr dl = Theme.BeginHud("##mainmenu", Vector2.Zero, Display);
            dl.AddRectFilled(new Vector2(-8), Display + new Vector2(8), Theme.U32(Theme.WithAlpha(Theme.Background, 0.78f)));

            float height = _menuPage switch
            {
                MenuPage.Main => 140 + (_menuIsStartScreen ? 6 : 7) * 56,
                MenuPage.Local => 534,
                MenuPage.Host => 662,
                MenuPage.Join => 486,
                _ => 356
            };
            Vector2 min = Vector2.Round(new Vector2((Display.X - MenuWidth) / 2, MathF.Max(Margin, (Display.Y - height) / 2 - 20)));
            Vector2 max = min + new Vector2(MenuWidth, height);
            Theme.Glow(dl, min, max, Theme.BorderLight, 16, 0.8f);
            Theme.DrawPanel(dl, min, max, Theme.BorderLight, 16, new Color(15, 33, 86, 250), 2);

            switch (_menuPage)
            {
                case MenuPage.Main: DrawMenuMain(dl, min); break;
                case MenuPage.Local: DrawMenuLocal(dl, min); break;
                case MenuPage.Host: DrawMenuHost(dl, min); break;
                case MenuPage.Join: DrawMenuJoin(dl, min); break;
                case MenuPage.Settings: DrawMenuSettings(dl, min); break;
            }
            ImGui.End();
        }

        private void DrawMenuMain(ImDrawListPtr dl, Vector2 min)
        {
            Theme.TextCentered(dl, Theme.ExtraBold36, min + new Vector2(MenuWidth / 2, 54), Theme.Text, "Hexfarer");
            Theme.TextCentered(dl, Theme.Regular15, min + new Vector2(MenuWidth / 2, 94), Theme.TextMuted, "Simulator and AI agent sandbox");

            float y = 126;
            bool Item(string id, string label, bool primary = false, bool enabled = true)
            {
                Vector2 pos = min + new Vector2((MenuWidth - 360) / 2, y);
                y += 56;
                return DrawButton(dl, id, pos, new Vector2(360, 46), label, enabled, primary ? Theme.Accent : Theme.PanelRaised,
                    primary ? Theme.AccentBorder : Theme.BorderLight, Theme.Bold17);
            }

            if (!_menuIsStartScreen && Item("menucontinue", "Continue", true)) CloseMenu();
            if (Item("menulocal", "Play on this computer", _menuIsStartScreen)) _menuPage = MenuPage.Local;
            if (Item("menuhost", "Host a game")) _menuPage = MenuPage.Host;
            if (Item("menujoin", "Join a game")) _menuPage = MenuPage.Join;
            if (Item("menuload", "Load game", enabled: IsAdmin)) _openLoadModal = true;
            if (Item("menusettings", "Settings")) _menuPage = MenuPage.Settings;
            if (Item("menuquit", "Quit")) _window.Close();
        }

        private void DrawMenuLocal(ImDrawListPtr dl, Vector2 min)
        {
            int humans = _menuLocalSeats.Count(kind => kind == SeatKind.Human);
            MenuHeader(dl, min, "Play on this computer", humans > 1 ? "The humans take turns; a curtain hides the hands." : "A human or a bot on each seat.");

            DrawMenuSeats(dl, min, 92, "menulocalseat", _menuLocalSeats);

            DrawToggle(dl, "menurandomseats", min + new Vector2(MenuLabelX, 278), "Shuffle the seats (otherwise seat 1 starts)", ref _menuRandomSeats);
            DrawToggle(dl, "menufreetrade", min + new Vector2(MenuLabelX, 322), "Trading between players", ref _freeTrade);
            DrawToggle(dl, "menudesert", min + new Vector2(MenuLabelX, 366), "Desert in the center", ref _centerDesert);
            DrawToggle(dl, "menutracked", min + new Vector2(MenuLabelX, 410), "Show the confirmed cards of the other hands", ref _showTrackedCards);

            if (MenuButtons(dl, min, 464, "Start")) StartGameFromMenu(MenuSeats(_menuLocalSeats), host: false);
        }

        private void DrawMenuHost(ImDrawListPtr dl, Vector2 min)
        {
            MenuHeader(dl, min, "Host a game", "Others join over the network; you play a seat and run the match.");

            DrawMenuSeats(dl, min, 92, "menuhostseat", _menuHostSeats);

            MenuLabel(dl, min, 280, "Port");
            ImGui.SetCursorScreenPos(min + new Vector2(MenuFieldX, 276));
            ImGui.SetNextItemWidth(140);
            ImGui.InputInt("##menuport", ref _hostPort);
            _hostPort = Math.Clamp(_hostPort, 1024, 65535);

            DrawToggle(dl, "menuhostfreetrade", min + new Vector2(MenuLabelX, 318), "Trading between players", ref _freeTrade);
            DrawToggle(dl, "menuhostnetwork", min + new Vector2(MenuLabelX, 362), "Reachable from other computers", ref _hostOnNetwork);
            DrawToggle(dl, "menuhostspectators", min + new Vector2(MenuLabelX, 406), "Anyone may watch without a token", ref _hostOpenToSpectators);
            DrawToggle(dl, "menuhosttls", min + new Vector2(MenuLabelX, 450), "Encrypt connections (TLS)", ref _hostSecure);
            DrawToggle(dl, "menuhosttracked", min + new Vector2(MenuLabelX, 494), "Show the confirmed cards of the other hands", ref _showTrackedCards);

            string note = _hostOnNetwork && _hostOpenToSpectators
                ? "Watchers see every hand: hidden cards are only hidden in the client."
                : "Each human seat gets a token to join with, shown after the start.";
            Theme.DrawText(dl, Theme.Regular15, min + new Vector2(MenuLabelX, 544), _hostOnNetwork && _hostOpenToSpectators ? new Color(255, 200, 90) : Theme.TextMuted, note);

            if (MenuButtons(dl, min, 592, "Start hosting")) StartGameFromMenu(MenuSeats(_menuHostSeats), host: true);
        }

        private void DrawMenuJoin(ImDrawListPtr dl, Vector2 min)
        {
            MenuHeader(dl, min, "Join a game", "Connect to a game someone hosts, or to a Hexfarer server.");

            void Field(string label, float y, string id, ref string value, uint length, ImGuiInputTextFlags flags = ImGuiInputTextFlags.None)
            {
                MenuLabel(dl, min, y + 4, label);
                ImGui.SetCursorScreenPos(min + new Vector2(MenuFieldX, y));
                ImGui.SetNextItemWidth(MenuFieldWidth);
                ImGui.InputText(id, ref value, length, flags);
            }

            Field("Address", 96, "##menuaddress", ref _joinAddress, 200);
            Field("Name", 136, "##menuname", ref _joinName, 40);

            MenuLabel(dl, min, 186, "Join as");
            DrawSegments(dl, "menurole", min + new Vector2(MenuFieldX, 176), MenuFieldWidth, ["Player", "Watch", "Admin", "Bridge"], ref _joinRole);
            if (JoinRoleValues[_joinRole] is Role.Player or Role.Agent)
            {
                MenuLabel(dl, min, 234, "Seat");
                string[] seats = Enumerable.Range(0, PLAYER_COUNT).Select(s => (s + 1).ToString()).ToArray();
                DrawSegments(dl, "menuseat", min + new Vector2(MenuFieldX, 224), MenuFieldWidth, seats, ref _joinSeat);
            }

            Field("Token", 274, "##menutoken", ref _joinToken, 100, ImGuiInputTextFlags.Password);
            Field("Fingerprint (TLS)", 314, "##menufingerprint", ref _joinFingerprint, 100);

            if (_networkError != null)
                WrapCentered(dl, Theme.Regular15, min + new Vector2(MenuWidth / 2, 360), MenuWidth - 56, new Color(255, 120, 120), _networkError);
            else
                WrapCentered(dl, Theme.Regular15, min + new Vector2(MenuWidth / 2, 360), MenuWidth - 56, Theme.TextMuted,
                    "Watching may work without a token; playing needs the token of your seat.");

            if (MenuButtons(dl, min, 416, _connecting != null ? "Connecting..." : "Connect", _connecting == null)) Connect();
        }

        private void DrawMenuSettings(ImDrawListPtr dl, Vector2 min)
        {
            MenuHeader(dl, min, "Settings", "Kept for this computer, in effect at once.");
            bool changed = false;

            bool sounds = _settings.Sounds;
            DrawToggle(dl, "menusounds", min + new Vector2(MenuLabelX, 96), "Sound effects", ref sounds);
            changed |= sounds != _settings.Sounds;
            _settings.Sounds = sounds;

            MenuLabel(dl, min, 148, "Volume");
            ImGui.SetCursorScreenPos(min + new Vector2(MenuFieldX, 144));
            ImGui.SetNextItemWidth(MenuFieldWidth);
            float volume = _settings.Volume * 100;
            if (ImGui.SliderFloat("##menuvolume", ref volume, 0, 100, "%.0f%%"))
            {
                _settings.Volume = Math.Clamp(volume / 100, 0, 1);
                changed = true;
            }

            bool particles = _settings.Particles;
            DrawToggle(dl, "menuparticles", min + new Vector2(MenuLabelX, 188), "Flying cards", ref particles);
            changed |= particles != _settings.Particles;
            _settings.Particles = particles;

            MenuLabel(dl, min, 240, "Pause between bot moves");
            ImGui.SetCursorScreenPos(min + new Vector2(MenuFieldX, 236));
            ImGui.SetNextItemWidth(MenuFieldWidth);
            float pace = _settings.BotPace;
            if (ImGui.SliderFloat("##menupace", ref pace, 0, 2, "%.1f s"))
            {
                _settings.BotPace = Math.Clamp(pace, 0, 3);
                changed = true;
            }

            if (_settingsError != null)
                WrapCentered(dl, Theme.Regular15, min + new Vector2(MenuWidth / 2, 268), MenuWidth - 56, new Color(255, 120, 120), $"Could not save the settings: {_settingsError}");

            if (changed) ApplySettings();

            Vector2 back = min + new Vector2(MenuLabelX, 286);
            if (DrawButton(dl, "menusettingsback", back, new Vector2(140, 46), "Back", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15)) MenuBack();
        }

        /// <summary>
        /// Who plays each seat: a human or an agent from the ordered list (agents that need the value net only with it)
        /// </summary>
        private void DrawMenuSeats(ImDrawListPtr dl, Vector2 min, float y, string id, SeatKind[] seats)
        {
            string[] names = Simulation.Seats.Ordered.Select(Simulation.Seats.Name).ToArray();
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                Vector2 pos = min + new Vector2(MenuLabelX, y + seat * MenuSeatRow);
                dl.AddCircleFilled(pos + new Vector2(14, 16), 13, Theme.U32(Theme.PlayerColor(seat)), 24);
                Theme.TextCentered(dl, Theme.Bold15, pos + new Vector2(14, 16), Theme.TextOnPlayer(seat), (seat + 1).ToString());
                Theme.DrawText(dl, Theme.SemiBold15, pos + new Vector2(38, 6), Theme.TextSoft, Theme.PlayerName(seat));

                ImGui.SetCursorScreenPos(min + new Vector2(MenuFieldX, y + seat * MenuSeatRow + 2));
                ImGui.SetNextItemWidth(MenuFieldWidth);
                int kind = Array.IndexOf(Simulation.Seats.Ordered, seats[seat]);
                if (ImGui.Combo($"##{id}{seat}", ref kind, names, names.Length))
                {
                    SeatKind chosen = Simulation.Seats.Ordered[kind];
                    if (chosen == SeatKind.Human || !Simulation.Seats.NeedsValueNet(chosen) || _valueNetAvailable) seats[seat] = chosen;
                }
            }
        }

        private static void MenuHeader(ImDrawListPtr dl, Vector2 min, string title, string subtitle)
        {
            Theme.DrawText(dl, Theme.Bold22, min + new Vector2(MenuLabelX, 22), Theme.Text, title);
            Theme.DrawText(dl, Theme.Regular15, min + new Vector2(MenuLabelX, 56), Theme.TextMuted, subtitle);
        }

        private static void MenuLabel(ImDrawListPtr dl, Vector2 min, float y, string text) =>
            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(MenuLabelX, y), Theme.TextSoft, text);

        /// <summary>
        /// Back on the left, the page's action on the right; whether the action was clicked
        /// </summary>
        private bool MenuButtons(ImDrawListPtr dl, Vector2 min, float y, string action, bool enabled = true)
        {
            if (DrawButton(dl, "menuback", min + new Vector2(MenuLabelX, y), new Vector2(140, 46), "Back", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                MenuBack();
            return DrawButton(dl, "menuaction", min + new Vector2(MenuWidth - MenuLabelX - 200, y), new Vector2(200, 46), action, enabled,
                Theme.Accent, Theme.AccentBorder, Theme.Bold17);
        }

        /// <summary>
        /// A row of options of which one is chosen, styled like the mode switch; whether the choice changed
        /// </summary>
        private static bool DrawSegments(ImDrawListPtr dl, string id, Vector2 min, float width, string[] labels, ref int selected)
        {
            const float height = 38;
            dl.RoundRectFilled(min, min + new Vector2(width, height), Theme.U32(Theme.Panel), height / 2);
            dl.RoundRect(min, min + new Vector2(width, height), Theme.U32(Theme.Border), height / 2);
            float segment = (width - 8) / labels.Length;
            bool changed = false;
            for (int i = 0; i < labels.Length; i++)
            {
                Vector2 a = min + new Vector2(4 + segment * i, 4);
                Vector2 size = new Vector2(segment, height - 8);
                bool clicked = Theme.HitArea($"{id}{i}", a, size, out bool hovered);
                bool active = selected == i;
                if (active) dl.RoundRectFilled(a, a + size, Theme.U32(Theme.PlayerColor(3)), size.Y / 2);
                else if (hovered) dl.RoundRectFilled(a, a + size, Theme.U32(Theme.PanelRaised), size.Y / 2);
                Theme.TextCentered(dl, Theme.Bold15, a + size / 2, active ? new Color(58, 38, 0) : Theme.TextMuted, labels[i]);
                if (clicked && !active)
                {
                    selected = i;
                    changed = true;
                }
            }
            return changed;
        }
    }
}
