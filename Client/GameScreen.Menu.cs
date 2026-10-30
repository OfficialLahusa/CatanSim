using Client.UI;
using ImGuiNET;
using SFML.Graphics;
using Simulation;
using System.Numerics;
using Color = SFML.Graphics.Color;

namespace Client
{
    /// <summary>
    /// The main menu over the board: a game against agents, several humans at one screen (hot-seat), hosting, joining, loading,
    /// and the settings. It opens at the start and from the menu button or Escape; a local game waits while it is open.
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

        // A new game: humans at this screen (hot-seat with more than one) or over the network, the agents' strength, and
        // whether the humans get random seats
        private int _menuHumans = 1;
        private int _menuHostHumans = 2;
        private int _menuBotStrength = 1;
        private bool _menuRandomSeats = true;

        private static readonly SeatKind[] BotStrengths = [SeatKind.MctsEasy, SeatKind.MctsMedium, SeatKind.MctsHard];

        private const float MenuWidth = 520, MenuLabelX = 28, MenuFieldX = 230, MenuFieldWidth = 262;

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

        private SeatKind MenuBot => _valueNetAvailable ? BotStrengths[_menuBotStrength] : SeatKind.Builder;

        /// <summary>
        /// Seats with this many humans, on random seats or the first ones; agents of the chosen strength play the others
        /// </summary>
        private SeatKind[] MenuSeats(int humans)
        {
            SeatKind[] seats = Enumerable.Repeat(MenuBot, PLAYER_COUNT).ToArray();
            IEnumerable<int> humanSeats = _menuRandomSeats ? Enumerable.Range(0, PLAYER_COUNT).OrderBy(_ => Random.Shared.Next()).Take(humans) : Enumerable.Range(0, humans);
            foreach (int seat in humanSeats) seats[seat] = SeatKind.Human;
            return seats;
        }

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
                MenuPage.Main => 140 + (_menuIsStartScreen ? 7 : 8) * 56,
                MenuPage.Local => 462,
                MenuPage.Host => 590,
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
            if (Item("menubots", "Play against bots", _menuIsStartScreen))
            {
                _menuHumans = 1;
                _menuPage = MenuPage.Local;
            }
            if (Item("menuhotseat", "Hot-seat"))
            {
                _menuHumans = Math.Max(2, _menuHumans);
                _menuPage = MenuPage.Local;
            }
            if (Item("menuhost", "Host a game")) _menuPage = MenuPage.Host;
            if (Item("menujoin", "Join a game")) _menuPage = MenuPage.Join;
            if (Item("menuload", "Load game", enabled: IsAdmin)) _openLoadModal = true;
            if (Item("menusettings", "Settings")) _menuPage = MenuPage.Settings;
            if (Item("menuquit", "Quit")) _window.Close();
        }

        private void DrawMenuLocal(ImDrawListPtr dl, Vector2 min)
        {
            int humans = _menuHumans;
            MenuHeader(dl, min, humans == 1 ? "Play against bots" : "Hot-seat",
                humans == 1 ? "You against three bots on this computer." : "Players take turns at this screen; a curtain hides the hands.");

            MenuLabel(dl, min, 102, "Humans at this screen");
            int humanIndex = humans - 1;
            if (DrawSegments(dl, "menuhumans", min + new Vector2(MenuFieldX, 92), MenuFieldWidth, ["1", "2", "3", "4"], ref humanIndex))
                _menuHumans = humanIndex + 1;

            DrawMenuBotRow(dl, min, 152, humans < PLAYER_COUNT);

            DrawToggle(dl, "menurandomseats", min + new Vector2(MenuLabelX, 206), "Random seats (otherwise the humans start)", ref _menuRandomSeats);
            DrawToggle(dl, "menufreetrade", min + new Vector2(MenuLabelX, 250), "Trading between players", ref _freeTrade);
            DrawToggle(dl, "menudesert", min + new Vector2(MenuLabelX, 294), "Desert in the center", ref _centerDesert);
            DrawToggle(dl, "menutracked", min + new Vector2(MenuLabelX, 338), "Show the confirmed cards of the other hands", ref _showTrackedCards);

            if (MenuButtons(dl, min, 392, "Start")) StartGameFromMenu(MenuSeats(humans), host: false);
        }

        private void DrawMenuHost(ImDrawListPtr dl, Vector2 min)
        {
            MenuHeader(dl, min, "Host a game", "Others join over the network; you play a seat and run the match.");

            MenuLabel(dl, min, 102, "Human players");
            int humanIndex = _menuHostHumans - 2;
            if (DrawSegments(dl, "menuhosthumans", min + new Vector2(MenuFieldX, 92), MenuFieldWidth, ["2", "3", "4"], ref humanIndex))
                _menuHostHumans = humanIndex + 2;

            DrawMenuBotRow(dl, min, 152, _menuHostHumans < PLAYER_COUNT);

            MenuLabel(dl, min, 204, "Port");
            ImGui.SetCursorScreenPos(min + new Vector2(MenuFieldX, 200));
            ImGui.SetNextItemWidth(140);
            ImGui.InputInt("##menuport", ref _hostPort);
            _hostPort = Math.Clamp(_hostPort, 1024, 65535);

            DrawToggle(dl, "menuhostfreetrade", min + new Vector2(MenuLabelX, 246), "Trading between players", ref _freeTrade);
            DrawToggle(dl, "menuhostnetwork", min + new Vector2(MenuLabelX, 290), "Reachable from other computers", ref _hostOnNetwork);
            DrawToggle(dl, "menuhostspectators", min + new Vector2(MenuLabelX, 334), "Anyone may watch without a token", ref _hostOpenToSpectators);
            DrawToggle(dl, "menuhosttls", min + new Vector2(MenuLabelX, 378), "Encrypt connections (TLS)", ref _hostSecure);
            DrawToggle(dl, "menuhosttracked", min + new Vector2(MenuLabelX, 422), "Show the confirmed cards of the other hands", ref _showTrackedCards);

            string note = _hostOnNetwork && _hostOpenToSpectators
                ? "Watchers see every hand: hidden cards are only hidden in the client."
                : "Each human seat gets a token to join with, shown after the start.";
            Theme.DrawText(dl, Theme.Regular15, min + new Vector2(MenuLabelX, 472), _hostOnNetwork && _hostOpenToSpectators ? new Color(255, 200, 90) : Theme.TextMuted, note);

            if (MenuButtons(dl, min, 520, "Start hosting")) StartGameFromMenu(MenuSeats(_menuHostHumans), host: true);
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
        /// The strength of the agents on the other seats, or why there is no choice
        /// </summary>
        private void DrawMenuBotRow(ImDrawListPtr dl, Vector2 min, float y, bool hasBots)
        {
            MenuLabel(dl, min, y + 10, "Bots");
            if (!hasBots)
                Theme.DrawText(dl, Theme.Regular15, min + new Vector2(MenuFieldX, y + 8), Theme.TextMuted, "None: every seat is human");
            else if (!_valueNetAvailable)
                Theme.DrawText(dl, Theme.Regular15, min + new Vector2(MenuFieldX, y + 8), Theme.TextMuted, "Builder (the value net is missing)");
            else
                DrawSegments(dl, "menubotstrength", min + new Vector2(MenuFieldX, y), MenuFieldWidth, ["Easy", "Medium", "Hard"], ref _menuBotStrength);
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
