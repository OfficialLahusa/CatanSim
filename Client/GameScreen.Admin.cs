using Client.UI;
using ImGuiNET;
using SFML.Graphics;
using Simulation;
using System.Numerics;

namespace Client
{
    /// <summary>
    /// The admin panel of a hosted match, for the client hosting it and for admins connected to a server: who plays which
    /// seat and the playback, the connections, the tokens and the server's options. Everything goes through the match's
    /// commands, so the dedicated server's console and this panel do the same.
    /// </summary>
    public partial class GameScreen
    {
        private bool _showAdminWindow;

        // A tab to bring to the front on the next frame (from the command line)
        private string? _adminTabRequest;

        /// <summary>
        /// Opens the admin panel, on a tab by its name (match, connections, tokens, server)
        /// </summary>
        public void ShowAdminWindow(string? tab)
        {
            _showAdminWindow = true;
            _adminTabRequest = tab?.ToLowerInvariant();
        }

        private bool AdminTab(string label, string key)
        {
            if (_adminTabRequest != key) return ImGui.BeginTabItem(label);
            _adminTabRequest = null;
            bool open = true;
            return ImGui.BeginTabItem(label, ref open, ImGuiTabItemFlags.SetSelected);
        }

        private int _tokenRole = 0;
        private int _tokenSeat = 0;
        private string _tokenLabel = "";
        private static readonly string[] TokenRoles = ["Player", "Agent bridge", "Spectator", "Admin"];
        private static readonly Role[] TokenRoleValues = [Role.Player, Role.Agent, Role.Spectator, Role.Admin];
        private static readonly SeatKind[] SeatKinds = Enum.GetValues<SeatKind>();

        /// <summary>
        /// Whether this client administers a match that others can join
        /// </summary>
        private bool CanAdminister => IsAdmin && _sim.Game.Server != null;

        private void DrawAdminWindow()
        {
            if (!_showAdminWindow) return;
            // Until the connection or the hosting is up (and the server's details have arrived) the panel waits; a local game
            // nobody can join, or a session that isn't an admin, has none
            if (!CanAdminister)
            {
                bool waiting = _connecting != null || _hostStarting != null || (IsAdmin && (_sim is RemoteSimulation || _hostServer != null));
                if (!waiting) _showAdminWindow = false;
                return;
            }

            ImGui.SetNextWindowPos(new Vector2(Display.X / 2 - 290, 80), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSize(new Vector2(580, 0), ImGuiCond.Always);
            if (!ImGui.Begin("Admin", ref _showAdminWindow, ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.End();
                return;
            }

            ImGui.PushFont(Theme.SemiBold17);
            ImGui.TextUnformatted(_sim.Description);
            ImGui.PopFont();

            // A token is shown once: it stays on top of every tab until it is put away
            if (_sim.Game.CreatedToken is TokenCreatedEvent created) DrawCreatedToken(created);

            if (ImGui.BeginTabBar("##admin"))
            {
                if (AdminTab("Match", "match"))
                {
                    DrawAdminMatch();
                    ImGui.EndTabItem();
                }
                if (AdminTab($"Connections ({_sim.Game.Sessions.Count})###connections", "connections"))
                {
                    DrawAdminConnections();
                    ImGui.EndTabItem();
                }
                if (AdminTab($"Tokens ({_sim.Game.Tokens.Count})###tokens", "tokens"))
                {
                    DrawAdminTokens();
                    ImGui.EndTabItem();
                }
                if (AdminTab("Server", "server"))
                {
                    DrawAdminServer(_sim.Game.Server!);
                    ImGui.EndTabItem();
                }
                ImGui.EndTabBar();
            }
            ImGui.End();
        }

        private void DrawCreatedToken(TokenCreatedEvent created)
        {
            ImGui.Spacing();
            using (new ImGuiTextColor(new Color(255, 200, 90)))
                ImGui.TextWrapped($"New {DescribeToken(created.Info)}: copy it now, it is shown only this once.");
            string token = created.Token;
            ImGui.SetNextItemWidth(420);
            ImGui.InputText("##createdtoken", ref token, 128, ImGuiInputTextFlags.ReadOnly);
            ImGui.SameLine();
            if (ImGui.Button("Copy##created")) SFML.Window.Clipboard.Contents = created.Token;
            ImGui.SameLine();
            if (ImGui.Button("Done##created")) _sim.Game.CreatedToken = null;
            ImGui.Separator();
        }

        private void DrawAdminMatch()
        {
            ImGui.TextUnformatted("Seats");
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                using (new ImGuiTextColor(Theme.PlayerTextColor(seat)))
                    ImGui.TextUnformatted($"{Theme.PlayerName(seat)} (seat {seat + 1})");
                ImGui.SameLine(130);
                ImGui.SetNextItemWidth(170);
                int current = Array.IndexOf(SeatKinds, Seats[seat]);
                if (ImGui.Combo($"##seat{seat}", ref current, SeatKinds.Select(Simulation.Seats.Name).ToArray(), SeatKinds.Length) && SeatKinds[current] != Seats[seat])
                    Send(new Simulation.SetSeat(seat, SeatKinds[current]));

                SessionInfo? holder = _sim.Game.Sessions.FirstOrDefault(s => s.Role is Role.Player or Role.Agent && s.Seat == seat);
                string? who = holder != null ? $"{holder.Name} ({(holder.Role == Role.Agent ? "bridge" : "player")})" : IsHuman(seat) ? "nobody connected" : null;
                if (who != null)
                {
                    ImGui.SameLine(320);
                    ImGui.TextDisabled(who);
                }
            }

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.TextUnformatted("Agents");
            if (ImGui.Button(Playback.Paused ? "Resume" : "Pause", new Vector2(110, 0)))
                Send(new SetPlayback(Playback with { Paused = !Playback.Paused }));
            ImGui.SameLine();
            if (ImGui.Button("Step", new Vector2(80, 0))) Send(new StepCommand());
            ImGui.SameLine();
            float pace = Playback.PaceSeconds;
            ImGui.SetNextItemWidth(160);
            if (ImGui.SliderFloat("s between moves", ref pace, 0, 3, "%.1f") && MathF.Abs(pace - Playback.PaceSeconds) > 0.05f)
                Send(new SetPlayback(Playback with { PaceSeconds = pace }));
            bool standIns = Playback.HumanStandIns;
            if (ImGui.Checkbox("Agents also play the human seats", ref standIns)) Send(new SetPlayback(Playback with { HumanStandIns = standIns }));

            ImGui.Spacing();
            ImGui.Separator();
            if (ImGui.Button("New game...", new Vector2(140, 0)))
            {
                Array.Copy(Seats, _newGameSeats, PLAYER_COUNT);
                _dialog = Dialog.NewGame;
            }
            ImGui.SameLine();
            ImGui.TextDisabled("starts over for everyone on a new map");
        }

        private void DrawAdminConnections()
        {
            List<SessionInfo> sessions = _sim.Game.Sessions;
            if (sessions.Count == 0)
            {
                ImGui.TextDisabled("Nobody is connected.");
                return;
            }
            foreach (SessionInfo session in sessions)
            {
                using (new ImGuiTextColor(session.Seat is int seat ? Theme.PlayerTextColor(seat) : Theme.TextSoft))
                    ImGui.TextUnformatted(session.Name);
                ImGui.SameLine(170);
                ImGui.TextDisabled(session.Role switch
                {
                    Role.Player => $"player of seat {session.Seat + 1}",
                    Role.Agent => $"bridge for seat {session.Seat + 1}",
                    _ => session.Role.ToString().ToLower()
                });
                ImGui.SameLine(330);
                ImGui.TextDisabled(session.Address);
                if (session.Id != _sim.Session?.Id)
                {
                    ImGui.SameLine(500);
                    if (ImGui.SmallButton($"Kick##{session.Id}")) Send(new KickSession(session.Id));
                }
                else
                {
                    ImGui.SameLine(500);
                    ImGui.TextDisabled("you");
                }
            }
            ImGui.Spacing();
            ImGui.TextDisabled("A kicked client may log in again with its token; revoke the token to keep it out.");
        }

        private void DrawAdminTokens()
        {
            List<TokenInfo> tokens = _sim.Game.Tokens;
            if (tokens.Count == 0) ImGui.TextDisabled("No tokens: only anonymous spectators can join, if allowed.");
            foreach (TokenInfo token in tokens)
            {
                using (new ImGuiTextColor(token.Seat is int seat ? Theme.PlayerTextColor(seat) : Theme.TextSoft))
                    ImGui.TextUnformatted(token.Label.Length > 0 ? token.Label : "(no label)");
                ImGui.SameLine(170);
                ImGui.TextDisabled(DescribeToken(token));
                ImGui.SameLine(360);
                ImGui.TextDisabled(token.Created.ToLocalTime().ToString("yyyy-MM-dd HH:mm"));
                ImGui.SameLine(500);
                if (ImGui.SmallButton($"Revoke##{token.Id}")) Send(new RevokeToken(token.Id));
            }

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.TextUnformatted("New token");
            ImGui.SetNextItemWidth(140);
            ImGui.Combo("##tokenrole", ref _tokenRole, TokenRoles, TokenRoles.Length);
            Role role = TokenRoleValues[_tokenRole];
            if (role is Role.Player or Role.Agent)
            {
                ImGui.SameLine();
                ImGui.SetNextItemWidth(120);
                string[] seats = Enumerable.Range(0, PLAYER_COUNT).Select(s => $"{Theme.PlayerName(s)} (seat {s + 1})").ToArray();
                ImGui.Combo("##tokenseat", ref _tokenSeat, seats, seats.Length);
            }
            ImGui.SameLine();
            ImGui.SetNextItemWidth(150);
            ImGui.InputText("Label##tokenlabel", ref _tokenLabel, 40);
            ImGui.SameLine();
            if (ImGui.Button("Create"))
            {
                Send(new CreateToken(role, role is Role.Player or Role.Agent ? _tokenSeat : null, _tokenLabel.Trim()));
                _tokenLabel = "";
            }
            if (role == Role.Admin && _sim.Game.Server is { AllowRemoteAdmins: false })
                ImGui.TextDisabled("Admin tokens only work on the server's own computer (AllowRemoteAdmins is off).");
        }

        private void DrawAdminServer(ServerInfo server)
        {
            string scheme = server.Secure ? "wss" : "ws";
            ImGui.TextUnformatted($"Listening on {scheme}://{server.BindAddress}:{server.Port}");
            ImGui.TextDisabled(server.BindAddress == "127.0.0.1" ? "Reachable from this computer only." : "Reachable from other computers.");
            if (server.Secure && server.CertificateFingerprint is string fingerprint) DrawFingerprint(fingerprint);
            else ImGui.TextDisabled("No TLS: tokens and moves travel unencrypted.");

            ImGui.Spacing();
            bool anonymous = server.AllowAnonymousSpectators;
            if (ImGui.Checkbox("Anyone may watch without a token", ref anonymous)) Send(new SetServerOptions(anonymous));
            if (anonymous && server.BindAddress != "127.0.0.1")
            {
                using (new ImGuiTextColor(new Color(255, 200, 90)))
                    ImGui.TextWrapped("Watchers see every hand: hidden cards are only hidden in the client.");
            }
            ImGui.TextDisabled(server.AllowRemoteAdmins ? "Admins may log in from other computers." : "Admins log in from the server's computer only.");

            ImGui.Spacing();
            if (_sim is RemoteSimulation)
            {
                if (ImGui.Button("Leave the server", new Vector2(160, 0))) Disconnect();
            }
            else if (_hostServer != null)
            {
                if (ImGui.Button("Stop hosting", new Vector2(160, 0))) StopHosting();
            }
        }

        private static string DescribeToken(TokenInfo token) => token.Role switch
        {
            Role.Player => $"player token for seat {token.Seat + 1}",
            Role.Agent => $"bridge token for seat {token.Seat + 1}",
            _ => $"{token.Role.ToString().ToLower()} token"
        };
    }
}
