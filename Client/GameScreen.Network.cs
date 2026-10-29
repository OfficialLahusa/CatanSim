using Client.Logging;
using Client.UI;
using Common;
using ImGuiNET;
using SFML.Graphics;
using Simulation;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using Color = SFML.Graphics.Color;

namespace Client
{
    /// <summary>
    /// Playing over the network: hosting the local match for others (a SimulationServer next to the local simulation, with a
    /// token per human seat), joining a match on a server, and the admin's view of the connections
    /// </summary>
    public partial class GameScreen
    {
        private bool _showNetworkWindow;

        // Hosting: the server on the local match and the tokens handed out (kept in plain text only here, for showing them)
        private SimulationServer? _hostServer;
        private Task<SimulationServer>? _hostStarting;
        private readonly List<(string label, string token)> _hostTokens = [];
        private int _hostPort = 17017;
        private bool _hostOnNetwork = true;
        private bool _hostOpenToSpectators = true;
        private string? _networkError;

        // Joining
        private Task<RemoteSimulation>? _connecting;
        private Hello? _lastHello;
        private Uri? _lastUri;
        private string _joinAddress = "localhost:17017";
        private string _joinName = Environment.UserName;
        private string _joinToken = "";
        private int _joinRole = 0;
        private int _joinSeat = 0;
        private static readonly string[] JoinRoles = ["Player", "Spectator", "Admin", "Agent bridge"];
        private static readonly Role[] JoinRoleValues = [Role.Player, Role.Spectator, Role.Admin, Role.Agent];
        private string? _disconnectReason;

        private string NetworkStatus => _sim switch
        {
            RemoteSimulation remote when remote.IsConnected => remote.Session?.Role switch
            {
                Role.Player => $"Online · seat {remote.Session.Seat + 1}",
                Role.Agent => $"Online · bridge for seat {remote.Session.Seat + 1}",
                Role.Admin => "Online · admin",
                _ => "Online · watching"
            },
            RemoteSimulation => "Disconnected",
            _ when _hostServer != null => $"Hosting · {_hostServer.Sessions.Count} connected",
            _ => "Local game"
        };

        /// <summary>
        /// Finishes a pending connection or server start
        /// </summary>
        private void UpdateNetwork()
        {
            if (_connecting is { IsCompleted: true } connecting)
            {
                _connecting = null;
                if (connecting.IsCompletedSuccessfully)
                {
                    StopHosting();
                    ReplaceSimulation(connecting.Result);
                    _networkError = null;
                    _disconnectReason = null;
                    _eventLog.WriteLine(new ColoredStrEntry(connecting.Result.Description, Color.Green));
                }
                else
                {
                    _networkError = connecting.Exception?.GetBaseException().Message ?? "The connection failed.";
                }
            }

            if (_hostStarting is { IsCompleted: true } starting)
            {
                _hostStarting = null;
                if (starting.IsCompletedSuccessfully)
                {
                    _hostServer = starting.Result;
                    _networkError = null;
                    if (_sim is LocalSimulation local) local.Description = $"Hosting on port {_hostServer.Port}";
                }
                else
                {
                    _hostTokens.Clear();
                    _networkError = starting.Exception?.GetBaseException().Message ?? "The server did not start.";
                }
            }
        }

        /// <summary>
        /// Switches to another simulation; a local match that is left behind stops
        /// </summary>
        private void ReplaceSimulation(ISimulation simulation)
        {
            ISimulation old = _sim;
            _sim = simulation;
            old.Dispose();
            if (old is LocalSimulation local) local.Match.Dispose();

            _mode = InteractionMode.Play;
            ClearHumanInput();
            _stateVersion++;
            ProcessSimulation();
        }

        private void StartHosting()
        {
            if (_sim is not LocalSimulation local || _hostServer != null || _hostStarting != null) return;

            // A player token per human seat and one for watching; the host itself is the admin
            _hostTokens.Clear();
            List<TokenEntry> entries = [];
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                if (!IsHuman(seat)) continue;
                string token = ServerConfig.NewToken();
                _hostTokens.Add(($"{Theme.PlayerName(seat)} (seat {seat + 1})", token));
                entries.Add(new TokenEntry(ServerConfig.HashToken(token), Role.Player, seat));
            }
            if (!_hostOpenToSpectators)
            {
                string token = ServerConfig.NewToken();
                _hostTokens.Add(("Spectators", token));
                entries.Add(new TokenEntry(ServerConfig.HashToken(token), Role.Spectator, null));
            }

            ServerConfig config = new ServerConfig
            {
                Port = _hostPort,
                BindAddress = _hostOnNetwork ? "0.0.0.0" : "127.0.0.1",
                AllowAnonymousSpectators = _hostOpenToSpectators,
                Tokens = entries
            };
            _hostStarting = SimulationServer.StartAsync(local.Match, config);
        }

        private void StopHosting()
        {
            if (_hostServer == null) return;
            SimulationServer server = _hostServer;
            _hostServer = null;
            _hostTokens.Clear();
            server.DisposeAsync().AsTask().Wait(3000);
            if (_sim is LocalSimulation local) local.Description = "Local game";
        }

        private void Connect()
        {
            if (_connecting != null) return;
            string address = _joinAddress.Trim();
            if (!address.Contains("://")) address = "ws://" + address;
            if (!Uri.TryCreate(address, UriKind.Absolute, out Uri? uri))
            {
                _networkError = "That is not an address (like 192.168.1.20:17017).";
                return;
            }
            if (uri.AbsolutePath is "" or "/") uri = new Uri(uri, Protocol.Path);
            if (uri.IsDefaultPort && !_joinAddress.Contains(':')) uri = new UriBuilder(uri) { Port = 17017 }.Uri;

            Role role = JoinRoleValues[_joinRole];
            _lastHello = new Hello(_joinName, string.IsNullOrWhiteSpace(_joinToken) ? null : _joinToken.Trim(), role, role is Role.Player or Role.Agent ? _joinSeat : null, null);
            _lastUri = uri;
            _networkError = null;
            _connecting = RemoteSimulation.ConnectAsync(uri, _lastHello);
        }

        private void Reconnect()
        {
            if (_lastHello == null || _lastUri == null || _connecting != null) return;
            _connecting = RemoteSimulation.ConnectAsync(_lastUri, _lastHello with { SessionId = _sim.Session?.Id });
        }

        /// <summary>
        /// Leaves the server; the game goes on locally from where it was, with its moves
        /// </summary>
        private void Disconnect()
        {
            if (_sim is not RemoteSimulation) return;
            GameState state = new GameState(State);
            List<HistoryEntry> history = History.ToList(), redo = Redo.ToList();
            SeatKind[] seats = (SeatKind[])Seats.Clone();
            ReplaceSimulation(StartLocal(state, seats, history, redo));
            _disconnectReason = null;
            _eventLog.WriteLine(new ColoredStrEntry("Left the server; the game continues here", Color.Yellow));
        }

        private void OnConnectionClosed(string reason)
        {
            if (_sim is RemoteSimulation remote && !remote.IsConnected)
            {
                _disconnectReason = remote.ClosedReason ?? reason;
                _showNetworkWindow = true;
                _eventLog.WriteLine(new ColoredStrEntry($"Disconnected: {_disconnectReason}", Color.Red));
            }
        }

        /// <summary>
        /// The network chip in the top bar: where the match runs, a click opens the network window
        /// </summary>
        private void DrawNetworkChip(ImDrawListPtr dl, Vector2 min)
        {
            string text = NetworkStatus;
            float width = Theme.TextSize(Theme.SemiBold15, text).X + 40;
            bool clicked = Theme.HitArea("network", min, new Vector2(width, 32), out bool hovered);
            Color dot = _sim switch
            {
                RemoteSimulation remote => remote.IsConnected ? new Color(80, 220, 120) : new Color(255, 96, 96),
                _ when _hostServer != null => new Color(80, 220, 120),
                _ => Theme.TextFaint
            };
            dl.RoundRectFilled(min, min + new Vector2(width, 32), Theme.U32(hovered ? Theme.PanelRaised : Theme.Panel), 16);
            dl.RoundRect(min, min + new Vector2(width, 32), Theme.U32(Theme.Border), 16);
            dl.AddCircleFilled(min + new Vector2(17, 16), 5, Theme.U32(dot), 12);
            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(29, 6), Theme.TextSoft, text);
            if (clicked) _showNetworkWindow = !_showNetworkWindow;
        }

        private static IEnumerable<string> LocalAddresses()
        {
            try
            {
                return Dns.GetHostAddresses(Dns.GetHostName()).Where(a => a.AddressFamily == AddressFamily.InterNetwork && !IPAddress.IsLoopback(a)).Select(a => a.ToString());
            }
            catch (SocketException)
            {
                return [];
            }
        }

        /// <summary>
        /// Hosts the local game from the command line: --host PORT for the network, --host localhost:PORT for this computer only
        /// </summary>
        public void HostFromCommandLine(string address)
        {
            string[] parts = address.Split(':');
            _hostPort = int.Parse(parts[^1]);
            _hostOnNetwork = parts.Length == 1;
            StartHosting();
            _showNetworkWindow = true;
        }

        /// <summary>
        /// Joins a server from the command line (--join ADDRESS with --name, --role, --seat and --token)
        /// </summary>
        public void JoinFromCommandLine(string address, string? name, string? role, int? seat, string? token)
        {
            _joinAddress = address;
            if (name != null) _joinName = name;
            if (role != null) _joinRole = Math.Max(0, Array.FindIndex(JoinRoleValues, r => r.ToString().Equals(role, StringComparison.OrdinalIgnoreCase)));
            if (seat is int s) _joinSeat = Math.Clamp(s - 1, 0, PLAYER_COUNT - 1);
            _joinToken = token ?? "";
            Connect();
        }

        public void ShowNetworkWindow() => _showNetworkWindow = true;

        private void DrawNetworkWindow()
        {
            if (!_showNetworkWindow) return;
            ImGui.SetNextWindowPos(new Vector2(Display.X / 2 - 260, 90), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.Always);
            if (!ImGui.Begin("Network", ref _showNetworkWindow, ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.End();
                return;
            }

            ImGui.PushFont(Theme.SemiBold17);
            ImGui.TextUnformatted(_sim.Description);
            ImGui.PopFont();
            if (_networkError != null)
            {
                using (new ImGuiTextColor(new Color(255, 120, 120))) ImGui.TextWrapped(_networkError);
            }
            ImGui.Separator();

            switch (_sim)
            {
                case RemoteSimulation remote:
                    DrawRemoteSection(remote);
                    break;
                case LocalSimulation:
                    if (_hostServer != null) DrawHostingSection(_hostServer);
                    else DrawHostSection();
                    ImGui.Separator();
                    DrawJoinSection();
                    break;
            }
            ImGui.End();
        }

        private void DrawHostSection()
        {
            ImGui.PushFont(Theme.Bold15);
            ImGui.TextUnformatted("Host this game");
            ImGui.PopFont();
            ImGui.TextDisabled("Others join the match on this computer; you stay its admin.");
            ImGui.SetNextItemWidth(120);
            ImGui.InputInt("Port", ref _hostPort);
            _hostPort = Math.Clamp(_hostPort, 1024, 65535);
            ImGui.Checkbox("Reachable from other computers", ref _hostOnNetwork);
            ImGui.Checkbox("Anyone may watch without a token", ref _hostOpenToSpectators);
            ImGui.TextDisabled($"Each human seat gets a token ({HumanCount()} now); set seats to Human first for more players.");
            if (_hostStarting != null) ImGui.TextUnformatted("Starting...");
            else if (ImGui.Button("Start hosting", new Vector2(160, 0))) StartHosting();
        }

        private void DrawHostingSection(SimulationServer server)
        {
            ImGui.PushFont(Theme.Bold15);
            ImGui.TextUnformatted($"Hosting on port {server.Port}");
            ImGui.PopFont();
            string addresses = server.Config.BindAddress == "0.0.0.0" ? string.Join(", ", LocalAddresses().Select(a => $"{a}:{server.Port}")) : $"localhost:{server.Port}";
            ImGui.TextWrapped($"Address: {(addresses.Length > 0 ? addresses : $"localhost:{server.Port}")}");

            ImGui.Spacing();
            ImGui.TextUnformatted("Tokens (give each player theirs):");
            foreach ((string label, string token) in _hostTokens)
            {
                ImGui.TextUnformatted($"{label}:");
                ImGui.SameLine(190);
                ImGui.TextUnformatted(token);
                ImGui.SameLine();
                if (ImGui.SmallButton($"Copy##{label}")) SFML.Window.Clipboard.Contents = token;
            }
            if (server.Config.AllowAnonymousSpectators) ImGui.TextDisabled("Spectators join without a token.");

            ImGui.Spacing();
            DrawSessions(server.Sessions);

            ImGui.Spacing();
            if (ImGui.Button("Stop hosting", new Vector2(160, 0))) StopHosting();
        }

        private void DrawSessions(IReadOnlyList<SessionInfo> sessions)
        {
            ImGui.TextUnformatted(sessions.Count == 0 ? "Nobody connected yet." : "Connected:");
            foreach (SessionInfo session in sessions)
            {
                string what = session.Role switch
                {
                    Role.Player => $"seat {session.Seat + 1}",
                    Role.Agent => $"bridge, seat {session.Seat + 1}",
                    _ => session.Role.ToString().ToLower()
                };
                using (new ImGuiTextColor(session.Seat is int seat ? Theme.PlayerTextColor(seat) : Theme.TextSoft))
                    ImGui.TextUnformatted($"{session.Name}");
                ImGui.SameLine(160);
                ImGui.TextDisabled($"{what}  {session.Address}");
                if (IsAdmin && session.Id != _sim.Session?.Id)
                {
                    ImGui.SameLine(440);
                    if (ImGui.SmallButton($"Kick##{session.Id}")) _sim.Send(new KickSession(session.Id));
                }
            }
        }

        private void DrawJoinSection()
        {
            ImGui.PushFont(Theme.Bold15);
            ImGui.TextUnformatted("Join a game");
            ImGui.PopFont();
            ImGui.SetNextItemWidth(260);
            ImGui.InputText("Address", ref _joinAddress, 200);
            ImGui.SetNextItemWidth(260);
            ImGui.InputText("Name", ref _joinName, 40);
            ImGui.SetNextItemWidth(160);
            ImGui.Combo("Join as", ref _joinRole, JoinRoles, JoinRoles.Length);
            if (JoinRoleValues[_joinRole] is Role.Player or Role.Agent)
            {
                ImGui.SetNextItemWidth(160);
                string[] seats = Enumerable.Range(0, PLAYER_COUNT).Select(s => $"{s + 1} · {Theme.PlayerName(s)}").ToArray();
                ImGui.Combo("Seat", ref _joinSeat, seats, seats.Length);
            }
            ImGui.SetNextItemWidth(260);
            ImGui.InputText("Token", ref _joinToken, 100, ImGuiInputTextFlags.Password);
            ImGui.TextDisabled("Watching may work without a token; playing needs the token of your seat.");
            if (_connecting != null) ImGui.TextUnformatted("Connecting...");
            else if (ImGui.Button("Connect", new Vector2(160, 0))) Connect();
        }

        private void DrawRemoteSection(RemoteSimulation remote)
        {
            if (!remote.IsConnected)
            {
                using (new ImGuiTextColor(new Color(255, 120, 120)))
                    ImGui.TextWrapped($"The connection ended: {_disconnectReason ?? remote.ClosedReason ?? "closed"}.");
                if (_connecting != null) ImGui.TextUnformatted("Reconnecting...");
                else
                {
                    if (ImGui.Button("Reconnect", new Vector2(160, 0))) Reconnect();
                    ImGui.SameLine();
                    if (ImGui.Button("Continue here alone", new Vector2(200, 0))) Disconnect();
                }
                return;
            }

            if (remote.Session?.Role == Role.Admin) DrawSessions(remote.Game.Sessions);
            else ImGui.TextDisabled(remote.Session?.Role == Role.Spectator ? "You are watching: hands stay hidden." : "You play your seat; the admin controls the match.");
            ImGui.Spacing();
            if (ImGui.Button("Leave", new Vector2(160, 0))) Disconnect();
        }
    }
}
