using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.WebSockets;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Channels;

namespace Simulation
{
    /// <summary>
    /// Hosts a match over WebSockets: each connection says hello with a token, gets a role, the match's snapshot and then its
    /// events (admins also the connections, the tokens and the server), and sends commands that the match checks against its
    /// role. A client that hosts a game runs one next to its LocalSimulation, both on the same match.
    /// </summary>
    public sealed class SimulationServer : IAsyncDisposable
    {
        private sealed class Connection
        {
            public required WebSocket Socket { get; init; }
            public required SessionInfo Session { get; init; }
            // Not single-reader: only the general unbounded channel can count its queue
            public Channel<string> Outgoing { get; } = Channel.CreateUnbounded<string>();
            public CancellationTokenSource Closing { get; } = new();
            public string? TokenHash { get; init; }
        }

        // A connection that falls this far behind with reading is closed instead of buffering without end
        private const int MaxQueuedMessages = 10_000;
        private const int MaxHelloBytes = 16 * 1024;
        private static readonly TimeSpan HelloTimeout = TimeSpan.FromSeconds(10);

        private readonly Match _match;
        private readonly ServerConfig _config;
        private readonly string? _configPath;
        private readonly X509Certificate2? _certificate;
        private readonly WebApplication _app;
        private readonly object _lock = new();
        private readonly Dictionary<string, Connection> _connections = [];
        private readonly LoginThrottle _throttle;

        // Sessions that may come back with their ID: the token hash they came with and their session
        private readonly Dictionary<string, (string? tokenHash, SessionInfo session)> _known = [];
        private int _nextId;
        private int _pending;

        public int Port { get; private set; }
        public bool Secure => _certificate != null;
        /// <summary>
        /// The address of the match on this machine (an IP, since "localhost" may try IPv6 first, where Kestrel doesn't listen)
        /// </summary>
        public Uri Uri => new($"{(Secure ? "wss" : "ws")}://{(IPAddress.Parse(_config.BindAddress).AddressFamily == System.Net.Sockets.AddressFamily.InterNetworkV6 ? "[::1]" : "127.0.0.1")}:{Port}{Protocol.Path}");
        public ServerConfig Config => _config;
        public string? CertificateFingerprint => _certificate != null ? ServerCertificates.Fingerprint(_certificate) : null;

        public IReadOnlyList<SessionInfo> Sessions
        {
            get { lock (_lock) return _connections.Values.Select(c => c.Session).ToList(); }
        }

        public IReadOnlyList<TokenInfo> Tokens => _config.TokenInfos();

        public ServerInfo Info => new(Port, _config.BindAddress, Secure, CertificateFingerprint, _config.AllowAnonymousSpectators, _config.AllowRemoteAdmins);

        private SimulationServer(Match match, ServerConfig config, string? configPath, X509Certificate2? certificate, WebApplication app)
        {
            _match = match;
            _config = config;
            _configPath = configPath;
            _certificate = certificate;
            _app = app;
            _throttle = new LoginThrottle(config.MaxFailedLogins, TimeSpan.FromSeconds(config.LockoutSeconds));
        }

        /// <param name="configPath">Where token and option changes are saved, or null to keep them in memory</param>
        public static async Task<SimulationServer> StartAsync(Match match, ServerConfig config, string? configPath = null)
        {
            X509Certificate2? certificate = config.Certificate
                ?? (config.CertificatePath != null ? ServerCertificates.LoadOrCreate(config.CertificatePath, config.CertificatePassword) : null);

            WebApplicationBuilder builder = WebApplication.CreateSlimBuilder();
            builder.Logging.ClearProviders();
            if (certificate != null) builder.WebHost.UseKestrelHttpsConfiguration();
            builder.WebHost.UseKestrel(options =>
            {
                options.Limits.MaxConcurrentUpgradedConnections = config.MaxConnections;
                options.Limits.MaxConcurrentConnections = config.MaxConnections + 16;
                options.Limits.MaxRequestHeadersTotalSize = 16 * 1024;
                options.Listen(IPAddress.Parse(config.BindAddress), config.Port, listen =>
                {
                    if (certificate != null) listen.UseHttps(certificate);
                });
            });
            WebApplication app = builder.Build();

            SimulationServer server = new SimulationServer(match, config, configPath, certificate, app);
            app.UseWebSockets(new WebSocketOptions { KeepAliveInterval = TimeSpan.FromSeconds(20), KeepAliveTimeout = TimeSpan.FromSeconds(60) });
            app.Map(Protocol.Path, server.HandleAsync);
            match.HostCommand += server.OnHostCommand;

            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            server.Port = new Uri(address.Replace("0.0.0.0", "localhost").Replace("[::]", "localhost")).Port;

            // A client hosting this match is its admin too: it learns about the server through the match
            server.NotifyAdmins();
            return server;
        }

        private async Task HandleAsync(HttpContext context)
        {
            if (!context.WebSockets.IsWebSocketRequest)
            {
                context.Response.StatusCode = 400;
                await context.Response.WriteAsync("Hexfarer match: connect with a WebSocket.");
                return;
            }

            // Browsers always send an Origin, the clients never do: a web page can't open the match behind the user's back
            if (!string.IsNullOrEmpty(context.Request.Headers.Origin))
            {
                context.Response.StatusCode = 403;
                return;
            }

            if (Interlocked.Increment(ref _pending) > _config.MaxConnections)
            {
                Interlocked.Decrement(ref _pending);
                context.Response.StatusCode = 503;
                return;
            }

            try
            {
                using WebSocket socket = await context.WebSockets.AcceptWebSocketAsync();
                await ServeAsync(context, socket);
            }
            finally
            {
                Interlocked.Decrement(ref _pending);
            }
        }

        private async Task ServeAsync(HttpContext context, WebSocket socket)
        {
            string? helloText;
            using (CancellationTokenSource helloTimeout = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted))
            {
                helloTimeout.CancelAfter(HelloTimeout);
                try
                {
                    helloText = await ReceiveAsync(socket, helloTimeout.Token, MaxHelloBytes);
                }
                catch (Exception e) when (e is WebSocketException or OperationCanceledException)
                {
                    return;
                }
            }
            if (helloText == null) return;

            Hello hello;
            try
            {
                hello = Protocol.DeserializeHello(helloText);
            }
            catch (Exception)
            {
                await RefuseAsync(socket, "Invalid hello.");
                return;
            }
            if (Protocol.VersionMismatch(hello.Version, otherIsServer: false) is string mismatch)
            {
                await RefuseAsync(socket, mismatch);
                return;
            }

            IPAddress address = context.Connection.RemoteIpAddress ?? IPAddress.None;
            (SessionInfo? session, string? tokenHash, string? refusal) = Authenticate(hello, address);
            if (session == null)
            {
                // A pause before each refusal slows down guessing further
                await Task.Delay(250);
                await RefuseAsync(socket, refusal!);
                return;
            }

            Connection connection = new Connection { Socket = socket, Session = session, TokenHash = tokenHash };
            Action<MatchEvent, string?> listener = (e, to) =>
            {
                if (!Match.IsFor(to, session)) return;
                connection.Outgoing.Writer.TryWrite(Protocol.Serialize(e));
                if (connection.Outgoing.Reader.Count > MaxQueuedMessages) connection.Closing.Cancel();
            };

            lock (_lock)
            {
                _connections[session.Id] = connection;
                _known[session.Id] = (tokenHash, session);
            }

            // Welcome and snapshot first (an admin also gets the server and its tokens), then the events from that point on
            _match.Subscribe(session, listener, snapshot =>
            {
                connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new WelcomeEvent(session)));
                connection.Outgoing.Writer.TryWrite(Protocol.Serialize(snapshot));
                if (session.Role == Role.Admin)
                {
                    connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new ServerInfoEvent(Info)));
                    connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new TokensEvent(_config.TokenInfos())));
                }
            });
            SendSessionsToAdmins();

            Task sending = SendLoopAsync(connection);
            try
            {
                while (!connection.Closing.IsCancellationRequested)
                {
                    string? text = await ReceiveAsync(socket, connection.Closing.Token, _config.MaxMessageBytes);
                    if (text == null) break;
                    try
                    {
                        _match.Handle(session, Protocol.DeserializeCommand(text));
                    }
                    catch (Exception e)
                    {
                        connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new ErrorEvent($"Invalid command: {e.Message}")));
                    }
                }
            }
            catch (Exception e) when (e is WebSocketException or OperationCanceledException)
            {
                // The connection broke or was closed
            }
            finally
            {
                _match.Published -= listener;
                lock (_lock) _connections.Remove(session.Id);
                connection.Outgoing.Writer.TryComplete();
                await sending;
                SendSessionsToAdmins();
            }
        }

        /// <summary>
        /// The session a hello asks for, if its token allows it: a token's role and seat, a spectator without a token if
        /// allowed, and the earlier session for a reconnect with the same token. Addresses with too many failed logins are
        /// locked out for a while, and admins only log in from this machine unless the configuration allows other computers.
        /// </summary>
        internal (SessionInfo? session, string? tokenHash, string? refusal) Authenticate(Hello hello, IPAddress address)
        {
            if (address.IsIPv4MappedToIPv6) address = address.MapToIPv4();
            if (_throttle.LockedFor(address) is TimeSpan wait)
                return (null, null, $"Too many failed logins from this address; try again in {Math.Ceiling(wait.TotalSeconds)} s.");

            if (!Enum.IsDefined(hello.Role)) return (null, null, "Unknown role.");
            TokenEntry? token = _config.Find(hello.Token);
            if (hello.Token != null && token == null)
            {
                _throttle.Failed(address);
                return (null, null, "The token is not valid.");
            }

            Role role;
            int? seat = null;
            if (token != null)
            {
                bool allowed = hello.Role == token.Role || hello.Role == Role.Spectator;
                if (!allowed)
                {
                    _throttle.Failed(address);
                    return (null, null, $"The token admits a {token.Role.ToString().ToLower()}, not a {hello.Role.ToString().ToLower()}.");
                }
                role = hello.Role;
                if (role is Role.Player or Role.Agent) seat = token.Seat ?? hello.Seat;
                if (role is Role.Player or Role.Agent && seat is not (>= 0 and < 4)) return (null, null, "A player needs a seat.");
                if (role == Role.Admin && !IPAddress.IsLoopback(address) && !_config.AllowRemoteAdmins)
                    return (null, null, "Admins can't log in from other computers on this server (AllowRemoteAdmins is off).");
                _throttle.Succeeded(address);
            }
            else if (hello.Role == Role.Spectator && _config.AllowAnonymousSpectators)
            {
                role = Role.Spectator;
            }
            else
            {
                return (null, null, hello.Role == Role.Spectator ? "This match needs a token to watch." : "A token is needed for this role.");
            }

            string name = ServerConfig.CleanLabel(hello.Name);
            lock (_lock)
            {
                if (_connections.Count >= _config.MaxConnections) return (null, null, "The server is full.");

                // A reconnect keeps its session, if the same token comes back with it
                if (hello.SessionId != null && _known.TryGetValue(hello.SessionId, out var known) && known.tokenHash == token?.Hash
                    && known.session.Role == role && !_connections.ContainsKey(hello.SessionId))
                    return (known.session with { Address = address.ToString() }, token?.Hash, null);

                string id = $"s{++_nextId}-{ServerConfig.NewToken()[..8]}";
                return (new SessionInfo(id, name.Length > 0 ? name : "Guest", role, seat, address.ToString()), token?.Hash, null);
            }
        }

        #region Admin
        /// <summary>
        /// A new token for a role; the token itself is returned once, the configuration keeps its hash
        /// </summary>
        public (TokenInfo info, string token) CreateToken(Role role, int? seat, string? label)
        {
            (TokenEntry entry, string token) = _config.AddToken(role, seat, label);
            SaveConfig();
            _match.Notify(new TokensEvent(_config.TokenInfos()), Match.Admins);
            return (entry.Info, token);
        }

        /// <summary>
        /// Removes a token and closes every connection that logged in with it
        /// </summary>
        public bool RevokeToken(string id)
        {
            string? hash = _config.RemoveToken(id);
            if (hash == null) return false;
            SaveConfig();

            List<Connection> affected;
            lock (_lock)
            {
                affected = _connections.Values.Where(c => c.TokenHash == hash).ToList();
                foreach (string session in _known.Where(k => k.Value.tokenHash == hash).Select(k => k.Key).ToList()) _known.Remove(session);
            }
            foreach (Connection connection in affected) Close(connection, "Your token was revoked");
            _match.Notify(new TokensEvent(_config.TokenInfos()), Match.Admins);
            return true;
        }

        public void SetAllowAnonymousSpectators(bool allow)
        {
            _config.AllowAnonymousSpectators = allow;
            SaveConfig();
            _match.Notify(new ServerInfoEvent(Info), Match.Admins);
        }

        private void SaveConfig()
        {
            if (_configPath != null) _config.Save(_configPath);
        }

        private void NotifyAdmins()
        {
            _match.Notify(new ServerInfoEvent(Info), Match.Admins);
            _match.Notify(new TokensEvent(_config.TokenInfos()), Match.Admins);
            SendSessionsToAdmins();
        }

        private void SendSessionsToAdmins()
        {
            List<SessionInfo> sessions;
            lock (_lock) sessions = _connections.Values.Select(c => c.Session).ToList();
            _match.Notify(new SessionsEvent(sessions), Match.Admins);
        }

        /// <summary>
        /// Runs a command the match passed on after checking that the sender is an admin. Called while the match is locked.
        /// </summary>
        private void OnHostCommand(SessionInfo sender, Command command)
        {
            try
            {
                switch (command)
                {
                    case KickSession kick:
                        Connection? connection;
                        lock (_lock)
                        {
                            _connections.TryGetValue(kick.SessionId, out connection);
                            _known.Remove(kick.SessionId);
                        }
                        if (connection != null) Close(connection, $"Closed by {sender.Name}");
                        break;

                    case CreateToken create:
                        (TokenInfo info, string token) = CreateToken(create.Role, create.Seat, create.Label);
                        _match.Notify(new TokenCreatedEvent(info, token), sender.Id);
                        break;

                    case RevokeToken revoke:
                        if (!RevokeToken(revoke.Id)) _match.Notify(new ErrorEvent("There is no such token."), sender.Id);
                        break;

                    case SetServerOptions options:
                        SetAllowAnonymousSpectators(options.AllowAnonymousSpectators);
                        break;
                }
            }
            catch (ArgumentException e)
            {
                _match.Notify(new ErrorEvent(e.Message), sender.Id);
            }
        }

        private static void Close(Connection connection, string reason)
        {
            connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new ClosedEvent(reason)));
            connection.Outgoing.Writer.TryComplete();
            connection.Closing.CancelAfter(TimeSpan.FromSeconds(2));
        }
        #endregion

        private static async Task SendLoopAsync(Connection connection)
        {
            try
            {
                await foreach (string message in connection.Outgoing.Reader.ReadAllAsync(connection.Closing.Token))
                {
                    if (connection.Socket.State != WebSocketState.Open) break;
                    await connection.Socket.SendAsync(Encoding.UTF8.GetBytes(message), WebSocketMessageType.Text, true, connection.Closing.Token);
                }
                if (connection.Socket.State == WebSocketState.Open)
                    await connection.Socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "Closed", CancellationToken.None);
            }
            catch (Exception e) when (e is WebSocketException or ObjectDisposedException or OperationCanceledException)
            {
                // The connection is gone
            }
            finally
            {
                connection.Closing.Cancel();
            }
        }

        private static async Task RefuseAsync(WebSocket socket, string reason)
        {
            try
            {
                await socket.SendAsync(Encoding.UTF8.GetBytes(Protocol.Serialize(new ErrorEvent(reason))), WebSocketMessageType.Text, true, CancellationToken.None);
                await socket.CloseAsync(WebSocketCloseStatus.PolicyViolation, "Refused", CancellationToken.None);
            }
            catch (WebSocketException)
            {
            }
        }

        /// <summary>
        /// One whole text message, or null once the socket closes. A message over the limit closes the socket.
        /// </summary>
        internal static async Task<string?> ReceiveAsync(WebSocket socket, CancellationToken cancellation, int maxBytes)
        {
            byte[] buffer = new byte[64 * 1024];
            using MemoryStream message = new MemoryStream();
            while (true)
            {
                WebSocketReceiveResult result = await socket.ReceiveAsync(buffer, cancellation);
                if (result.MessageType == WebSocketMessageType.Close) return null;
                if (message.Length + result.Count > maxBytes)
                {
                    try
                    {
                        await socket.CloseAsync(WebSocketCloseStatus.MessageTooBig, "Message too big", CancellationToken.None);
                    }
                    catch (WebSocketException)
                    {
                    }
                    return null;
                }
                message.Write(buffer, 0, result.Count);
                if (result.EndOfMessage) return Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length);
            }
        }

        public async ValueTask DisposeAsync()
        {
            _match.HostCommand -= OnHostCommand;
            List<Connection> connections;
            lock (_lock) connections = _connections.Values.ToList();
            foreach (Connection connection in connections)
            {
                connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new ClosedEvent("The server stopped")));
                connection.Outgoing.Writer.TryComplete();
            }
            _match.Notify(new ServerInfoEvent(null), Match.Admins);
            await Task.Delay(100);
            await _app.StopAsync();
            await _app.DisposeAsync();
        }
    }

    /// <summary>
    /// Counts failed logins per address and locks an address out for a while after too many
    /// </summary>
    internal sealed class LoginThrottle(int maxFailures, TimeSpan lockout)
    {
        private readonly Dictionary<IPAddress, (int failures, DateTime first, DateTime? lockedUntil)> _addresses = [];

        public TimeSpan? LockedFor(IPAddress address)
        {
            lock (_addresses)
            {
                if (!_addresses.TryGetValue(address, out var entry) || entry.lockedUntil is not DateTime until) return null;
                TimeSpan left = until - DateTime.UtcNow;
                if (left > TimeSpan.Zero) return left;
                _addresses.Remove(address);
                return null;
            }
        }

        public void Failed(IPAddress address)
        {
            lock (_addresses)
            {
                DateTime now = DateTime.UtcNow;
                var entry = _addresses.TryGetValue(address, out var e) && now - e.first < lockout ? e : (0, now, null);
                entry.failures++;
                if (entry.failures >= maxFailures) entry.lockedUntil = now + lockout;
                _addresses[address] = entry;

                // Old entries go, so the table can't grow without end
                if (_addresses.Count > 10_000)
                    foreach (IPAddress old in _addresses.Where(a => now - a.Value.first > lockout && a.Value.lockedUntil is null).Select(a => a.Key).ToList())
                        _addresses.Remove(old);
            }
        }

        public void Succeeded(IPAddress address)
        {
            lock (_addresses)
            {
                if (_addresses.TryGetValue(address, out var entry) && entry.lockedUntil == null) _addresses.Remove(address);
            }
        }
    }
}
