using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Channels;

namespace Simulation
{
    /// <summary>
    /// Hosts a match over WebSockets: each connection says hello with a token, gets a role, the match's snapshot and then its
    /// events (admins also the list of connections), and sends commands that the match checks against its role. A client that
    /// hosts a game runs one next to its LocalSimulation, both on the same match.
    /// </summary>
    public sealed class SimulationServer : IAsyncDisposable
    {
        private sealed class Connection
        {
            public required WebSocket Socket { get; init; }
            public required SessionInfo Session { get; init; }
            public Channel<string> Outgoing { get; } = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });
            public CancellationTokenSource Closing { get; } = new();
            public string? TokenHash { get; init; }
        }

        private readonly Match _match;
        private readonly ServerConfig _config;
        private readonly WebApplication _app;
        private readonly object _lock = new();
        private readonly Dictionary<string, Connection> _connections = [];

        // Sessions that may come back with their ID: the token hash they came with and their session
        private readonly Dictionary<string, (string? tokenHash, SessionInfo session)> _known = [];
        private int _nextId;

        public int Port { get; private set; }
        public Uri Uri => new($"ws://localhost:{Port}{Protocol.Path}");
        public ServerConfig Config => _config;

        public IReadOnlyList<SessionInfo> Sessions
        {
            get { lock (_lock) return _connections.Values.Select(c => c.Session).ToList(); }
        }

        private SimulationServer(Match match, ServerConfig config, WebApplication app)
        {
            _match = match;
            _config = config;
            _app = app;
        }

        public static async Task<SimulationServer> StartAsync(Match match, ServerConfig config)
        {
            WebApplicationBuilder builder = WebApplication.CreateSlimBuilder();
            builder.Logging.ClearProviders();
            builder.WebHost.UseKestrel(options => options.Listen(IPAddress.Parse(config.BindAddress), config.Port));
            WebApplication app = builder.Build();

            SimulationServer server = new SimulationServer(match, config, app);
            app.UseWebSockets(new WebSocketOptions { KeepAliveInterval = TimeSpan.FromSeconds(20) });
            app.Map(Protocol.Path, server.HandleAsync);
            match.HostCommand += server.OnHostCommand;

            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            server.Port = new Uri(address.Replace("0.0.0.0", "localhost").Replace("[::]", "localhost")).Port;
            return server;
        }

        private async Task HandleAsync(HttpContext context)
        {
            if (!context.WebSockets.IsWebSocketRequest)
            {
                context.Response.StatusCode = 400;
                await context.Response.WriteAsync("CatanSim match: connect with a WebSocket.");
                return;
            }

            using WebSocket socket = await context.WebSockets.AcceptWebSocketAsync();
            string? helloText = await ReceiveAsync(socket, context.RequestAborted);
            if (helloText == null) return;

            Hello hello;
            try
            {
                hello = Protocol.DeserializeHello(helloText);
            }
            catch (Exception e)
            {
                await RefuseAsync(socket, $"Invalid hello: {e.Message}");
                return;
            }

            string address = context.Connection.RemoteIpAddress?.ToString() ?? "";
            (SessionInfo? session, string? tokenHash, string? refusal) = Authenticate(hello, address);
            if (session == null)
            {
                await RefuseAsync(socket, refusal!);
                return;
            }

            Connection connection = new Connection { Socket = socket, Session = session, TokenHash = tokenHash };
            Action<MatchEvent, string?> listener = (e, to) =>
            {
                if (to == null || to == session.Id) connection.Outgoing.Writer.TryWrite(Protocol.Serialize(e));
            };

            lock (_lock)
            {
                _connections[session.Id] = connection;
                _known[session.Id] = (tokenHash, session);
            }

            // Welcome and snapshot first, then the events from that point on
            _match.Subscribe(listener, snapshot =>
            {
                connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new WelcomeEvent(session)));
                connection.Outgoing.Writer.TryWrite(Protocol.Serialize(snapshot));
            });
            SendSessionsToAdmins();

            Task sending = SendLoopAsync(connection);
            try
            {
                while (!connection.Closing.IsCancellationRequested)
                {
                    string? text = await ReceiveAsync(socket, connection.Closing.Token);
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
        /// allowed, and the earlier session for a reconnect with the same token
        /// </summary>
        private (SessionInfo? session, string? tokenHash, string? refusal) Authenticate(Hello hello, string address)
        {
            TokenEntry? token = _config.Find(hello.Token);
            if (hello.Token != null && token == null) return (null, null, "The token is not valid.");

            Role role;
            int? seat = null;
            if (token != null)
            {
                bool allowed = hello.Role == token.Role || hello.Role == Role.Spectator;
                if (!allowed) return (null, null, $"The token admits a {token.Role.ToString().ToLower()}, not a {hello.Role.ToString().ToLower()}.");
                role = hello.Role;
                if (role is Role.Player or Role.Agent) seat = token.Seat ?? hello.Seat;
                if (role is Role.Player or Role.Agent && seat is not (>= 0 and < 4)) return (null, null, "A player needs a seat.");
            }
            else if (hello.Role == Role.Spectator && _config.AllowAnonymousSpectators)
            {
                role = Role.Spectator;
            }
            else
            {
                return (null, null, hello.Role == Role.Spectator ? "This match needs a token to watch." : "A token is needed for this role.");
            }

            lock (_lock)
            {
                // A reconnect keeps its session, if the same token comes back with it
                if (hello.SessionId != null && _known.TryGetValue(hello.SessionId, out var known) && known.tokenHash == token?.Hash
                    && known.session.Role == role && !_connections.ContainsKey(hello.SessionId))
                    return (known.session with { Address = address }, token?.Hash, null);

                string id = $"s{++_nextId}-{ServerConfig.NewToken()[..6]}";
                return (new SessionInfo(id, string.IsNullOrWhiteSpace(hello.Name) ? "Guest" : hello.Name.Trim(), role, seat, address), token?.Hash, null);
            }
        }

        private void SendSessionsToAdmins()
        {
            lock (_lock)
            {
                string message = Protocol.Serialize(new SessionsEvent(_connections.Values.Select(c => c.Session).ToList()));
                foreach (Connection connection in _connections.Values.Where(c => c.Session.Role == Role.Admin))
                    connection.Outgoing.Writer.TryWrite(message);
            }
        }

        private void OnHostCommand(SessionInfo sender, Command command)
        {
            if (command is not KickSession kick) return;
            Connection? connection;
            lock (_lock)
            {
                _connections.TryGetValue(kick.SessionId, out connection);
                _known.Remove(kick.SessionId);
            }
            if (connection == null) return;
            connection.Outgoing.Writer.TryWrite(Protocol.Serialize(new ClosedEvent($"Closed by {sender.Name}")));
            connection.Outgoing.Writer.TryComplete();
            connection.Closing.CancelAfter(TimeSpan.FromSeconds(2));
        }

        private static async Task SendLoopAsync(Connection connection)
        {
            try
            {
                await foreach (string message in connection.Outgoing.Reader.ReadAllAsync())
                {
                    if (connection.Socket.State != WebSocketState.Open) break;
                    await connection.Socket.SendAsync(Encoding.UTF8.GetBytes(message), WebSocketMessageType.Text, true, CancellationToken.None);
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
        /// One whole text message, or null once the socket closes
        /// </summary>
        internal static async Task<string?> ReceiveAsync(WebSocket socket, CancellationToken cancellation)
        {
            byte[] buffer = new byte[64 * 1024];
            using MemoryStream message = new MemoryStream();
            while (true)
            {
                WebSocketReceiveResult result = await socket.ReceiveAsync(buffer, cancellation);
                if (result.MessageType == WebSocketMessageType.Close) return null;
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
            await Task.Delay(100);
            await _app.StopAsync();
            await _app.DisposeAsync();
        }
    }
}
