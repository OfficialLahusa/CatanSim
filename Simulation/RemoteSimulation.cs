using Agents.MCTS;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Channels;

namespace Simulation
{
    /// <summary>
    /// A match on a server: commands go out over the WebSocket, events come back and keep Game up to date
    /// </summary>
    public sealed class RemoteSimulation : ISimulation
    {
        private readonly ClientWebSocket _socket;
        private readonly Uri _uri;
        private readonly ConcurrentQueue<MatchEvent> _events = new();
        private readonly Channel<string> _outgoing = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });
        private readonly CancellationTokenSource _closing = new();
        private readonly Task _receiving, _sending;

        public SessionInfo? Session { get; private set; }
        public ClientGame Game { get; } = new ClientGame();
        public bool IsConnected { get; private set; } = true;
        public MCTSAgent? ThinkingAgent => null;

        /// <summary>
        /// Refusals and failures the server reported, oldest first
        /// </summary>
        public List<string> Errors { get; } = [];

        /// <summary>
        /// Why the connection ended, once it has
        /// </summary>
        public string? ClosedReason { get; private set; }

        public string Description => IsConnected
            ? $"Connected to {_uri.Host}:{_uri.Port} as {Session?.Role.ToString().ToLower()}" + (Session?.Seat is int seat ? $" of seat {seat + 1}" : "")
            : $"Disconnected from {_uri.Host}:{_uri.Port}" + (ClosedReason != null ? $" ({ClosedReason})" : "");

        private RemoteSimulation(ClientWebSocket socket, Uri uri, SessionInfo session)
        {
            _socket = socket;
            _uri = uri;
            Session = session;
            _receiving = Task.Run(ReceiveLoopAsync);
            _sending = Task.Run(SendLoopAsync);
        }

        /// <summary>
        /// Connects and says hello; throws if the server refuses (with its reason) or can't be reached
        /// </summary>
        public static async Task<RemoteSimulation> ConnectAsync(Uri uri, Hello hello, CancellationToken cancellation = default)
        {
            ClientWebSocket socket = new ClientWebSocket();
            socket.Options.KeepAliveInterval = TimeSpan.FromSeconds(20);
            try
            {
                using CancellationTokenSource timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
                timeout.CancelAfter(TimeSpan.FromSeconds(10));
                await socket.ConnectAsync(uri, timeout.Token);
                await socket.SendAsync(Encoding.UTF8.GetBytes(Protocol.Serialize(hello)), WebSocketMessageType.Text, true, timeout.Token);

                string? first = await SimulationServer.ReceiveAsync(socket, timeout.Token);
                MatchEvent? welcome = first != null ? Protocol.DeserializeEvent(first) : null;
                return welcome switch
                {
                    WelcomeEvent w => new RemoteSimulation(socket, uri, w.Session),
                    ErrorEvent error => throw new InvalidOperationException(error.Message),
                    _ => throw new InvalidOperationException("The server closed the connection.")
                };
            }
            catch
            {
                socket.Dispose();
                throw;
            }
        }

        public void Send(Command command)
        {
            if (!_outgoing.Writer.TryWrite(Protocol.Serialize(command)))
                _events.Enqueue(new ErrorEvent("Not connected."));
        }

        public List<MatchEvent> ProcessEvents()
        {
            List<MatchEvent> events = [];
            while (_events.TryDequeue(out MatchEvent? e))
            {
                switch (e)
                {
                    case ErrorEvent error: Errors.Add(error.Message); break;
                    case WelcomeEvent welcome: Session = welcome.Session; break;
                    case ClosedEvent closed: ClosedReason ??= closed.Reason; break;
                }
                Game.Apply(e);
                events.Add(e);
            }
            return events;
        }

        private async Task ReceiveLoopAsync()
        {
            try
            {
                while (!_closing.IsCancellationRequested)
                {
                    string? text = await SimulationServer.ReceiveAsync(_socket, _closing.Token);
                    if (text == null) break;
                    _events.Enqueue(Protocol.DeserializeEvent(text));
                }
            }
            catch (Exception e) when (e is WebSocketException or OperationCanceledException or ObjectDisposedException)
            {
                // The connection broke or this side closed it
            }
            catch (Exception e)
            {
                _events.Enqueue(new ErrorEvent($"Could not read a message from the server: {e.Message}"));
            }
            finally
            {
                IsConnected = false;
                _outgoing.Writer.TryComplete();
                _events.Enqueue(new ClosedEvent("The connection closed"));
            }
        }

        private async Task SendLoopAsync()
        {
            try
            {
                await foreach (string message in _outgoing.Reader.ReadAllAsync(_closing.Token))
                    await _socket.SendAsync(Encoding.UTF8.GetBytes(message), WebSocketMessageType.Text, true, _closing.Token);
            }
            catch (Exception e) when (e is WebSocketException or OperationCanceledException or ObjectDisposedException)
            {
            }
        }

        public void Dispose()
        {
            _outgoing.Writer.TryComplete();
            try
            {
                if (_socket.State == WebSocketState.Open)
                    _socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "Bye", CancellationToken.None).Wait(1000);
            }
            catch (Exception)
            {
                // Closing is best effort
            }
            _closing.Cancel();
            Task.WaitAll([_receiving, _sending], 2000);
            _socket.Dispose();
        }
    }
}
