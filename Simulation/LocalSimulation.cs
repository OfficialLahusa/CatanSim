using Agents.MCTS;
using System.Collections.Concurrent;

namespace Simulation
{
    /// <summary>
    /// What a client plays through: the match in this process or one on a server. Commands go out with Send; the match's
    /// events arrive in the background and take effect on Game when ProcessEvents is called (on the UI thread).
    /// </summary>
    public interface ISimulation : IDisposable
    {
        /// <summary>
        /// This client's connection: its role and seat
        /// </summary>
        SessionInfo? Session { get; }

        ClientGame Game { get; }

        bool IsConnected { get; }

        /// <summary>
        /// Where the match runs, for the UI
        /// </summary>
        string Description { get; }

        void Send(Command command);

        /// <summary>
        /// Applies the events that arrived to Game and returns them, oldest first
        /// </summary>
        List<MatchEvent> ProcessEvents();

        /// <summary>
        /// The agent searching right now, when the match runs in this process (null over the network)
        /// </summary>
        MCTSAgent? ThinkingAgent { get; }
    }

    /// <summary>
    /// The match in this process: the client is its admin. Other clients can join the same match through a SimulationServer.
    /// </summary>
    public sealed class LocalSimulation : ISimulation
    {
        private readonly ConcurrentQueue<MatchEvent> _events = new();
        private readonly Action<MatchEvent, string?> _listener;

        public Match Match { get; }
        public SessionInfo? Session { get; } = new SessionInfo("local", "Local", Role.Admin, null);
        public ClientGame Game { get; } = new ClientGame();
        public bool IsConnected => true;
        public string Description { get; set; } = "Local game";
        public MCTSAgent? ThinkingAgent => Match.ThinkingAgent;

        public LocalSimulation(Match match)
        {
            Match = match;
            _listener = (e, to) =>
            {
                if (to == null || to == Session!.Id) _events.Enqueue(e);
            };

            // The snapshot and the listener under one lock, so no event is missed or doubled
            match.Subscribe(_listener, snapshot => _events.Enqueue(snapshot));
        }

        public void Send(Command command) => Match.Handle(Session!, command);

        public List<MatchEvent> ProcessEvents()
        {
            List<MatchEvent> events = [];
            while (_events.TryDequeue(out MatchEvent? e))
            {
                Game.Apply(e);
                events.Add(e);
            }
            return events;
        }

        public void Dispose()
        {
            Match.Published -= _listener;
        }
    }
}
