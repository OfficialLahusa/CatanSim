using Agents;
using Common;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// Plays the agent seats of a match on a background thread: whenever an agent should move (its turn, not paused) it searches
    /// on a copy of the position, waits the pause between moves (so its finished search can be followed meanwhile) and plays
    /// its move if the position is still the same and the match still wants it. A change of the position cancels a running
    /// search.
    /// </summary>
    internal sealed class AgentRunner : IDisposable
    {
        private readonly Match _match;
        private readonly Thread _thread;
        private readonly AutoResetEvent _wake = new(false);
        private readonly CancellationTokenSource _stop = new();
        private readonly object _lock = new();
        private CancellationTokenSource? _search;
        private bool _stepRequested;

        public bool IsThinking { get; private set; }
        public Agent? ThinkingAgent { get; private set; }

        public AgentRunner(Match match)
        {
            _match = match;
            _thread = new Thread(Run) { IsBackground = true, Name = "Agent runner" };
            _thread.Start();
        }

        public void Wake() => _wake.Set();

        public void RequestStep()
        {
            lock (_lock) _stepRequested = true;
            _wake.Set();
        }

        /// <summary>
        /// The position changed: a running search is now for an old position
        /// </summary>
        public void PositionChanged()
        {
            lock (_lock) _search?.Cancel();
            _wake.Set();
        }

        private void Run()
        {
            while (!_stop.IsCancellationRequested)
            {
                bool step;
                lock (_lock) step = _stepRequested;

                if (!_match.WantsAgentMove(step))
                {
                    lock (_lock) _stepRequested = false;
                    _wake.WaitOne(200);
                    continue;
                }

                (Agent agent, GameState state, int version)? next = _match.TakeAgentMove(step);
                if (next == null) continue;
                (Agent agent, GameState snapshot, int moveVersion) = next.Value;
                CancellationTokenSource search;
                lock (_lock)
                {
                    _stepRequested = false;
                    search = _search = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token);
                }

                Action? action = null;
                IsThinking = true;
                ThinkingAgent = agent;
                try
                {
                    action = agent.Act(snapshot, search.Token);
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    _match.PublishError($"The agent of seat {agent.PlayerIndex + 1} failed: {e.Message}");
                    _wake.WaitOne(1000);
                }
                finally
                {
                    IsThinking = false;
                    ThinkingAgent = null;
                    _match.ReleaseAgent(agent);
                }

                bool cancelled = search.IsCancellationRequested;
                lock (_lock)
                {
                    if (ReferenceEquals(_search, search)) _search = null;
                }
                search.Dispose();
                if (action != null && !cancelled && (step || PauseBeforeMove())) _match.PlayAgentMove(agent, action, moveVersion);
            }
        }

        /// <summary>
        /// The pause between moves, after the search so its result stays on screen; false if the match no longer wants the move
        /// meanwhile (paused, a human's turn without stand-ins), true at once for a step
        /// </summary>
        private bool PauseBeforeMove()
        {
            DateTime searched = DateTime.UtcNow;
            while (!_stop.IsCancellationRequested)
            {
                bool step;
                lock (_lock) step = _stepRequested;
                if (step) return true;
                if (!_match.WantsAgentMove(false)) return false;
                double wait = _match.Playback.PaceSeconds - (DateTime.UtcNow - searched).TotalSeconds;
                if (wait <= 0) return true;
                _wake.WaitOne(TimeSpan.FromSeconds(Math.Min(wait, 0.2)));
            }
            return false;
        }

        public void Dispose()
        {
            _stop.Cancel();
            lock (_lock) _search?.Cancel();
            _wake.Set();
            _thread.Join(5000);
        }
    }
}
