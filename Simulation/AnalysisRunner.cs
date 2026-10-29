using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// Works out what the match's agents see, for spectators and admins: for every new position the win chances and the
    /// policy's priors from the agents' value net, and while an MCTS agent searches its root moves a few times a second. Runs on
    /// its own thread with its own net (one session per thread), so neither the match nor the agents wait for it. Moves are
    /// sent as indices into MCTSTree.GetValidActions of the position, which every client computes the same way.
    /// </summary>
    internal sealed class AnalysisRunner : IDisposable
    {
        private const int SearchIntervalMs = 250, IdleIntervalMs = 1000;

        private readonly Match _match;
        private readonly bool _valueNetAvailable;
        private readonly Thread _thread;
        private readonly AutoResetEvent _wake = new(false);
        private volatile bool _stop;

        public AnalysisRunner(Match match, bool valueNetAvailable)
        {
            _match = match;
            _valueNetAvailable = valueNetAvailable;
            _thread = new Thread(Run) { IsBackground = true, Name = "Match analysis" };
            _thread.Start();
        }

        /// <summary>
        /// The position changed or an agent started thinking
        /// </summary>
        public void Wake() => _wake.Set();

        private void Run()
        {
            StateValueNet? net = null;
            if (_valueNetAvailable)
            {
                try
                {
                    net = new StateValueNet(StateValueNet.DefaultModelFile);
                }
                catch (Exception e)
                {
                    _match.PublishError($"The analysis couldn't load the value net: {e.Message}");
                }
            }

            int evaluated = -1;
            GameState? state = null;
            List<Action> moves = [];
            while (!_stop)
            {
                MCTSAgent? thinking = null;
                try
                {
                    (int version, GameState? changed, thinking) = _match.AnalysisSnapshot(evaluated);
                    if (changed != null)
                    {
                        state = changed;
                        moves = state.HasEnded ? [] : MCTSTree.GetValidActions(state);
                    }
                    if (state != null && version != evaluated)
                    {
                        evaluated = version;
                        _match.PublishAnalysis(version, Evaluate(net, state, moves));
                    }
                    if (state != null && thinking != null && Search(thinking, state, moves) is SearchEvent search)
                        _match.PublishAnalysis(version, search);
                }
                catch (Exception e)
                {
                    // The analysis is extra: a failure shows up as an error, the match goes on
                    _match.PublishError($"The analysis failed: {e.Message}");
                }

                _wake.WaitOne(thinking != null ? SearchIntervalMs : IdleIntervalMs);
            }
        }

        /// <summary>
        /// The win chances and priors of a position: from the value net the agents use, or the simple agent's estimate without
        /// it (and no priors); a finished game is decided
        /// </summary>
        internal static EvaluationEvent Evaluate(StateValueNet? net, GameState state, List<Action> moves)
        {
            float[] chances = net != null ? net.Run(StateVectorizer.For(net).Vectorize(state)).ToArray() : SimpleAgent.StateValueFunc(state).ToArray();
            if (state.HasEnded)
            {
                for (int i = 0; i < chances.Length; i++)
                    chances[i] = i == state.Turn.PlayerIndex ? 1 : 0;
            }

            // No priors for the answers and picks of a trade between players: the policy has no encoding for them
            float[] priors = [];
            if (net?.Policy is PolicyHead policy && moves.Count > 0 && state.Turn.PendingTrade == null)
            {
                priors = new float[moves.Count];
                if (moves.Count == 1) priors[0] = 1;
                else
                {
                    LeafValues values = net.Evaluate(StateVectorizer.For(net).Vectorize(state), false, withPolicy: true);
                    ActionEncoding.Logits(values.PolicyQueries!.AsSpan(0, policy.QueryLength), policy.Embeddings, policy.Dim, moves, priors);
                    ActionEncoding.Softmax(priors);
                }
            }
            return new EvaluationEvent(chances, priors);
        }

        /// <summary>
        /// The thinking agent's root moves for the position, or null if its tree is for another one (or a chance node)
        /// </summary>
        internal static SearchEvent? Search(MCTSAgent agent, GameState state, List<Action> moves)
        {
            (GameState? root, IReadOnlyList<MCTSAgent.RootMove> rootMoves) = agent.GetRootMoves();
            if (root == null || rootMoves.Count == 0 || !root.Equals(state)) return null;

            uint[] visits = new uint[moves.Count];
            float[] priors = new float[moves.Count];
            float[][] means = new float[moves.Count][];
            for (int i = 0; i < moves.Count; i++) means[i] = new float[state.Players.Length];
            foreach (MCTSAgent.RootMove move in rootMoves)
            {
                int index = moves.FindIndex(m => m.IsSameDecision(move.Move));
                if (index < 0) continue;
                visits[index] = move.Visits;
                priors[index] = move.Prior;
                means[index] = move.MeanValues.Select(v => (float)v).ToArray();
            }
            return new SearchEvent(agent.PlayerIndex, rootMoves.Sum(m => (long)m.Visits), visits, priors, means);
        }

        public void Dispose()
        {
            _stop = true;
            _wake.Set();
            _thread.Join(5000);
        }
    }
}
