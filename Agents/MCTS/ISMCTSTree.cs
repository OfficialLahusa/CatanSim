using Common;
using Common.Actions;
using System.Collections.Concurrent;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    /// <summary>
    /// A node of the information set tree: everything the searching seat can't tell apart after the moves that lead here.
    /// Its children are the moves as the seat sees them; a move with a random outcome is a chance node, whose children are the
    /// outcomes the seat can tell apart (an outcome it doesn't see is a single child).
    /// </summary>
    public sealed class ISNode
    {
        private readonly object _lock = new();
        private ISNode[] _children = [];
        private uint _visits, _resolved, _availability;
        private readonly double[] _scores = new double[MCTSNode.PlayerCount];

        // The policy's logits of moves at this node (from the evaluation of a world here), for its children
        private (Action move, float logit)[]? _logits;

        // The legal moves of the acting player in a world, by its hand: everything else at the node is the same in every world
        internal readonly ConcurrentDictionary<(ulong, ulong), List<Action>> LegalByHand = new();

        public ISNode? Parent { get; }

        /// <summary>
        /// The move into this node as the seat sees it (for an outcome: the move with its outcome, redacted), null at the root
        /// </summary>
        public Action? Move { get; }

        public bool IsChance { get; }

        /// <summary>
        /// The prior logit of the move into this node, NaN while unknown
        /// </summary>
        public float Logit { get; internal set; } = float.NaN;

        public IReadOnlyList<ISNode> Children => Volatile.Read(ref _children);

        /// <summary>
        /// Visits, including those whose result is still pending
        /// </summary>
        public uint Visits => Volatile.Read(ref _visits);

        public uint ResolvedVisits => Volatile.Read(ref _resolved);

        /// <summary>
        /// How often the parent was visited in a world where this move was legal
        /// </summary>
        public uint Availability => Volatile.Read(ref _availability);

        public bool HasLogits => Volatile.Read(ref _logits) != null;

        internal ISNode(ISNode? parent, Action? move, bool isChance)
        {
            Parent = parent;
            Move = move;
            IsChance = isChance;
        }

        public double MeanValue(int player)
        {
            lock (_lock) return _resolved == 0 ? 0 : _scores[player] / _resolved;
        }

        internal void MarkPendingVisit() => Interlocked.Increment(ref _visits);

        internal void MarkAvailable() => Interlocked.Increment(ref _availability);

        internal void AddResults(ReadOnlySpan<double> utilities)
        {
            lock (_lock)
            {
                for (int player = 0; player < _scores.Length; player++) _scores[player] += utilities[player];
                _resolved++;
            }
        }

        /// <summary>
        /// The children for the legal moves of a world, created where missing; in the order of the moves
        /// </summary>
        internal ISNode[] Available(List<Action> legal)
        {
            ISNode[] available = new ISNode[legal.Count];
            ISNode[] children = Volatile.Read(ref _children);
            bool missing = false;
            for (int i = 0; i < legal.Count; i++)
            {
                available[i] = Find(children, legal[i])!;
                missing |= available[i] == null;
            }
            if (!missing) return available;

            lock (_lock)
            {
                children = _children;
                List<ISNode> added = [];
                for (int i = 0; i < legal.Count; i++)
                {
                    if (available[i] != null) continue;
                    available[i] = Find(children, legal[i]) ?? added.Find(child => child.Move!.IsSameDecision(legal[i]))!;
                    if (available[i] != null) continue;
                    ISNode child = new ISNode(this, legal[i].Clone(), legal[i] is IOutputRandomnessAction) { Logit = LogitOf(legal[i]) };
                    added.Add(child);
                    available[i] = child;
                }
                if (added.Count > 0) Volatile.Write(ref _children, [.. children, .. added]);
            }
            return available;
        }

        private static ISNode? Find(ISNode[] children, Action move)
        {
            foreach (ISNode child in children)
                if (child.Move!.IsSameDecision(move)) return child;
            return null;
        }

        /// <summary>
        /// The child of a chance node for the outcome as the seat saw it, created if it's new
        /// </summary>
        internal ISNode Outcome(Action seen)
        {
            static bool Same(ISNode child, Action seen) => child.Move is IOutputRandomnessAction outcome && outcome.HasSameOutcome(seen);

            foreach (ISNode child in Volatile.Read(ref _children))
                if (Same(child, seen)) return child;
            lock (_lock)
            {
                foreach (ISNode child in _children)
                    if (Same(child, seen)) return child;
                ISNode outcome = new ISNode(this, seen, false);
                Volatile.Write(ref _children, [.. _children, outcome]);
                return outcome;
            }
        }

        /// <summary>
        /// The logits of the moves of a world evaluated at this node, for the children it has and those it gets later
        /// </summary>
        internal void SetLogits(List<Action> moves, float[] priors)
        {
            (Action, float)[] logits = new (Action, float)[moves.Count];
            for (int i = 0; i < moves.Count; i++) logits[i] = (moves[i], MathF.Log(Math.Max(priors[i], 1e-12f)));
            lock (_lock)
            {
                _logits = logits;
                foreach (ISNode child in _children)
                    if (float.IsNaN(child.Logit)) child.Logit = LogitOf(child.Move!);
            }
        }

        private float LogitOf(Action move)
        {
            (Action move, float logit)[]? logits = Volatile.Read(ref _logits);
            if (logits == null) return float.NaN;
            foreach ((Action known, float logit) in logits)
                if (known.IsSameDecision(move)) return logit;
            return float.NaN;
        }
    }

    /// <summary>
    /// Single-observer information set MCTS (Cowling, Powley, Whitehouse 2012): one tree for the searching seat, whose nodes
    /// are its information sets, and a world drawn from its beliefs (BeliefTracker) for every iteration. The iteration descends
    /// with the moves legal in its world, counts every legal child as available (the selection's exploration uses that count
    /// instead of the parent's visits), lets the world roll the outcomes of random moves and follows the outcome as the seat
    /// sees it. A game that ends in the world ends the iteration there. The other seats choose by their own values (max^n).
    /// The search depends only on the seat's view and the seed.
    /// </summary>
    public sealed class ISMCTSTree : ISearchTree
    {
        private readonly BeliefTracker _beliefs;
        private readonly GameState _view;
        private readonly bool _fullKnowledge;
        private readonly int _seed;
        private readonly double _exploration;
        private long _iterations;

        public ISNode RootNode { get; } = new ISNode(null, null, false);

        /// <summary>
        /// A world of the root (drawn from the view): the root's moves are the seat's, the same in every world
        /// </summary>
        public GameState RootState { get; }

        public sbyte OwnerIndex { get; }

        public bool UsesPolicy { get; set; }

        /// <summary>
        /// How much lower than its parent an unvisited move counts in PUCT (see MCTSTree.FirstPlayReduction)
        /// </summary>
        public double FirstPlayReduction { get; set; } = 0.1;

        /// <summary>
        /// Branches on every outcome of a random move, also those the seat can't see (an ablation: MCTSTree branches so)
        /// </summary>
        public bool BranchHiddenOutcomes { get; init; }

        /// <summary>
        /// Selects by each player's values scaled to the range seen in the tree (an ablation: MCTSTree normalizes so)
        /// </summary>
        public bool NormalizeValues
        {
            get => _playerBounds != null;
            init => _playerBounds = value ? Enumerable.Range(0, MCTSNode.PlayerCount).Select(_ => new MinMaxStats()).ToArray() : null;
        }

        private readonly MinMaxStats[]? _playerBounds;

        /// <param name="truth">The true position; only its view through the beliefs is used</param>
        /// <param name="beliefs">What the searching seat knows</param>
        /// <param name="fullKnowledge">Every world is the true position (an ablation: the same search without hidden information)</param>
        public ISMCTSTree(GameState truth, BeliefTracker beliefs, double exploration = 1.414, int seed = 0, bool fullKnowledge = false)
        {
            _beliefs = new BeliefTracker(beliefs);
            _fullKnowledge = fullKnowledge;
            _view = fullKnowledge ? new GameState(truth) : _beliefs.View(truth);
            _seed = seed;
            _exploration = exploration;
            OwnerIndex = (sbyte)beliefs.Observer;
            RootState = DrawWorld(new Random(seed));
        }

        private GameState DrawWorld(Random random) => _fullKnowledge ? new GameState(_view) : _beliefs.Determinize(_view, random);

        public bool RootHasPolicyPriors => RootNode.HasLogits;

        public void SetRootPriors(float[] priors) => RootNode.SetLogits(MCTSTree.GetValidActions(RootState), priors);

        public SearchLeaf SelectLeaf()
        {
            // A world of its own and random numbers of its own per iteration, so the search repeats with the seed
            long iteration = Interlocked.Increment(ref _iterations);
            Random random = new Random(HashCode.Combine(_seed, iteration));
            GameState world = DrawWorld(random);

            ISNode node = RootNode;
            while (!world.HasEnded && node.Visits > 0)
            {
                List<Action> legal = Legal(node, world);
                ISNode[] available = node.Available(legal);
                foreach (ISNode child in available) child.MarkAvailable();
                ISNode chosen = Select(node, available, legal[0].PlayerIndex, random);

                // The world decides the outcome of a random move (drawn here from the iteration's random numbers, so the search
                // repeats and the game's own random numbers stay untouched); the tree follows it as the seat sees it
                Action move = chosen.IsChance ? DrawOutcome(chosen.Move!, world, random) : chosen.Move!.Clone();
                move.Apply(world);
                node = chosen.IsChance ? chosen.Outcome(BranchHiddenOutcomes ? move : move.RedactedFor(OwnerIndex)) : chosen;
            }

            for (ISNode? pending = node; pending != null; pending = pending.Parent) pending.MarkPendingVisit();
            return new SearchLeaf(node, world);
        }

        /// <summary>
        /// A random move with its outcome drawn in proportion to the outcome's chance in the world (a card from the world's deck
        /// or the victim's hand in the world, the dice)
        /// </summary>
        private static Action DrawOutcome(Action move, GameState world, Random random)
        {
            Action unresolved = move.Clone();
            if (unresolved is IReplayAction replay) replay.ClearHistory();
            List<OutcomeVariant> variants = ((IOutputRandomnessAction)unresolved).GetOutcomeVariants(world, unresolved.PlayerIndex);
            double draw = random.NextDouble() * variants.Sum(variant => variant.Probability);
            foreach ((Action variant, double probability) in variants)
            {
                draw -= probability;
                if (draw < 0) return variant.Clone();
            }
            return variants[^1].Action.Clone();
        }

        /// <summary>
        /// The legal moves at a node in a world: cached by the acting player's hand, the only part of the world that changes them
        /// </summary>
        private static List<Action> Legal(ISNode node, GameState world)
        {
            int actor = 0;
            while (!world.CanPlayerAct(actor)) actor++;
            PlayerState player = world.Players[actor];
            ulong hand = (ulong)actor, news = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes) hand = hand << 5 | player.ResourceCards.Get(type);
            foreach (DevelopmentCardType type in Enum.GetValues<DevelopmentCardType>())
            {
                hand = hand << 5 | player.DevelopmentCards.Get(type);
                news = news << 5 | player.NewDevelopmentCards.Get(type);
            }
            return node.LegalByHand.GetOrAdd((hand, news), _ =>
            {
                List<Action> moves = MCTSTree.GetValidActions(world);
                return MCTSTree.WinningMove(world, moves) is Action winning ? [winning] : moves;
            });
        }

        /// <summary>
        /// The available child with the highest PUCT (with priors) or UCT for the deciding player, exploring by availability
        /// </summary>
        private ISNode Select(ISNode parent, ISNode[] available, int decider, Random random)
        {
            if (available.Length == 1) return available[0];

            uint parentResolved = parent.ResolvedVisits;
            MinMaxStats.Range? range = _playerBounds?[decider].GetRange();
            double Mean(ISNode node) => range is MinMaxStats.Range bounds ? bounds.Normalize(node.MeanValue(decider)) : node.MeanValue(decider);
            double parentMean = parentResolved == 0 ? 0.5 : Mean(parent);
            float[]? priors = UsesPolicy ? Priors(available) : null;

            ISNode best = available[0];
            double bestScore = double.NegativeInfinity;
            for (int i = 0; i < available.Length; i++)
            {
                ISNode child = available[i];
                uint visits = child.Visits, resolved = child.ResolvedVisits;
                double score;
                if (priors != null)
                {
                    double value = resolved == 0 ? parentMean - FirstPlayReduction : Mean(child);
                    score = value + _exploration * priors[i] * Math.Sqrt(Math.Max(child.Availability, 1)) / (1 + visits);
                }
                else
                {
                    score = visits == 0 ? double.MaxValue / 2
                        : (resolved == 0 ? parentMean : Mean(child)) + _exploration * Math.Sqrt(Math.Log(Math.Max(child.Availability, 1)) / visits);
                }
                score += random.NextDouble() * 1e-6;
                if (score > bestScore)
                {
                    best = child;
                    bestScore = score;
                }
            }
            return best;
        }

        /// <summary>
        /// The priors of the available moves: the softmax of their logits, uniform where none are known yet
        /// </summary>
        private static float[] Priors(ISNode[] available)
        {
            float[] priors = new float[available.Length];
            float max = float.NegativeInfinity;
            int known = 0;
            foreach (ISNode child in available)
            {
                if (float.IsNaN(child.Logit)) continue;
                max = MathF.Max(max, child.Logit);
                known++;
            }
            if (known == 0)
            {
                Array.Fill(priors, 1f / available.Length);
                return priors;
            }

            // A move no evaluation has seen yet gets the mean of the known ones
            float mean = available.Where(child => !float.IsNaN(child.Logit)).Average(child => child.Logit);
            float sum = 0;
            for (int i = 0; i < available.Length; i++)
            {
                priors[i] = MathF.Exp((float.IsNaN(available[i].Logit) ? mean : available[i].Logit) - max);
                sum += priors[i];
            }
            for (int i = 0; i < priors.Length; i++) priors[i] /= sum;
            return priors;
        }

        public bool HasPolicyPriors(SearchLeaf leaf) => ((ISNode)leaf.Node).HasLogits;

        public void ApplyPriors(SearchLeaf leaf, float[] priors, List<Action> moves) => ((ISNode)leaf.Node).SetLogits(moves, priors);

        public void Backpropagate(SearchLeaf leaf, ReadOnlySpan<double> utilities)
        {
            for (ISNode? node = (ISNode)leaf.Node; node != null; node = node.Parent)
            {
                node.AddResults(utilities);
                for (int player = 0; _playerBounds != null && player < MCTSNode.PlayerCount; player++) _playerBounds[player].Update(node.MeanValue(player));
            }
        }

        /// <summary>
        /// The root's moves with their visits
        /// </summary>
        public IReadOnlyList<(Action move, uint visits)> RootVisits() => RootNode.Children.Select(child => (child.Move!, child.Visits)).ToList();

        /// <summary>
        /// The mean value of the root for each player
        /// </summary>
        public double[] RootValues() => Enumerable.Range(0, MCTSNode.PlayerCount).Select(RootNode.MeanValue).ToArray();

        /// <summary>
        /// The most visited move of the root, without an outcome: the real game decides it
        /// </summary>
        public Action GetMostPromisingMove()
        {
            ISNode best = RootNode.Children.MaxBy(child => child.Visits) ?? throw new InvalidOperationException("The root has no moves; search first.");
            Action move = best.Move!.Clone();
            if (move is IReplayAction replay) replay.ClearHistory();
            return move;
        }
    }
}
