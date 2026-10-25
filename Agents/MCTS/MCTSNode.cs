using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    public class MCTSNode
    {
        public MCTSNode? Parent { get; protected set; }
        public Action? LastAction { get; protected set; }

        // Set once selection has applied the node's action and the game was over (or for a terminal root)
        public bool IsTerminal { get; protected set; }

        // Active player of a node that was not expanded yet
        public const sbyte UnknownPlayer = -1;

        private int _visitCount = 0;
        public uint VisitCount
        {
            get => (uint)Volatile.Read(ref _visitCount);
            set => Volatile.Write(ref _visitCount, (int)value);
        }

        // Visits counted by MarkPendingPath whose result is not back yet; they lower the mean in selection (virtual loss)
        private int _pendingVisits = 0;

        // Sum of the owner's results (win probability, plus the placement tie-breaker if used), stored as bits so it can be added to without a lock
        private long _totalScoreBits = 0;
        public double TotalScore
        {
            get => BitConverter.Int64BitsToDouble(Volatile.Read(ref _totalScoreBits));
            set => Volatile.Write(ref _totalScoreBits, BitConverter.DoubleToInt64Bits(value));
        }

        // Sums of every player's result, for searches that model the opponents (see OpponentModel), stored as bits like TotalScore
        public const int PlayerCount = 4;
        private long _score0Bits, _score1Bits, _score2Bits, _score3Bits;

        // Player to act in the node's state, known for the root and for expanded nodes (UnknownPlayer before)
        public sbyte ActivePlayerIndex { get; set; }

        // Player whose actions the children are, i.e. who decides here (for discards the discarding player, not the turn
        // player), set on expansion; UnknownPlayer for outcome groups, whose children are sampled
        public sbyte DeciderIndex { get; private set; } = UnknownPlayer;
        public bool IsOutputRandomnessGroup { get; set; }

        // Probability of this outcome among its siblings (only meaningful for children of a randomness group)
        public double OutcomeProbability { get; set; } = 1.0;

        // Whether the tree's owner picks the next action here (otherwise it is sampled), decided when the node is expanded
        public bool OwnerDecides { get; private set; }

        // Taken only to expand the node, never to read its children
        public readonly object ChildLock = new object();

        // Published all at once when the node is expanded, so selection can read it without locking
        private MCTSNode[] _children = [];
        public IReadOnlyList<MCTSNode> Children => Volatile.Read(ref _children);
        internal MCTSNode[] ChildArray => Volatile.Read(ref _children);
        public bool IsExpanded => Volatile.Read(ref _children).Length > 0;

        public MCTSNode(sbyte activePlayerIndex, MCTSNode? parent = null, Action? lastAction = null, bool isTerminal = false, bool isOutputRandomnessGroup = false)
        {
            ActivePlayerIndex = activePlayerIndex;
            Parent = parent;
            LastAction = lastAction;
            IsTerminal = isTerminal;
            IsOutputRandomnessGroup = isOutputRandomnessGroup;
        }

        internal void MarkTerminal()
        {
            IsTerminal = true;
        }

        /// <summary>
        /// Makes the children visible to all threads at once
        /// </summary>
        internal void PublishChildren(MCTSNode[] children, bool ownerDecides, sbyte deciderIndex = UnknownPlayer)
        {
            OwnerDecides = ownerDecides;
            DeciderIndex = deciderIndex;
            Volatile.Write(ref _children, children);
        }

        public double GetUCT(double explorationParameter, MinMaxStats? valueBounds = null)
        {
            // Root node does not have an exploration term
            if (Parent == null)
            {
                uint visits = VisitCount;
                return visits == 0 ? 1e9 : TotalScore / visits;
            }

            return GetUCT(explorationParameter, Math.Log(Parent.VisitCount), valueBounds?.GetRange());
        }

        /// <summary>
        /// UCT of a child with the logarithm of its parent's visits and the tree's value range read once for all siblings
        /// </summary>
        internal double GetUCT(double explorationParameter, double logParentVisits, MinMaxStats.Range? valueRange)
        {
            // Snapshots for thread-safety
            uint visitCountSnapshot = VisitCount;
            double totalScoreSnapshot = TotalScore;

            // Max priority for unexplored nodes
            if (visitCountSnapshot == 0)
                // Not double.PositiveInfinity, because that would invalidate random additive tie breaker noise
                return 1e9;

            double meanValue = totalScoreSnapshot / visitCountSnapshot;

            // Rescale the exploitation term to the range of values seen in the tree, so that it stays
            // comparable to the exploration term even when all values are tiny (player far behind)
            if (valueRange != null)
                meanValue = valueRange.Value.Normalize(meanValue);

            return meanValue + explorationParameter * Math.Sqrt(logParentVisits / visitCountSnapshot);
        }

        /// <summary>
        /// UCT of a child on one player's result (normalized with that player's value range), inverted for a player who
        /// wants that result low (the paranoid opponent model)
        /// </summary>
        internal double GetUCT(double explorationParameter, double logParentVisits, MinMaxStats.Range? valueRange, int player, bool invert)
        {
            uint visitCountSnapshot = VisitCount;
            if (visitCountSnapshot == 0)
                return 1e9;

            double meanValue = PlayerScore(player) / visitCountSnapshot;
            if (valueRange != null)
                meanValue = valueRange.Value.Normalize(meanValue);
            if (invert)
                meanValue = 1 - meanValue;

            return meanValue + explorationParameter * Math.Sqrt(logParentVisits / visitCountSnapshot);
        }

        /// <summary>
        /// Sum of the player's results
        /// </summary>
        public double PlayerScore(int player)
        {
            return BitConverter.Int64BitsToDouble(Volatile.Read(ref ScoreBits(player)));
        }

        /// <summary>
        /// Adds one result per player, without a lock
        /// </summary>
        public void AddResults(ReadOnlySpan<double> values)
        {
            for (int player = 0; player < PlayerCount; player++)
                AddTo(ref ScoreBits(player), values[player]);
        }

        private ref long ScoreBits(int player)
        {
            switch (player)
            {
                case 0: return ref _score0Bits;
                case 1: return ref _score1Bits;
                case 2: return ref _score2Bits;
                case 3: return ref _score3Bits;
                default: throw new ArgumentOutOfRangeException(nameof(player));
            }
        }

        private static void AddTo(ref long bits, double value)
        {
            long initial, added;
            do
            {
                initial = Volatile.Read(ref bits);
                added = BitConverter.DoubleToInt64Bits(BitConverter.Int64BitsToDouble(initial) + value);
            }
            while (Interlocked.CompareExchange(ref bits, added, initial) != initial);
        }

        /// <summary>
        /// Cuts the link to the parent, so this node can become the root of a reused subtree
        /// </summary>
        public void DetachFromParent()
        {
            Parent = null;
        }

        public void IncrementVisitCount()
        {
            Interlocked.Increment(ref _visitCount);
        }

        /// <summary>
        /// Counts a visit whose result comes later (see MCTSTree.MarkPendingPath)
        /// </summary>
        internal void MarkPendingVisit()
        {
            Interlocked.Increment(ref _pendingVisits);
            Interlocked.Increment(ref _visitCount);
        }

        /// <summary>
        /// The result of a pending visit has been added
        /// </summary>
        internal void ResolvePendingVisit()
        {
            Interlocked.Decrement(ref _pendingVisits);
        }

        /// <summary>
        /// Visits whose results are already in the scores, i.e. without the pending ones: the divisor of the true mean value
        /// </summary>
        public uint ResolvedVisitCount
        {
            get
            {
                uint visits = VisitCount;
                int pending = Volatile.Read(ref _pendingVisits);
                return pending <= 0 ? visits : visits > (uint)pending ? visits - (uint)pending : 0;
            }
        }

        public void AddResult(double value)
        {
            AddTo(ref _totalScoreBits, value);
        }
    }
}
