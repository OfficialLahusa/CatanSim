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

        public bool IsTerminal { get; protected set; }

        private int _visitCount = 0;
        public uint VisitCount
        {
            get => (uint)Volatile.Read(ref _visitCount);
            set => Volatile.Write(ref _visitCount, (int)value);
        }

        // Sum of binary (1 for win, 0 for loss) or double (win percentage) results
        private readonly object _scoreLock = new object();
        private double _totalScore = 0;
        public double TotalScore
        {
            get { lock (_scoreLock) return _totalScore; }
            set { lock (_scoreLock) _totalScore = value; }
        }

        public sbyte ActivePlayerIndex { get; set; }
        public bool IsOutputRandomnessGroup { get; set; }

        public readonly object ChildLock = new object();
        public List<MCTSNode> Children { get; set; } = new List<MCTSNode>();

        public MCTSNode(sbyte activePlayerIndex, MCTSNode? parent = null, Action? lastAction = null, bool isTerminal = false, bool isOutputRandomnessGroup = false)
        {
            ActivePlayerIndex = activePlayerIndex;
            Parent = parent;
            LastAction = lastAction;
            IsTerminal = isTerminal;
            IsOutputRandomnessGroup = isOutputRandomnessGroup;
        }

        public double GetUCT(double explorationParameter)
        {
            // Snapshots for thread-safety
            uint visitCountSnapshot = VisitCount;
            double totalScoreSnapshot = TotalScore;

            // Max priority for unexplored nodes
            if (visitCountSnapshot == 0)
                // Not double.PositiveInfinity, because that would invalidate random additive tie breaker noise
                return 1e9;

            // Root node does not have an exploration term
            if (Parent == null)
                return totalScoreSnapshot / visitCountSnapshot;

            return totalScoreSnapshot / visitCountSnapshot + explorationParameter * Math.Sqrt(Math.Log(Parent.VisitCount) / visitCountSnapshot);
        }

        public void IncrementVisitCount()
        {
            Interlocked.Increment(ref _visitCount);
        }

        public void AddResult(double value)
        {
            lock (_scoreLock)
                _totalScore += value;
        }
    }
}
