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
        public uint VisitCount { get; set; } = 0;
        public uint WinCount { get; set; } = 0;
        public sbyte ActivePlayerIndex { get; set; }

        public List<MCTSNode> Children { get; set; } = new List<MCTSNode>();

        public MCTSNode(sbyte activePlayerIndex, MCTSNode? parent = null, Action? lastAction = null, bool isTerminal = false)
        {
            ActivePlayerIndex = activePlayerIndex;
            Parent = parent;
            LastAction = lastAction;
            IsTerminal = isTerminal;
        }

        public double GetUCT(double explorationParameter)
        {
            if (VisitCount == 0)
                return double.PositiveInfinity;

            if (Parent == null)
                return (double)WinCount / VisitCount;

            return (double)WinCount / VisitCount + explorationParameter * Math.Sqrt(Math.Log(Parent.VisitCount) / VisitCount);
        }
    }
}
