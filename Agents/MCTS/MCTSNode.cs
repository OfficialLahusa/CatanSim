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
        // Sum of binary (1 for win, 0 for loss) or double (win percentage) results
        public double TotalScore { get; set; } = 0f;
        public sbyte ActivePlayerIndex { get; set; }
        public bool IsOutputRandomnessGroup { get; set; }

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
            if (VisitCount == 0)
                return double.PositiveInfinity;

            if (Parent == null)
                return TotalScore / VisitCount;

            return TotalScore / VisitCount + explorationParameter * Math.Sqrt(Math.Log(Parent.VisitCount) / VisitCount);
        }
    }
}
