using Agents.Inference;
using Common;
using Common.Actions;
using Common.Serialization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    public class MCTSTree
    {
        public GameState RootState;
        public MCTSNode RootNode;
        

        public MCTSTree(GameState rootState)
        {
            RootState = rootState;
            RootNode = new MCTSNode((sbyte)rootState.Turn.PlayerIndex, null, null, false);
        }

        public Action GetMostPromisingMove()
        {
            // Return the child with the highest visit count
            MCTSNode bestChild = RootNode.Children.OrderByDescending(c => c.VisitCount).First();

            return bestChild.LastAction!;
        }

        public int GetTreeSize()
        {
            return GetTreeSizeRec(RootNode);
        }

        protected int GetTreeSizeRec(MCTSNode currentSubtreeRoot)
        {
            if (currentSubtreeRoot.IsTerminal || currentSubtreeRoot.Children.Count == 0)
            {
                return 1;
            }

            int sum = 0;
            foreach (MCTSNode child in currentSubtreeRoot.Children)
            {
                sum += GetTreeSizeRec(child);
            }
            return sum;
        }
    }
}
