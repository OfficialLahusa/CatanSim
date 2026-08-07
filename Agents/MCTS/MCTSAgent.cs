using Common;
using Common.Actions;
using Agents.Inference;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    public class MCTSAgent : Agent
    {
        // TODO: Erstmal non-ML MCTS implementieren, dann ML einbauen
        private const int ITERATION_COUNT = 1000;

        public MCTSAgent(sbyte playerIndex)
            : base(playerIndex)
        {
            
        }

        static MCTSAgent()
        {

        }

        public override Action Act(GameState state, uint playedActionsCount)
        {
            MCTSTree tree = new MCTSTree(state, PlayerIndex);

            tree.RunIterations(ITERATION_COUNT);

            return tree.GetMostPromisingMove();
        }
    }
}
