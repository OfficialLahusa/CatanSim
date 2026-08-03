using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;


namespace Common.Agents
{
    public abstract class Agent
    {
        public readonly sbyte PlayerIndex;

        public Agent(sbyte playerIndex)
        {
            PlayerIndex = playerIndex;
        }

        public abstract Action Act(GameState state);
    }
}
