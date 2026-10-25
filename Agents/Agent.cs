using Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;


namespace Agents
{
    public abstract class Agent
    {
        public readonly sbyte PlayerIndex;

        public Agent(sbyte playerIndex)
        {
            PlayerIndex = playerIndex;
        }

        public abstract Action Act(GameState state);

        /// <summary>
        /// Act that may stop early once the token is cancelled; only agents that think for long check it. The move of a
        /// cancelled call is not meant to be played.
        /// </summary>
        public virtual Action Act(GameState state, CancellationToken cancellation)
        {
            return Act(state);
        }

        public virtual void HandleAction(Action action) { }

        public virtual void ResetToState(GameState state) { }
    }
}
