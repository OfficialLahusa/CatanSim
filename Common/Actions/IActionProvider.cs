using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public interface IActionProvider
    {
        /// <summary>
        /// Returns a list of actions that are legal to play for a given player in a given game state.
        /// Output randomness is left as a nondeterministic factor to resolve dynamically when the action is applied, which is used for random game playouts.
        /// </summary>
        /// <param name="state">The game state from which the actions are legal to play</param>
        /// <param name="playerIdx">The player for which the actions are legal to play</param>
        /// <returns></returns>
        static abstract List<Action> GetActionsForState(GameState state, sbyte playerIdx);
    }
}
