using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    /// <summary>
    /// One possible resolution of an action's output randomness, with the probability of that outcome
    /// </summary>
    public record struct OutcomeVariant(Action Action, double Probability);

    public interface IOutputRandomnessAction
    {
        /// <summary>
        /// Enumerates every possible resolution of this action's randomness for the given state.
        /// The probabilities of the returned variants sum to 1.
        /// </summary>
        public List<OutcomeVariant> GetOutcomeVariants(GameState state, sbyte playerIdx);
    }
}
