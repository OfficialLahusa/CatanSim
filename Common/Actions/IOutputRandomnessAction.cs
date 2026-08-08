using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public interface IOutputRandomnessAction
    {
        public List<Action> GetOutcomeVariants(GameState state, sbyte playerIdx);
    }
}
