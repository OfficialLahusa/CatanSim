using Common.Actions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Common.Agents
{
    public class RandomAgent : Agent
    {
        public RandomAgent(sbyte playerIndex)
            : base(playerIndex)
        {

        }

        public override Action Act(GameState state)
        {
            List<Action> actions = [
                .. EndTurnAction.GetActionsForState(state, PlayerIndex),

                .. RollAction.GetActionsForState(state, PlayerIndex),
                .. RobberAction.GetActionsForState(state, PlayerIndex),

                .. FirstInitialSettlementAction.GetActionsForState(state, PlayerIndex),
                .. FirstInitialRoadAction.GetActionsForState(state, PlayerIndex),
                .. SecondInitialSettlementAction.GetActionsForState(state, PlayerIndex),
                .. SecondInitialRoadAction.GetActionsForState(state, PlayerIndex),

                .. RoadAction.GetActionsForState(state, PlayerIndex),
                .. SettlementAction.GetActionsForState(state, PlayerIndex),
                .. CityAction.GetActionsForState(state, PlayerIndex),
                .. BuyDevelopmentCardAction.GetActionsForState(state, PlayerIndex),

                .. KnightAction.GetActionsForState(state, PlayerIndex),
                .. MonopolyAction.GetActionsForState(state, PlayerIndex),
                .. RoadBuildingAction.GetActionsForState(state, PlayerIndex),
                .. YearOfPlentyAction.GetActionsForState(state, PlayerIndex),

                .. FourToOneTradeAction.GetActionsForState(state, PlayerIndex),
                .. ThreeToOneTradeAction.GetActionsForState(state, PlayerIndex),
                .. TwoToOneTradeAction.GetActionsForState(state, PlayerIndex)
            ];

            if(DiscardAction.IsTurnValid(state.Turn, PlayerIndex))
            {
                actions.Add(DiscardAction.GetRandomDiscard(state, PlayerIndex));
            }

            if (actions.Count == 0) throw new InvalidOperationException();

            int minIdx = actions.Count > 1 && actions[0] is EndTurnAction ? 1 : 0;
            int actionIdx = Utils.Random.Next(minIdx, actions.Count);
            return actions[actionIdx];
        }
    }
}
