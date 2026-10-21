using Common;
using Common.Actions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents
{
    public class RandomAgent : Agent
    {
        public RandomAgent(sbyte playerIndex)
            : base(playerIndex)
        {

        }

        public override Action Act(GameState state)
        {
            // While a discard is pending, no other action is valid for this player
            if (DiscardAction.IsTurnValid(state.Turn, PlayerIndex))
            {
                return DiscardAction.GetRandomDiscard(state, PlayerIndex);
            }

            // Pick an action type uniformly, then an instance of that type uniformly.
            // Sampling uniformly over all instances would drown EndTurn (one instance) among the many
            // trade/road variants, and never ending the turn while trades remain produced degenerate
            // games with hundreds of rounds.
            List<List<Action>> actionGroups = [
                EndTurnAction.GetActionsForState(state, PlayerIndex),

                RollAction.GetActionsForState(state, PlayerIndex),
                RobberAction.GetActionsForState(state, PlayerIndex),

                FirstInitialSettlementAction.GetActionsForState(state, PlayerIndex),
                FirstInitialRoadAction.GetActionsForState(state, PlayerIndex),
                SecondInitialSettlementAction.GetActionsForState(state, PlayerIndex),
                SecondInitialRoadAction.GetActionsForState(state, PlayerIndex),

                RoadAction.GetActionsForState(state, PlayerIndex),
                SettlementAction.GetActionsForState(state, PlayerIndex),
                CityAction.GetActionsForState(state, PlayerIndex),
                BuyDevelopmentCardAction.GetActionsForState(state, PlayerIndex),

                KnightAction.GetActionsForState(state, PlayerIndex),
                MonopolyAction.GetActionsForState(state, PlayerIndex),
                RoadBuildingAction.GetActionsForState(state, PlayerIndex),
                YearOfPlentyAction.GetActionsForState(state, PlayerIndex),

                FourToOneTradeAction.GetActionsForState(state, PlayerIndex),
                ThreeToOneTradeAction.GetActionsForState(state, PlayerIndex),
                TwoToOneTradeAction.GetActionsForState(state, PlayerIndex)
            ];

            actionGroups.RemoveAll(group => group.Count == 0);

            if (actionGroups.Count == 0) throw new InvalidOperationException();

            List<Action> actions = actionGroups[Utils.Random.Next(actionGroups.Count)];
            return actions[Utils.Random.Next(actions.Count)];
        }
    }
}
