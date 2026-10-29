using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class LegalActionProvider : IActionProvider
    {
        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [
                .. EndTurnAction.GetActionsForState(state, playerIdx),

                .. RollAction.GetActionsForState(state, playerIdx),
                .. DiscardAction.GetActionsForState(state, playerIdx),
                .. RobberAction.GetActionsForState(state, playerIdx),

                .. FirstInitialSettlementAction.GetActionsForState(state, playerIdx),
                .. FirstInitialRoadAction.GetActionsForState(state, playerIdx),
                .. SecondInitialSettlementAction.GetActionsForState(state, playerIdx),
                .. SecondInitialRoadAction.GetActionsForState(state, playerIdx),

                .. RoadAction.GetActionsForState(state, playerIdx),
                .. SettlementAction.GetActionsForState(state, playerIdx),
                .. CityAction.GetActionsForState(state, playerIdx),
                .. BuyDevelopmentCardAction.GetActionsForState(state, playerIdx),

                .. KnightAction.GetActionsForState(state, playerIdx),
                .. MonopolyAction.GetActionsForState(state, playerIdx),
                .. RoadBuildingAction.GetActionsForState(state, playerIdx),
                .. YearOfPlentyAction.GetActionsForState(state, playerIdx),

                .. FourToOneTradeAction.GetActionsForState(state, playerIdx),
                .. ThreeToOneTradeAction.GetActionsForState(state, playerIdx),
                .. TwoToOneTradeAction.GetActionsForState(state, playerIdx),

                // Answers to an open offer and the offerer's pick; offers and counteroffers themselves are any card
                // combination and come from the client or an agent
                .. RespondToTradeAction.GetActionsForState(state, playerIdx),
                .. ConfirmTradeAction.GetActionsForState(state, playerIdx),
                .. CancelTradeAction.GetActionsForState(state, playerIdx)
            ];

            return actions;
        }
    }
}
