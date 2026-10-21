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
        // Action providers of every type except discards (which are handled separately)
        private static readonly Func<GameState, sbyte, List<Action>>[] ActionProviders =
        [
            EndTurnAction.GetActionsForState,

            RollAction.GetActionsForState,
            RobberAction.GetActionsForState,

            FirstInitialSettlementAction.GetActionsForState,
            FirstInitialRoadAction.GetActionsForState,
            SecondInitialSettlementAction.GetActionsForState,
            SecondInitialRoadAction.GetActionsForState,

            RoadAction.GetActionsForState,
            SettlementAction.GetActionsForState,
            CityAction.GetActionsForState,
            BuyDevelopmentCardAction.GetActionsForState,

            KnightAction.GetActionsForState,
            MonopolyAction.GetActionsForState,
            RoadBuildingAction.GetActionsForState,
            YearOfPlentyAction.GetActionsForState,

            FourToOneTradeAction.GetActionsForState,
            ThreeToOneTradeAction.GetActionsForState,
            TwoToOneTradeAction.GetActionsForState
        ];

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

            // Pick an action type uniformly among the types with a legal action, then an instance of that type uniformly.
            // Sampling uniformly over all instances would drown EndTurn (one instance) among the many
            // trade/road variants, and never ending the turn while trades remain produced degenerate
            // games with hundreds of rounds.
            // Types are drawn without replacement and empty ones skipped, which gives the same uniform choice among the
            // non-empty types while only enumerating the actions of the types actually tried.
            Span<int> remainingTypes = stackalloc int[ActionProviders.Length];
            for (int i = 0; i < remainingTypes.Length; i++)
                remainingTypes[i] = i;

            for (int remaining = remainingTypes.Length; remaining > 0; remaining--)
            {
                int pick = Utils.Random.Next(remaining);
                int providerIdx = remainingTypes[pick];
                remainingTypes[pick] = remainingTypes[remaining - 1];

                List<Action> actions = ActionProviders[providerIdx](state, PlayerIndex);
                if (actions.Count > 0)
                    return actions[Utils.Random.Next(actions.Count)];
            }

            throw new InvalidOperationException();
        }
    }
}
