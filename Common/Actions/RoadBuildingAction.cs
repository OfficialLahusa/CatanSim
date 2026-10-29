using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class RoadBuildingAction : Action, IActionProvider
    {
        public RoadBuildingAction(sbyte playerIdx)
            : base(playerIdx)
        { }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private RoadBuildingAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Remove card
            state.Players[PlayerIndex].DevelopmentCards.Remove(DevelopmentCardType.RoadBuilding, 1);

            // Add free road stock (up to 2)
            BuildingStock buildingStock = state.Players[PlayerIndex].BuildingStock;
            uint awardedAmount = Math.Min(buildingStock.RemainingRoads, 2);
            buildingStock.RemainingRoads -= awardedAmount;
            buildingStock.FreeRoads += awardedAmount;

            // Update turn state
            state.Turn.HasPlayedDevelopmentCard = true;
        }

        public override void Revert(GameState state)
        {
            // Return card
            state.Players[PlayerIndex].DevelopmentCards.Add(DevelopmentCardType.RoadBuilding, 1);

            // Remove free road stock
            BuildingStock buildingStock = state.Players[PlayerIndex].BuildingStock;
            buildingStock.RemainingRoads += buildingStock.FreeRoads;
            buildingStock.FreeRoads = 0;

            // Update turn state
            state.Turn.HasPlayedDevelopmentCard = false;
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && turn.PendingTrade == null
                && !turn.MustDiscard
                && !turn.MustMoveRobber
                && !turn.HasPlayedDevelopmentCard;
        }

        public bool IsBoardValid(GameState state)
        {
            bool hasCard = state.Players[PlayerIndex].DevelopmentCards.Contains(DevelopmentCardType.RoadBuilding);

            // Check for dev card age
            bool cardAgeSufficient = state.Players[PlayerIndex].DevelopmentCards.Get(DevelopmentCardType.RoadBuilding) > state.Players[PlayerIndex].NewDevelopmentCards.Get(DevelopmentCardType.RoadBuilding);

            return hasCard && cardAgeSufficient;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            RoadBuildingAction action = new(playerIdx);

            return action.IsValidFor(state) ? [action] : [];
        }
    }
}
