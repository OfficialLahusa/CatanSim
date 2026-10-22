using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class CityAction : Action, IActionProvider
    {
        public int IntersectionIndex { get; init; }

        public CityAction(sbyte playerIdx, int intersectionIndex)
            : base(playerIdx)
        {
            IntersectionIndex = intersectionIndex;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private CityAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Place city
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.City, PlayerIndex);

            // Remove cards
            CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;
            playerCards.Remove(ResourceCardType.Grain, 2);
            playerCards.Remove(ResourceCardType.Ore, 3);

            // Return cards to bank
            state.ResourceBank.Add(ResourceCardType.Grain, 2);
            state.ResourceBank.Add(ResourceCardType.Ore, 3);

            // Remove piece from stock
            state.Players[PlayerIndex].BuildingStock.RemainingCities--;

            // Add previous piece to stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements++;

            // Award VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints--;
            state.Players[PlayerIndex].VictoryPoints.CityPoints += 2;

            // Check for match completion
            state.CheckForCompletion();
        }

        public override void Revert(GameState state)
        {
            // Remove city
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.Settlement, PlayerIndex);

            // Return cards
            CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;
            playerCards.Add(ResourceCardType.Grain, 2);
            playerCards.Add(ResourceCardType.Ore, 3);

            // Remove cards from bank
            state.ResourceBank.Remove(ResourceCardType.Grain, 2);
            state.ResourceBank.Remove(ResourceCardType.Ore, 3);

            // Add piece to stock
            state.Players[PlayerIndex].BuildingStock.RemainingCities++;

            // Remove previous piece from stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements--;

            // Remove VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints++;
            state.Players[PlayerIndex].VictoryPoints.CityPoints -= 2;

            // Un-complete match
            state.Turn.TypeOfRound = TurnState.RoundType.Normal;
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && !turn.MustRoll
                && !turn.MustDiscard
                && !turn.MustMoveRobber;
        }

        public bool IsBoardValid(GameState state)
        {
            Intersection intersection = state.Board.Intersections[IntersectionIndex];

            bool hasSettlement = state.Board.GetBuilding(intersection.Index) == Intersection.BuildingType.Settlement;
            bool ownsSettlement = state.Board.GetBuildingOwner(intersection.Index) == PlayerIndex;
            bool canAfford = state.Players[PlayerIndex].CanAffordCity();

            return hasSettlement && ownsSettlement && canAfford;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = new List<Action>();

            if (!IsTurnValid(state.Turn, playerIdx)) return actions;

            for (int intersectionIdx = 0; intersectionIdx < state.Board.Intersections.Count; intersectionIdx++)
            {
                CityAction action = new(playerIdx, intersectionIdx);
                if (action.IsBoardValid(state))
                {
                    actions.Add(action);
                }
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", I{IntersectionIndex}";
        }
    }
}
