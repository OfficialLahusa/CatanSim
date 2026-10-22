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
    public class SimpleAgent : Agent
    {
        public SimpleAgent(sbyte playerIndex)
            : base(playerIndex)
        {

        }

        public override Action Act(GameState state)
        {
            List<Action> firstInitialSettlementActions = FirstInitialSettlementAction.GetActionsForState(state, PlayerIndex);
            if (firstInitialSettlementActions.Count > 0)
            {
                // The first of the most valuable spots
                return firstInitialSettlementActions
                    .MaxBy(a => IntersectionValueFunc(state.Board.Intersections[((FirstInitialSettlementAction)a).IntersectionIndex], state))!;
            }

            List<Action> secondInitialSettlementActions = SecondInitialSettlementAction.GetActionsForState(state, PlayerIndex);
            if (secondInitialSettlementActions.Count > 0)
            {
                return secondInitialSettlementActions
                    .MaxBy(a => IntersectionValueFunc(state.Board.Intersections[((SecondInitialSettlementAction)a).IntersectionIndex], state))!;
            }

            if (DiscardAction.IsTurnValid(state.Turn, PlayerIndex))
            {
                return DiscardAction.GetRandomDiscard(state, PlayerIndex);
            }

            List<Action> actions = [
                .. EndTurnAction.GetActionsForState(state, PlayerIndex),

                .. RollAction.GetActionsForState(state, PlayerIndex),
                .. RobberAction.GetActionsForState(state, PlayerIndex),

                .. FirstInitialRoadAction.GetActionsForState(state, PlayerIndex),
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

            if (actions.Count == 0) throw new InvalidOperationException();

            int minIdx = actions.Count > 1 && actions[0] is EndTurnAction ? 1 : 0;
            int actionIdx = Utils.Random.Next(minIdx, actions.Count);
            return actions[actionIdx];
        }

        public static float IntersectionValueFunc(Intersection intersection, GameState state, bool considerRobber = true)
        {
            float value = 0f;

            foreach (Tile adjTile in state.Board.Adjacency.GetTiles(intersection))
            {
                if (considerRobber && state.Board.HasRobber(adjTile))
                    continue;

                value += adjTile.YieldPoints;
            }

            return value;
        }

        public static float[] StateValueFunc(GameState state)
        {
            float[] valuation = new float[state.Players.Length];

            // Return full valuation if match is already decided
            if (state.HasEnded)
            {
                valuation[state.Turn.PlayerIndex] = 1f;
                return valuation;
            }

            const float victoryPointWeight = 5f;
            const float yieldPointWeight = 1f;
            const float roadKnightWeight = 1f;

            // Add victory points
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                valuation[playerIdx] = victoryPointWeight * state.Players[playerIdx].VictoryPoints.Total;
            }

            // Add yield score
            foreach (Intersection intersection in state.Board.Intersections)
            {
                Intersection.BuildingType building = state.Board.GetBuilding(intersection.Index);
                if (building != Intersection.BuildingType.None)
                {
                    float exploitation = building == Intersection.BuildingType.Settlement ? 1 : 2;

                    valuation[state.Board.GetBuildingOwner(intersection.Index)] += yieldPointWeight * exploitation * IntersectionValueFunc(intersection, state, true);
                }
            }

            // Add longest road length and largest army size to incentivize higher lead
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                valuation[playerIdx] += roadKnightWeight * (state.Players[playerIdx].LongestRoadLength + state.Players[playerIdx].PlayedKnights);
            }

            // Normalize
            float sum = valuation.Sum();

            // Edge case: All players have a valuation of 0 => equalize valuation to 1
            if (sum < 0.0001f)
            {
                Array.Fill(valuation, 1f);
                sum = valuation.Sum();
            }

            for (int playerIdx = 0;playerIdx < state.Players.Length; playerIdx++)
            {
                valuation[playerIdx] /= sum;
            }

            return valuation;
        }
    }
}
