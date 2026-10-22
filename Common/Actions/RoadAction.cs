using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class RoadAction : Action, IReplayAction, IActionProvider
    {
        public record RoadActionHistory(int PrevLongestRoadHolder, bool WasFree)
        {
            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private RoadActionHistory() : this(-1, false) { }
        }

        public RoadActionHistory? History { get; private set; }

        public int EdgeIndex { get; init; }

        public RoadAction(sbyte playerIdx, int edgeIndex)
            : base(playerIdx)
        {
            EdgeIndex = edgeIndex;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private RoadAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Determine previous longest road holder
            int currentLongestRoadHolder = -1;

            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                if (state.Players[playerIdx].VictoryPoints.LongestRoadPoints > 0)
                {
                    currentLongestRoadHolder = playerIdx;
                    break;
                }
            }

            // Place road
            state.Board.SetRoad(EdgeIndex, Edge.BuildingType.Road, PlayerIndex);

            // Remove cards and/or building stock
            BuildingStock buildingStock = state.Players[PlayerIndex].BuildingStock;
            bool hasFreeStock = buildingStock.FreeRoads > 0;

            History = new RoadActionHistory(currentLongestRoadHolder, hasFreeStock);

            // Use free road building dev card stock first
            if (hasFreeStock)
            {
                buildingStock.FreeRoads--;
            }
            // Otherwise, buy normally
            else
            {
                // Remove cards
                CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;
                playerCards.Remove(ResourceCardType.Lumber, 1);
                playerCards.Remove(ResourceCardType.Brick, 1);

                // Return cards to bank
                state.ResourceBank.Add(ResourceCardType.Lumber, 1);
                state.ResourceBank.Add(ResourceCardType.Brick, 1);

                // Remove piece from stock
                buildingStock.RemainingRoads--;
            }

            // Recalculate longest road and award VPs accordingly
            state.CalculateLongestRoad(PlayerIndex);

            // Check for match completion
            state.CheckForCompletion();
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Remove road
            state.Board.SetRoad(EdgeIndex, Edge.BuildingType.None, -1);

            // Return cards and/or building stock
            BuildingStock buildingStock = state.Players[PlayerIndex].BuildingStock;

            // Return free building stock
            if (History!.WasFree)
            {
                buildingStock.FreeRoads++;
            }
            // Otherwise, return normal cost
            else
            {
                // Return cards
                CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;
                playerCards.Add(ResourceCardType.Lumber, 1);
                playerCards.Add(ResourceCardType.Brick, 1);

                // Remove cards from bank
                state.ResourceBank.Remove(ResourceCardType.Lumber, 1);
                state.ResourceBank.Remove(ResourceCardType.Brick, 1);

                // Return piece to stock
                buildingStock.RemainingRoads++;
            }

            // Recalculate longest road
            state.CalculateLongestRoad(PlayerIndex);

            // Move VPs to previous longest road holder in case of draws (or nobody if -1)
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                state.Players[playerIdx].VictoryPoints.LongestRoadPoints = (byte)(playerIdx == History.PrevLongestRoadHolder ? 2 : 0);
            }

            // Un-complete match
            state.Turn.TypeOfRound = TurnState.RoundType.Normal;
        }

        public override bool IsValidFor(GameState state)
        {
            return (IsTurnValid(state.Turn, PlayerIndex) || IsPlacingFreeRoads(state, PlayerIndex)) && IsBoardValid(state);
        }

        /// <summary>
        /// Right after a Road Building card the player can place its roads, also before the roll
        /// (taking any other action gives up the roads that are left)
        /// </summary>
        public static bool IsPlacingFreeRoads(GameState state, int playerIdx)
        {
            return state.Turn.PlayerIndex == playerIdx
                && state.Turn.TypeOfRound == TurnState.RoundType.Normal
                && state.Players[playerIdx].BuildingStock.FreeRoads > 0;
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
            Edge road = state.Board.Edges[EdgeIndex];

            bool spaceFree = state.Board.GetRoad(road.Index) == Edge.BuildingType.None;
            bool canAfford = state.Players[PlayerIndex].CanAffordRoad();
            if (!spaceFree || !canAfford) return false;

            // Get the two intersections on the ends of the road
            (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(road);

            // Check if one of the intersections has a building owned by the player
            bool hasTopBuilding = state.Board.GetBuildingOwner(top.Index) == PlayerIndex && state.Board.GetBuilding(top.Index) != Intersection.BuildingType.None;
            bool hasBottomBuilding = state.Board.GetBuildingOwner(bottom.Index) == PlayerIndex && state.Board.GetBuilding(bottom.Index) != Intersection.BuildingType.None;

            bool hasDirectAdjBuilding = hasTopBuilding || hasBottomBuilding;
            if (hasDirectAdjBuilding) return true;

            // Check if any of the intersections connect to roads of the player
            bool topBlocked = state.Board.GetBuildingOwner(top.Index) != PlayerIndex && state.Board.GetBuilding(top.Index) != Intersection.BuildingType.None;
            bool bottomBlocked = state.Board.GetBuildingOwner(bottom.Index) != PlayerIndex && state.Board.GetBuilding(bottom.Index) != Intersection.BuildingType.None;

            bool topHasAdjRoad = false;
            bool bottomHasAdjRoad = false;

            // Check if one of the adjacent roads of the top intersection was built by the player
            foreach (Edge adjacentRoad in state.Board.Adjacency.GetEdges(top))
            {
                if (state.Board.GetRoadOwner(adjacentRoad.Index) == PlayerIndex && state.Board.GetRoad(adjacentRoad.Index) != Edge.BuildingType.None)
                {
                    topHasAdjRoad = true;
                    break;
                }
            }

            // Check if one of the adjacent roads of the bottom intersection was built by the player
            foreach (Edge adjacentRoad in state.Board.Adjacency.GetEdges(bottom))
            {
                if (state.Board.GetRoadOwner(adjacentRoad.Index) == PlayerIndex && state.Board.GetRoad(adjacentRoad.Index) != Edge.BuildingType.None)
                {
                    bottomHasAdjRoad = true;
                    break;
                }
            }

            bool hasDirectAdjRoad = (topHasAdjRoad && !topBlocked) || (bottomHasAdjRoad && !bottomBlocked);
            return hasDirectAdjRoad;
        }

        public bool HasHistory()
        {
            return History != null;
        }

        public void ClearHistory()
        {
            History = null;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            if (!IsTurnValid(state.Turn, playerIdx) && !IsPlacingFreeRoads(state, playerIdx)) return actions;
            if (!state.Players[playerIdx].CanAffordRoad()) return actions;

            // Same result as IsBoardValid for every edge, from the board's bitmasks
            foreach (int edgeIdx in BitMask.Of(state.Board.GetRoadSites(playerIdx)))
            {
                actions.Add(new RoadAction(playerIdx, edgeIdx));
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", I{EdgeIndex}";
        }
    }
}
