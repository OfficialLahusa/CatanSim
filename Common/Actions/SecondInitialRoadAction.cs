using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class SecondInitialRoadAction : Action, IActionProvider
    {
        public int EdgeIndex { get; init; }

        public SecondInitialRoadAction(sbyte playerIdx, int edgeIndex)
            : base(playerIdx)
        {
            EdgeIndex = edgeIndex;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private SecondInitialRoadAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Place road
            state.Board.SetRoad(EdgeIndex, Edge.BuildingType.Road, PlayerIndex);

            // Remove piece from stock
            state.Players[PlayerIndex].BuildingStock.RemainingRoads--;

            // Update turn state
            if (state.Turn.PlayerIndex == 0)
            {
                state.Turn.RoundCounter++;
                state.Turn.TypeOfRound = TurnState.RoundType.Normal;
                state.Turn.MustRoll = true;
            }
            else
            {
                state.Turn.PlayerIndex--;
                // TODO: Remove?
                state.Turn.MustRoll = false;
            }

            // TODO: Remove?
            state.Turn.MustMoveRobber = false;
            Array.Fill(state.Turn.AwaitedPlayerDiscards, false);
            state.Turn.HasPlayedDevelopmentCard = false;

            state.CalculateLongestRoad(PlayerIndex);
        }

        public override void Revert(GameState state)
        {
            // Remove road
            state.Board.SetRoad(EdgeIndex, Edge.BuildingType.None, -1);

            // Return piece to stock
            state.Players[PlayerIndex].BuildingStock.RemainingRoads++;

            // Update turn state
            if (state.Turn.TypeOfRound == TurnState.RoundType.Normal)
            {
                state.Turn.RoundCounter--;
                state.Turn.TypeOfRound = TurnState.RoundType.SecondInitial;
                state.Turn.MustRoll = false;
            }
            else
            {
                state.Turn.PlayerIndex++;
            }

            state.CalculateLongestRoad(PlayerIndex);
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.SecondInitial;
        }

        public bool IsBoardValid(GameState state)
        {
            Edge road = state.Board.Edges[EdgeIndex];

            bool spaceFree = state.Board.GetRoad(road.Index) == Edge.BuildingType.None;
            bool twoSettlementsPlaced = state.Players[PlayerIndex].BuildingStock.RemainingSettlements == BuildingStock.MAX_SETTLEMENTS - 2;
            bool oneRoadPlaced = state.Players[PlayerIndex].BuildingStock.RemainingRoads == BuildingStock.MAX_ROADS - 1;

            // Get the two intersections on the ends of the road
            (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(road);

            // Check if one of the intersections has a building owned by the player
            bool hasTopBuilding = state.Board.GetBuildingOwner(top.Index) == PlayerIndex && state.Board.GetBuilding(top.Index) != Intersection.BuildingType.None;
            bool hasBottomBuilding = state.Board.GetBuildingOwner(bottom.Index) == PlayerIndex && state.Board.GetBuilding(bottom.Index) != Intersection.BuildingType.None;

            bool hasDirectAdjBuilding = hasTopBuilding || hasBottomBuilding;

            bool isAdjacentToFirstSettlement = false;
            if(hasDirectAdjBuilding)
            {
                foreach (Edge adjEdge in state.Board.Adjacency.GetEdges(hasTopBuilding ? top : bottom))
                {
                    if (state.Board.GetRoadOwner(adjEdge.Index) == PlayerIndex && state.Board.GetRoad(adjEdge.Index) != Edge.BuildingType.None)
                    {
                        isAdjacentToFirstSettlement = true;
                        break;
                    }
                }
            }

            bool isAdjacentToSecondSettlement = hasDirectAdjBuilding && !isAdjacentToFirstSettlement;

            return spaceFree && twoSettlementsPlaced && oneRoadPlaced && isAdjacentToSecondSettlement;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            if (!IsTurnValid(state.Turn, playerIdx)) return actions;

            for (int edgeIdx = 0; edgeIdx < state.Board.Edges.Count; edgeIdx++)
            {
                SecondInitialRoadAction action = new(playerIdx, edgeIdx);

                if (action.IsBoardValid(state))
                {
                    actions.Add(action);
                }
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", I{EdgeIndex}";
        }
    }
}
