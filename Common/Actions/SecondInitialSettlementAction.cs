using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class SecondInitialSettlementAction : Action, IReplayAction, IActionProvider
    {
        public record SecondInitialSettlementActionHistory(List<ResourceCardType> InitialYields)
        {
            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private SecondInitialSettlementActionHistory() : this(InitialYields: null!) { }
        }

        public SecondInitialSettlementActionHistory? History { get; private set; }

        public int IntersectionIndex { get; init; }

        public SecondInitialSettlementAction(sbyte playerIdx, int intersectionIndex)
            : base(playerIdx)
        {
            IntersectionIndex = intersectionIndex;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private SecondInitialSettlementAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Place settlement
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.Settlement, PlayerIndex);

            // Remove piece from stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements--;

            // Award adjacent resources
            List<ResourceCardType> awardedResources = [];
            foreach (Tile adjTile in state.Board.Adjacency.GetTiles(state.Board.Intersections[IntersectionIndex]))
            {
                if(adjTile != null && adjTile.HasYield())
                {
                    ResourceCardType adjResource = adjTile.Type.ToCardType();

                    state.Players[PlayerIndex].ResourceCards.Add(adjResource, 1);
                    state.ResourceBank.Remove(adjResource, 1);

                    awardedResources.Add(adjResource);
                }
            }

            History = new SecondInitialSettlementActionHistory(awardedResources);

            // Award VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints++;

            // Update port privileges
            SettlementAction.UpdatePortPrivileges(state, IntersectionIndex, PlayerIndex);
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Remove settlement
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.None, -1);

            // Return piece to stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements++;

            // Remove initial resources
            foreach (ResourceCardType resourceType in History!.InitialYields)
            {
                state.Players[PlayerIndex].ResourceCards.Remove(resourceType, 1);
                state.ResourceBank.Add(resourceType, 1);
            }

            // Remove VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints--;

            // Recalculate port privileges
            SettlementAction.FullyRecalculatePortPrivileges(state);
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
            Intersection intersection = state.Board.Intersections[IntersectionIndex];

            bool spaceFree = state.Board.GetBuilding(intersection.Index) == Intersection.BuildingType.None;
            bool oneSettlementPlaced = state.Players[PlayerIndex].BuildingStock.RemainingSettlements == BuildingStock.MAX_SETTLEMENTS - 1;
            bool oneRoadPlaced = state.Players[PlayerIndex].BuildingStock.RemainingRoads == BuildingStock.MAX_ROADS - 1;

            // No adjacent settlements
            bool hasAdjacentSettlement = false;
            foreach (Edge adjacentRoad in state.Board.Adjacency.GetEdges(intersection))
            {
                (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(adjacentRoad);

                if (state.Board.GetBuilding(top.Index) != Intersection.BuildingType.None || state.Board.GetBuilding(bottom.Index) != Intersection.BuildingType.None)
                {
                    hasAdjacentSettlement = true;
                }
            }

            return spaceFree && oneSettlementPlaced && oneRoadPlaced && !hasAdjacentSettlement;
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
            List<Action> actions = new List<Action>();

            if (!IsTurnValid(state.Turn, playerIdx)) return actions;

            BuildingStock stock = state.Players[playerIdx].BuildingStock;
            if (stock.RemainingSettlements != BuildingStock.MAX_SETTLEMENTS - 1 || stock.RemainingRoads != BuildingStock.MAX_ROADS - 1) return actions;

            // Same result as IsBoardValid for every intersection, from the board's bitmasks
            foreach (int intersectionIdx in BitMask.Of(state.Board.GetFreeBuildingSites()))
            {
                actions.Add(new SecondInitialSettlementAction(playerIdx, intersectionIdx));
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", I{IntersectionIndex}";
        }
    }
}
