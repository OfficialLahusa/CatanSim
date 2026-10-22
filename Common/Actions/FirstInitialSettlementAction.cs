using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class FirstInitialSettlementAction : Action, IActionProvider
    {
        public int IntersectionIndex { get; init; }

        public FirstInitialSettlementAction(sbyte playerIdx, int intersectionIndex)
            : base(playerIdx)
        {
            IntersectionIndex = intersectionIndex;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private FirstInitialSettlementAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Place settlement
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.Settlement, PlayerIndex);

            // Remove piece from stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements--;

            // Award VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints++;

            // Update port privileges
            SettlementAction.UpdatePortPrivileges(state, IntersectionIndex, PlayerIndex);
        }

        public override void Revert(GameState state)
        {
            // Remove settlement
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.None, -1);

            // Return piece to stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements++;

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
                && turn.TypeOfRound == TurnState.RoundType.FirstInitial;
        }

        public bool IsBoardValid(GameState state)
        {
            Intersection intersection = state.Board.Intersections[IntersectionIndex];

            bool spaceFree = state.Board.GetBuilding(intersection.Index) == Intersection.BuildingType.None;
            bool noSettlementPlaced = state.Players[PlayerIndex].BuildingStock.RemainingSettlements == BuildingStock.MAX_SETTLEMENTS;

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

            return spaceFree && noSettlementPlaced && !hasAdjacentSettlement;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = new List<Action>();

            if (!IsTurnValid(state.Turn, playerIdx)) return actions;
            if (state.Players[playerIdx].BuildingStock.RemainingSettlements != BuildingStock.MAX_SETTLEMENTS) return actions;

            // Same result as IsBoardValid for every intersection, from the board's bitmasks
            foreach (int intersectionIdx in BitMask.Of(state.Board.GetFreeBuildingSites()))
            {
                actions.Add(new FirstInitialSettlementAction(playerIdx, intersectionIdx));
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", I{IntersectionIndex}";
        }
    }
}
