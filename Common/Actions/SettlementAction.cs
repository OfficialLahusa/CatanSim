using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class SettlementAction : Action, IReplayAction, IActionProvider
    {
        public record SettlementActionHistory(int PrevLongestRoadHolder)
        {
            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private SettlementActionHistory() : this(-1) { }
        }

        public SettlementActionHistory? History {  get; private set; }

        public int IntersectionIndex { get; init; }

        public SettlementAction(sbyte playerIdx, int intersectionIndex)
            : base(playerIdx)
        {
            IntersectionIndex = intersectionIndex;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private SettlementAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Road Building roads that were not placed yet are given up
            ForfeitFreeRoads(state);

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

            History = new SettlementActionHistory(currentLongestRoadHolder);

            // Place settlement
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.Settlement, PlayerIndex);

            // Remove cards
            CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;
            playerCards.Remove(ResourceCardType.Lumber, 1);
            playerCards.Remove(ResourceCardType.Brick, 1);
            playerCards.Remove(ResourceCardType.Wool, 1);
            playerCards.Remove(ResourceCardType.Grain, 1);

            // Return cards to bank
            state.ResourceBank.Add(ResourceCardType.Lumber, 1);
            state.ResourceBank.Add(ResourceCardType.Brick, 1);
            state.ResourceBank.Add(ResourceCardType.Wool, 1);
            state.ResourceBank.Add(ResourceCardType.Grain, 1);

            // Remove piece from stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements--;

            // Award VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints++;

            // Update port privileges
            UpdatePortPrivileges(state, IntersectionIndex, PlayerIndex);

            // Check if the settlement might have broken the longest road, which can only happen if there are two adjacent roads from another player
            int? adjPlayer = null;
            foreach (Edge adjRoad in state.Board.Adjacency.GetEdges(state.Board.Intersections[IntersectionIndex]))
            {
                if (state.Board.GetRoad(adjRoad.Index) == Edge.BuildingType.None || state.Board.GetRoadOwner(adjRoad.Index) == PlayerIndex) continue;

                // First road from another player
                if (adjPlayer == null)
                {
                    adjPlayer = state.Board.GetRoadOwner(adjRoad.Index);
                }
                // Second road from another player
                else if (adjPlayer == state.Board.GetRoadOwner(adjRoad.Index))
                {
                    // Recalculate longest road and award VPs accordingly, since it can be broken here
                    state.CalculateLongestRoad(state.Board.GetRoadOwner(adjRoad.Index), true);
                }
            }

            // Check for match completion
            state.CheckForCompletion();
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Remove settlement
            state.Board.SetBuilding(IntersectionIndex, Intersection.BuildingType.None, -1);

            // Return cards
            CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;
            playerCards.Add(ResourceCardType.Lumber, 1);
            playerCards.Add(ResourceCardType.Brick, 1);
            playerCards.Add(ResourceCardType.Wool, 1);
            playerCards.Add(ResourceCardType.Grain, 1);

            // Remove cards from bank
            state.ResourceBank.Remove(ResourceCardType.Lumber, 1);
            state.ResourceBank.Remove(ResourceCardType.Brick, 1);
            state.ResourceBank.Remove(ResourceCardType.Wool, 1);
            state.ResourceBank.Remove(ResourceCardType.Grain, 1);

            // Return piece to stock
            state.Players[PlayerIndex].BuildingStock.RemainingSettlements++;

            // Remove VP
            state.Players[PlayerIndex].VictoryPoints.SettlementPoints--;

            // Recalculate port privileges
            FullyRecalculatePortPrivileges(state);

            // Recalculate longest roads in case of break
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                state.CalculateLongestRoad(playerIdx);
            }

            // Move VPs to previous longest road holder in case of draws (or nobody if -1)
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                state.Players[playerIdx].VictoryPoints.LongestRoadPoints = (byte)(playerIdx == History!.PrevLongestRoadHolder ? 2 : 0);
            }

            // Un-complete match
            state.Turn.TypeOfRound = TurnState.RoundType.Normal;

            ReturnForfeitedFreeRoads(state);
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

            bool spaceFree = state.Board.GetBuilding(intersection.Index) == Intersection.BuildingType.None;
            bool canAfford = state.Players[PlayerIndex].CanAffordSettlement();
            if (!spaceFree || !canAfford) return false;

            // Has adjacent road
            bool hasAdjacentRoad = false;
            bool hasAdjacentSettlement = false;
            foreach (Edge adjacentRoad in state.Board.Adjacency.GetEdges(intersection))
            {
                if (state.Board.GetRoadOwner(adjacentRoad.Index) == PlayerIndex && state.Board.GetRoad(adjacentRoad.Index) != Edge.BuildingType.None)
                {
                    hasAdjacentRoad = true;
                }

                (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(adjacentRoad);

                if (state.Board.GetBuilding(top.Index) != Intersection.BuildingType.None || state.Board.GetBuilding(bottom.Index) != Intersection.BuildingType.None)
                {
                    hasAdjacentSettlement = true;
                }
            }

            return hasAdjacentRoad && !hasAdjacentSettlement;
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

            if (!IsTurnValid(state.Turn, playerIdx) || !state.Players[playerIdx].CanAffordSettlement()) return actions;

            // Same result as IsBoardValid for every intersection, from the board's bitmasks
            foreach (int intersectionIdx in BitMask.Of(state.Board.GetSettlementSites(playerIdx)))
            {
                actions.Add(new SettlementAction(playerIdx, intersectionIdx));
            }

            return actions;
        }

        public static void UpdatePortPrivileges(GameState state, int intersectionIdx, int playerIdx)
        {
            foreach (Port port in state.Board.Ports)
            {
                Tile seaAnchor = state.Board.Map.GetTile(port.AnchorTileX, port.AnchorTileY);
                Tile landAnchor = state.Board.Adjacency.GetTile(seaAnchor, port.AnchorDirection)!;
                Edge portEdge = state.Board.Adjacency.GetEdge(landAnchor, port.AnchorDirection.Mirror())!;
                Intersection settlementIntersection = state.Board.Intersections[intersectionIdx];

                // On port
                (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(portEdge);
                if (settlementIntersection.Index == top.Index || settlementIntersection.Index == bottom.Index)
                {
                    state.Players[playerIdx].PortPrivileges |= port.Type.GetPortPrivilege();
                    break;
                }
            }
        }

        public static void FullyRecalculatePortPrivileges(GameState state)
        {
            // Reset port privileges
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                state.Players[playerIdx].PortPrivileges = PortPrivileges.None;
            }

            // Recalculate for all ports and players
            foreach (Port port in state.Board.Ports)
            {
                // TODO: Kann man dafür nicht einfach die Wasser-Tile nutzen?
                Tile seaAnchor = state.Board.Map.GetTile(port.AnchorTileX, port.AnchorTileY);
                Tile landAnchor = state.Board.Adjacency.GetTile(seaAnchor, port.AnchorDirection)!;
                Edge portEdge = state.Board.Adjacency.GetEdge(landAnchor, port.AnchorDirection.Mirror())!;
                (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(portEdge);

                if (state.Board.GetBuilding(top.Index) != Intersection.BuildingType.None)
                {
                    state.Players[state.Board.GetBuildingOwner(top.Index)].PortPrivileges |= port.Type.GetPortPrivilege();
                }

                if (state.Board.GetBuilding(bottom.Index) != Intersection.BuildingType.None)
                {
                    state.Players[state.Board.GetBuildingOwner(bottom.Index)].PortPrivileges |= port.Type.GetPortPrivilege();
                }
            }
        }

        public override string ToString()
        {
            return base.ToString() + $", I{IntersectionIndex}";
        }
    }
}
