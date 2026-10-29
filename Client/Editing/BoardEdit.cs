using Common;
using Common.Actions;

namespace Client.Editing
{
    /// <summary>
    /// A change made in Edit mode, kept as the position before and after it, so it can be undone and redone like a move
    /// </summary>
    public sealed class BoardEdit
    {
        public string Description { get; }
        public int Player { get; }
        private readonly GameState _before, _after;

        public BoardEdit(string description, int player, GameState before, GameState after)
        {
            Description = description;
            Player = player;
            _before = new GameState(before);
            _after = new GameState(after);
        }

        public GameState Before => new GameState(_before);
        public GameState After => new GameState(_after);

        /// <summary>
        /// Recomputes what follows from the pieces on the board after they were edited: victory points of buildings, the
        /// building stock, port privileges and the longest road
        /// </summary>
        public static void RecomputeFromBoard(GameState state)
        {
            Board board = state.Board;
            for (int player = 0; player < state.Players.Length; player++)
            {
                PlayerState p = state.Players[player];
                int settlements = 0, cities = 0, roads = 0;
                for (int i = 0; i < board.Intersections.Count; i++)
                {
                    if (!board.HasBuildingOf(i, player)) continue;
                    if (board.GetBuilding(i) == Intersection.BuildingType.Settlement) settlements++;
                    else cities++;
                }
                for (int i = 0; i < board.Edges.Count; i++)
                {
                    if (board.HasRoadOf(i, player)) roads++;
                }

                p.VictoryPoints.SettlementPoints = (byte)settlements;
                p.VictoryPoints.CityPoints = (byte)(2 * cities);
                p.BuildingStock.RemainingSettlements = (uint)Math.Max(0, (int)BuildingStock.MAX_SETTLEMENTS - settlements);
                p.BuildingStock.RemainingCities = (uint)Math.Max(0, (int)BuildingStock.MAX_CITIES - cities);
                p.BuildingStock.RemainingRoads = (uint)Math.Max(0, (int)BuildingStock.MAX_ROADS - roads - (int)p.BuildingStock.FreeRoads);
            }

            SettlementAction.FullyRecalculatePortPrivileges(state);

            // The edit tools keep every player within the stock, which CalculateLongestRoad assumes
            for (int player = 0; player < state.Players.Length; player++)
                state.CalculateLongestRoad(player, true);
        }
    }
}
