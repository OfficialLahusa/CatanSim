using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    /// <summary>
    /// Position of a side of a tile where roads are built.
    /// Shared by all copies of a board; the road and owner at it are stored in the board (Board.GetRoad, ...).
    /// </summary>
    public class Edge
    {
        public byte Index { get; init; }
        public Direction.Edge Direction { get; init; }

        public enum BuildingType : byte
        {
            None = 0,
            Road = 1,
            // Ship = 2 (For future use)
        }

        public Edge(byte index, Direction.Edge direction)
        {
            Index = index;
            Direction = direction;
        }

        public override bool Equals(object? obj)
        {
            return obj is Edge edge
                && Index == edge.Index
                && Direction == edge.Direction;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Index, Direction);
        }
    }
}
