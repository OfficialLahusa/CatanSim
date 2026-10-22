using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    /// <summary>
    /// A side of a tile where roads are built.
    /// The road and owner are stored in the board's arrays; an Edge is a view onto them for one board.
    /// Without a board (during map generation) it only describes the position.
    /// </summary>
    public class Edge
    {
        private readonly Board? _board;

        public byte Index { get; init; }
        public Direction.Edge Direction { get; init; }

        public enum BuildingType : byte
        {
            None = 0,
            Road = 1,
            // Ship = 2 (For future use)
        }

        public BuildingType Building
        {
            get => BoundBoard.GetRoad(Index);
            set => BoundBoard.SetRoad(Index, value, BoundBoard.GetRoadOwner(Index));
        }

        // -1 => None, 0/1/.. => Player 1/2/..
        public sbyte Owner
        {
            get => BoundBoard.GetRoadOwner(Index);
            set => BoundBoard.SetRoad(Index, BoundBoard.GetRoad(Index), value);
        }

        private Board BoundBoard => _board ?? throw new InvalidOperationException("This edge is not bound to a board.");

        /// <summary>
        /// Position only, as used while a board is generated
        /// </summary>
        public Edge(byte index, Direction.Edge direction)
        {
            Index = index;
            Direction = direction;
        }

        /// <summary>
        /// View onto the road and owner at this position of a board
        /// </summary>
        public Edge(Board board, byte index, Direction.Edge direction)
            : this(index, direction)
        {
            _board = board;
        }

        public override bool Equals(object? obj)
        {
            return obj is Edge edge
                && Index == edge.Index
                && Direction == edge.Direction
                && Building == edge.Building
                && Owner == edge.Owner;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Direction, Building, Owner);
        }
    }
}
