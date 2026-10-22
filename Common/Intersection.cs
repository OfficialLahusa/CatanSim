
namespace Common
{
    /// <summary>
    /// A corner of the board where settlements and cities are built.
    /// The building and owner are stored in the board's arrays; an Intersection is a view onto them for one board.
    /// Without a board (during map generation) it only describes the position.
    /// </summary>
    public class Intersection
    {
        private readonly Board? _board;

        public byte Index { get; init; }
        public bool FacesDownwards { get; init; }

        public enum BuildingType : byte
        {
            None = 0,
            Settlement = 1,
            City = 2
        }

        public BuildingType Building
        {
            get => BoundBoard.GetBuilding(Index);
            set => BoundBoard.SetBuilding(Index, value, BoundBoard.GetBuildingOwner(Index));
        }

        // -1 => None, 0/1/.. => Player 1/2/..
        public sbyte Owner
        {
            get => BoundBoard.GetBuildingOwner(Index);
            set => BoundBoard.SetBuilding(Index, BoundBoard.GetBuilding(Index), value);
        }

        private Board BoundBoard => _board ?? throw new InvalidOperationException("This intersection is not bound to a board.");

        /// <summary>
        /// Position only, as used while a board is generated
        /// </summary>
        public Intersection(byte index, bool facesDownwards)
        {
            Index = index;
            FacesDownwards = facesDownwards;
        }

        /// <summary>
        /// View onto the building and owner at this position of a board
        /// </summary>
        public Intersection(Board board, byte index, bool facesDownwards)
            : this(index, facesDownwards)
        {
            _board = board;
        }

        public override bool Equals(object? obj)
        {
            return obj is Intersection intersection &&
                   Index == intersection.Index &&
                   FacesDownwards == intersection.FacesDownwards &&
                   Building == intersection.Building &&
                   Owner == intersection.Owner;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Index, FacesDownwards, Building, Owner);
        }
    }
}
