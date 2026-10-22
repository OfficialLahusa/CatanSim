
namespace Common
{
    /// <summary>
    /// Position of a corner of the board where settlements and cities are built.
    /// Shared by all copies of a board; the building and owner at it are stored in the board (Board.GetBuilding, ...).
    /// </summary>
    public class Intersection
    {
        public byte Index { get; init; }
        public bool FacesDownwards { get; init; }

        public enum BuildingType : byte
        {
            None = 0,
            Settlement = 1,
            City = 2
        }

        public Intersection(byte index, bool facesDownwards)
        {
            Index = index;
            FacesDownwards = facesDownwards;
        }

        public override bool Equals(object? obj)
        {
            return obj is Intersection intersection &&
                   Index == intersection.Index &&
                   FacesDownwards == intersection.FacesDownwards;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Index, FacesDownwards);
        }
    }
}
