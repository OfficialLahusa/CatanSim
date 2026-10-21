using YamlDotNet.Serialization;
using static Common.Tile;

namespace Common
{
    public class Tile
    {
        public byte X { get; init; }
        public byte Y { get; init; }
        public enum TileType : byte
        {
            Water,
            Lumber,
            Brick,
            Wool,
            Grain,
            Ore,
            Desert,
            NonPlayable = 255,
        };
        // Tiles are shared by all copies of a board, so they never change after creation
        public TileType Type { get; init; }
        public byte? Number { get; init; }
        [YamlIgnore]
        public int YieldPoints
        {
            get
            {
                return Number switch
                {
                    2 or 12 => 1,
                    3 or 11 => 2,
                    4 or 10 => 3,
                    5 or 9 => 4,
                    6 or 8 => 5,
                    _ => 0
                };
            }
        }

        public Tile(byte x, byte y, TileType type, byte? number)
        {
            X = x;
            Y = y;
            Type = type;
            Number = number;
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public Tile(Tile copy)
        {
            X = copy.X;
            Y = copy.Y;
            Type = copy.Type;
            Number = copy.Number;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private Tile()
        {

        }

        public bool IsLandTile()
        {
            return Type != TileType.NonPlayable && Type != TileType.Water;
        }

        public bool HasYield()
        {
            return IsLandTile() && Type != TileType.Desert && Number != null;
        }

        public override bool Equals(object? obj)
        {
            return obj is Tile tile
                && X == tile.X
                && Y == tile.Y
                && Type == tile.Type
                && Number == tile.Number;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Type, Number);
        }
    }

    public static class TileTypeExtensions
    {
        public static ResourceCardType ToCardType(this TileType tileType)
        {
            return tileType switch
            {
                TileType.Lumber => ResourceCardType.Lumber,
                TileType.Brick => ResourceCardType.Brick,
                TileType.Wool => ResourceCardType.Wool,
                TileType.Grain => ResourceCardType.Grain,
                TileType.Ore => ResourceCardType.Ore,
                _ => throw new InvalidOperationException()
            };
        }
    }
}
