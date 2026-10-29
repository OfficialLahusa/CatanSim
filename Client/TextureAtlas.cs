using Common;
using Microsoft.VisualBasic;
using SFML.Graphics;
using static Common.Direction;
using static Common.Tile;

namespace Client
{
    public static class TextureAtlas
    {
        public static readonly Texture Texture;

        public enum Sprite : byte
        {
            None = 255,
            Lumber = 0,
            Brick = 1,
            Wool = 2,
            Grain = 3,
            Ore = 4,
            Development = 5,
            QuestionMark = 6,

            Settlement = 8,
            City = 9,
            Road = 10,
            Desert = 11,
            Port = 12,
            Robber = 13,
            ThumbsUp = 14,
            ThumbsDown = 15,

            Knight = 16,
            Monopoly = 17,
            YearOfPlenty = 18,
            VictoryPoint = 19,
            ArmySize = 20,
            Bank = 21,
            Checkmark = 22,
            Crossmark = 23,

            DiceOne = 24,
            DiceTwo = 25,
            DiceThree = 26,
            DiceFour = 27,
            DiceFive = 28,
            DiceSix = 29,
            Hourglass = 30,
            Skip = 31
        }

        static TextureAtlas()
        {
            using Image image = new Image(ClientUtils.Resource("atlas.png"));
            using Image bled = BleedColors(image);
            Texture = new Texture(bled);
        }

        /// <summary>
        /// The atlas with every fully transparent pixel colored like the nearest visible one. The file's transparent pixels are
        /// black, which smoothing and mipmaps mixed into the edges of the tinted icons as a dark rim (most visible on light
        /// icons like the sheep). A breadth-first fill from the visible pixels, so each transparent pixel takes its color from
        /// one a shortest step path away.
        /// </summary>
        private static Image BleedColors(Image image)
        {
            int width = (int)image.Size.X, height = (int)image.Size.Y, count = width * height;
            byte[] pixels = image.Pixels;
            bool[] colored = new bool[count];
            int[] queue = new int[count];
            int head = 0, tail = 0;
            for (int i = 0; i < count; i++)
            {
                if (pixels[4 * i + 3] == 0) continue;
                colored[i] = true;
                queue[tail++] = i;
            }

            while (head < tail)
            {
                int from = queue[head++];
                int x = from % width, y = from / width;
                for (int n = 0; n < 4; n++)
                {
                    int to = n switch { 0 => x > 0 ? from - 1 : -1, 1 => x < width - 1 ? from + 1 : -1, 2 => y > 0 ? from - width : -1, _ => y < height - 1 ? from + width : -1 };
                    if (to < 0 || colored[to]) continue;
                    colored[to] = true;
                    pixels[4 * to] = pixels[4 * from];
                    pixels[4 * to + 1] = pixels[4 * from + 1];
                    pixels[4 * to + 2] = pixels[4 * from + 2];
                    queue[tail++] = to;
                }
            }
            return new Image((uint)width, (uint)height, pixels);
        }

        public static IntRect GetTextureRect(this Sprite sprite)
        {
            return (sprite != Sprite.None) ? new IntRect(((int)sprite % 8) * 512, ((int)sprite / 8) * 512, 512, 512) : new IntRect(0, 0, 0, 0);
        }

        public static Sprite GetSprite(TileType type)
        {
            return type switch
            {
                TileType.Lumber => Sprite.Lumber,
                TileType.Brick => Sprite.Brick,
                TileType.Wool => Sprite.Wool,
                TileType.Grain => Sprite.Grain,
                TileType.Ore => Sprite.Ore,
                TileType.Desert => Sprite.Desert,
                _ => Sprite.None
            };
        }

        public static Sprite GetSprite(Port.TradeType type)
        {
            return type switch
            {
                Port.TradeType.Generic => Sprite.QuestionMark,
                Port.TradeType.Lumber => Sprite.Lumber,
                Port.TradeType.Brick => Sprite.Brick,
                Port.TradeType.Wool => Sprite.Wool,
                Port.TradeType.Grain => Sprite.Grain,
                Port.TradeType.Ore => Sprite.Ore,
                _ => throw new InvalidOperationException(),
            };
        }

        public static Sprite GetSprite(ResourceCardType type)
        {
            return type switch
            {
                ResourceCardType.Unknown => Sprite.QuestionMark,
                ResourceCardType.Lumber => Sprite.Lumber,
                ResourceCardType.Brick => Sprite.Brick,
                ResourceCardType.Wool => Sprite.Wool,
                ResourceCardType.Grain => Sprite.Grain,
                ResourceCardType.Ore => Sprite.Ore,
                _ => throw new InvalidOperationException()
            };
        }

        public static Sprite GetSprite(DevelopmentCardType type)
        {
            return type switch
            {
                DevelopmentCardType.Unknown => Sprite.QuestionMark,
                DevelopmentCardType.Knight => Sprite.Knight,
                DevelopmentCardType.RoadBuilding => Sprite.Road,
                DevelopmentCardType.YearOfPlenty => Sprite.YearOfPlenty,
                DevelopmentCardType.Monopoly => Sprite.Monopoly,
                DevelopmentCardType.VictoryPoint => Sprite.VictoryPoint,
                _ => throw new InvalidOperationException()
            };
        }
    }
}
