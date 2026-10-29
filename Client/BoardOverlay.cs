using Client.UI;
using Common;
using SFML.Graphics;
using SFML.System;
using Edge = Common.Edge;
using Tile = Common.Tile;

namespace Client
{
    public enum TargetKind : byte { Intersection, Edge, Tile }

    /// <summary>
    /// A clickable spot on the board: an intersection, an edge or a tile (index into Board.Map.Values, as RobberAction uses)
    /// </summary>
    public readonly record struct BoardTarget(TargetKind Kind, int Index);

    /// <summary>
    /// What the board shows on top of the pieces: ghosts on the legal spots of the active build mode (bright roads and circles,
    /// pulsing in opacity and size), the hovered spot as a solid preview, a policy heat map, and rings on the tiles of the last roll. Drawn in
    /// map coordinates, so it moves and zooms with the board.
    /// </summary>
    public class BoardOverlay
    {
        public readonly record struct Ghost(BoardTarget Target, TextureAtlas.Sprite Sprite, Color Color);
        public readonly record struct HeatSpot(BoardTarget Target, float Share, float Intensity);

        // Pulse of the ghosts: opacity and size follow one sine, all ghosts in phase
        public const float PulsePeriod = 1.6f;
        private const float MinOpacity = 0.45f, MaxOpacity = 0.9f, MinScale = 0.94f, MaxScale = 1.06f;

        // Road, settlement and city ghosts share one light gray that no player color uses, so the legal spots stand out from
        // the pieces; only the hovered one shows the piece in the player's color. The circles stay translucent (at most
        // about 75% opaque) even at the top of the pulse.
        public static readonly Color GhostColor = new Color(222, 224, 230);

        private readonly BoardRenderer _renderer;
        private float _time;

        public List<Ghost> Ghosts { get; } = [];
        public List<HeatSpot> Heat { get; } = [];
        public BoardTarget? Hovered { get; set; }
        public Color HoverColor { get; set; } = Color.White;

        // Pieces placed a moment ago, with their age: a ring spreads from them and fades
        private readonly List<(BoardTarget target, Color color, float age)> _placements = [];
        private const float PlacementSeconds = 1.2f;

        // Tiles of the last roll and how long ago it was
        private readonly List<int> _rolledTiles = [];
        private float _rollAge = float.MaxValue;

        private readonly VertexArray _roadVertices = new(PrimitiveType.Triangles);
        private readonly RectangleShape _sprite;
        private readonly RectangleShape _bar = new();
        private readonly CircleShape _cap = new(1, 24);
        private readonly CircleShape _ring = new(1, 64);
        private readonly Text _label;

        public BoardOverlay(BoardRenderer renderer)
        {
            _renderer = renderer;
            _sprite = new RectangleShape(new Vector2f(renderer.SideLength * 0.5f, renderer.SideLength * 0.5f)) { Texture = TextureAtlas.Texture };
            _sprite.Origin = _sprite.Size / 2;
            _label = new Text("", GameScreen.BoldFont, 15);
        }

        public void Update(float deltaSeconds)
        {
            _time += deltaSeconds;
            if (_rollAge < float.MaxValue) _rollAge += deltaSeconds;
            for (int i = _placements.Count - 1; i >= 0; i--)
            {
                var p = _placements[i];
                if (p.age + deltaSeconds >= PlacementSeconds) _placements.RemoveAt(i);
                else _placements[i] = (p.target, p.color, p.age + deltaSeconds);
            }
        }

        /// <summary>
        /// Marks a piece that was just placed (or the robber's new tile) with a spreading ring
        /// </summary>
        public void ShowPlacement(BoardTarget target, Color color)
        {
            _placements.Add((target, color, 0));
        }

        /// <summary>
        /// Lights up the tiles that produced on a roll
        /// </summary>
        public void ShowRoll(IEnumerable<int> tileIndices)
        {
            _rolledTiles.Clear();
            _rolledTiles.AddRange(tileIndices);
            _rollAge = 0;
        }

        public void ClearPlacements()
        {
            _placements.Clear();
        }

        public void ClearRoll()
        {
            _rolledTiles.Clear();
            _rollAge = float.MaxValue;
        }

        /// <summary>
        /// Pulse phase from 0 (smallest, faintest) to 1 (largest, most opaque)
        /// </summary>
        private float Pulse => 0.5f - 0.5f * MathF.Cos(2 * MathF.PI * _time / PulsePeriod);

        public void Draw(RenderTarget target, Board board)
        {
            DrawRollRings(target, board);
            DrawHeat(target, board);
            foreach ((BoardTarget spot, Color color, float age) in _placements)
            {
                float t = age / PlacementSeconds;
                float eased = 1 - (1 - t) * (1 - t);
                float radius = _renderer.SideLength * (spot.Kind == TargetKind.Tile ? 0.5f : 0.2f) * (1 + 1.4f * eased);
                DrawRing(target, Position(board, spot), radius, 5 * (1 - t) + 1, Theme.WithAlpha(color, 0.9f * (1 - t)));
            }

            float pulse = Pulse;
            float opacity = MinOpacity + (MaxOpacity - MinOpacity) * pulse;
            float scale = MinScale + (MaxScale - MinScale) * pulse;
            foreach (Ghost ghost in Ghosts)
            {
                if (Hovered == ghost.Target || ghost.Target.Kind == TargetKind.Edge) continue;
                DrawPiece(target, board, ghost, opacity, scale, false);
            }

            // The hovered spot: a solid preview, a bit larger, with a glow and no pulse
            if (Hovered is BoardTarget hovered)
            {
                int index = Ghosts.FindIndex(g => g.Target == hovered);
                if (index >= 0 && hovered.Kind != TargetKind.Edge) DrawPiece(target, board, Ghosts[index], 1f, 1.1f, true);
                else if (index < 0) DrawHoverMarker(target, board, hovered);
            }
        }

        /// <summary>
        /// What lies under the buildings: road ghosts and the heat map's roads, in the shape of built roads
        /// </summary>
        public void DrawUnderBuildings(RenderTarget target, Board board)
        {
            _roadVertices.Clear();
            foreach (HeatSpot spot in Heat.Where(s => s.Target.Kind == TargetKind.Edge).OrderBy(s => s.Intensity))
                _renderer.AppendRoad(_roadVertices, board.Edges[spot.Target.Index], Theme.Heat(spot.Intensity));

            // Road ghosts pulse in opacity only; the hovered one is solid
            float opacity = MinOpacity + (MaxOpacity - MinOpacity) * Pulse;
            foreach (Ghost ghost in Ghosts.Where(g => g.Target.Kind == TargetKind.Edge))
                _renderer.AppendRoad(_roadVertices, board.Edges[ghost.Target.Index], Hovered == ghost.Target ? ghost.Color : Theme.WithAlpha(GhostColor, opacity * 0.8f));
            target.Draw(_roadVertices);
        }

        private void DrawPiece(RenderTarget target, Board board, Ghost ghost, float opacity, float scale, bool hovered)
        {
            switch (ghost.Target.Kind)
            {
                case TargetKind.Edge:
                    {
                        (Vector2f a, Vector2f b) = _renderer.GetEdgePoints(board.Edges[ghost.Target.Index]);
                        float width = _renderer.SideLength * 0.13f * scale;
                        if (hovered) DrawRoad(target, a, b, width * 2.2f, Theme.WithAlpha(Color.White, 0.25f), scale);
                        DrawRoad(target, a, b, width + 5, Theme.WithAlpha(Color.White, opacity * (hovered ? 1f : 0.85f)), scale);
                        DrawRoad(target, a, b, width, Theme.WithAlpha(ghost.Color, opacity), scale);
                        break;
                    }
                case TargetKind.Intersection:
                    {
                        Vector2f center = _renderer.GetIntersectionCenter(board.Intersections[ghost.Target.Index], board);
                        float radius = _renderer.SideLength * 0.17f;
                        if (hovered)
                        {
                            // The piece itself in the player's color; a city on a dark disc that covers the settlement it replaces
                            if (ghost.Sprite == TextureAtlas.Sprite.City)
                            {
                                DrawDisc(target, center, radius * 1.5f, new Color(8, 25, 76, 200));
                                DrawRing(target, center, radius * 1.5f, 3, GhostColor);
                            }
                            DrawSprite(target, ghost.Sprite, center, 1.1f, ghost.Color);
                        }
                        else if (ghost.Sprite == TextureAtlas.Sprite.City)
                        {
                            // A ring around the settlement that would become a city, filled enough to stand out from it while
                            // its outline still shows through
                            DrawDisc(target, center, radius * 1.45f * scale + 2, new Color(8, 25, 76, (byte)(160 * opacity)));
                            DrawDisc(target, center, radius * 1.45f * scale, Theme.WithAlpha(GhostColor, 0.3f + 0.4f * opacity));
                            DrawRing(target, center, radius * 1.45f * scale, 5, Theme.WithAlpha(GhostColor, 0.35f + 0.45f * opacity));
                        }
                        else
                        {
                            // A disc on the free spot, with a dark edge so it reads on light tiles as well
                            DrawDisc(target, center, radius * scale + 2, new Color(8, 25, 76, (byte)(160 * opacity)));
                            DrawDisc(target, center, radius * scale, Theme.WithAlpha(GhostColor, opacity * 0.8f));
                        }
                        break;
                    }
                case TargetKind.Tile:
                    {
                        // A ring around the token; the robber itself where it would stand, faint until hovered
                        Vector2f center = _renderer.GetTileCenter(board.Map.Values[ghost.Target.Index]);
                        float radius = _renderer.SideLength * 0.42f * scale;
                        DrawRing(target, center, radius, hovered ? 6 : 4, Theme.WithAlpha(Color.White, hovered ? 1f : opacity * 0.75f));
                        float offset = _renderer.SideLength / 2.5f + _renderer.IconSize / 3.25f;
                        float alpha = hovered ? 1f : opacity * 0.35f;
                        DrawSprite(target, ghost.Sprite, center + new Vector2f(-offset, 0), 0.93f, Theme.WithAlpha(ghost.Color, alpha));
                        DrawSprite(target, ghost.Sprite, center + new Vector2f(offset, 0), 0.93f, Theme.WithAlpha(ghost.Color, alpha));
                        break;
                    }
            }
        }

        private void DrawHoverMarker(RenderTarget target, Board board, BoardTarget hovered)
        {
            Color color = Theme.WithAlpha(HoverColor, 0.9f);
            switch (hovered.Kind)
            {
                case TargetKind.Edge:
                    (Vector2f a, Vector2f b) = _renderer.GetEdgePoints(board.Edges[hovered.Index]);
                    DrawRoad(target, a, b, _renderer.SideLength * 0.2f, Theme.WithAlpha(color, 0.45f), 1f);
                    break;
                case TargetKind.Intersection:
                    DrawRing(target, _renderer.GetIntersectionCenter(board.Intersections[hovered.Index], board), _renderer.SideLength * 0.22f, 4, color);
                    break;
                case TargetKind.Tile:
                    DrawRing(target, _renderer.GetTileCenter(board.Map.Values[hovered.Index]), _renderer.SideLength * 0.7f, 5, color);
                    break;
            }
        }

        /// <summary>
        /// The policy heat map: every spot in its heat color, the share as a label
        /// </summary>
        private void DrawHeat(RenderTarget target, Board board)
        {
            if (Heat.Count == 0) return;

            // Weakest first, so the strongest spots are on top
            foreach (HeatSpot spot in Heat.OrderBy(s => s.Intensity))
            {
                Color color = Theme.Heat(spot.Intensity);
                switch (spot.Target.Kind)
                {
                    case TargetKind.Edge:
                        break;
                    case TargetKind.Intersection:
                        Vector2f c = _renderer.GetIntersectionCenter(board.Intersections[spot.Target.Index], board);
                        DrawDisc(target, c, _renderer.SideLength * 0.2f, new Color(8, 25, 76, 150));
                        DrawDisc(target, c, _renderer.SideLength * 0.15f, color);
                        break;
                    case TargetKind.Tile:
                        DrawRing(target, _renderer.GetTileCenter(board.Map.Values[spot.Target.Index]), _renderer.SideLength * 0.45f, 8, color);
                        break;
                }
            }

            foreach (HeatSpot spot in Heat)
            {
                Vector2f position = spot.Target.Kind switch
                {
                    TargetKind.Edge => Mid(_renderer.GetEdgePoints(board.Edges[spot.Target.Index])),
                    TargetKind.Intersection => _renderer.GetIntersectionCenter(board.Intersections[spot.Target.Index], board) + new Vector2f(0, -_renderer.SideLength * 0.3f),
                    _ => _renderer.GetTileCenter(board.Map.Values[spot.Target.Index]) + new Vector2f(0, _renderer.SideLength * 0.55f),
                };
                DrawLabel(target, position, $"{spot.Share * 100:0}%", Theme.Heat(spot.Intensity));
            }
        }

        private void DrawLabel(RenderTarget target, Vector2f center, string text, Color border)
        {
            _label.DisplayedString = text;
            FloatRect bounds = _label.GetLocalBounds();
            Vector2f size = new Vector2f(bounds.Width + 16, 22);
            Vector2f min = center - size / 2;

            DrawPill(target, min - new Vector2f(2, 2), size + new Vector2f(4, 4), border);
            DrawPill(target, min, size, new Color(11, 22, 64));

            _label.FillColor = Color.White;
            _label.Origin = new Vector2f(bounds.Left + bounds.Width / 2, bounds.Top + bounds.Height / 2);
            _label.Position = ClientUtils.RoundVec2f(center);
            target.Draw(_label);
        }

        private void DrawPill(RenderTarget target, Vector2f min, Vector2f size, Color color)
        {
            float r = size.Y / 2;
            _bar.Size = new Vector2f(Math.Max(0, size.X - size.Y), size.Y);
            _bar.Origin = new Vector2f(0, 0);
            _bar.Rotation = 0;
            _bar.Position = min + new Vector2f(r, 0);
            _bar.FillColor = color;
            target.Draw(_bar);
            DrawDisc(target, min + new Vector2f(r, r), r, color);
            DrawDisc(target, min + new Vector2f(size.X - r, r), r, color);
        }

        private void DrawRollRings(RenderTarget target, Board board)
        {
            if (_rolledTiles.Count == 0) return;

            // A bright flash right after the roll that settles into a steady ring
            float flash = MathF.Max(0, 1 - _rollAge / 0.8f);
            float radius = _renderer.SideLength / 3 + 7 + 10 * flash;
            foreach (int tileIdx in _rolledTiles)
            {
                Vector2f center = _renderer.GetTileCenter(board.Map.Values[tileIdx]);
                DrawRing(target, center, radius + 6, 6, Theme.WithAlpha(Theme.Highlight, 0.18f + 0.3f * flash));
                DrawRing(target, center, radius, 4, Theme.WithAlpha(Theme.Highlight, 0.95f));
            }
        }

        private void DrawRoad(RenderTarget target, Vector2f a, Vector2f b, float width, Color color, float scale)
        {
            // A separate piece: shortened at both ends, scaled around its middle
            Vector2f mid = (a + b) / 2;
            Vector2f dir = b - a;
            float length = MathF.Sqrt(dir.X * dir.X + dir.Y * dir.Y);
            float pieceLength = MathF.Max(0, (length - _renderer.SideLength * 0.22f) * scale - width);
            float angle = MathF.Atan2(dir.Y, dir.X) * 180 / MathF.PI;

            _bar.Size = new Vector2f(pieceLength, width);
            _bar.Origin = _bar.Size / 2;
            _bar.Position = mid;
            _bar.Rotation = angle;
            _bar.FillColor = color;
            target.Draw(_bar);

            Vector2f unit = length > 0 ? dir / length : new Vector2f(1, 0);
            DrawDisc(target, mid + unit * (pieceLength / 2), width / 2, color);
            DrawDisc(target, mid - unit * (pieceLength / 2), width / 2, color);
        }

        private void DrawSprite(RenderTarget target, TextureAtlas.Sprite sprite, Vector2f center, float scale, Color color)
        {
            _sprite.TextureRect = sprite.GetTextureRect();
            _sprite.Position = center;
            _sprite.Scale = new Vector2f(scale, scale);
            _sprite.FillColor = color;
            target.Draw(_sprite);
        }

        private void DrawDisc(RenderTarget target, Vector2f center, float radius, Color color)
        {
            _cap.Radius = radius;
            _cap.Origin = new Vector2f(radius, radius);
            _cap.Position = center;
            _cap.FillColor = color;
            _cap.OutlineThickness = 0;
            target.Draw(_cap);
        }

        private void DrawRing(RenderTarget target, Vector2f center, float radius, float thickness, Color color)
        {
            _ring.Radius = radius;
            _ring.Origin = new Vector2f(radius, radius);
            _ring.Position = center;
            _ring.FillColor = Color.Transparent;
            _ring.OutlineColor = color;
            _ring.OutlineThickness = thickness;
            target.Draw(_ring);
        }

        private static Vector2f Mid((Vector2f a, Vector2f b) points) => (points.a + points.b) / 2;

        /// <summary>
        /// The spot under a point in map coordinates among the candidates: intersections first, then edges, then tiles
        /// </summary>
        public BoardTarget? HitTest(Board board, Vector2f point, IEnumerable<BoardTarget> candidates)
        {
            BoardTarget? best = null;
            float bestScore = float.MaxValue;
            foreach (BoardTarget candidate in candidates)
            {
                float distance, limit, priority;
                switch (candidate.Kind)
                {
                    case TargetKind.Intersection:
                        distance = Length(point - _renderer.GetIntersectionCenter(board.Intersections[candidate.Index], board));
                        limit = _renderer.SideLength * 0.28f;
                        priority = 0;
                        break;
                    case TargetKind.Edge:
                        (Vector2f a, Vector2f b) = _renderer.GetEdgePoints(board.Edges[candidate.Index]);
                        distance = SegmentDistance(point, a, b);
                        limit = _renderer.SideLength * 0.2f;
                        priority = 1000;
                        break;
                    default:
                        distance = Length(point - _renderer.GetTileCenter(board.Map.Values[candidate.Index]));
                        limit = _renderer.SideLength * 0.8f;
                        priority = 2000;
                        break;
                }
                if (distance > limit) continue;
                float score = priority + distance;
                if (score < bestScore)
                {
                    bestScore = score;
                    best = candidate;
                }
            }
            return best;
        }

        /// <summary>
        /// Where a spot is in map coordinates
        /// </summary>
        public Vector2f Position(Board board, BoardTarget target) => target.Kind switch
        {
            TargetKind.Intersection => _renderer.GetIntersectionCenter(board.Intersections[target.Index], board),
            TargetKind.Edge => Mid(_renderer.GetEdgePoints(board.Edges[target.Index])),
            _ => _renderer.GetTileCenter(board.Map.Values[target.Index]),
        };

        private static float Length(Vector2f v) => MathF.Sqrt(v.X * v.X + v.Y * v.Y);

        private static float SegmentDistance(Vector2f p, Vector2f a, Vector2f b)
        {
            Vector2f ab = b - a;
            float t = Math.Clamp(((p.X - a.X) * ab.X + (p.Y - a.Y) * ab.Y) / (ab.X * ab.X + ab.Y * ab.Y), 0, 1);
            return Length(p - (a + ab * t));
        }
    }
}
