using Common;
using Common.Actions;
using SFML.Graphics;
using SFML.System;
using SFML.Window;
using static Common.Direction;
using static Common.Tile;
using Color = SFML.Graphics.Color;
using Edge = Common.Edge;
using Tile = Common.Tile;

namespace Client
{
    public class BoardRenderer : Drawable
    {
        public Board Board { get; set; }

        // Rendering options
        public bool DrawYieldPoints = true;
        public bool DrawTokenShadows = true;
        public bool DrawIntersectionMarkers = false;
        public bool DrawEdgeMarkers = false;
        public bool DrawCenterMarkers = false;

        // Static geometry
        private VertexBuffer _tiles;
        private VertexArray _grid;
        private VertexArray _overlay;
        private VertexArray _portBridges;
        private VertexArray _roads;

        // Number tokens: the classic yellow face with a faint rim, bold black numbers, the 6 and 8 in red
        private static readonly Color TokenFace = new Color(230, 230, 120);
        private static readonly Color TokenRim = new Color(200, 196, 100);
        private float _tokenRadius;
        private float _tokenTextSize;
        private Text _tokenText;
        private readonly List<(Vector2i pixel, int number, Color ink)> _tokenNumbers = [];
        private CircleShape _tokenBase;
        private CircleShape _yieldPoint;

        // Port text
        private Text _portText;

        // Robber
        private RectangleShape _robber;

        private CircleShape _intersectionMarker;
        private RectangleShape _intersectionRect;
        private RectangleShape _edgeMarker;
        private CircleShape _centerMarker;

        #region Hexagon Dimensions
        private float _sideLength;
        private float _borderWidth;

        // Outer circle radius
        public float SideLength
        {
            get => _sideLength;
            set
            {
                _sideLength = value;
                FlatSideLength = (MathF.Sqrt(3.0f) / 2.0f) * _sideLength;
            }
        }

        // Inner circle radius (calculated automatically through property "SideLength", do not set manually)
        public float FlatSideLength
        {
            get;
            private set;
        }
        #endregion

        /// <summary>
        /// Drawn after the roads and tokens, before the buildings
        /// </summary>
        public System.Action<RenderTarget, RenderStates>? BeforeBuildings;

        public float IconSize
        {
            get => SideLength * 7 / 15;
        }

        public BoardRenderer(Board board, float sideLength, float borderWidth)
        {
            Board = board;
            SideLength = sideLength;
            _borderWidth = borderWidth;

            _tiles = new VertexBuffer(Board.Map.Width * Board.Map.Height * 4 * 3, PrimitiveType.Triangles, VertexBuffer.UsageSpecifier.Static);
            _grid = new VertexArray(PrimitiveType.Triangles);
            _overlay = new VertexArray(PrimitiveType.Triangles);
            _portBridges = new VertexArray(PrimitiveType.Triangles);
            _roads = new VertexArray(PrimitiveType.Triangles);

            _tokenRadius = SideLength / 3;
            _tokenText = new Text("", GameScreen.BoldFont);
            _tokenTextSize = sideLength * 0.35f;

            _tokenBase = new CircleShape(_tokenRadius, 64);
            _tokenBase.Origin = new Vector2f(_tokenBase.Radius, _tokenBase.Radius);

            _yieldPoint = new CircleShape(SideLength / 42, 12);
            _yieldPoint.Origin = new Vector2f(_yieldPoint.Radius, _yieldPoint.Radius);

            _portText = new Text("", GameScreen.Font);
            _portText.CharacterSize = (uint)Math.Round(sideLength * (8.0f / 30.0f));
            _portText.FillColor = Color.White;

            _robber = new RectangleShape(new Vector2f(IconSize, IconSize));
            _robber.Origin = _robber.Size / 2;
            _robber.Texture = TextureAtlas.Texture;
            _robber.TextureRect = TextureAtlas.Sprite.Robber.GetTextureRect();
            _robber.FillColor = Color.Black;

            _intersectionMarker = new CircleShape(SideLength / 4, 32);
            _intersectionMarker.Origin = new Vector2f(_intersectionMarker.Radius, _intersectionMarker.Radius);
            _intersectionMarker.FillColor = Color.Red;

            _intersectionRect = new RectangleShape(new Vector2f(SideLength * 0.5f, SideLength * 0.5f));
            _intersectionRect.Origin = new Vector2f(_intersectionRect.Size.X / 2, _intersectionRect.Size.Y / 2);
            _intersectionRect.Texture = TextureAtlas.Texture;

            _edgeMarker = new RectangleShape(new Vector2f(SideLength * 0.35f, SideLength));
            _edgeMarker.Origin = _edgeMarker.Size / 2;
            _edgeMarker.FillColor = Color.Cyan;

            _centerMarker = new CircleShape(SideLength * 0.85f, 32);
            _centerMarker.Origin = new Vector2f(_centerMarker.Radius, _centerMarker.Radius);
            _centerMarker.FillColor = Color.Magenta;

            // Create Geometry
            Update();
        }

        // Update Tile Geometry
        public void Update()
        {
            uint width = Board.Map.Width;
            uint height = Board.Map.Height;
            Vertex[] tempTiles = new Vertex[width * height * 4 * 3];
            _grid.Clear();
            _overlay.Clear();
            _portBridges.Clear();
            _roads.Clear();

            // Build tiles
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Tile tile = Board.Map.GetTile(x, y);

                    // Calculate Color from Tile Value
                    Color tileColor = ColorPalette.GetTileColor(Board.Map.GetTile(x, y).Type);
                    Color? gridColor = ColorPalette.GetTileBorderColor(Board.Map.GetTile(x, y));

                    Vector2f center = new Vector2f(x * 2 * FlatSideLength + ((y % 2 == 0) ? FlatSideLength : 0), y * 1.5f * SideLength);
                    
                    Vector2f[] points = new Vector2f[6];
                    Vertex[] vertices = new Vertex[6];

                    // Generate vertices
                    for (int i = 0; i < 6; i++)
                    {
                        // Hex points
                        points[i] = center + SideLength * ClientUtils.EulerAngleToVec2f(30.0f + 60.0f * i);

                        // Hex vertices
                        vertices[i] = new Vertex(points[i], tileColor);
                    }

                    // Appending vertices in groups of three in correct order for triangle drawing
                    int offset = (int)(x * height + y) * 4 * 3;
                    // Top triangle
                    tempTiles[offset + 0] = vertices[0];
                    tempTiles[offset + 1] = vertices[1];
                    tempTiles[offset + 2] = vertices[2];
                    // Middle top left triangle
                    tempTiles[offset + 3] = vertices[0];
                    tempTiles[offset + 4] = vertices[2];
                    tempTiles[offset + 5] = vertices[3];
                    // Middle bottom right triangle
                    tempTiles[offset + 6] = vertices[0];
                    tempTiles[offset + 7] = vertices[3];
                    tempTiles[offset + 8] = vertices[5];
                    // Bottom triangle
                    tempTiles[offset + 9]  = vertices[3];
                    tempTiles[offset + 10] = vertices[4];
                    tempTiles[offset + 11] = vertices[5];

                    // Build grid
                    if(gridColor.HasValue)
                    {
                        Vector2f[] insetPoints  = new Vector2f[6];
                        Vector2f[] outsetPoints = new Vector2f[6];
                        Vertex[] gridVertices       = new Vertex[6];
                        Vertex[] gridInsetVertices  = new Vertex[6];
                        Vertex[] gridOutsetVertices = new Vertex[6];

                        // Generate vertices
                        for(int i = 0; i < 6; i++)
                        {
                            // Inner points of hex (side length inset by half the border width)
                            insetPoints[i]  = center + (SideLength - _borderWidth / 2) * ClientUtils.EulerAngleToVec2f(30.0f + 60.0f * i);

                            // Outer points of hex (side length outset by half the border width)
                            outsetPoints[i] = center + (SideLength + _borderWidth / 2) * ClientUtils.EulerAngleToVec2f(30.0f + 60.0f * i);

                            // Grid vertices
                            gridVertices[i] = new Vertex(points[i], gridColor.Value);
                            gridInsetVertices[i] = new Vertex(insetPoints[i], gridColor.Value);
                            gridOutsetVertices[i] = new Vertex(outsetPoints[i], gridColor.Value);
                        }

                        // Append grid triangles
                        for(int i = 0; i < 6; i++)
                        {
                            // Outer Triangle
                            _grid.Append(gridOutsetVertices[i]);
                            _grid.Append(gridOutsetVertices[(i + 1) % 6]);
                            _grid.Append(gridInsetVertices[i]);

                            // Inner Triangle
                            _grid.Append(gridInsetVertices[i]);
                            _grid.Append(gridInsetVertices[(i + 1) % 6]);
                            _grid.Append(gridOutsetVertices[(i + 1) % 6]);
                        }
                    }

                    // Yield icons for land tiles
                    if(tile.IsLandTile())
                    {
                        Color color = ColorPalette.GetTileIconColor(tile.Type);

                        IntRect textureRect = TextureAtlas.GetSprite(tile.Type).GetTextureRect();
                        float iconSize = SideLength * 7 / 15;

                        // Place icon above and below
                        for(int i = -1; i < 2; i += 2)
                        {
                            Vector2f position = new Vector2f(center.X - iconSize / 2, center.Y - iconSize / 2 + i * (SideLength / 1.75f));
                            Vertex vertTL = new Vertex(position, color, new Vector2f(textureRect.Left, textureRect.Top));
                            Vertex vertTR = new Vertex(position + new Vector2f(iconSize, 0), color, new Vector2f(textureRect.Left + textureRect.Width, textureRect.Top));
                            Vertex vertBL = new Vertex(position + new Vector2f(0, iconSize), color, new Vector2f(textureRect.Left, textureRect.Top + textureRect.Height));
                            Vertex vertBR = new Vertex(position + new Vector2f(iconSize, iconSize), color, new Vector2f(textureRect.Left + textureRect.Width, textureRect.Top + textureRect.Height));

                            _overlay.Append(vertTL);
                            _overlay.Append(vertTR);
                            _overlay.Append(vertBL);
                            _overlay.Append(vertTR);
                            _overlay.Append(vertBR);
                            _overlay.Append(vertBL);
                        }
                    }
                }
            }
            _tiles.Update(tempTiles);

            // Build port bridges and icons
            foreach (Port port in Board.Ports)
            {
                Color portColor = new Color(0x65, 0x35, 0x0F);
                Vector2f center = new Vector2f(port.AnchorTileX * 2 * FlatSideLength + ((port.AnchorTileY % 2 == 0) ? FlatSideLength : 0), port.AnchorTileY * 1.5f * SideLength);

                // Coastal bridges
                Direction.Corner leftCorner, rightCorner;
                float leftAngle, rightAngle;

                (leftCorner, rightCorner) = port.AnchorDirection.GetAdjacentCorners();
                leftAngle = leftCorner.ToAngle();
                rightAngle = rightCorner.ToAngle();

                Vector2f leftDir = ClientUtils.EulerAngleToVec2f(leftAngle);
                Vector2f rightDir = ClientUtils.EulerAngleToVec2f(rightAngle);



                Vertex vertLeftOuter = new Vertex(center + (SideLength - _borderWidth / 2f) * leftDir, portColor);
                Vertex vertRightOuter = new Vertex(center + (SideLength - _borderWidth / 2f) * rightDir, portColor);
                Vertex vertLeftInner = new Vertex(center + SideLength * 0.75f * leftDir, portColor);
                Vertex vertRightInner = new Vertex(center + SideLength * 0.75f * rightDir, portColor);

                _portBridges.Append(vertLeftOuter);
                _portBridges.Append(vertRightOuter);
                _portBridges.Append(vertLeftInner);

                _portBridges.Append(vertLeftInner);
                _portBridges.Append(vertRightOuter);
                _portBridges.Append(vertRightInner);


                // Icons
                // Port icon (bottom)
                IntRect textureRect = TextureAtlas.Sprite.Port.GetTextureRect();

                Vector2f position = new Vector2f(center.X - IconSize / 2, center.Y - IconSize / 2 + (SideLength / 2.5f));
                Vertex vertTopLeft = new Vertex(position, portColor, new Vector2f(textureRect.Left, textureRect.Top));
                Vertex vertTopRight = new Vertex(position + new Vector2f(IconSize, 0), portColor, new Vector2f(textureRect.Left + textureRect.Width, textureRect.Top));
                Vertex vertBottomLeft = new Vertex(position + new Vector2f(0, IconSize), portColor, new Vector2f(textureRect.Left, textureRect.Top + textureRect.Height));
                Vertex vertBottomRight = new Vertex(position + new Vector2f(IconSize, IconSize), portColor, new Vector2f(textureRect.Left + textureRect.Width, textureRect.Top + textureRect.Height));

                _overlay.Append(vertTopLeft);
                _overlay.Append(vertTopRight);
                _overlay.Append(vertBottomLeft);
                _overlay.Append(vertTopRight);
                _overlay.Append(vertBottomRight);
                _overlay.Append(vertBottomLeft);

                // Type icon (top)
                Color resourceColor = ColorPalette.GetPortIconColor(port.Type);
                textureRect = TextureAtlas.GetSprite(port.Type).GetTextureRect();

                position = new Vector2f(center.X - IconSize / 2, center.Y - IconSize / 2 - (SideLength / 2.5f));
                vertTopLeft = new Vertex(position, resourceColor, new Vector2f(textureRect.Left, textureRect.Top));
                vertTopRight = new Vertex(position + new Vector2f(IconSize, 0), resourceColor, new Vector2f(textureRect.Left + textureRect.Width, textureRect.Top));
                vertBottomLeft = new Vertex(position + new Vector2f(0, IconSize), resourceColor, new Vector2f(textureRect.Left, textureRect.Top + textureRect.Height));
                vertBottomRight = new Vertex(position + new Vector2f(IconSize, IconSize), resourceColor, new Vector2f(textureRect.Left + textureRect.Width, textureRect.Top + textureRect.Height));

                _overlay.Append(vertTopLeft);
                _overlay.Append(vertTopRight);
                _overlay.Append(vertBottomLeft);
                _overlay.Append(vertTopRight);
                _overlay.Append(vertBottomRight);
                _overlay.Append(vertBottomLeft);
            }

            // Build roads
            foreach (Edge edge in Board.Edges)
            {
                // Skip edges without roads
                if (Board.GetRoad(edge.Index) == Edge.BuildingType.None) continue;
                AppendRoad(_roads, edge, ColorPalette.GetPlayerColor(Board.GetRoadOwner(edge.Index)));
            }
        }

        /// <summary>
        /// The triangles of a road on an edge, with pointed ends that meet the neighboring roads
        /// </summary>
        public void AppendRoad(VertexArray array, Edge edge, Color playerColor)
        {
            // Anchor edge to one of its two neighboring tiles
            (Tile anchor, Direction.Tile anchorDir) = GetEdgeAnchor(edge);

            // Determine corner positions
            (Direction.Corner leftCorner, Direction.Corner rightCorner) = anchorDir.GetAdjacentCorners();
            float leftAngle  = leftCorner.ToAngle();
            float rightAngle = rightCorner.ToAngle();

            Vector2f tileCenter     = GetTileCenter(anchor);
            Vector2f leftDir        = ClientUtils.EulerAngleToVec2f(leftAngle);
            Vector2f rightDir       = ClientUtils.EulerAngleToVec2f(rightAngle);
            Vector2f leftInnerDir   = ClientUtils.EulerAngleToVec2f(leftCorner.Mirror().ToAngle());
            Vector2f rightInnerDir  = ClientUtils.EulerAngleToVec2f(rightCorner.Mirror().ToAngle());
            Vector2f leftOuterDir   = ClientUtils.EulerAngleToVec2f(leftCorner.Rotate(1).ToAngle());
            Vector2f rightOuterDir  = ClientUtils.EulerAngleToVec2f(rightCorner.Rotate(-1).ToAngle());

            // Build vertices
            Vertex left         = new Vertex(tileCenter + SideLength * leftDir, playerColor);
            Vertex right        = new Vertex(tileCenter + SideLength * rightDir, playerColor);
            Vertex leftInset    = new Vertex(left.Position  + leftInnerDir  * (_borderWidth / 2), playerColor);
            Vertex rightInset   = new Vertex(right.Position + rightInnerDir * (_borderWidth / 2), playerColor);
            Vertex leftOutset   = new Vertex(left.Position  + leftOuterDir  * (_borderWidth / 2), playerColor);
            Vertex rightOutset  = new Vertex(right.Position + rightOuterDir * (_borderWidth / 2), playerColor);

            array.Append(leftInset);
            array.Append(leftOutset);
            array.Append(rightInset);

            array.Append(leftOutset);
            array.Append(rightOutset);
            array.Append(rightInset);

            array.Append(left);
            array.Append(leftInset);
            array.Append(leftOutset);

            array.Append(right);
            array.Append(rightInset);
            array.Append(rightOutset);
        }

        /// <summary>
        /// The token numbers in screen pixels: rasterized at the size they appear at and snapped to whole pixels, so every
        /// number is equally sharp and centered at any zoom (in map coordinates each would land on another pixel fraction)
        /// </summary>
        private void DrawTokenNumbers(RenderTarget target, RenderStates states)
        {
            // A copy: GetView returns the window's current view itself, which SetView replaces
            View mapView = new View(target.GetView());
            float zoom = mapView.Size.X / target.Size.X;
            target.SetView(new View(new FloatRect(0, 0, target.Size.X, target.Size.Y)));
            _tokenText.CharacterSize = (uint)Math.Max(1, Math.Round(_tokenTextSize / zoom));

            foreach ((Vector2i pixel, int value, Color ink) in _tokenNumbers)
            {
                string number = value.ToString();
                _tokenText.DisplayedString = number;
                _tokenText.FillColor = ink;

                // Centered on the ink, except the 11: the flags of its ones stretch the ink box to the left, so its
                // stems would sit right of the middle. Open Sans figures are tabular with the 1 in the middle of its
                // advance, so the 11 is centered on its advance width instead.
                FloatRect bounds = _tokenText.GetLocalBounds();
                float originX = bounds.Left + bounds.Width / 2.0f;
                if (number == "11")
                    originX = (_tokenText.FindCharacterPos((uint)number.Length).X - _tokenText.FindCharacterPos(0).X) / 2.0f;
                _tokenText.Origin = new Vector2f(MathF.Round(originX), MathF.Round(bounds.Top + bounds.Height / 2.0f));
                _tokenText.Position = new Vector2f(pixel.X, pixel.Y);
                target.Draw(_tokenText, states);
            }

            _tokenNumbers.Clear();
            target.SetView(mapView);
        }

        public Vector2f GetTileCenter(Tile tile)
        {
            return GetTileCenter(tile.X, tile.Y);
        }

        // Get the Coordinates of the Center of a Given Tile Relative to the Origin of the HexMap
        public Vector2f GetTileCenter(int x, int y)
        {
            if (x >= Board.Map.Width || x < 0) throw new ArgumentOutOfRangeException("x", "x needs to be 0 <= x < Width");
            else if (y >= Board.Map.Height || y < 0) throw new ArgumentOutOfRangeException("y", "y needs to be 0 <= y < Height");
            else
            {
                return new Vector2f(x * 2 * FlatSideLength + ((y % 2 == 0) ? FlatSideLength : 0), y * 1.5f * SideLength);
            }
        }

        public Vector2f GetIntersectionCenter(Intersection intersection, Board board)
        {
            if (board.Adjacency.GetTiles(intersection).Count() == 0)
            {
                throw new InvalidOperationException("Intersection was not properly initialized and lacks adjacent tiles");
            }

            Direction.Corner? adjacentParentDir = null;
            Tile? adjacentParent = null;
            foreach (Direction.Corner corner in Enum.GetValues(typeof(Direction.Corner)))
            {
                Tile? tile = board.Adjacency.GetTile(intersection, corner);

                if (tile != null)
                {
                    adjacentParentDir = corner;
                    adjacentParent = tile;
                    break;
                }
            }

            Vector2f parentCenter = GetTileCenter(adjacentParent!);
            float angle = adjacentParentDir!.Value.ToAngle();

            return parentCenter + SideLength * ClientUtils.EulerAngleToVec2f(angle);
        }

        public (Tile anchor, Direction.Tile anchorDir) GetEdgeAnchor(Edge edge)
        {
            // Anchor the edge to one of two neighboring tiles
            Tile anchor;
            Direction.Tile anchorDir;

            // Anchor to east tile, if it exists
            Tile? eastTile = Board.Adjacency.GetEastTile(edge);
            Tile? westTile = Board.Adjacency.GetWestTile(edge);
            if (eastTile != null)
            {
                anchor = eastTile;
                anchorDir = edge.Direction.ToWestTileDir();
            }
            // Anchor to west tile otherwise
            else if (westTile != null)
            {
                anchor = westTile;
                anchorDir = edge.Direction.ToEastTileDir();
            }
            // If there are no adjacent tiles, the edge state must be invalid
            else
            {
                throw new InvalidOperationException("Edge doesn't have adjacent tiles");
            }

            return (anchor, anchorDir);
        }

        public (Vector2f left, Vector2f right) GetEdgePoints(Edge edge)
        {
            // Anchor edge to one of its two neighboring tiles
            (Tile anchor, Direction.Tile anchorDir) = GetEdgeAnchor(edge);

            // Determine corner positions
            (Direction.Corner leftCorner, Direction.Corner rightCorner) = anchorDir.GetAdjacentCorners();
            float leftAngle = leftCorner.ToAngle();
            float rightAngle = rightCorner.ToAngle();

            Vector2f tileCenter = GetTileCenter(anchor);
            Vector2f leftDir = ClientUtils.EulerAngleToVec2f(leftAngle);
            Vector2f rightDir = ClientUtils.EulerAngleToVec2f(rightAngle);

            Vector2f left = tileCenter + SideLength * leftDir;
            Vector2f right = tileCenter + SideLength * rightDir;

            return (left, right);
        }
        
        public void Draw(RenderTarget target, RenderStates states) { }

        // Render the HexMap to a RenderTarget
        public void Draw(RenderTarget target, RenderStates states, GameState state, sbyte playerIdx) // TODO: Remove state debug parameter
        {
            target.Draw(_tiles, states);
            target.Draw(_grid, states);
            target.Draw(_portBridges, states);
            target.Draw(_roads, states);
            states.Texture = TextureAtlas.Texture;
            target.Draw(_overlay, states);

            // Draw number tokens
            for (int y = 0; y < Board.Map.Height; y++)
            {
                for (int x = 0; x < Board.Map.Width; x++)
                {
                    Tile value = Board.Map.GetTile(x, y);
                    if (value.HasYield())
                    {
                        Vector2f center = GetTileCenter(x, y);
                        bool hot = value.Number == 6 || value.Number == 8;
                        Color ink = hot ? Color.Red : Color.Black;

                        // Soft shadow, a rim and an ivory face
                        if (DrawTokenShadows)
                        {
                            _tokenBase.Radius = _tokenRadius + 1;
                            _tokenBase.Origin = new Vector2f(_tokenBase.Radius, _tokenBase.Radius);
                            _tokenBase.FillColor = new Color(0, 0, 0, 70);
                            _tokenBase.Position = center + new Vector2f(3, 3);
                            target.Draw(_tokenBase, states);
                        }
                        _tokenBase.Radius = _tokenRadius;
                        _tokenBase.Origin = new Vector2f(_tokenRadius, _tokenRadius);
                        _tokenBase.FillColor = TokenRim;
                        _tokenBase.Position = center;
                        target.Draw(_tokenBase, states);
                        _tokenBase.Radius = _tokenRadius - 2f;
                        _tokenBase.Origin = new Vector2f(_tokenBase.Radius, _tokenBase.Radius);
                        _tokenBase.FillColor = TokenFace;
                        target.Draw(_tokenBase, states);

                        // The number a bit above the middle (drawn after the loop in screen pixels), the probability pips below it
                        Vector2f numberCenter = center + new Vector2f(0, DrawYieldPoints ? -_tokenRadius * 0.08f : 0);
                        _tokenNumbers.Add((target.MapCoordsToPixel(numberCenter), value.Number ?? 0, ink));

                        if (DrawYieldPoints)
                        {
                            int yieldPoints = value.YieldPoints;
                            float spacing = _tokenRadius / 6f;
                            _yieldPoint.FillColor = ink;
                            _yieldPoint.Position = center + new Vector2f(-spacing * (yieldPoints - 1) / 2f, _tokenRadius * 0.64f);
                            for (int i = 0; i < yieldPoints; i++)
                            {
                                target.Draw(_yieldPoint, states);
                                _yieldPoint.Position += new Vector2f(spacing, 0);
                            }
                        }
                    }
                }
            }

            DrawTokenNumbers(target, states);

            // Draw center markers
            if (DrawCenterMarkers)
            {
                foreach (Tile tile in Board.Map.Where(x => x.IsLandTile()))
                {
                    _centerMarker.Position = GetTileCenter(tile);
                    target.Draw(_centerMarker, states);
                }
            }

            // Draw edge markers
            if (DrawEdgeMarkers)
            {
                //foreach(Edge edge in Board.Edges)
                for (int i = 0; i < Board.Edges.Count; i++)
                {
                    Edge edge = Board.Edges[i];

                    RoadAction act = new RoadAction(playerIdx, i);
                    if (!act.IsValidFor(state)) continue;

                    (Vector2f left, Vector2f right) = GetEdgePoints(edge);

                    _edgeMarker.Position = (left + right) / 2f;

                    _edgeMarker.Rotation = edge.Direction switch
                    {
                        Direction.Edge.UpDown => 0f,
                        Direction.Edge.LeftTop => -60f,
                        Direction.Edge.RightTop => 60f,
                        _ => throw new InvalidOperationException()
                    };

                    target.Draw(_edgeMarker, states);
                }
            }

            // Draw intersection markers
            if (DrawIntersectionMarkers)
            {
                //foreach (Intersection intersection in Board.Intersections)
                for (int i = 0; i < Board.Intersections.Count; i++)
                {
                    Intersection intersection = Board.Intersections[i];

                    SettlementAction act = new SettlementAction(playerIdx, i);
                    CityAction act2 = new CityAction(playerIdx, i);
                    if (!act.IsValidFor(state) && !act2.IsValidFor(state)) continue;

                    _intersectionMarker.Position = GetIntersectionCenter(intersection, Board);
                    target.Draw(_intersectionMarker, states);
                }
            }

            // Things that belong under the buildings, like previews of roads
            BeforeBuildings?.Invoke(target, states);

            // Draw intersection buildings
            foreach (Intersection intersection in Board.Intersections)
            {
                Intersection.BuildingType building = Board.GetBuilding(intersection.Index);
                switch(building)
                {
                    case Intersection.BuildingType.Settlement:
                    case Intersection.BuildingType.City:
                        _intersectionRect.Position = GetIntersectionCenter(intersection, Board);
                        _intersectionRect.TextureRect = TextureAtlas.GetTextureRect(building == Intersection.BuildingType.Settlement ? TextureAtlas.Sprite.Settlement : TextureAtlas.Sprite.City);
                        _intersectionRect.FillColor = ColorPalette.GetPlayerColor(Board.GetBuildingOwner(intersection.Index));
                        target.Draw(_intersectionRect, states);
                        break;
                    default:
                        break;
                }
            }

            // Draw robber
            if(Board.Robber != null)
            {
                _robber.Position = GetTileCenter(Board.Robber) + new Vector2f(-SideLength / 2.5f - IconSize / 3.25f, 0);
                target.Draw(_robber, states);

                _robber.Position = GetTileCenter(Board.Robber) + new Vector2f(SideLength / 2.5f + IconSize / 3.25f, 0);
                target.Draw(_robber, states);
            }

            // Draw port text
            foreach(Port port in Board.Ports)
            {
                Vector2f center = GetTileCenter(port.AnchorTileX, port.AnchorTileY);
                _portText.DisplayedString = (port.Type == Port.TradeType.Generic) ? "3:1" : "2:1";
                FloatRect textBounds = _portText.GetLocalBounds();
                _portText.Origin = new Vector2f(textBounds.Left + textBounds.Width / 2f, textBounds.Top + textBounds.Height / 2f);
                _portText.Position = center + new Vector2f(-2, 0);

                target.Draw(_portText, states);
            }
        }
    }
}

