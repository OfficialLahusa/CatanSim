
using YamlDotNet.Serialization;

namespace Common
{
    public class Board
    {
        // Part of the board that never changes during a game and is shared by all copies of it
        private sealed class Layout
        {
            public HexMap<Tile> Map = null!;
            public List<Port> Ports = null!;
            public bool[] IntersectionFacesDownwards = [];
            public Direction.Edge[] EdgeDirections = [];
        }

        private Layout _layout;

        // Buildings and roads, the only part of the board that is copied
        private Intersection.BuildingType[] _buildings;
        private sbyte[] _buildingOwners;
        private Edge.BuildingType[] _roads;
        private sbyte[] _roadOwners;

        // Views onto the arrays above, so code can keep working with Intersection and Edge objects
        private List<Intersection> _intersections;
        private List<Edge> _edges;

        [YamlMember(Order = 0)]
        public HexMap<Tile> Map
        {
            get => _layout.Map;
            set => _layout.Map = value;
        }

        [YamlIgnore]
        public List<Intersection> Intersections => _intersections;

        [YamlIgnore]
        public List<Edge> Edges => _edges;

        [YamlMember(Order = 3)]
        public List<Port> Ports
        {
            get => _layout.Ports;
            set => _layout.Ports = value;
        }

        [YamlMember(Order = 4)]
        public AdjacencyMatrix Adjacency { get; set; }

        // The robber is stored as its position, so a copy only copies three values and the robber always
        // resolves to the tile inside this board's map (also after deserialization)
        private bool _hasRobber;
        private byte _robberX;
        private byte _robberY;

        // Written first, where it was when the robber was a field, so save files stay byte-identical
        [YamlMember(Order = -1)]
        public Tile? Robber
        {
            get => _hasRobber ? Map.GetTile(_robberX, _robberY) : null;
            set
            {
                _hasRobber = value != null;
                _robberX = value?.X ?? 0;
                _robberY = value?.Y ?? 0;
            }
        }

        /// <summary>
        /// Index of the robber's tile in Map.Values, or null if there is no robber
        /// </summary>
        public int? GetRobberTileIndex() => _hasRobber ? _robberY * (int)Map.Width + _robberX : null;

        /// <summary>
        /// Whether the robber is on the given tile
        /// </summary>
        public bool HasRobber(Tile tile) => _hasRobber && _robberX == tile.X && _robberY == tile.Y;

        // -- Buildings and roads by index --

        [YamlIgnore]
        public int IntersectionCount => _buildings.Length;

        [YamlIgnore]
        public int EdgeCount => _roads.Length;

        public Intersection.BuildingType GetBuilding(int intersectionIdx) => _buildings[intersectionIdx];

        public sbyte GetBuildingOwner(int intersectionIdx) => _buildingOwners[intersectionIdx];

        public void SetBuilding(int intersectionIdx, Intersection.BuildingType building, sbyte owner)
        {
            _buildings[intersectionIdx] = building;
            _buildingOwners[intersectionIdx] = owner;
        }

        public Edge.BuildingType GetRoad(int edgeIdx) => _roads[edgeIdx];

        public sbyte GetRoadOwner(int edgeIdx) => _roadOwners[edgeIdx];

        public void SetRoad(int edgeIdx, Edge.BuildingType road, sbyte owner)
        {
            _roads[edgeIdx] = road;
            _roadOwners[edgeIdx] = owner;
        }

        public bool IntersectionFacesDownwards(int intersectionIdx) => _layout.IntersectionFacesDownwards[intersectionIdx];

        public Direction.Edge GetEdgeDirection(int edgeIdx) => _layout.EdgeDirections[edgeIdx];

        // -- Construction --

        /// <summary>
        /// Board from the positions created by the map generator, without buildings or roads
        /// </summary>
        public Board(HexMap<Tile> map, List<Intersection> intersections, List<Edge> edges, List<Port> ports, AdjacencyMatrix adjacency, Tile? robber)
        {
            _layout = new Layout
            {
                Map = map,
                Ports = ports,
                IntersectionFacesDownwards = intersections.OrderBy(intersection => intersection.Index).Select(intersection => intersection.FacesDownwards).ToArray(),
                EdgeDirections = edges.OrderBy(edge => edge.Index).Select(edge => edge.Direction).ToArray()
            };

            _buildings = new Intersection.BuildingType[intersections.Count];
            _buildingOwners = new sbyte[intersections.Count];
            _roads = new Edge.BuildingType[edges.Count];
            _roadOwners = new sbyte[edges.Count];
            Array.Fill(_buildingOwners, (sbyte)-1);
            Array.Fill(_roadOwners, (sbyte)-1);

            _intersections = CreateIntersectionViews();
            _edges = CreateEdgeViews();

            // Shares the adjacency built during generation, resolving to this board's views
            Adjacency = new AdjacencyMatrix(adjacency, Map, _intersections, _edges);
            Robber = robber;
        }

        /// <summary>
        /// Copy: shares the layout and copies the buildings, roads and robber
        /// </summary>
        public Board(Board copy)
        {
            _layout = copy._layout;

            _buildings = (Intersection.BuildingType[])copy._buildings.Clone();
            _buildingOwners = (sbyte[])copy._buildingOwners.Clone();
            _roads = (Edge.BuildingType[])copy._roads.Clone();
            _roadOwners = (sbyte[])copy._roadOwners.Clone();

            _intersections = CreateIntersectionViews();
            _edges = CreateEdgeViews();

            Adjacency = new AdjacencyMatrix(copy.Adjacency, Map, _intersections, _edges);

            _hasRobber = copy._hasRobber;
            _robberX = copy._robberX;
            _robberY = copy._robberY;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private Board()
        {
            _layout = new Layout();
            _buildings = [];
            _buildingOwners = [];
            _roads = [];
            _roadOwners = [];
            _intersections = [];
            _edges = [];
            Adjacency = null!;
        }

        private List<Intersection> CreateIntersectionViews()
        {
            List<Intersection> views = new(_buildings.Length);
            for (int i = 0; i < _buildings.Length; i++)
                views.Add(new Intersection(this, (byte)i, _layout.IntersectionFacesDownwards[i]));
            return views;
        }

        private List<Edge> CreateEdgeViews()
        {
            List<Edge> views = new(_roads.Length);
            for (int i = 0; i < _roads.Length; i++)
                views.Add(new Edge(this, (byte)i, _layout.EdgeDirections[i]));
            return views;
        }

        // -- Serialization: intersections and edges are written as before, one entry per position --

        public sealed class IntersectionEntry
        {
            public byte Index { get; set; }
            public bool FacesDownwards { get; set; }
            public Intersection.BuildingType Building { get; set; }
            public sbyte Owner { get; set; }
        }

        public sealed class EdgeEntry
        {
            public byte Index { get; set; }
            public Direction.Edge Direction { get; set; }
            public Edge.BuildingType Building { get; set; }
            public sbyte Owner { get; set; }
        }

        [YamlMember(Alias = "Intersections", Order = 1)]
        public List<IntersectionEntry> IntersectionEntries
        {
            get => Enumerable.Range(0, _buildings.Length).Select(i => new IntersectionEntry
            {
                Index = (byte)i,
                FacesDownwards = _layout.IntersectionFacesDownwards[i],
                Building = _buildings[i],
                Owner = _buildingOwners[i]
            }).ToList();
            set
            {
                _layout.IntersectionFacesDownwards = new bool[value.Count];
                _buildings = new Intersection.BuildingType[value.Count];
                _buildingOwners = new sbyte[value.Count];

                foreach (IntersectionEntry entry in value)
                {
                    _layout.IntersectionFacesDownwards[entry.Index] = entry.FacesDownwards;
                    _buildings[entry.Index] = entry.Building;
                    _buildingOwners[entry.Index] = entry.Owner;
                }

                _intersections = CreateIntersectionViews();
            }
        }

        [YamlMember(Alias = "Edges", Order = 2)]
        public List<EdgeEntry> EdgeEntries
        {
            get => Enumerable.Range(0, _roads.Length).Select(i => new EdgeEntry
            {
                Index = (byte)i,
                Direction = _layout.EdgeDirections[i],
                Building = _roads[i],
                Owner = _roadOwners[i]
            }).ToList();
            set
            {
                _layout.EdgeDirections = new Direction.Edge[value.Count];
                _roads = new Edge.BuildingType[value.Count];
                _roadOwners = new sbyte[value.Count];

                foreach (EdgeEntry entry in value)
                {
                    _layout.EdgeDirections[entry.Index] = entry.Direction;
                    _roads[entry.Index] = entry.Building;
                    _roadOwners[entry.Index] = entry.Owner;
                }

                _edges = CreateEdgeViews();
            }
        }

        public void Clear()
        {
            // Reset buildings
            Array.Fill(_buildings, Intersection.BuildingType.None);
            Array.Fill(_buildingOwners, (sbyte)-1);
            Array.Fill(_roads, Edge.BuildingType.None);
            Array.Fill(_roadOwners, (sbyte)-1);

            // Reset robber
            foreach(Tile tile in Map.Values)
            {
                if (tile.Type == Tile.TileType.Desert)
                {
                    Robber = tile;
                    break;
                }
            }
        }

        public override bool Equals(object? obj)
        {
            return obj is Board board
                && (ReferenceEquals(_layout, board._layout)
                    || Map.Equals(board.Map)
                        && Ports.SequenceEqual(board.Ports)
                        && _layout.IntersectionFacesDownwards.SequenceEqual(board._layout.IntersectionFacesDownwards)
                        && _layout.EdgeDirections.SequenceEqual(board._layout.EdgeDirections))
                && _buildings.AsSpan().SequenceEqual(board._buildings)
                && _buildingOwners.AsSpan().SequenceEqual(board._buildingOwners)
                && _roads.AsSpan().SequenceEqual(board._roads)
                && _roadOwners.AsSpan().SequenceEqual(board._roadOwners)
                && _hasRobber == board._hasRobber && _robberX == board._robberX && _robberY == board._robberY;
        }

        public override int GetHashCode()
        {
            HashCode hash = new();
            hash.Add(Map);
            foreach (Port port in Ports) hash.Add(port);
            foreach (Intersection.BuildingType building in _buildings) hash.Add(building);
            foreach (sbyte owner in _buildingOwners) hash.Add(owner);
            foreach (Edge.BuildingType road in _roads) hash.Add(road);
            foreach (sbyte owner in _roadOwners) hash.Add(owner);
            hash.Add(_hasRobber);
            hash.Add(_robberX);
            hash.Add(_robberY);
            return hash.ToHashCode();
        }
    }
}
