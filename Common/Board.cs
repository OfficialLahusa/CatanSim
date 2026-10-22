
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
            public List<Intersection> Intersections = [];
            public List<Edge> Edges = [];
        }

        private Layout _layout;

        // Buildings and roads, the only part of the board that is copied
        private Intersection.BuildingType[] _buildings;
        private sbyte[] _buildingOwners;
        private Edge.BuildingType[] _roads;
        private sbyte[] _roadOwners;

        [YamlMember(Order = 0)]
        public HexMap<Tile> Map
        {
            get => _layout.Map;
            set => _layout.Map = value;
        }

        /// <summary>
        /// Positions of the intersections, shared by all copies of the board (read buildings with GetBuilding)
        /// </summary>
        [YamlIgnore]
        public List<Intersection> Intersections => _layout.Intersections;

        /// <summary>
        /// Positions of the edges, shared by all copies of the board (read roads with GetRoad)
        /// </summary>
        [YamlIgnore]
        public List<Edge> Edges => _layout.Edges;

        [YamlMember(Order = 3)]
        public List<Port> Ports
        {
            get => _layout.Ports;
            set => _layout.Ports = value;
        }

        // Shared by all copies of the board
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

        public bool HasBuilding(int intersectionIdx) => _buildings[intersectionIdx] != Intersection.BuildingType.None;

        public bool HasBuildingOf(int intersectionIdx, int playerIdx) => HasBuilding(intersectionIdx) && _buildingOwners[intersectionIdx] == playerIdx;

        public void SetBuilding(int intersectionIdx, Intersection.BuildingType building, sbyte owner)
        {
            _buildings[intersectionIdx] = building;
            _buildingOwners[intersectionIdx] = owner;
        }

        public Edge.BuildingType GetRoad(int edgeIdx) => _roads[edgeIdx];

        public sbyte GetRoadOwner(int edgeIdx) => _roadOwners[edgeIdx];

        public bool HasRoadOf(int edgeIdx, int playerIdx) => _roads[edgeIdx] != Edge.BuildingType.None && _roadOwners[edgeIdx] == playerIdx;

        public void SetRoad(int edgeIdx, Edge.BuildingType road, sbyte owner)
        {
            _roads[edgeIdx] = road;
            _roadOwners[edgeIdx] = owner;
        }

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
                Intersections = intersections,
                Edges = edges
            };

            _buildings = new Intersection.BuildingType[intersections.Count];
            _buildingOwners = new sbyte[intersections.Count];
            _roads = new Edge.BuildingType[edges.Count];
            _roadOwners = new sbyte[edges.Count];
            Array.Fill(_buildingOwners, (sbyte)-1);
            Array.Fill(_roadOwners, (sbyte)-1);

            Adjacency = adjacency;
            Robber = robber;
        }

        /// <summary>
        /// Copy: shares the layout and adjacency, copies the buildings, roads and robber
        /// </summary>
        public Board(Board copy)
        {
            _layout = copy._layout;
            Adjacency = copy.Adjacency;

            _buildings = (Intersection.BuildingType[])copy._buildings.Clone();
            _buildingOwners = (sbyte[])copy._buildingOwners.Clone();
            _roads = (Edge.BuildingType[])copy._roads.Clone();
            _roadOwners = (sbyte[])copy._roadOwners.Clone();

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
            Adjacency = null!;
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
            get => Intersections.Select(intersection => new IntersectionEntry
            {
                Index = intersection.Index,
                FacesDownwards = intersection.FacesDownwards,
                Building = _buildings[intersection.Index],
                Owner = _buildingOwners[intersection.Index]
            }).ToList();
            set
            {
                _layout.Intersections = value.OrderBy(entry => entry.Index).Select(entry => new Intersection(entry.Index, entry.FacesDownwards)).ToList();
                _buildings = new Intersection.BuildingType[value.Count];
                _buildingOwners = new sbyte[value.Count];

                foreach (IntersectionEntry entry in value)
                {
                    _buildings[entry.Index] = entry.Building;
                    _buildingOwners[entry.Index] = entry.Owner;
                }
            }
        }

        [YamlMember(Alias = "Edges", Order = 2)]
        public List<EdgeEntry> EdgeEntries
        {
            get => Edges.Select(edge => new EdgeEntry
            {
                Index = edge.Index,
                Direction = edge.Direction,
                Building = _roads[edge.Index],
                Owner = _roadOwners[edge.Index]
            }).ToList();
            set
            {
                _layout.Edges = value.OrderBy(entry => entry.Index).Select(entry => new Edge(entry.Index, entry.Direction)).ToList();
                _roads = new Edge.BuildingType[value.Count];
                _roadOwners = new sbyte[value.Count];

                foreach (EdgeEntry entry in value)
                {
                    _roads[entry.Index] = entry.Building;
                    _roadOwners[entry.Index] = entry.Owner;
                }
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
                        && Intersections.SequenceEqual(board.Intersections)
                        && Edges.SequenceEqual(board.Edges))
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
