
using YamlDotNet.Serialization;

namespace Common
{
    public class Board
    {
        // Tiles, ports and the adjacency never change during a game and are shared by all copies of the board
        public HexMap<Tile> Map { get; set; }
        public List<Intersection> Intersections { get; set; }
        public List<Edge> Edges { get; set; }
        public List<Port> Ports { get; set; }
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

        public Board(HexMap<Tile> map, List<Intersection> intersections, List<Edge> edges, List<Port> ports, AdjacencyMatrix adjacency, Tile? robber)
        {
            Map = map;
            Intersections = intersections;
            Edges = edges;
            Ports = ports;
            Adjacency = adjacency;
            Robber = robber;
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public Board(Board copy)
        {
            // Shared, immutable after generation
            Map = copy.Map;
            Ports = copy.Ports;

            Intersections = new List<Intersection>();
            foreach (Intersection intersection in copy.Intersections)
            {
                Intersections.Add(new Intersection(intersection));
            }

            Edges = new List<Edge>();
            foreach (Edge edge in copy.Edges)
            {
                Edges.Add(new Edge(edge));
            }

            Adjacency = new AdjacencyMatrix(copy.Adjacency, Map, Intersections, Edges);

            _hasRobber = copy._hasRobber;
            _robberX = copy._robberX;
            _robberY = copy._robberY;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private Board()
        {

        }

        public void Clear()
        {
            // Reset buildings
            foreach(Intersection intersection in Intersections)
            {
                intersection.Owner = -1;
                intersection.Building = Intersection.BuildingType.None;
            }

            foreach(Edge edge in Edges)
            {
                edge.Owner = -1;
                edge.Building = Edge.BuildingType.None;
            }

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
                && Map.Equals(board.Map)
                && Intersections.SequenceEqual(board.Intersections)
                && Edges.SequenceEqual(board.Edges)
                && Ports.SequenceEqual(board.Ports)
                && _hasRobber == board._hasRobber && _robberX == board._robberX && _robberY == board._robberY;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(
                Map,
                Intersections.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                Edges.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                Ports.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                HashCode.Combine(_hasRobber, _robberX, _robberY));
        }
    }
}
