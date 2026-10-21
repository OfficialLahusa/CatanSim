
using YamlDotNet.Serialization;

namespace Common
{
    public class Board
    {
        public HexMap<Tile> Map { get; set; }
        public List<Intersection> Intersections { get; set; }
        public List<Edge> Edges { get; set; }
        public List<Port> Ports { get; set; }
        public AdjacencyMatrix Adjacency { get; set; }
        public Tile? Robber;

        /// <summary>
        /// Index of the robber's tile in Map.Values, or null if there is no robber
        /// </summary>
        public int? GetRobberTileIndex() => Robber == null ? null : Robber.Y * (int)Map.Width + Robber.X;

        /// <summary>
        /// Whether the robber is on the given tile. Compared by position, since the robber may be a
        /// separate Tile instance after deserialization.
        /// </summary>
        public bool HasRobber(Tile tile) => Robber != null && Robber.X == tile.X && Robber.Y == tile.Y;

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
            Map = new HexMap<Tile>(copy.Map);
            for (int x = 0; x < copy.Map.Width; x++)
            {
                for (int y = 0; y < copy.Map.Height; y++)
                {
                    Map.SetTile(x, y, new Tile(copy.Map.GetTile(x, y)));
                }
            }

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

            Ports = new List<Port>();
            foreach (Port port in copy.Ports)
            {
                Ports.Add(new Port(port));
            }

            Adjacency = new AdjacencyMatrix(copy.Adjacency, Map, Intersections, Edges);

            Robber = copy.Robber != null ? Map.GetTile(copy.Robber.X, copy.Robber.Y) : null;
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
                && Tile.Equals(Robber, board.Robber);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(
                Map,
                Intersections.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                Edges.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                Ports.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                Robber);
        }
    }
}
