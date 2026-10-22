using System;
using System.Collections;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;

namespace Common
{
    using AdjIdx = byte;

    /// <summary>
    /// Adjacency between the tiles, intersections and edges of a board.
    /// The adjacency never changes during a game, so it lives in a Topology that all copies of a board share.
    /// An AdjacencyMatrix is a cheap per-board view that resolves the topology's indices to that board's objects.
    /// </summary>
    public class AdjacencyMatrix
    {
        protected const AdjIdx NO_ADJACENCY = 255;
        protected static readonly ArrayEqualityComparer<AdjIdx> _arrayEqualityComparer;

        // -- References to board elements --
        // Not serialized, restored via Attach(Board) after deserialization
        public HexMap<Tile> Map;
        public List<Intersection> Intersections;
        public List<Edge> Edges;

        // Shared by all copies of the board, only modified while the board is generated
        protected Topology _topology;

        static AdjacencyMatrix()
        {
            _arrayEqualityComparer = new ArrayEqualityComparer<AdjIdx>();
        }

        public AdjacencyMatrix(HexMap<Tile> map, List<Intersection> intersections, List<Edge> edges)
        {
            Map = map;
            Intersections = intersections;
            Edges = edges;

            _topology = new Topology((int)(map.Width * map.Height));
        }

        /// <summary>
        /// View for a copy of the board: shares the topology and resolves indices to the copy's objects
        /// </summary>
        public AdjacencyMatrix(AdjacencyMatrix copy, HexMap<Tile> map, List<Intersection> intersections, List<Edge> edges)
        {
            Map = map;
            Intersections = intersections;
            Edges = edges;

            _topology = copy._topology;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private AdjacencyMatrix()
        {
            Map = null!;
            Intersections = null!;
            Edges = null!;
            _topology = null!;
        }

        /// <summary>
        /// Restore board references after deserialization
        /// </summary>
        /// <param name="board"></param>
        public void Attach(Board board)
        {
            if (Map != null || Intersections != null || Edges != null)
                throw new InvalidOperationException("AdjacencyMatrix is already attached to a board");

            Map = board.Map;
            Intersections = board.Intersections;
            Edges = board.Edges;
        }

        public void RegisterTile(Tile source, Direction.Tile direction, Tile neighbor)
        {
            _topology.EnsureMutable();

            AdjIdx sourceIndex = GetIndexByTile(source);
            AdjIdx neighborIndex = GetIndexByTile(neighbor);

            if (_topology.TileToTile[sourceIndex] == null)
            {
                _topology.TileToTile[sourceIndex] = new AdjIdx[6];
                Array.Fill(_topology.TileToTile[sourceIndex], NO_ADJACENCY);
            }

            _topology.TileToTile[sourceIndex][(int)direction] = neighborIndex;
        }

        public void RegisterTile(Intersection source, Direction.Corner direction, Tile neighbor)
        {
            _topology.EnsureMutable();

            AdjIdx neighborIndex = GetIndexByTile(neighbor);

            // Add empty entries until the list is large enough to map the source intersection
            while (_topology.IntersectionToTile.Count <= source.Index)
            {
                _topology.IntersectionToTile.Add(null);
            }

            if (_topology.IntersectionToTile[source.Index] == null)
            {
                _topology.IntersectionToTile[source.Index] = new AdjIdx[6];
                Array.Fill(_topology.IntersectionToTile[source.Index]!, NO_ADJACENCY);
            }

            _topology.IntersectionToTile[source.Index]![(int)direction] = neighborIndex;
        }

        public void RegisterWestTile(Edge source, Tile neighbor)
        {
            _topology.EnsureMutable();

            AdjIdx neighborIndex = GetIndexByTile(neighbor);

            // Add empty entries until the list is large enough to map the source intersection
            while (_topology.EdgeToWestTile.Count <= source.Index)
            {
                _topology.EdgeToWestTile.Add(NO_ADJACENCY);
            }

            _topology.EdgeToWestTile[source.Index] = neighborIndex;
        }

        public void RegisterEastTile(Edge source, Tile neighbor)
        {
            _topology.EnsureMutable();

            AdjIdx neighborIndex = GetIndexByTile(neighbor);

            // Add empty entries until the list is large enough to map the source intersection
            while (_topology.EdgeToEastTile.Count <= source.Index)
            {
                _topology.EdgeToEastTile.Add(NO_ADJACENCY);
            }

            _topology.EdgeToEastTile[source.Index] = neighborIndex;
        }

        public void RegisterIntersection(Tile source, Direction.Corner direction, Intersection intersection)
        {
            _topology.EnsureMutable();

            AdjIdx sourceIndex = GetIndexByTile(source);

            if (_topology.TileToIntersection[sourceIndex] == null)
            {
                _topology.TileToIntersection[sourceIndex] = new AdjIdx[6];
                Array.Fill(_topology.TileToIntersection[sourceIndex], NO_ADJACENCY);
            }

            _topology.TileToIntersection[sourceIndex][(int)direction] = (AdjIdx)intersection.Index;
        }

        public void RegisterEdge(Tile source, Direction.Tile direction, Edge edge)
        {
            _topology.EnsureMutable();

            AdjIdx sourceIndex = GetIndexByTile(source);
            if (_topology.TileToEdge[sourceIndex] == null)
            {
                _topology.TileToEdge[sourceIndex] = new AdjIdx[6];
                Array.Fill(_topology.TileToEdge[sourceIndex], NO_ADJACENCY);
            }
            _topology.TileToEdge[sourceIndex][(int)direction] = (AdjIdx)edge.Index;
        }

        public Tile? GetTile(Tile source, Direction.Tile direction)
        {
            AdjIdx sourceIndex = GetIndexByTile(source);

            if (_topology.TileToTile[sourceIndex] == null)
                return null;

            AdjIdx neighborIndex = _topology.TileToTile[sourceIndex][(int)direction];
            if (neighborIndex == NO_ADJACENCY)
                return null;

            return GetTileByIndex(neighborIndex);
        }

        public Tile? GetTile(Intersection source, Direction.Corner direction)
        {
            if (_topology.IntersectionToTile[source.Index] == null)
                return null;

            AdjIdx neighborIndex = _topology.IntersectionToTile[source.Index]![(int)direction];
            if (neighborIndex == NO_ADJACENCY)
                return null;

            return GetTileByIndex(neighborIndex);
        }

        public Tile? GetWestTile(Edge source)
        {
            if (_topology.EdgeToWestTile.Count <= source.Index)
                return null;

            if (_topology.EdgeToWestTile[source.Index] == NO_ADJACENCY)
                return null;

            return GetTileByIndex(_topology.EdgeToWestTile[source.Index]);
        }

        public Tile? GetEastTile(Edge source)
        {
            if (_topology.EdgeToEastTile.Count <= source.Index)
                return null;

            if (_topology.EdgeToEastTile[source.Index] == NO_ADJACENCY)
                return null;

            return GetTileByIndex(_topology.EdgeToEastTile[source.Index]);
        }

        public IEnumerable<Tile> GetTiles(Tile source)
        {
            AdjIdx sourceIndex = GetIndexByTile(source);
            if (_topology.TileToTile[sourceIndex] == null)
                yield break;

            foreach (AdjIdx neighborIndex in _topology.TileToTile[sourceIndex])
            {
                if (neighborIndex != NO_ADJACENCY)
                    yield return GetTileByIndex(neighborIndex);
            }
        }

        public IEnumerable<Tile> GetTiles(Intersection source)
        {
            if (_topology.IntersectionToTile[source.Index] == null)
                yield break;

            foreach (AdjIdx neighborIndex in _topology.IntersectionToTile[source.Index]!)
            {
                if (neighborIndex != NO_ADJACENCY)
                    yield return GetTileByIndex(neighborIndex);
            }
        }

        public Intersection? GetIntersection(Tile source, Direction.Corner direction)
        {
            AdjIdx sourceIndex = GetIndexByTile(source);

            if (_topology.TileToIntersection[sourceIndex] == null)
                return null;

            AdjIdx neighborIndex = _topology.TileToIntersection[sourceIndex][(int)direction];
            if (neighborIndex == NO_ADJACENCY)
                return null;

            return Intersections[neighborIndex];
        }

        public IEnumerable<Intersection> GetIntersections(Tile source)
        {
            AdjIdx sourceIndex = GetIndexByTile(source);
            if (_topology.TileToIntersection[sourceIndex] == null)
                yield break;

            foreach (AdjIdx neighborIndex in _topology.TileToIntersection[sourceIndex])
            {
                if (neighborIndex != NO_ADJACENCY)
                    yield return Intersections[neighborIndex];
            }
        }

        public (Intersection top, Intersection bottom) GetIntersections(Edge edge)
        {
            (int top, int bottom) = GetIntersectionIndices(edge.Index);
            return (Intersections[top], Intersections[bottom]);
        }

        /// <summary>
        /// Indices of the two intersections at the ends of an edge
        /// </summary>
        public (int top, int bottom) GetIntersectionIndices(int edgeIdx)
        {
            (AdjIdx top, AdjIdx bottom) = _topology.GetDerived(this).EdgeEnds[edgeIdx];
            return (top, bottom);
        }

        public Edge? GetEdge(Tile source, Direction.Tile direction)
        {
            AdjIdx sourceIndex = GetIndexByTile(source);

            if (_topology.TileToEdge[sourceIndex] == null)
                return null;

            AdjIdx neighborIndex = _topology.TileToEdge[sourceIndex][(int)direction];
            if (neighborIndex == NO_ADJACENCY)
                return null;

            return Edges[neighborIndex];
        }

        public IEnumerable<Edge> GetEdges(Tile source)
        {
            AdjIdx sourceIndex = GetIndexByTile(source);
            if (_topology.TileToEdge[sourceIndex] == null)
                yield break;

            foreach (AdjIdx neighborIndex in _topology.TileToEdge[sourceIndex])
            {
                if (neighborIndex != NO_ADJACENCY)
                    yield return Edges[neighborIndex];
            }
        }

        public BoardElementList<Edge> GetEdges(Intersection intersection)
        {
            return new BoardElementList<Edge>(Edges, _topology.GetDerived(this).IntersectionEdges[intersection.Index]);
        }

        /// <summary>
        /// Indices of the edges meeting at an intersection
        /// </summary>
        public ReadOnlySpan<AdjIdx> GetEdgeIndices(int intersectionIdx)
        {
            return _topology.GetDerived(this).IntersectionEdges[intersectionIdx];
        }

        /// <summary>
        /// Mask of the two intersections at the ends of an edge
        /// </summary>
        public UInt128 GetIntersectionMask(int edgeIdx)
        {
            return _topology.GetDerived(this).EdgeEndMasks[edgeIdx];
        }

        /// <summary>
        /// Mask of the edges meeting at an intersection
        /// </summary>
        public UInt128 GetEdgeMask(int intersectionIdx)
        {
            return _topology.GetDerived(this).IntersectionEdgeMasks[intersectionIdx];
        }

        /// <summary>
        /// Mask of the intersections one edge away from an intersection
        /// </summary>
        public UInt128 GetNeighborMask(int intersectionIdx)
        {
            return _topology.GetDerived(this).IntersectionNeighborMasks[intersectionIdx];
        }

        protected Tile GetTileByIndex(AdjIdx idx)
        {
            int x = (int)(idx % Map.Width);
            int y = (int)(idx / Map.Width);
            return Map.GetTile(x, y);
        }

        protected AdjIdx GetIndexByTile(Tile tile)
        {
            return (AdjIdx)(tile.Y * Map.Width + tile.X);
        }

        public override bool Equals(object? obj)
        {
            return obj is AdjacencyMatrix matrix
                && (ReferenceEquals(_topology, matrix._topology) || _topology.Equals(matrix._topology));
        }

        public override int GetHashCode()
        {
            return _topology.GetHashCode();
        }

        /// <summary>
        /// The adjacency data of a board, shared by all of its copies.
        /// Lookups that are derived from it (edge ends, edges per intersection) are computed once, on first use.
        /// After that, the topology can no longer change.
        /// </summary>
        protected sealed class Topology
        {
            // Dim 1: Source tile
            // Dim 2: Tile direction
            // Stored: Flattened index of adjacent tile (y * width + x)
            public AdjIdx[][] TileToTile;

            // Dim 1: Source tile
            // Dim 2: Corner direction
            // Stored: Index of adjacent intersection
            public AdjIdx[][] TileToIntersection;

            // Dim 1: Source tile
            // Dim 2: Tile direction
            // Stored: Index of adjacent edge
            public AdjIdx[][] TileToEdge;

            // Dim 1: Source intersection
            // Dim 2: Corner direction (that the source intersection is on at the adjacent tile)
            // Stored: Tile index of adjacent tile
            public List<AdjIdx[]?> IntersectionToTile;

            // Dim 1: Source edge
            // Stored: Tile index of adjacent tile
            public List<AdjIdx> EdgeToWestTile;

            // Dim 1: Source edge
            // Stored: Tile index of adjacent tile
            public List<AdjIdx> EdgeToEastTile;

            private readonly object _derivedLock = new object();
            private Derived? _derived;

            public Topology(int tileCount)
                : this(new AdjIdx[tileCount][], new AdjIdx[tileCount][], new AdjIdx[tileCount][], new List<AdjIdx[]?>(), new List<AdjIdx>(), new List<AdjIdx>())
            { }

            public Topology(AdjIdx[][] tileToTile, AdjIdx[][] tileToIntersection, AdjIdx[][] tileToEdge, List<AdjIdx[]?> intersectionToTile, List<AdjIdx> edgeToWestTile, List<AdjIdx> edgeToEastTile)
            {
                TileToTile = tileToTile;
                TileToIntersection = tileToIntersection;
                TileToEdge = tileToEdge;
                IntersectionToTile = intersectionToTile;
                EdgeToWestTile = edgeToWestTile;
                EdgeToEastTile = edgeToEastTile;
            }

            public void EnsureMutable()
            {
                if (Volatile.Read(ref _derived) != null)
                    throw new InvalidOperationException("The adjacency of a board can't change after it has been used.");
            }

            /// <summary>
            /// The derived lookups, computed on first use with the given view (any view of this topology gives the same result)
            /// </summary>
            public Derived GetDerived(AdjacencyMatrix view)
            {
                Derived? derived = Volatile.Read(ref _derived);
                if (derived != null) return derived;

                lock (_derivedLock)
                {
                    if (_derived == null)
                        Volatile.Write(ref _derived, Derived.Compute(view));

                    return _derived!;
                }
            }

            public override bool Equals(object? obj)
            {
                return obj is Topology topology
                    && TileToTile.SequenceEqual(topology.TileToTile, _arrayEqualityComparer)
                    && TileToIntersection.SequenceEqual(topology.TileToIntersection, _arrayEqualityComparer)
                    && TileToEdge.SequenceEqual(topology.TileToEdge, _arrayEqualityComparer)
                    && IntersectionToTile.SequenceEqual(topology.IntersectionToTile, _arrayEqualityComparer)
                    && EdgeToWestTile.SequenceEqual(topology.EdgeToWestTile)
                    && EdgeToEastTile.SequenceEqual(topology.EdgeToEastTile);
            }

            public override int GetHashCode()
            {
                return HashCode.Combine(
                    TileToTile.Aggregate(0, (acc, v) => v == null ? acc : HashCode.Combine(acc, v.Aggregate(0, (acc2, v2) => HashCode.Combine(acc2, v2.GetHashCode())))),
                    TileToIntersection.Aggregate(0, (acc, v) => v == null ? acc : HashCode.Combine(acc, v.Aggregate(0, (acc2, v2) => HashCode.Combine(acc2, v2.GetHashCode())))),
                    TileToEdge.Aggregate(0, (acc, v) => v == null ? acc : HashCode.Combine(acc, v.Aggregate(0, (acc2, v2) => HashCode.Combine(acc2, v2.GetHashCode())))),
                    IntersectionToTile.Aggregate(0, (acc, v) => v == null ? acc : HashCode.Combine(acc, v.Aggregate(0, (acc2, v2) => HashCode.Combine(acc2, v2.GetHashCode())))),
                    EdgeToWestTile.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode())),
                    EdgeToEastTile.Aggregate(0, (acc, v) => HashCode.Combine(acc, v.GetHashCode()))
                );
            }
        }

        protected sealed class Derived
        {
            // Dim 1: Source edge
            // Stored: (top intersection, bottom intersection)
            public readonly (AdjIdx top, AdjIdx bottom)[] EdgeEnds;

            // Dim 1: Source intersection
            // Stored: Indices of adjacent edges
            public readonly AdjIdx[][] IntersectionEdges;

            // The same lookups as bitmasks (see BitMask)
            public readonly UInt128[] EdgeEndMasks;
            public readonly UInt128[] IntersectionEdgeMasks;
            public readonly UInt128[] IntersectionNeighborMasks;

            private Derived((AdjIdx top, AdjIdx bottom)[] edgeEnds, AdjIdx[][] intersectionEdges)
            {
                EdgeEnds = edgeEnds;
                IntersectionEdges = intersectionEdges;

                EdgeEndMasks = new UInt128[edgeEnds.Length];
                for (int edgeIdx = 0; edgeIdx < edgeEnds.Length; edgeIdx++)
                    EdgeEndMasks[edgeIdx] = BitMask.Bit(edgeEnds[edgeIdx].top) | BitMask.Bit(edgeEnds[edgeIdx].bottom);

                IntersectionEdgeMasks = new UInt128[intersectionEdges.Length];
                IntersectionNeighborMasks = new UInt128[intersectionEdges.Length];
                for (int intersectionIdx = 0; intersectionIdx < intersectionEdges.Length; intersectionIdx++)
                {
                    foreach (AdjIdx edgeIdx in intersectionEdges[intersectionIdx])
                    {
                        IntersectionEdgeMasks[intersectionIdx] |= BitMask.Bit(edgeIdx);
                        IntersectionNeighborMasks[intersectionIdx] |= EdgeEndMasks[edgeIdx];
                    }

                    IntersectionNeighborMasks[intersectionIdx] &= ~BitMask.Bit(intersectionIdx);
                }
            }

            public static Derived Compute(AdjacencyMatrix view)
            {
                var edgeEnds = new (AdjIdx top, AdjIdx bottom)[view.Edges.Count];
                foreach (Edge edge in view.Edges)
                {
                    Tile? westTile = view.GetWestTile(edge);
                    bool fromWest = westTile != null;
                    Tile anchorTile = westTile ?? view.GetEastTile(edge)!;

                    Direction.Tile tileDir = fromWest ? edge.Direction.ToEastTileDir() : edge.Direction.ToWestTileDir();
                    (Direction.Corner left, Direction.Corner right) = tileDir.GetAdjacentCorners();

                    AdjIdx leftIntersection = view.GetIntersection(anchorTile, left)!.Index;
                    AdjIdx rightIntersection = view.GetIntersection(anchorTile, right)!.Index;

                    edgeEnds[edge.Index] = (fromWest ? leftIntersection : rightIntersection, fromWest ? rightIntersection : leftIntersection);
                }

                var intersectionEdges = new AdjIdx[view.Intersections.Count][];
                foreach (Intersection intersection in view.Intersections)
                {
                    List<AdjIdx> adjacentEdges = new List<AdjIdx>(3);

                    // Iterate through all adjacent tiles to the intersection (by corner direction)
                    foreach (Direction.Corner anchoredDir in Enum.GetValues<Direction.Corner>())
                    {
                        Tile? tile = view.GetTile(intersection, anchoredDir);
                        if (tile == null) continue;

                        // Get the two roads adjacent to the tile and intersection
                        (Direction.Tile leftAdjEdgeDir, Direction.Tile rightAdjEdgeDir) = anchoredDir.GetAdjacentTiles();
                        foreach (Direction.Tile edgeDir in new[] { leftAdjEdgeDir, rightAdjEdgeDir })
                        {
                            AdjIdx edgeIdx = view.GetEdge(tile, edgeDir)!.Index;
                            if (!adjacentEdges.Contains(edgeIdx))
                                adjacentEdges.Add(edgeIdx);
                        }
                    }

                    intersectionEdges[intersection.Index] = adjacentEdges.ToArray();
                }

                return new Derived(edgeEnds, intersectionEdges);
            }
        }

        public sealed class Converter : IYamlTypeConverter
        {
            public bool Accepts(Type type)
            {
                return type == typeof(AdjacencyMatrix);
            }

            public object? ReadYaml(IParser parser, Type type, ObjectDeserializer rootDeserializer)
            {
                // Expect mapping start
                parser.Expect<MappingStart>();

                AdjIdx[][]? tileToTile = null, tileToIntersection = null, tileToEdge = null;
                List<AdjIdx[]?>? intersectionToTile = null;
                List<AdjIdx>? edgeToWestTile = null, edgeToEastTile = null;

                while (!parser.Accept<MappingEnd>())
                {
                    var key = parser.Expect<Scalar>().Value ?? string.Empty;
                    switch (key)
                    {
                        case "TileToTile":
                        case "_tileToTile":
                            tileToTile = (AdjIdx[][]?)rootDeserializer(typeof(AdjIdx[][]))!;
                            break;
                        case "TileToIntersection":
                        case "_tileToIntersection":
                            tileToIntersection = (AdjIdx[][]?)rootDeserializer(typeof(AdjIdx[][]))!;
                            break;
                        case "TileToEdge":
                        case "_tileToEdge":
                            tileToEdge = (AdjIdx[][]?)rootDeserializer(typeof(AdjIdx[][]))!;
                            break;
                        case "IntersectionToTile":
                        case "_intersectionToTile":
                            intersectionToTile = (List<AdjIdx[]?>?)rootDeserializer(typeof(List<AdjIdx[]?>))!;
                            break;
                        case "EdgeToWestTile":
                        case "_edgeToWestTile":
                            edgeToWestTile = (List<AdjIdx>?)rootDeserializer(typeof(List<AdjIdx>))!;
                            break;
                        case "EdgeToEastTile":
                        case "_edgeToEastTile":
                            edgeToEastTile = (List<AdjIdx>?)rootDeserializer(typeof(List<AdjIdx>))!;
                            break;
                        default:
                            // Unknown key: skip value
                            _ = rootDeserializer(typeof(object));
                            break;
                    }
                }

                parser.Expect<MappingEnd>();

                // Derived lookups are not serialized, they are computed on first use
                var result = new AdjacencyMatrix();
                result._topology = new Topology(
                    tileToTile ?? throw new YamlException("Adjacency is missing TileToTile."),
                    tileToIntersection ?? throw new YamlException("Adjacency is missing TileToIntersection."),
                    tileToEdge ?? throw new YamlException("Adjacency is missing TileToEdge."),
                    intersectionToTile ?? throw new YamlException("Adjacency is missing IntersectionToTile."),
                    edgeToWestTile ?? throw new YamlException("Adjacency is missing EdgeToWestTile."),
                    edgeToEastTile ?? throw new YamlException("Adjacency is missing EdgeToEastTile."));

                return result;
            }


            public void WriteYaml(IEmitter emitter, object? value, Type type, ObjectSerializer serializer)
            {
                emitter.Emit(new MappingStart());

                Topology topology = ((AdjacencyMatrix)value!)._topology;

                // Adjacency arrays and lists
                emitter.Emit(new Scalar("TileToTile"));
                serializer(topology.TileToTile);

                emitter.Emit(new Scalar("TileToIntersection"));
                serializer(topology.TileToIntersection);

                emitter.Emit(new Scalar("TileToEdge"));
                serializer(topology.TileToEdge);

                emitter.Emit(new Scalar("IntersectionToTile"));
                serializer(topology.IntersectionToTile);

                emitter.Emit(new Scalar("EdgeToWestTile"));
                serializer(topology.EdgeToWestTile);

                emitter.Emit(new Scalar("EdgeToEastTile"));
                serializer(topology.EdgeToEastTile);

                emitter.Emit(new MappingEnd());
            }
        }
    }

    /// <summary>
    /// Read-only list of board elements selected by index, without copying them (e.g. the edges at an intersection)
    /// </summary>
    public readonly struct BoardElementList<T> : IReadOnlyList<T>
    {
        private readonly List<T> _elements;
        private readonly byte[] _indices;

        public BoardElementList(List<T> elements, byte[] indices)
        {
            _elements = elements;
            _indices = indices;
        }

        public int Count => _indices.Length;

        public T this[int index] => _elements[_indices[index]];

        public Enumerator GetEnumerator() => new Enumerator(_elements, _indices);

        IEnumerator<T> IEnumerable<T>.GetEnumerator() => GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public struct Enumerator : IEnumerator<T>
        {
            private readonly List<T> _elements;
            private readonly byte[] _indices;
            private int _position;

            public Enumerator(List<T> elements, byte[] indices)
            {
                _elements = elements;
                _indices = indices;
                _position = -1;
            }

            public T Current => _elements[_indices[_position]];

            object? IEnumerator.Current => Current;

            public bool MoveNext() => ++_position < _indices.Length;

            public void Reset() => _position = -1;

            public void Dispose() { }
        }
    }
}
