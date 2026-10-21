using Common;
using Common.Serialization;

namespace Tests
{
    /// <summary>
    /// The board adjacency: shared between copies, resolved per board, consistent across lookups
    /// </summary>
    public class AdjacencyTests
    {
        [Fact]
        public void CopiesShareTheAdjacencyButResolveToTheirOwnObjects()
        {
            TestUtil.Seed(160);
            GameState state = TestUtil.NewGame();
            GameState copy = new GameState(state);

            Assert.Equal(state.Board.Adjacency, copy.Board.Adjacency);

            for (int i = 0; i < copy.Board.Intersections.Count; i++)
            {
                foreach (Edge edge in copy.Board.Adjacency.GetEdges(copy.Board.Intersections[i]))
                    Assert.Same(copy.Board.Edges[edge.Index], edge);
            }

            for (int e = 0; e < copy.Board.Edges.Count; e++)
            {
                (Intersection top, Intersection bottom) = copy.Board.Adjacency.GetIntersections(copy.Board.Edges[e]);
                Assert.Same(copy.Board.Intersections[top.Index], top);
                Assert.Same(copy.Board.Intersections[bottom.Index], bottom);
            }

            // Changing the copy's board leaves the original untouched
            copy.Board.Edges[0].Building = Edge.BuildingType.Road;
            Assert.Equal(Edge.BuildingType.None, state.Board.Edges[0].Building);
        }

        [Fact]
        public void IndexLookupsMatchObjectLookups()
        {
            TestUtil.Seed(161);
            Board board = TestUtil.NewGame().Board;

            foreach (Intersection intersection in board.Intersections)
            {
                int[] fromIndices = board.Adjacency.GetEdgeIndices(intersection.Index).ToArray().Select(i => (int)i).ToArray();
                Assert.Equal(board.Adjacency.GetEdges(intersection).Select(edge => (int)edge.Index), fromIndices);
                Assert.InRange(fromIndices.Length, 2, 3);
                Assert.Equal(fromIndices.Length, fromIndices.Distinct().Count());
            }

            foreach (Edge edge in board.Edges)
            {
                (int top, int bottom) = board.Adjacency.GetIntersectionIndices(edge.Index);
                (Intersection topObject, Intersection bottomObject) = board.Adjacency.GetIntersections(edge);
                Assert.Equal((top, bottom), ((int)topObject.Index, (int)bottomObject.Index));
                Assert.NotEqual(top, bottom);

                // Both ends list the edge among their edges
                Assert.Contains(edge, board.Adjacency.GetEdges(topObject));
                Assert.Contains(edge, board.Adjacency.GetEdges(bottomObject));
            }
        }

        [Fact]
        public void TheAdjacencyCannotChangeAfterItWasUsed()
        {
            TestUtil.Seed(162);
            Board board = TestUtil.NewGame().Board;
            _ = board.Adjacency.GetEdges(board.Intersections[0]);

            Tile tile = board.Map.Values.First(t => t.IsLandTile());
            Assert.Throws<InvalidOperationException>(() => board.Adjacency.RegisterEdge(tile, Direction.Tile.West, board.Edges[0]));
        }

        [Fact]
        public void FirstUseFromManyThreadsGivesConsistentResults()
        {
            TestUtil.Seed(163);
            GameState state = TestUtil.NewGame();

            // A freshly loaded board has not computed its derived lookups yet
            for (int run = 0; run < 5; run++)
            {
                GameState loaded = SaveFileSerializer.Deserialize(SaveFileSerializer.Serialize(new SaveFile(state, [], []))).GameState;
                Board board = loaded.Board;

                int[][] results = new int[64][];
                Parallel.For(0, results.Length, i =>
                {
                    results[i] = board.Intersections.SelectMany(intersection => board.Adjacency.GetEdges(intersection).Select(edge => (int)edge.Index)).ToArray();
                });

                Assert.All(results, result => Assert.Equal(results[0], result));
                Assert.Equal(state.Board.Intersections.SelectMany(i => state.Board.Adjacency.GetEdges(i).Select(e => (int)e.Index)), results[0]);
            }
        }
    }
}
