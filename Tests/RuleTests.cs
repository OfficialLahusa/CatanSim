using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Targeted rule scenarios built on constructed states
    /// </summary>
    public class RuleTests
    {
        // ---------------------------------------------------------------- Port trades

        [Fact]
        public void TwoToOnePortTradeNeedsExactlyTwoCards()
        {
            TestUtil.Seed(10);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Players[0].PortPrivileges = PortPrivileges.LumberTwoToOne;
            TestUtil.GiveCards(state, 0, ResourceCardType.Lumber, 2);

            TwoToOneTradeAction trade = new(0, ResourceCardType.Lumber, ResourceCardType.Brick);
            Assert.True(trade.IsValidFor(state));
            Assert.Contains(TwoToOneTradeAction.GetActionsForState(state, 0), action => action.IsSameDecision(trade));

            trade.Apply(state);
            Assert.Equal(0u, state.Players[0].ResourceCards.Get(ResourceCardType.Lumber));
            Assert.Equal(1u, state.Players[0].ResourceCards.Get(ResourceCardType.Brick));
            Assert.Equal(19u, state.ResourceBank.Get(ResourceCardType.Lumber));
            Assert.Equal(18u, state.ResourceBank.Get(ResourceCardType.Brick));
        }

        [Fact]
        public void TwoToOnePortTradeNeedsThePortAndTwoCards()
        {
            TestUtil.Seed(11);
            GameState state = TestUtil.EmptyNormalTurn(0);
            TestUtil.GiveCards(state, 0, ResourceCardType.Lumber, 1);
            state.Players[0].PortPrivileges = PortPrivileges.LumberTwoToOne;
            Assert.False(new TwoToOneTradeAction(0, ResourceCardType.Lumber, ResourceCardType.Brick).IsValidFor(state));

            TestUtil.GiveCards(state, 0, ResourceCardType.Lumber, 1);
            state.Players[0].PortPrivileges = PortPrivileges.BrickTwoToOne;
            Assert.False(new TwoToOneTradeAction(0, ResourceCardType.Lumber, ResourceCardType.Brick).IsValidFor(state));
            Assert.False(new TwoToOneTradeAction(0, ResourceCardType.Lumber, ResourceCardType.Lumber).IsValidFor(state));
        }

        [Fact]
        public void ThreeAndFourToOneTradesUseTheirRatios()
        {
            TestUtil.Seed(12);
            GameState state = TestUtil.EmptyNormalTurn(0);
            TestUtil.GiveCards(state, 0, ResourceCardType.Ore, 3);

            Assert.False(new FourToOneTradeAction(0, ResourceCardType.Ore, ResourceCardType.Wool).IsValidFor(state));
            Assert.False(new ThreeToOneTradeAction(0, ResourceCardType.Ore, ResourceCardType.Wool).IsValidFor(state));

            state.Players[0].PortPrivileges = PortPrivileges.GenericThreeToOne;
            ThreeToOneTradeAction trade = new(0, ResourceCardType.Ore, ResourceCardType.Wool);
            Assert.True(trade.IsValidFor(state));
            trade.Apply(state);
            Assert.Equal(0u, state.Players[0].ResourceCards.Get(ResourceCardType.Ore));
            Assert.Equal(1u, state.Players[0].ResourceCards.Get(ResourceCardType.Wool));

            TestUtil.GiveCards(state, 0, ResourceCardType.Grain, 4);
            Assert.True(new FourToOneTradeAction(0, ResourceCardType.Grain, ResourceCardType.Brick).IsValidFor(state));
        }

        // ---------------------------------------------------------------- Longest Road

        [Fact]
        public void LongestRoadIsSetAsideWhenTheHolderIsBrokenBelowFive()
        {
            TestUtil.Seed(20);
            GameState state = TestUtil.EmptyNormalTurn(3);
            List<RoadPath> paths = RoadPath.FindDisjoint(state.Board, [6]);

            BuildRoad(state, 0, paths[0]);
            Assert.Equal(6u, state.Players[0].LongestRoadLength);
            AssertLongestRoadHolder(state, 0);

            BreakRoad(state, 0, paths[0], splitAt: 3, breaker: 3);
            Assert.Equal(3u, state.Players[0].LongestRoadLength);
            AssertLongestRoadHolder(state, -1);
        }

        [Fact]
        public void LongestRoadMovesToTheSoleLeaderAfterABreak()
        {
            TestUtil.Seed(21);
            GameState state = TestUtil.EmptyNormalTurn(3);
            List<RoadPath> paths = RoadPath.FindDisjoint(state.Board, [7, 6]);

            BuildRoad(state, 0, paths[0]);
            BuildRoad(state, 1, paths[1]);
            AssertLongestRoadHolder(state, 0);

            BreakRoad(state, 0, paths[0], splitAt: 3, breaker: 3);
            Assert.Equal(4u, state.Players[0].LongestRoadLength);
            AssertLongestRoadHolder(state, 1);
        }

        [Fact]
        public void HolderKeepsLongestRoadWhenStillTiedAfterABreak()
        {
            TestUtil.Seed(22);
            GameState state = TestUtil.EmptyNormalTurn(3);
            List<RoadPath> paths = RoadPath.FindDisjoint(state.Board, [7, 6]);

            BuildRoad(state, 0, paths[0]);
            BuildRoad(state, 1, paths[1]);

            BreakRoad(state, 0, paths[0], splitAt: 1, breaker: 3);
            Assert.Equal(6u, state.Players[0].LongestRoadLength);
            AssertLongestRoadHolder(state, 0);
        }

        [Fact]
        public void SetAsideLongestRoadGoesToTheSoleLeaderOfALaterBreak()
        {
            TestUtil.Seed(23);
            GameState state = TestUtil.EmptyNormalTurn(3);
            List<RoadPath> paths = RoadPath.FindDisjoint(state.Board, [8, 6, 6]);

            BuildRoad(state, 0, paths[0]);
            BuildRoad(state, 1, paths[1]);
            BuildRoad(state, 2, paths[2]);
            AssertLongestRoadHolder(state, 0);

            // Holder falls behind a tie between two others: the card is set aside
            BreakRoad(state, 0, paths[0], splitAt: 4, breaker: 3);
            AssertLongestRoadHolder(state, -1);

            // One of the tied players is broken: the other one is now alone in front and receives it
            BreakRoad(state, 1, paths[1], splitAt: 3, breaker: 3);
            AssertLongestRoadHolder(state, 2);
        }

        [Fact]
        public void NewRoadOnlyTakesLongestRoadWhenStrictlyLonger()
        {
            TestUtil.Seed(24);
            GameState state = TestUtil.EmptyNormalTurn(3);
            List<RoadPath> paths = RoadPath.FindDisjoint(state.Board, [5, 5, 4]);

            BuildRoad(state, 0, paths[0]);
            AssertLongestRoadHolder(state, 0);

            // Equal length does not take the card
            BuildRoad(state, 1, paths[1]);
            AssertLongestRoadHolder(state, 0);

            // Shorter than 5 never takes it
            BuildRoad(state, 2, paths[2]);
            AssertLongestRoadHolder(state, 0);
        }

        private static void BuildRoad(GameState state, int playerIdx, RoadPath path)
        {
            foreach (int edgeIdx in path.Edges)
            {
                state.Board.Edges[edgeIdx].Building = Edge.BuildingType.Road;
                state.Board.Edges[edgeIdx].Owner = (sbyte)playerIdx;
            }

            state.CalculateLongestRoad(playerIdx);
        }

        /// <summary>
        /// Places a settlement of the breaker on the path's vertex, splitting it into splitAt and length - splitAt roads,
        /// and recalculates like SettlementAction does for a broken opponent road
        /// </summary>
        private static void BreakRoad(GameState state, int playerIdx, RoadPath path, int splitAt, int breaker)
        {
            Intersection intersection = state.Board.Intersections[path.Vertices[splitAt]];
            intersection.Building = Intersection.BuildingType.Settlement;
            intersection.Owner = (sbyte)breaker;

            state.CalculateLongestRoad(playerIdx, checkForBreak: true);
        }

        private static void AssertLongestRoadHolder(GameState state, int expectedHolder)
        {
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                byte expected = (byte)(playerIdx == expectedHolder ? 2 : 0);
                Assert.True(state.Players[playerIdx].VictoryPoints.LongestRoadPoints == expected,
                    $"player {playerIdx} has {state.Players[playerIdx].VictoryPoints.LongestRoadPoints} Longest Road points, expected {expected} (holder {expectedHolder})");
            }
        }

        /// <summary>
        /// A simple path of edges on the board: Vertices[i] and Vertices[i + 1] are the ends of Edges[i]
        /// </summary>
        private record RoadPath(int[] Edges, int[] Vertices)
        {
            public static List<RoadPath> FindDisjoint(Board board, int[] lengths)
            {
                HashSet<int> usedVertices = [];
                List<RoadPath> paths = [];

                foreach (int length in lengths)
                {
                    RoadPath? path = null;
                    for (int start = 0; start < board.Intersections.Count && path == null; start++)
                    {
                        if (usedVertices.Contains(start)) continue;
                        path = Search(board, [start], [], length, usedVertices);
                    }

                    Assert.True(path != null, $"no free simple path of length {length} on the board");
                    paths.Add(path!);
                    usedVertices.UnionWith(path!.Vertices);

                    // Keep paths apart by one vertex, so that roads of different players never touch
                    foreach (int vertex in path.Vertices)
                        foreach (Edge edge in board.Adjacency.GetEdges(board.Intersections[vertex]))
                        {
                            (Intersection top, Intersection bottom) = board.Adjacency.GetIntersections(edge);
                            usedVertices.Add(top.Index);
                            usedVertices.Add(bottom.Index);
                        }
                }

                return paths;
            }

            private static RoadPath? Search(Board board, List<int> vertices, List<int> edges, int length, HashSet<int> blocked)
            {
                if (edges.Count == length) return new RoadPath([.. edges], [.. vertices]);

                Intersection current = board.Intersections[vertices[^1]];
                foreach (Edge edge in board.Adjacency.GetEdges(current).OrderBy(edge => edge.Index))
                {
                    (Intersection top, Intersection bottom) = board.Adjacency.GetIntersections(edge);
                    int next = top.Index != current.Index ? top.Index : bottom.Index;
                    if (vertices.Contains(next) || blocked.Contains(next)) continue;

                    vertices.Add(next);
                    edges.Add(edge.Index);
                    RoadPath? found = Search(board, vertices, edges, length, blocked);
                    if (found != null) return found;
                    vertices.RemoveAt(vertices.Count - 1);
                    edges.RemoveAt(edges.Count - 1);
                }

                return null;
            }
        }

        // ---------------------------------------------------------------- Production and the bank

        [Fact]
        public void ShortageForSeveralPlayersPaysNobody()
        {
            TestUtil.Seed(30);
            (GameState state, Tile tile, int cornerA, int cornerB) = ProductionSetup();
            ResourceCardType resource = tile.Type.ToCardType();

            PlaceBuilding(state, cornerA, 1, Intersection.BuildingType.Settlement);
            PlaceBuilding(state, cornerB, 2, Intersection.BuildingType.Settlement);
            uint demandA = Demand(state, 1, tile.Number!.Value, resource);
            uint demandB = Demand(state, 2, tile.Number!.Value, resource);
            LimitBank(state, resource, demandA + demandB - 1);

            Roll(state, tile.Number!.Value);

            Assert.Equal(0u, state.Players[1].ResourceCards.Get(resource));
            Assert.Equal(0u, state.Players[2].ResourceCards.Get(resource));
            Assert.Equal(demandA + demandB - 1, state.ResourceBank.Get(resource));
        }

        [Fact]
        public void ShortageForASinglePlayerPaysTheRemainder()
        {
            TestUtil.Seed(31);
            (GameState state, Tile tile, int cornerA, _) = ProductionSetup();
            ResourceCardType resource = tile.Type.ToCardType();

            PlaceBuilding(state, cornerA, 1, Intersection.BuildingType.City);
            uint demand = Demand(state, 1, tile.Number!.Value, resource);
            Assert.True(demand >= 2);
            LimitBank(state, resource, demand - 1);

            Roll(state, tile.Number!.Value);

            Assert.Equal(demand - 1, state.Players[1].ResourceCards.Get(resource));
            Assert.Equal(0u, state.ResourceBank.Get(resource));
        }

        [Fact]
        public void ProductionPaysSettlementsOnceAndCitiesTwice()
        {
            TestUtil.Seed(32);
            (GameState state, Tile tile, int cornerA, int cornerB) = ProductionSetup();
            ResourceCardType resource = tile.Type.ToCardType();

            PlaceBuilding(state, cornerA, 1, Intersection.BuildingType.Settlement);
            PlaceBuilding(state, cornerB, 2, Intersection.BuildingType.City);
            uint demandA = Demand(state, 1, tile.Number!.Value, resource);
            uint demandB = Demand(state, 2, tile.Number!.Value, resource);
            Assert.True(demandB >= 2 * 1 && demandB % 2 == 0);

            Roll(state, tile.Number!.Value);

            Assert.Equal(demandA, state.Players[1].ResourceCards.Get(resource));
            Assert.Equal(demandB, state.Players[2].ResourceCards.Get(resource));
        }

        [Fact]
        public void RobberBlocksProductionOfItsTile()
        {
            TestUtil.Seed(33);
            (GameState state, Tile tile, int cornerA, _) = ProductionSetup();
            ResourceCardType resource = tile.Type.ToCardType();

            PlaceBuilding(state, cornerA, 1, Intersection.BuildingType.Settlement);
            uint demandWithoutRobber = Demand(state, 1, tile.Number!.Value, resource);
            state.Board.Robber = tile;
            uint demandWithRobber = Demand(state, 1, tile.Number!.Value, resource);
            Assert.True(demandWithRobber < demandWithoutRobber);

            Roll(state, tile.Number!.Value);

            Assert.Equal(demandWithRobber, state.Players[1].ResourceCards.Get(resource));
        }

        /// <summary>
        /// Empty board before player 0's roll, plus a producing tile without the robber and two of its corners that do not touch
        /// </summary>
        private static (GameState state, Tile tile, int cornerA, int cornerB) ProductionSetup()
        {
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            Tile tile = state.Board.Map.Values.First(t => t.HasYield() && t.Number is not null && !state.Board.HasRobber(t));

            List<Intersection> corners = state.Board.Adjacency.GetIntersections(tile).ToList();
            foreach (Intersection a in corners)
                foreach (Intersection b in corners)
                {
                    bool adjacent = state.Board.Adjacency.GetEdges(a).Intersect(state.Board.Adjacency.GetEdges(b)).Any();
                    if (a.Index != b.Index && !adjacent) return (state, tile, a.Index, b.Index);
                }

            throw new InvalidOperationException("No pair of non-adjacent corners found.");
        }

        private static void PlaceBuilding(GameState state, int intersectionIdx, int playerIdx, Intersection.BuildingType type)
        {
            state.Board.Intersections[intersectionIdx].Building = type;
            state.Board.Intersections[intersectionIdx].Owner = (sbyte)playerIdx;
        }

        /// <summary>
        /// Cards of the resource a player is owed for the number, from all of their buildings and all matching tiles
        /// </summary>
        private static uint Demand(GameState state, int playerIdx, byte number, ResourceCardType resource)
        {
            uint demand = 0;
            foreach (Intersection intersection in state.Board.Intersections.Where(i => i.Owner == playerIdx && i.Building != Intersection.BuildingType.None))
            {
                foreach (Tile tile in state.Board.Adjacency.GetTiles(intersection))
                {
                    if (tile.Number == number && tile.HasYield() && tile.Type.ToCardType() == resource && !state.Board.HasRobber(tile))
                        demand += intersection.Building == Intersection.BuildingType.City ? 2u : 1u;
                }
            }
            return demand;
        }

        private static void LimitBank(GameState state, ResourceCardType resource, uint remaining)
        {
            state.ResourceBank.Remove(resource, state.ResourceBank.Get(resource) - remaining);
        }

        private static void Roll(GameState state, int total)
        {
            RollAction roll = (RollAction)RollAction.GetActionsForState(state, (sbyte)state.Turn.PlayerIndex).Single();
            Action outcome = roll.GetOutcomeVariants(state, (sbyte)state.Turn.PlayerIndex)
                .Single(variant => ((RollAction)variant.Action).History!.RollResult.Total == total).Action;
            outcome.Apply(state);
        }

        // ---------------------------------------------------------------- Seven, discards and the robber

        [Fact]
        public void SevenRequiresDiscardingHalfRoundedDownAboveSevenCards()
        {
            TestUtil.Seed(40);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            TestUtil.GiveCards(state, 1, ResourceCardType.Lumber, 5);
            TestUtil.GiveCards(state, 1, ResourceCardType.Ore, 4);
            TestUtil.GiveCards(state, 2, ResourceCardType.Wool, 7);

            Roll(state, 7);

            Assert.True(state.Turn.MustDiscard);
            Assert.True(state.Turn.MustMoveRobber);
            Assert.True(state.Turn.AwaitedPlayerDiscards[1]);
            Assert.False(state.Turn.AwaitedPlayerDiscards[2]);
            Assert.False(state.Turn.AwaitedPlayerDiscards[0]);

            // Only the discarding player can act, the robber waits for all discards
            Assert.True(state.CanPlayerAct(1));
            Assert.False(state.CanPlayerAct(0));
            Assert.Empty(RobberAction.GetActionsForState(state, 0));

            List<Action> discards = DiscardAction.GetActionsForState(state, 1);
            Assert.NotEmpty(discards);
            Assert.All(discards, discard => Assert.Equal(4u, ((DiscardAction)discard).SelectedCards.Count()));
            Assert.False(new DiscardAction(1, Cards(ResourceCardType.Lumber, 3)).IsValidFor(state));
            Assert.False(new DiscardAction(1, Cards(ResourceCardType.Lumber, 5)).IsValidFor(state));
            Assert.False(new DiscardAction(2, Cards(ResourceCardType.Wool, 3)).IsValidFor(state));

            DiscardAction valid = new(1, Cards(ResourceCardType.Lumber, 4));
            Assert.True(valid.IsValidFor(state));
            valid.Apply(state);

            Assert.Equal(5u, state.Players[1].ResourceCards.Count());
            Assert.False(state.Turn.MustDiscard);
            Assert.NotEmpty(RobberAction.GetActionsForState(state, 0));
        }

        [Fact]
        public void RobberMustMoveToAnotherLandTile()
        {
            TestUtil.Seed(41);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            Roll(state, 7);
            Assert.True(state.Turn.MustMoveRobber);

            int robberIdx = state.Board.GetRobberTileIndex()!.Value;
            Assert.False(new RobberAction(0, robberIdx, null).IsValidFor(state));

            List<Action> moves = RobberAction.GetActionsForState(state, 0);
            Assert.NotEmpty(moves);
            Assert.All(moves, move =>
            {
                Tile target = state.Board.Map.Values[((RobberAction)move).TargetTileIndex];
                Assert.True(target.IsLandTile());
                Assert.NotEqual(robberIdx, ((RobberAction)move).TargetTileIndex);
            });

            Tile water = state.Board.Map.Values.First(tile => !tile.IsLandTile());
            Assert.False(new RobberAction(0, Array.IndexOf(state.Board.Map.Values, water), null).IsValidFor(state));
        }

        [Fact]
        public void RobberStealsOnlyFromAdjacentOpponents()
        {
            TestUtil.Seed(42);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            Roll(state, 7);

            Tile target = state.Board.Map.Values.First(tile => tile.IsLandTile() && !state.Board.HasRobber(tile));
            int targetIdx = Array.IndexOf(state.Board.Map.Values, target);
            Intersection corner = state.Board.Adjacency.GetIntersections(target).First();
            PlaceBuilding(state, corner.Index, 2, Intersection.BuildingType.Settlement);
            TestUtil.GiveCards(state, 2, ResourceCardType.Grain, 1);

            Assert.True(new RobberAction(0, targetIdx, 2).IsValidFor(state));
            Assert.False(new RobberAction(0, targetIdx, 3).IsValidFor(state));
            Assert.False(new RobberAction(0, targetIdx, 0).IsValidFor(state));
            // Not stealing is only allowed when no opponent is adjacent
            Assert.False(new RobberAction(0, targetIdx, null).IsValidFor(state));

            RobberAction steal = new(0, targetIdx, 2);
            steal.Apply(state);
            Assert.Equal(1u, state.Players[0].ResourceCards.Get(ResourceCardType.Grain));
            Assert.Equal(0u, state.Players[2].ResourceCards.Count());
            Assert.False(state.Turn.MustMoveRobber);
        }

        // ---------------------------------------------------------------- Development cards

        [Fact]
        public void DevelopmentCardsBoughtThisTurnCannotBePlayed()
        {
            TestUtil.Seed(50);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.DevelopmentBank.Remove(DevelopmentCardType.Knight, 1);
            state.Players[0].DevelopmentCards.Add(DevelopmentCardType.Knight, 1);
            state.Players[0].NewDevelopmentCards.Add(DevelopmentCardType.Knight, 1);

            Assert.False(new KnightAction(0).IsValidFor(state));

            new EndTurnAction(0).Apply(state);
            state.Turn.PlayerIndex = 0;
            state.Turn.MustRoll = false;

            Assert.True(new KnightAction(0).IsValidFor(state));
        }

        [Fact]
        public void OnlyOneDevelopmentCardPerTurnAndKnightsBeforeTheRoll()
        {
            TestUtil.Seed(51);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            state.DevelopmentBank.Remove(DevelopmentCardType.Knight, 2);
            state.Players[0].DevelopmentCards.Add(DevelopmentCardType.Knight, 2);

            KnightAction knight = new(0);
            Assert.True(knight.IsValidFor(state));
            knight.Apply(state);

            Assert.True(state.Turn.MustMoveRobber);
            Assert.True(state.Turn.HasPlayedDevelopmentCard);
            Assert.False(new KnightAction(0).IsValidFor(state));
            // The roll waits until the robber has moved
            Assert.Empty(RollAction.GetActionsForState(state, 0));
        }

        [Fact]
        public void ThirdKnightAwardsLargestArmyAndTiesKeepTheHolder()
        {
            TestUtil.Seed(52);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Players[0].PlayedKnights = 2;
            state.Players[1].PlayedKnights = 3;
            state.CalculateLargestArmy(1);
            Assert.Equal(2, state.Players[1].VictoryPoints.LargestArmyPoints);

            state.Players[0].PlayedKnights = 3;
            state.CalculateLargestArmy(0);
            Assert.Equal(2, state.Players[1].VictoryPoints.LargestArmyPoints);
            Assert.Equal(0, state.Players[0].VictoryPoints.LargestArmyPoints);

            state.Players[0].PlayedKnights = 4;
            state.CalculateLargestArmy(0);
            Assert.Equal(2, state.Players[0].VictoryPoints.LargestArmyPoints);
            Assert.Equal(0, state.Players[1].VictoryPoints.LargestArmyPoints);
        }

        [Fact]
        public void VictoryPointCardWinsImmediatelyOnTheOwnTurn()
        {
            TestUtil.Seed(53);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Players[0].VictoryPoints.SettlementPoints = 9;
            foreach (DevelopmentCardType type in CardSet<DevelopmentCardType>.Values)
                if (type != DevelopmentCardType.VictoryPoint)
                    state.DevelopmentBank.Remove(type, state.DevelopmentBank.Get(type));
            TestUtil.GiveCards(state, 0, ResourceCardType.Wool, 1);
            TestUtil.GiveCards(state, 0, ResourceCardType.Grain, 1);
            TestUtil.GiveCards(state, 0, ResourceCardType.Ore, 1);

            new BuyDevelopmentCardAction(0).Apply(state);

            Assert.True(state.HasEnded);
            Assert.Equal(0, state.Turn.PlayerIndex);
        }

        // ---------------------------------------------------------------- Placement

        [Fact]
        public void SettlementsRespectTheDistanceRuleAndNeedAnOwnRoad()
        {
            TestUtil.Seed(60);
            GameState state = TestUtil.EmptyNormalTurn(0);
            foreach (ResourceCardType type in new[] { ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain })
                TestUtil.GiveCards(state, 0, type, 2);

            Intersection site = state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).Any(t => t.IsLandTile()) && state.Board.Adjacency.GetEdges(i).Count == 3);
            Edge road = state.Board.Adjacency.GetEdges(site).First();

            // No own road yet
            Assert.False(new SettlementAction(0, site.Index).IsValidFor(state));

            road.Building = Edge.BuildingType.Road;
            road.Owner = 0;
            Assert.True(new SettlementAction(0, site.Index).IsValidFor(state));

            // A building on a neighboring intersection forbids it
            (Intersection top, Intersection bottom) = state.Board.Adjacency.GetIntersections(road);
            Intersection neighbor = top.Index != site.Index ? top : bottom;
            PlaceBuilding(state, neighbor.Index, 1, Intersection.BuildingType.Settlement);
            Assert.False(new SettlementAction(0, site.Index).IsValidFor(state));
        }

        [Fact]
        public void BuildingStockLimitsSettlements()
        {
            TestUtil.Seed(61);
            GameState state = TestUtil.EmptyNormalTurn(0);
            foreach (ResourceCardType type in new[] { ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain })
                TestUtil.GiveCards(state, 0, type, 2);

            Intersection site = state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).Any(t => t.IsLandTile()));
            Edge road = state.Board.Adjacency.GetEdges(site).First();
            road.Building = Edge.BuildingType.Road;
            road.Owner = 0;
            Assert.True(new SettlementAction(0, site.Index).IsValidFor(state));

            state.Players[0].BuildingStock.RemainingSettlements = 0;
            Assert.False(new SettlementAction(0, site.Index).IsValidFor(state));
        }

        [Fact]
        public void CityReturnsTheSettlementToStockAndAddsOneVictoryPoint()
        {
            TestUtil.Seed(62);
            GameState state = TestUtil.EmptyNormalTurn(0);
            Intersection site = state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).Any(t => t.IsLandTile()));
            PlaceBuilding(state, site.Index, 0, Intersection.BuildingType.Settlement);
            state.Players[0].BuildingStock.RemainingSettlements--;
            state.Players[0].VictoryPoints.SettlementPoints = 1;
            TestUtil.GiveCards(state, 0, ResourceCardType.Ore, 3);
            TestUtil.GiveCards(state, 0, ResourceCardType.Grain, 2);

            CityAction city = new(0, site.Index);
            Assert.True(city.IsValidFor(state));
            city.Apply(state);

            Assert.Equal(Intersection.BuildingType.City, site.Building);
            Assert.Equal(BuildingStock.MAX_SETTLEMENTS, state.Players[0].BuildingStock.RemainingSettlements);
            Assert.Equal(BuildingStock.MAX_CITIES - 1, state.Players[0].BuildingStock.RemainingCities);
            Assert.Equal(2, state.Players[0].VictoryPoints.Total);
            Assert.Equal(0u, state.Players[0].ResourceCards.Count());
        }

        private static CardSet<ResourceCardType> Cards(ResourceCardType type, uint amount)
        {
            CardSet<ResourceCardType> cards = new();
            cards.Add(type, amount);
            return cards;
        }
    }
}
