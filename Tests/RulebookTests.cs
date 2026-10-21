using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The base game rulebook, section by section: components and setup, turn structure, production, the 7, trading,
    /// building, development cards, special cards and winning. Rules the simulator does not implement (yet) are listed
    /// as skipped tests with the reason, so they stay visible in every test run.
    /// </summary>
    public class RulebookTests
    {
        private static readonly ResourceCardType[] Resources =
            [ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain, ResourceCardType.Ore];

        // ================================================================ Components and board setup

        [Fact]
        public void BoardHasTheBaseGameTilesNumbersHarborsAndRobber()
        {
            TestUtil.Seed(100);
            byte[] expectedNumbers = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

            for (int board = 0; board < 40; board++)
            {
                Board generated = MapGenerator.GenerateRandomClassic();
                List<Tile> land = generated.Map.Values.Where(tile => tile.IsLandTile()).ToList();

                Assert.Equal(19, land.Count);
                Assert.Equal(4, land.Count(tile => tile.Type == Tile.TileType.Lumber));
                Assert.Equal(4, land.Count(tile => tile.Type == Tile.TileType.Wool));
                Assert.Equal(4, land.Count(tile => tile.Type == Tile.TileType.Grain));
                Assert.Equal(3, land.Count(tile => tile.Type == Tile.TileType.Brick));
                Assert.Equal(3, land.Count(tile => tile.Type == Tile.TileType.Ore));
                Assert.Equal(1, land.Count(tile => tile.Type == Tile.TileType.Desert));

                Assert.Null(land.Single(tile => tile.Type == Tile.TileType.Desert).Number);
                Assert.Equal(expectedNumbers, land.Where(tile => tile.Type != Tile.TileType.Desert).Select(tile => tile.Number!.Value).OrderBy(n => n));

                // The robber starts on the desert
                Assert.Equal(Tile.TileType.Desert, generated.Robber!.Type);

                // 9 harbors: four generic 3:1 and one 2:1 harbor per resource
                Assert.Equal(9, generated.Ports.Count);
                Assert.Equal(4, generated.Ports.Count(port => port.Type == Port.TradeType.Generic));
                foreach (Port.TradeType type in new[] { Port.TradeType.Lumber, Port.TradeType.Brick, Port.TradeType.Wool, Port.TradeType.Grain, Port.TradeType.Ore })
                    Assert.Equal(1, generated.Ports.Count(port => port.Type == type));

                // Red numbers (6 and 8) are never adjacent
                foreach (Tile red in land.Where(MapGenerator.RedNumberPredicate))
                    Assert.Empty(generated.Map.GetNeighbors(red.X, red.Y, MapGenerator.RedNumberPredicate));
            }
        }

        [Fact]
        public void SuppliesMatchTheRulebook()
        {
            TestUtil.Seed(101);
            GameState state = TestUtil.NewGame();

            foreach (ResourceCardType type in Resources)
                Assert.Equal(19u, state.ResourceBank.Get(type));

            Assert.Equal(25u, state.DevelopmentBank.Count());
            Assert.Equal(14u, state.DevelopmentBank.Get(DevelopmentCardType.Knight));
            Assert.Equal(5u, state.DevelopmentBank.Get(DevelopmentCardType.VictoryPoint));
            Assert.Equal(2u, state.DevelopmentBank.Get(DevelopmentCardType.RoadBuilding));
            Assert.Equal(2u, state.DevelopmentBank.Get(DevelopmentCardType.YearOfPlenty));
            Assert.Equal(2u, state.DevelopmentBank.Get(DevelopmentCardType.Monopoly));

            Assert.All(state.Players, player =>
            {
                Assert.Equal(15u, player.BuildingStock.RemainingRoads);
                Assert.Equal(5u, player.BuildingStock.RemainingSettlements);
                Assert.Equal(4u, player.BuildingStock.RemainingCities);
                Assert.Equal(0u, player.ResourceCards.Count());
                Assert.Equal(0u, player.DevelopmentCards.Count());
                Assert.Equal(0, player.VictoryPoints.Total);
            });

            Assert.Equal(10, state.Settings.VictoryPoints);
            Assert.Equal(7, state.Settings.RobberCardLimit);
        }

        // ================================================================ Setup phase

        [Fact]
        public void SetupPlacesSettlementAndRoadInSnakeOrderThenPlayerOneRolls()
        {
            TestUtil.Seed(102);

            for (int game = 0; game < 10; game++)
            {
                GameState state = TestUtil.NewGame();
                List<(int player, string type)> placements = [];
                TestUtil.PlayUntil(state, s => s.Turn.TypeOfRound == TurnState.RoundType.Normal, played: null);

                // Replay to record the order
                GameState replay = TestUtil.NewGame();
                List<Action> played = [];
                TestUtil.Seed(1000 + game);
                replay = TestUtil.NewGame();
                TestUtil.PlayUntil(replay, s => s.Turn.TypeOfRound == TurnState.RoundType.Normal, played);
                placements = played.Select(a => ((int)a.PlayerIndex, a.GetType().Name)).ToList();

                List<(int, string)> expected = [];
                foreach (int player in new[] { 0, 1, 2, 3 })
                    expected.AddRange([(player, nameof(FirstInitialSettlementAction)), (player, nameof(FirstInitialRoadAction))]);
                foreach (int player in new[] { 3, 2, 1, 0 })
                    expected.AddRange([(player, nameof(SecondInitialSettlementAction)), (player, nameof(SecondInitialRoadAction))]);

                Assert.Equal(expected, placements);
                Assert.Equal(0, replay.Turn.PlayerIndex);
                Assert.True(replay.Turn.MustRoll);
                Assert.All(replay.Players, player =>
                {
                    Assert.Equal(2, player.VictoryPoints.Total);
                    Assert.Equal(13u, player.BuildingStock.RemainingRoads);
                    Assert.Equal(3u, player.BuildingStock.RemainingSettlements);
                });
            }
        }

        [Fact]
        public void SetupSettlementsNeedNoRoadButKeepTheDistanceRule()
        {
            TestUtil.Seed(103);
            GameState state = TestUtil.NewGame();
            Intersection first = LandIntersection(state, i => state.Board.Adjacency.GetEdges(i).Count == 3);

            FirstInitialSettlementAction settle = new(0, first.Index);
            Assert.True(settle.IsValidFor(state));
            settle.Apply(state);

            // The road has to touch the settlement just placed
            Edge touching = state.Board.Adjacency.GetEdges(first).First();
            Edge elsewhere = state.Board.Edges.First(edge => !state.Board.Adjacency.GetEdges(first).Contains(edge));
            Assert.False(new FirstInitialRoadAction(0, elsewhere.Index).IsValidFor(state));
            Assert.All(FirstInitialRoadAction.GetActionsForState(state, 0), road => Assert.Contains(state.Board.Edges[((FirstInitialRoadAction)road).EdgeIndex], state.Board.Adjacency.GetEdges(first)));
            new FirstInitialRoadAction(0, touching.Index).Apply(state);

            // Next player: neighbors of the first settlement are forbidden, far intersections are fine
            Intersection neighbor = Neighbors(state, first).First();
            Assert.False(new FirstInitialSettlementAction(1, neighbor.Index).IsValidFor(state));
            Assert.False(new FirstInitialSettlementAction(1, first.Index).IsValidFor(state));
            Intersection far = LandIntersection(state, i => i.Index != first.Index && !Neighbors(state, first).Contains(i) && !Neighbors(state, i).Any(n => n.Building != Intersection.BuildingType.None));
            Assert.True(new FirstInitialSettlementAction(1, far.Index).IsValidFor(state));
        }

        [Fact]
        public void OnlyTheSecondSetupSettlementYieldsOneCardPerAdjacentProducingTile()
        {
            TestUtil.Seed(104);

            for (int game = 0; game < 10; game++)
            {
                GameState state = TestUtil.NewGame();
                Agents.Agent[] agents = TestUtil.RandomAgents();

                while (state.Turn.TypeOfRound != TurnState.RoundType.Normal)
                {
                    sbyte playerIdx = TestUtil.ActingPlayer(state);
                    Action action = agents[playerIdx].Act(state);
                    CardSet<ResourceCardType> before = new(state.Players[playerIdx].ResourceCards);
                    action.Apply(state);
                    CardSet<ResourceCardType> after = state.Players[playerIdx].ResourceCards;

                    if (action is SecondInitialSettlementAction second)
                    {
                        Intersection site = state.Board.Intersections[second.IntersectionIndex];
                        foreach (ResourceCardType type in Resources)
                        {
                            uint expected = (uint)state.Board.Adjacency.GetTiles(site).Count(tile => tile.HasYield() && tile.Type.ToCardType() == type);
                            Assert.Equal(before.Get(type) + expected, after.Get(type));
                        }

                        // The second road has to touch the second settlement
                        Assert.All(SecondInitialRoadAction.GetActionsForState(state, playerIdx),
                            road => Assert.Contains(state.Board.Edges[((SecondInitialRoadAction)road).EdgeIndex], state.Board.Adjacency.GetEdges(site)));
                    }
                    else
                    {
                        Assert.Equal(before, after);
                    }
                }
            }
        }

        // ================================================================ Turn structure

        [Fact]
        public void TurnStartsWithExactlyOneRollAndOnlyDevelopmentCardsMayComeFirst()
        {
            TestUtil.Seed(105);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.TypeOfRound == TurnState.RoundType.Normal && s.Turn.MustRoll && s.Players.Any(p => p.ResourceCards.Count() > 4)));
            sbyte playerIdx = (sbyte)state.Turn.PlayerIndex;
            GiveCost(state, playerIdx, RoadCost);
            GiveCost(state, playerIdx, DevelopmentCardCost);

            // Before the roll: no building, buying, trading or ending the turn
            Assert.All(TestUtil.AllProvidedActions(state, playerIdx), action => Assert.True(action is RollAction or KnightAction or MonopolyAction or YearOfPlentyAction or RoadBuildingAction, $"{action} offered before the roll"));
            Assert.False(new EndTurnAction(playerIdx).IsValidFor(state));
            Assert.Empty(BuyDevelopmentCardAction.GetActionsForState(state, playerIdx));
            Assert.Empty(FourToOneTradeAction.GetActionsForState(state, playerIdx));

            Action roll = RollAction.GetActionsForState(state, playerIdx).Single();
            roll.Apply(state);
            Assert.Empty(RollAction.GetActionsForState(state, playerIdx));
        }

        [Fact]
        public void EndingTheTurnPassesToTheNextPlayerWhoMustRoll()
        {
            TestUtil.Seed(106);
            GameState state = TestUtil.EmptyNormalTurn(3);
            int round = state.Turn.RoundCounter;

            new EndTurnAction(3).Apply(state);
            Assert.Equal(0, state.Turn.PlayerIndex);
            Assert.True(state.Turn.MustRoll);
            Assert.Equal(round + 1, state.Turn.RoundCounter);

            state.Turn.MustRoll = false;
            new EndTurnAction(0).Apply(state);
            Assert.Equal(1, state.Turn.PlayerIndex);
            Assert.Equal(round + 1, state.Turn.RoundCounter);
        }

        // ================================================================ Production and the 7

        [Fact]
        public void ASevenProducesNothing()
        {
            TestUtil.Seed(107);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            foreach (Intersection intersection in state.Board.Intersections.Where(i => state.Board.Adjacency.GetTiles(i).Any(t => t.HasYield())).Take(1))
            {
                intersection.Building = Intersection.BuildingType.City;
                intersection.Owner = 1;
            }

            RollTotal(state, 7);

            Assert.All(state.Players, player => Assert.Equal(0u, player.ResourceCards.Count()));
            Assert.All(Resources, type => Assert.Equal(19u, state.ResourceBank.Get(type)));
        }

        [Fact]
        public void ProductionPaysAllPlayersNotOnlyTheRoller()
        {
            TestUtil.Seed(108);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            Tile tile = state.Board.Map.Values.First(t => t.HasYield() && !state.Board.HasRobber(t));
            List<Intersection> corners = state.Board.Adjacency.GetIntersections(tile).ToList();
            corners[0].Building = Intersection.BuildingType.Settlement;
            corners[0].Owner = 0;
            corners[3].Building = Intersection.BuildingType.Settlement;
            corners[3].Owner = 2;

            RollTotal(state, tile.Number!.Value);

            Assert.True(state.Players[0].ResourceCards.Get(tile.Type.ToCardType()) >= 1);
            Assert.True(state.Players[2].ResourceCards.Get(tile.Type.ToCardType()) >= 1);
        }

        // ================================================================ Trade

        [Fact(Skip = "Not implemented: domestic trade between players (TODO.md: \"Action für variable Player Trades\")")]
        public void PlayersMayTradeResourcesWithEachOtherOnTheirTurn()
        {
        }

        [Fact]
        public void MaritimeTradeOnlyAfterTheRollAndWithEnoughCards()
        {
            TestUtil.Seed(109);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            TestUtil.GiveCards(state, 0, ResourceCardType.Brick, 4);
            FourToOneTradeAction trade = new(0, ResourceCardType.Brick, ResourceCardType.Ore);
            Assert.False(trade.IsValidFor(state));

            state.Turn.MustRoll = false;
            Assert.True(trade.IsValidFor(state));

            // The bank must have the requested card
            state.ResourceBank.Remove(ResourceCardType.Ore, 19);
            Assert.False(trade.IsValidFor(state));
        }

        [Fact]
        public void HarborPrivilegesComeFromBuildingOnTheHarbor()
        {
            TestUtil.Seed(110);

            for (int board = 0; board < 10; board++)
            {
                GameState state = TestUtil.NewGame();
                Port port = state.Board.Ports[board % state.Board.Ports.Count];
                (Intersection top, _) = HarborIntersections(state, port);

                Assert.True(new FirstInitialSettlementAction(0, top.Index).IsValidFor(state));
                new FirstInitialSettlementAction(0, top.Index).Apply(state);
                Assert.True(state.Players[0].PortPrivileges.HasFlag(port.Type.GetPortPrivilege()));

                // A settlement away from every harbor gives none
                GameState other = TestUtil.NewGame();
                HashSet<int> harborSites = other.Board.Ports.SelectMany(p => { var (a, b) = HarborIntersections(other, p); return new[] { (int)a.Index, b.Index }; }).ToHashSet();
                Intersection inland = LandIntersection(other, i => !harborSites.Contains(i.Index));
                new FirstInitialSettlementAction(0, inland.Index).Apply(other);
                Assert.Equal(PortPrivileges.None, other.Players[0].PortPrivileges);
            }
        }

        // ================================================================ Building

        private static readonly (ResourceCardType type, uint amount)[] RoadCost = [(ResourceCardType.Lumber, 1), (ResourceCardType.Brick, 1)];
        private static readonly (ResourceCardType type, uint amount)[] SettlementCost = [(ResourceCardType.Lumber, 1), (ResourceCardType.Brick, 1), (ResourceCardType.Wool, 1), (ResourceCardType.Grain, 1)];
        private static readonly (ResourceCardType type, uint amount)[] CityCost = [(ResourceCardType.Ore, 3), (ResourceCardType.Grain, 2)];
        private static readonly (ResourceCardType type, uint amount)[] DevelopmentCardCost = [(ResourceCardType.Wool, 1), (ResourceCardType.Grain, 1), (ResourceCardType.Ore, 1)];

        [Fact]
        public void BuildingCostsMatchTheRulebook()
        {
            TestUtil.Seed(111);

            AssertCost(RoadCost, state =>
            {
                Intersection own = OwnSettlement(state);
                return new RoadAction(0, state.Board.Adjacency.GetEdges(own).First().Index);
            });
            AssertCost(SettlementCost, state =>
            {
                Intersection site = LandIntersection(state, _ => true);
                Edge road = state.Board.Adjacency.GetEdges(site).First();
                road.Building = Edge.BuildingType.Road;
                road.Owner = 0;
                return new SettlementAction(0, site.Index);
            });
            AssertCost(CityCost, state => new CityAction(0, OwnSettlement(state).Index));
            AssertCost(DevelopmentCardCost, _ => new BuyDevelopmentCardAction(0));
        }

        /// <summary>
        /// The action is valid with exactly the cost, uses up exactly the cost (paid to the bank), and is invalid when any card is missing
        /// </summary>
        private static void AssertCost((ResourceCardType type, uint amount)[] cost, Func<GameState, Action> createAction)
        {
            GameState state = TestUtil.EmptyNormalTurn(0);
            Action action = createAction(state);
            GiveCost(state, 0, cost);

            Assert.True(action.IsValidFor(state), $"{action} is not valid with exactly its cost");
            action.Apply(state);
            Assert.Equal(0u, state.Players[0].ResourceCards.Count());
            Assert.All(Resources, type => Assert.Equal(19u, state.ResourceBank.Get(type)));

            foreach ((ResourceCardType missingType, _) in cost)
            {
                GameState lacking = TestUtil.EmptyNormalTurn(0);
                Action lackingAction = createAction(lacking);
                GiveCost(lacking, 0, cost);
                lacking.Players[0].ResourceCards.Remove(missingType, 1);
                lacking.ResourceBank.Add(missingType, 1);
                Assert.False(lackingAction.IsValidFor(lacking), $"{lackingAction} is valid without one {missingType}");
            }
        }

        [Fact]
        public void RoadsMustConnectAndCannotContinuePastAnOpponentsBuilding()
        {
            TestUtil.Seed(112);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveCost(state, 0, RoadCost);

            Intersection own = OwnSettlement(state);
            Edge first = state.Board.Adjacency.GetEdges(own).First();
            Edge unconnected = state.Board.Edges.First(edge =>
            {
                var (a, b) = state.Board.Adjacency.GetIntersections(edge);
                return a.Index != own.Index && b.Index != own.Index && !Neighbors(state, own).Contains(a) && !Neighbors(state, own).Contains(b);
            });
            Assert.True(new RoadAction(0, first.Index).IsValidFor(state));
            Assert.False(new RoadAction(0, unconnected.Index).IsValidFor(state));

            // Extend from the end of the first road ...
            new RoadAction(0, first.Index).Apply(state);
            GiveCost(state, 0, RoadCost);
            var (top, bottom) = state.Board.Adjacency.GetIntersections(first);
            Intersection end = top.Index != own.Index ? top : bottom;
            Edge? next = state.Board.Adjacency.GetEdges(end).FirstOrDefault(edge => edge.Index != first.Index);
            Assert.NotNull(next);
            Assert.True(new RoadAction(0, next!.Index).IsValidFor(state));

            // ... but not through an opponent's settlement on that intersection
            end.Building = Intersection.BuildingType.Settlement;
            end.Owner = 1;
            Assert.False(new RoadAction(0, next.Index).IsValidFor(state));

            // Occupied edges can't be built on
            Assert.False(new RoadAction(0, first.Index).IsValidFor(state));
        }

        [Fact]
        public void CitiesOnlyUpgradeOwnSettlements()
        {
            TestUtil.Seed(113);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveCost(state, 0, CityCost);

            Intersection empty = LandIntersection(state, _ => true);
            Assert.False(new CityAction(0, empty.Index).IsValidFor(state));

            empty.Building = Intersection.BuildingType.Settlement;
            empty.Owner = 1;
            Assert.False(new CityAction(0, empty.Index).IsValidFor(state));

            empty.Owner = 0;
            Assert.True(new CityAction(0, empty.Index).IsValidFor(state));

            empty.Building = Intersection.BuildingType.City;
            Assert.False(new CityAction(0, empty.Index).IsValidFor(state));
        }

        [Fact]
        public void BuildingStockLimitsRoadsAndCities()
        {
            TestUtil.Seed(114);
            GameState state = TestUtil.EmptyNormalTurn(0);
            Intersection own = OwnSettlement(state);
            GiveCost(state, 0, RoadCost);
            GiveCost(state, 0, CityCost);

            RoadAction road = new(0, state.Board.Adjacency.GetEdges(own).First().Index);
            CityAction city = new(0, own.Index);
            Assert.True(road.IsValidFor(state));
            Assert.True(city.IsValidFor(state));

            state.Players[0].BuildingStock.RemainingRoads = 0;
            state.Players[0].BuildingStock.RemainingCities = 0;
            Assert.False(road.IsValidFor(state));
            Assert.False(city.IsValidFor(state));
        }

        // ================================================================ Development cards

        [Fact]
        public void DevelopmentCardsCannotBeBoughtFromAnEmptyDeck()
        {
            TestUtil.Seed(115);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveCost(state, 0, DevelopmentCardCost);
            Assert.True(new BuyDevelopmentCardAction(0).IsValidFor(state));

            foreach (DevelopmentCardType type in CardSet<DevelopmentCardType>.Values)
                state.DevelopmentBank.Remove(type, state.DevelopmentBank.Get(type));
            Assert.False(new BuyDevelopmentCardAction(0).IsValidFor(state));
        }

        [Fact]
        public void KnightMovesTheRobberAndCountsForLargestArmy()
        {
            TestUtil.Seed(116);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.Knight);

            new KnightAction(0).Apply(state);

            Assert.True(state.Turn.MustMoveRobber);
            Assert.Equal(1u, state.Players[0].PlayedKnights);
            Assert.Equal(0u, state.Players[0].DevelopmentCards.Get(DevelopmentCardType.Knight));
            Assert.NotEmpty(RobberAction.GetActionsForState(state, 0));
            Assert.Empty(RoadAction.GetActionsForState(state, 0));
        }

        [Fact]
        public void RoadBuildingPlacesTwoRoadsForFree()
        {
            TestUtil.Seed(117);
            GameState state = TestUtil.EmptyNormalTurn(0);
            Intersection own = OwnSettlement(state);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.RoadBuilding);

            new RoadBuildingAction(0).Apply(state);
            Assert.Equal(2u, state.Players[0].BuildingStock.FreeRoads);

            for (int i = 0; i < 2; i++)
            {
                RoadAction free = (RoadAction)RoadAction.GetActionsForState(state, 0).First();
                free.Apply(state);
            }

            Assert.Equal(2, TestUtil.CountRoads(state, 0));
            Assert.Equal(0u, state.Players[0].ResourceCards.Count());
            Assert.Equal(0u, state.Players[0].BuildingStock.FreeRoads);
            // A third road has to be paid for again
            Assert.Empty(RoadAction.GetActionsForState(state, 0));
        }

        [Fact]
        public void RoadBuildingWithOneRoadLeftGivesOneAndUnusedRoadsReturnAtTheEndOfTheTurn()
        {
            TestUtil.Seed(118);
            GameState state = TestUtil.EmptyNormalTurn(0);
            OwnSettlement(state);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.RoadBuilding);
            state.Players[0].BuildingStock.RemainingRoads = 1;

            new RoadBuildingAction(0).Apply(state);
            Assert.Equal(1u, state.Players[0].BuildingStock.FreeRoads);
            Assert.Equal(0u, state.Players[0].BuildingStock.RemainingRoads);

            new EndTurnAction(0).Apply(state);
            Assert.Equal(0u, state.Players[0].BuildingStock.FreeRoads);
            Assert.Equal(1u, state.Players[0].BuildingStock.RemainingRoads);
        }

        [Fact(Skip = "Known deviation: roads from Road Building can only be placed after the roll and at any time later in the turn, the rulebook places them immediately")]
        public void RoadBuildingRoadsArePlacedImmediatelyEvenBeforeTheRoll()
        {
            TestUtil.Seed(119);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            OwnSettlement(state);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.RoadBuilding);

            new RoadBuildingAction(0).Apply(state);
            Assert.NotEmpty(RoadAction.GetActionsForState(state, 0));
            Assert.Empty(RollAction.GetActionsForState(state, 0));
        }

        [Fact]
        public void YearOfPlentyTakesAnyTwoResourcesFromTheBank()
        {
            TestUtil.Seed(120);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.YearOfPlenty);
            new YearOfPlentyAction(0, ResourceCardType.Ore, ResourceCardType.Ore).Apply(state);
            Assert.Equal(2u, state.Players[0].ResourceCards.Get(ResourceCardType.Ore));
            Assert.Equal(17u, state.ResourceBank.Get(ResourceCardType.Ore));

            GameState mixed = TestUtil.EmptyNormalTurn(0);
            GiveDevelopmentCard(mixed, 0, DevelopmentCardType.YearOfPlenty);
            new YearOfPlentyAction(0, ResourceCardType.Brick, ResourceCardType.Wool).Apply(mixed);
            Assert.Equal(1u, mixed.Players[0].ResourceCards.Get(ResourceCardType.Brick));
            Assert.Equal(1u, mixed.Players[0].ResourceCards.Get(ResourceCardType.Wool));

            // Cards the bank doesn't have can't be taken
            GameState scarce = TestUtil.EmptyNormalTurn(0);
            GiveDevelopmentCard(scarce, 0, DevelopmentCardType.YearOfPlenty);
            scarce.ResourceBank.Remove(ResourceCardType.Grain, 18);
            Assert.False(new YearOfPlentyAction(0, ResourceCardType.Grain, ResourceCardType.Grain).IsValidFor(scarce));
            Assert.True(new YearOfPlentyAction(0, ResourceCardType.Grain, ResourceCardType.Ore).IsValidFor(scarce));
            Assert.False(new YearOfPlentyAction(0, ResourceCardType.Grain, null).IsValidFor(scarce));

            // With a single card left in the whole bank, that one card is all there is to take
            foreach (ResourceCardType type in Resources)
                scarce.ResourceBank.Remove(type, scarce.ResourceBank.Get(type));
            scarce.ResourceBank.Add(ResourceCardType.Grain, 1);
            Assert.True(new YearOfPlentyAction(0, ResourceCardType.Grain, null).IsValidFor(scarce));
            Assert.Contains(YearOfPlentyAction.GetActionsForState(scarce, 0), action => action.IsSameDecision(new YearOfPlentyAction(0, ResourceCardType.Grain, null)));
        }

        [Fact]
        public void MonopolyTakesEveryCardOfTheTypeFromAllOpponents()
        {
            TestUtil.Seed(121);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.Monopoly);
            TestUtil.GiveCards(state, 1, ResourceCardType.Wool, 3);
            TestUtil.GiveCards(state, 2, ResourceCardType.Wool, 1);
            TestUtil.GiveCards(state, 2, ResourceCardType.Ore, 2);
            TestUtil.GiveCards(state, 0, ResourceCardType.Wool, 1);

            new MonopolyAction(0, ResourceCardType.Wool).Apply(state);

            Assert.Equal(5u, state.Players[0].ResourceCards.Get(ResourceCardType.Wool));
            Assert.Equal(0u, state.Players[1].ResourceCards.Get(ResourceCardType.Wool));
            Assert.Equal(0u, state.Players[2].ResourceCards.Get(ResourceCardType.Wool));
            Assert.Equal(2u, state.Players[2].ResourceCards.Get(ResourceCardType.Ore));
        }

        [Fact]
        public void OnlyOneDevelopmentCardPerTurnOfAnyKind()
        {
            TestUtil.Seed(122);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.Monopoly);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.YearOfPlenty);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.Knight);
            GiveDevelopmentCard(state, 0, DevelopmentCardType.RoadBuilding);

            new MonopolyAction(0, ResourceCardType.Ore).Apply(state);

            Assert.Empty(YearOfPlentyAction.GetActionsForState(state, 0));
            Assert.Empty(KnightAction.GetActionsForState(state, 0));
            Assert.Empty(RoadBuildingAction.GetActionsForState(state, 0));

            // The next turn allows the next card
            new EndTurnAction(0).Apply(state);
            state.Turn.PlayerIndex = 0;
            Assert.NotEmpty(KnightAction.GetActionsForState(state, 0));
        }

        [Fact]
        public void EveryDevelopmentCardMayBePlayedBeforeTheRoll()
        {
            TestUtil.Seed(123);
            foreach (DevelopmentCardType type in new[] { DevelopmentCardType.Knight, DevelopmentCardType.Monopoly, DevelopmentCardType.YearOfPlenty, DevelopmentCardType.RoadBuilding })
            {
                GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
                GiveDevelopmentCard(state, 0, type);
                Assert.Contains(TestUtil.AllProvidedActions(state, 0), action => action is KnightAction or MonopolyAction or YearOfPlentyAction or RoadBuildingAction);
            }
        }

        [Fact]
        public void VictoryPointCardsCountImmediatelyAndAreNeverPlayed()
        {
            TestUtil.Seed(124);
            GameState state = TestUtil.EmptyNormalTurn(0);
            GiveCost(state, 0, DevelopmentCardCost);
            BuyDevelopmentCardAction buy = new(0);
            Action victoryPoint = buy.GetOutcomeVariants(state, 0).Single(v => ((BuyDevelopmentCardAction)v.Action).History!.DrawnType == DevelopmentCardType.VictoryPoint).Action;

            victoryPoint.Apply(state);
            Assert.Equal(1, state.Players[0].VictoryPoints.DevelopmentCardPoints);
            Assert.Equal(1, state.Players[0].VictoryPoints.Total);

            // Holding only a VP card offers no development card action
            Assert.DoesNotContain(TestUtil.AllProvidedActions(state, 0), action => action is KnightAction or MonopolyAction or YearOfPlentyAction or RoadBuildingAction);
        }

        // ================================================================ Longest Road and Largest Army

        [Fact]
        public void LongestRoadNeedsFiveContinuousSegmentsAndBranchesDoNotAdd()
        {
            TestUtil.Seed(125);
            GameState state = TestUtil.EmptyNormalTurn(0);
            Board board = state.Board;

            // A star of three arms with two roads each: six roads, but the longest continuous road is four
            (List<int[]> arms, List<int> armEnds, HashSet<int> used) = FindStar(board, armLength: 2);
            foreach (int edge in arms.SelectMany(arm => arm))
            {
                board.Edges[edge].Building = Edge.BuildingType.Road;
                board.Edges[edge].Owner = 0;
            }
            state.CalculateLongestRoad(0);
            Assert.Equal(4u, state.Players[0].LongestRoadLength);
            Assert.Equal(0, state.Players[0].VictoryPoints.LongestRoadPoints);

            // Extending one arm by one road gives five in a row and the card
            Edge extension = armEnds.Select(end => board.Adjacency.GetEdges(board.Intersections[end]).FirstOrDefault(edge =>
            {
                var (p, q) = board.Adjacency.GetIntersections(edge);
                return !used.Contains((p.Index != end ? p : q).Index);
            })).First(edge => edge != null)!;
            extension.Building = Edge.BuildingType.Road;
            extension.Owner = 0;
            state.CalculateLongestRoad(0);
            Assert.Equal(5u, state.Players[0].LongestRoadLength);
            Assert.Equal(2, state.Players[0].VictoryPoints.LongestRoadPoints);
        }

        [Fact]
        public void ARingOfRoadsCountsAllItsSegments()
        {
            TestUtil.Seed(126);
            GameState state = TestUtil.EmptyNormalTurn(0);
            Tile tile = state.Board.Map.Values.First(t => t.IsLandTile());
            foreach (Edge edge in state.Board.Adjacency.GetEdges(tile))
            {
                edge.Building = Edge.BuildingType.Road;
                edge.Owner = 0;
            }

            state.CalculateLongestRoad(0);
            Assert.Equal(6u, state.Players[0].LongestRoadLength);
            Assert.Equal(2, state.Players[0].VictoryPoints.LongestRoadPoints);
        }

        // ================================================================ Winning

        [Fact]
        public void ReachingTenPointsOnTheOwnTurnEndsTheGameImmediately()
        {
            TestUtil.Seed(127);
            GameState state = TestUtil.EmptyNormalTurn(0);
            Intersection own = OwnSettlement(state);
            state.Players[0].VictoryPoints.SettlementPoints = 9;
            GiveCost(state, 0, CityCost);

            new CityAction(0, own.Index).Apply(state);

            Assert.True(state.HasEnded);
            Assert.Equal(0, state.Turn.PlayerIndex);
            Assert.All(Enumerable.Range(0, TestUtil.PlayerCount), idx => Assert.Empty(TestUtil.AllProvidedActions(state, (sbyte)idx)));
        }

        [Fact]
        public void TenPointsOnAnotherPlayersTurnOnlyWinWhenTheirOwnTurnStarts()
        {
            TestUtil.Seed(128);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Players[1].VictoryPoints.SettlementPoints = 10;

            // Still player 0's turn: the game goes on
            Assert.False(state.HasEnded);
            Assert.NotEmpty(TestUtil.AllProvidedActions(state, 0));

            new EndTurnAction(0).Apply(state);
            Assert.True(state.HasEnded);
            Assert.Equal(1, state.Turn.PlayerIndex);
        }

        // ================================================================ Helpers

        private static void GiveCost(GameState state, int playerIdx, (ResourceCardType type, uint amount)[] cost)
        {
            foreach ((ResourceCardType type, uint amount) in cost)
                TestUtil.GiveCards(state, playerIdx, type, amount);
        }

        /// <summary>
        /// Gives a development card bought in an earlier turn (so it can be played now)
        /// </summary>
        private static void GiveDevelopmentCard(GameState state, int playerIdx, DevelopmentCardType type)
        {
            state.DevelopmentBank.Remove(type, 1);
            state.Players[playerIdx].DevelopmentCards.Add(type, 1);
        }

        private static Intersection LandIntersection(GameState state, Func<Intersection, bool> condition)
        {
            return state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).Any(t => t.IsLandTile()) && condition(i));
        }

        private static IEnumerable<Intersection> Neighbors(GameState state, Intersection intersection)
        {
            foreach (Edge edge in state.Board.Adjacency.GetEdges(intersection))
            {
                var (top, bottom) = state.Board.Adjacency.GetIntersections(edge);
                yield return top.Index != intersection.Index ? top : bottom;
            }
        }

        /// <summary>
        /// Places a settlement of player 0 on a land intersection with three edges and returns it
        /// </summary>
        private static Intersection OwnSettlement(GameState state)
        {
            Intersection site = LandIntersection(state, i => state.Board.Adjacency.GetEdges(i).Count == 3 && state.Board.Adjacency.GetTiles(i).All(t => t.IsLandTile()));
            site.Building = Intersection.BuildingType.Settlement;
            site.Owner = 0;
            state.Players[0].BuildingStock.RemainingSettlements--;
            state.Players[0].VictoryPoints.SettlementPoints++;
            return site;
        }

        private static (Intersection top, Intersection bottom) HarborIntersections(GameState state, Port port)
        {
            Tile seaAnchor = state.Board.Map.GetTile(port.AnchorTileX, port.AnchorTileY);
            Tile landAnchor = state.Board.Adjacency.GetTile(seaAnchor, port.AnchorDirection)!;
            Edge portEdge = state.Board.Adjacency.GetEdge(landAnchor, port.AnchorDirection.Mirror())!;
            return state.Board.Adjacency.GetIntersections(portEdge);
        }

        private static void RollTotal(GameState state, int total)
        {
            RollAction roll = (RollAction)RollAction.GetActionsForState(state, (sbyte)state.Turn.PlayerIndex).Single();
            roll.GetOutcomeVariants(state, (sbyte)state.Turn.PlayerIndex)
                .Single(variant => ((RollAction)variant.Action).History!.RollResult.Total == total).Action.Apply(state);
        }

        /// <summary>
        /// A vertex with three edges and three vertex-disjoint arms of the given length starting from it.
        /// Returns the arms' edges, the arms' end vertices and all vertices the star touches.
        /// </summary>
        private static (List<int[]> arms, List<int> armEnds, HashSet<int> used) FindStar(Board board, int armLength)
        {
            foreach (Intersection center in board.Intersections.Where(i => board.Adjacency.GetEdges(i).Count == 3))
            {
                HashSet<int> used = [center.Index];
                List<int[]> arms = [];
                List<int> armEnds = [];

                foreach (Edge start in board.Adjacency.GetEdges(center).OrderBy(e => e.Index))
                {
                    List<int> arm = [start.Index];
                    var (a, b) = board.Adjacency.GetIntersections(start);
                    Intersection current = a.Index != center.Index ? a : b;
                    if (used.Contains(current.Index)) break;
                    used.Add(current.Index);

                    while (arm.Count < armLength)
                    {
                        Edge? next = board.Adjacency.GetEdges(current).OrderBy(e => e.Index).FirstOrDefault(e =>
                        {
                            var (p, q) = board.Adjacency.GetIntersections(e);
                            return !used.Contains((p.Index != current.Index ? p : q).Index);
                        });
                        if (next == null) break;

                        var (p2, q2) = board.Adjacency.GetIntersections(next);
                        current = p2.Index != current.Index ? p2 : q2;
                        used.Add(current.Index);
                        arm.Add(next.Index);
                    }

                    if (arm.Count < armLength) break;
                    arms.Add([.. arm]);
                    armEnds.Add(current.Index);
                }

                if (arms.Count == 3) return (arms, armEnds, used);
            }

            throw new InvalidOperationException("No star found on the board.");
        }
    }
}
