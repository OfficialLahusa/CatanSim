using Common;
using Common.Actions;
using Common.Serialization;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The bitmask-based move generation and board masks agree with brute force over all intersections and edges
    /// </summary>
    public class MoveGenerationTests
    {
        private const int Games = 25;
        private const int ScrambledBoards = 300;

        [Fact]
        public void GeneratorsMatchBruteForceThroughoutRandomGames()
        {
            TestUtil.Seed(170);

            for (int game = 0; game < Games; game++)
            {
                GameState state = TestUtil.NewGame();
                int step = 0;

                List<Action> played = TestUtil.PlayRandomGame(state, afterApply: (s, action) =>
                {
                    step++;
                    string context = $"game {game}, step {step}, after {action}";

                    AssertMasksMatchBoard(s.Board, s.Players.Length, context);
                    AssertSitesMatchReference(s.Board, s.Players.Length, context);
                    AssertGeneratorsMatchBruteForce(s, TestUtil.ActingPlayer(s), context);
                });

                // The masks follow every revert back to the empty board
                for (int i = played.Count - 1; i >= 0; i--)
                {
                    played[i].Revert(state);
                    AssertMasksMatchBoard(state.Board, state.Players.Length, $"game {game}, reverting {played[i]}");
                }

                Assert.Equal(UInt128.Zero, state.Board.BuildingMask);
                Assert.Equal(UInt128.Zero, state.Board.RoadMask);
            }
        }

        [Fact]
        public void GeneratorsMatchBruteForceOnScrambledBoards()
        {
            TestUtil.Seed(171);
            Random random = new(171);

            for (int board = 0; board < ScrambledBoards; board++)
            {
                sbyte playerIdx = (sbyte)random.Next(4);
                GameState state = TestUtil.EmptyNormalTurn(playerIdx);
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                {
                    TestUtil.GiveCards(state, playerIdx, type, (uint)random.Next(6));
                    if (random.Next(4) == 0) state.ResourceBank.Remove(type, state.ResourceBank.Get(type) - (uint)random.Next(3));
                }
                state.Players[playerIdx].PortPrivileges = (PortPrivileges)random.Next(64);
                foreach (DevelopmentCardType card in new[] { DevelopmentCardType.YearOfPlenty, DevelopmentCardType.Monopoly })
                {
                    if (random.Next(2) == 0) continue;
                    state.DevelopmentBank.Remove(card, 1);
                    state.Players[playerIdx].DevelopmentCards.Add(card, 1);
                    if (random.Next(3) == 0) state.Players[playerIdx].NewDevelopmentCards.Add(card, 1);
                }
                state.Turn.MustMoveRobber = random.Next(4) == 0;

                // Overwrites included, so buildings and roads also change owner or disappear again
                int writes = random.Next(1, 60);
                for (int w = 0; w < writes; w++)
                {
                    if (random.Next(2) == 0)
                    {
                        var building = (Intersection.BuildingType)random.Next(3);
                        sbyte owner = building == Intersection.BuildingType.None && random.Next(2) == 0 ? (sbyte)-1 : (sbyte)random.Next(4);
                        state.Board.SetBuilding(random.Next(state.Board.IntersectionCount), building, owner);
                    }
                    else
                    {
                        var road = (Edge.BuildingType)random.Next(2);
                        sbyte owner = road == Edge.BuildingType.None && random.Next(2) == 0 ? (sbyte)-1 : (sbyte)random.Next(4);
                        state.Board.SetRoad(random.Next(state.Board.EdgeCount), road, owner);
                    }
                }

                string context = $"scrambled board {board}";
                AssertMasksMatchBoard(state.Board, state.Players.Length, context);
                AssertSitesMatchReference(state.Board, state.Players.Length, context);
                AssertGeneratorsMatchBruteForce(state, playerIdx, context);

                // Copies keep their own masks
                GameState copy = new GameState(state);
                AssertMasksMatchBoard(copy.Board, copy.Players.Length, context + " (copy)");
                copy.Board.SetRoad(0, Edge.BuildingType.Road, 0);
                copy.Board.SetBuilding(0, Intersection.BuildingType.City, 1);
                AssertMasksMatchBoard(state.Board, state.Players.Length, context + " (original after changing the copy)");
            }
        }

        [Fact]
        public void MasksAreRebuiltWhenLoadingASaveFile()
        {
            TestUtil.Seed(172);
            GameState state = TestUtil.NewGame();
            List<Action> played = TestUtil.PlayRandomGame(state, maxActions: 400);

            SaveFile loaded = SaveFileSerializer.Deserialize(SaveFileSerializer.Serialize(new SaveFile(state, played, [])));

            Assert.Equal(state.Board.BuildingMask, loaded.GameState.Board.BuildingMask);
            Assert.Equal(state.Board.RoadMask, loaded.GameState.Board.RoadMask);
            AssertMasksMatchBoard(loaded.GameState.Board, loaded.GameState.Players.Length, "loaded");
            AssertSitesMatchReference(loaded.GameState.Board, loaded.GameState.Players.Length, "loaded");
        }

        [Fact]
        public void BitMaskEnumeratesSetBitsInAscendingOrder()
        {
            int[] indices = [0, 1, 17, 63, 64, 65, 100, 127];
            UInt128 mask = UInt128.Zero;
            foreach (int idx in indices) mask |= BitMask.Bit(idx);

            List<int> enumerated = [];
            foreach (int idx in BitMask.Of(mask)) enumerated.Add(idx);

            Assert.Equal(indices, enumerated);
            Assert.Equal(UInt128.MaxValue, BitMask.Lowest(128));
            Assert.Equal((UInt128)0b111, BitMask.Lowest(3));
        }

        private static void AssertMasksMatchBoard(Board board, int playerCount, string context)
        {
            for (int i = 0; i < board.IntersectionCount; i++)
            {
                Assert.True(BitMask.Contains(board.BuildingMask, i) == board.HasBuilding(i), $"{context}: building mask at intersection {i}");
                for (int p = 0; p < playerCount; p++)
                    Assert.True(BitMask.Contains(board.GetBuildingMask(p), i) == board.HasBuildingOf(i, p), $"{context}: building mask of player {p} at intersection {i}");
            }

            for (int e = 0; e < board.EdgeCount; e++)
            {
                Assert.True(BitMask.Contains(board.RoadMask, e) == (board.GetRoad(e) != Edge.BuildingType.None), $"{context}: road mask at edge {e}");
                for (int p = 0; p < playerCount; p++)
                    Assert.True(BitMask.Contains(board.GetRoadMask(p), e) == board.HasRoadOf(e, p), $"{context}: road mask of player {p} at edge {e}");
            }
        }

        /// <summary>
        /// Board sites against a reference written directly from the rules with the index API
        /// </summary>
        private static void AssertSitesMatchReference(Board board, int playerCount, string context)
        {
            bool HasOwnRoadAt(int intersectionIdx, int p)
            {
                foreach (byte edgeIdx in board.Adjacency.GetEdgeIndices(intersectionIdx))
                    if (board.HasRoadOf(edgeIdx, p)) return true;
                return false;
            }

            bool SatisfiesDistanceRule(int intersectionIdx)
            {
                if (board.HasBuilding(intersectionIdx)) return false;
                foreach (byte edgeIdx in board.Adjacency.GetEdgeIndices(intersectionIdx))
                {
                    (int top, int bottom) = board.Adjacency.GetIntersectionIndices(edgeIdx);
                    if (board.HasBuilding(top == intersectionIdx ? bottom : top)) return false;
                }
                return true;
            }

            bool CanExtendFrom(int intersectionIdx, int p)
            {
                if (board.HasBuildingOf(intersectionIdx, p)) return true;
                bool opponentBuilding = board.HasBuilding(intersectionIdx) && board.GetBuildingOwner(intersectionIdx) != p;
                return !opponentBuilding && HasOwnRoadAt(intersectionIdx, p);
            }

            UInt128 freeSites = board.GetFreeBuildingSites();
            for (int i = 0; i < board.IntersectionCount; i++)
                Assert.True(BitMask.Contains(freeSites, i) == SatisfiesDistanceRule(i), $"{context}: free building site {i}");

            for (int p = 0; p < playerCount; p++)
            {
                UInt128 settlementSites = board.GetSettlementSites(p);
                for (int i = 0; i < board.IntersectionCount; i++)
                    Assert.True(BitMask.Contains(settlementSites, i) == (SatisfiesDistanceRule(i) && HasOwnRoadAt(i, p)), $"{context}: settlement site {i} of player {p}");

                UInt128 roadSites = board.GetRoadSites(p);
                for (int e = 0; e < board.EdgeCount; e++)
                {
                    (int top, int bottom) = board.Adjacency.GetIntersectionIndices(e);
                    bool expected = board.GetRoad(e) == Edge.BuildingType.None && (CanExtendFrom(top, p) || CanExtendFrom(bottom, p));
                    Assert.True(BitMask.Contains(roadSites, e) == expected, $"{context}: road site {e} of player {p}");
                }
            }
        }

        /// <summary>
        /// The generated actions are exactly the valid ones among all intersections and edges, in index order
        /// </summary>
        private static void AssertGeneratorsMatchBruteForce(GameState state, sbyte playerIdx, string context)
        {
            int intersections = state.Board.IntersectionCount;
            int edges = state.Board.EdgeCount;

            AssertSame(RoadAction.GetActionsForState(state, playerIdx).Cast<RoadAction>().Select(a => a.EdgeIndex),
                Enumerable.Range(0, edges).Where(e => new RoadAction(playerIdx, e).IsValidFor(state)), context, "roads");
            AssertSame(SettlementAction.GetActionsForState(state, playerIdx).Cast<SettlementAction>().Select(a => a.IntersectionIndex),
                Enumerable.Range(0, intersections).Where(i => new SettlementAction(playerIdx, i).IsValidFor(state)), context, "settlements");
            AssertSame(CityAction.GetActionsForState(state, playerIdx).Cast<CityAction>().Select(a => a.IntersectionIndex),
                Enumerable.Range(0, intersections).Where(i => new CityAction(playerIdx, i).IsValidFor(state)), context, "cities");
            AssertSame(FirstInitialSettlementAction.GetActionsForState(state, playerIdx).Cast<FirstInitialSettlementAction>().Select(a => a.IntersectionIndex),
                Enumerable.Range(0, intersections).Where(i => new FirstInitialSettlementAction(playerIdx, i).IsValidFor(state)), context, "first setup settlements");
            AssertSame(SecondInitialSettlementAction.GetActionsForState(state, playerIdx).Cast<SecondInitialSettlementAction>().Select(a => a.IntersectionIndex),
                Enumerable.Range(0, intersections).Where(i => new SecondInitialSettlementAction(playerIdx, i).IsValidFor(state)), context, "second setup settlements");

            // Robber: tile and victim (none = -1) as one number
            AssertSame(RobberAction.GetActionsForState(state, playerIdx).Cast<RobberAction>().Select(a => a.TargetTileIndex * 8 + (a.TargetPlayerIndex ?? -1) + 1),
                from tile in Enumerable.Range(0, state.Board.Map.Length)
                from target in Enumerable.Range(-1, state.Players.Length + 1)
                where new RobberAction(playerIdx, tile, target < 0 ? null : (sbyte)target).IsValidFor(state)
                select tile * 8 + target + 1, context, "robber moves");

            // Trades: input and output as one number
            ResourceCardType[] resources = CardTypeExtensions.ResourceTypes;
            AssertSame(FourToOneTradeAction.GetActionsForState(state, playerIdx).Cast<FourToOneTradeAction>().Select(a => (int)a.InputType * 8 + (int)a.OutputType),
                from input in resources from output in resources where new FourToOneTradeAction(playerIdx, input, output).IsValidFor(state) select (int)input * 8 + (int)output,
                context, "4:1 trades");
            AssertSame(ThreeToOneTradeAction.GetActionsForState(state, playerIdx).Cast<ThreeToOneTradeAction>().Select(a => (int)a.InputType * 8 + (int)a.OutputType),
                from input in resources from output in resources where new ThreeToOneTradeAction(playerIdx, input, output).IsValidFor(state) select (int)input * 8 + (int)output,
                context, "3:1 trades");
            AssertSame(TwoToOneTradeAction.GetActionsForState(state, playerIdx).Cast<TwoToOneTradeAction>().Select(a => (int)a.InputType * 8 + (int)a.OutputType),
                from input in resources from output in resources where new TwoToOneTradeAction(playerIdx, input, output).IsValidFor(state) select (int)input * 8 + (int)output,
                context, "2:1 trades");

            // Development cards: Year of Plenty pairs (second choice null = 7) and Monopoly resources
            IEnumerable<int> plenty = from first in resources from second in resources
                                      where first <= second && new YearOfPlentyAction(playerIdx, first, second).IsValidFor(state)
                                      select (int)first * 8 + (int)second;
            if (state.ResourceBank.Count() == 1)
                plenty = plenty.Concat(resources.Where(r => state.ResourceBank.Get(r) == 1 && new YearOfPlentyAction(playerIdx, r, null).IsValidFor(state)).Select(r => (int)r * 8 + 7));
            AssertSame(YearOfPlentyAction.GetActionsForState(state, playerIdx).Cast<YearOfPlentyAction>().Select(a => (int)a.FirstChoice * 8 + (int)(a.SecondChoice ?? (ResourceCardType)7)),
                plenty, context, "Year of Plenty choices");
            AssertSame(MonopolyAction.GetActionsForState(state, playerIdx).Cast<MonopolyAction>().Select(a => (int)a.ChosenType),
                resources.Where(r => new MonopolyAction(playerIdx, r).IsValidFor(state)).Select(r => (int)r), context, "Monopoly choices");
            AssertSame(EndTurnAction.GetActionsForState(state, playerIdx).Select(_ => 0),
                new EndTurnAction(playerIdx).IsValidFor(state) ? [0] : [], context, "end of turn");
        }

        private static void AssertSame(IEnumerable<int> generated, IEnumerable<int> bruteForce, string context, string what)
        {
            int[] expected = bruteForce.ToArray();
            int[] actual = generated.Select(idx => (int)idx).ToArray();
            Assert.True(expected.SequenceEqual(actual), $"{context}: {what} generated [{string.Join(", ", actual)}], valid [{string.Join(", ", expected)}]");
        }
    }
}
