using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Invariants that must hold after every action of complete random games
    /// </summary>
    public class GameInvariantTests
    {
        private const int Games = 60;
        private const int RoundLimit = 1500;

        [Fact]
        public void RandomGamesKeepAllInvariants()
        {
            TestUtil.Seed(1);

            for (int game = 0; game < Games; game++)
            {
                GameState state = TestUtil.NewGame();
                uint playedNonKnightDevCards = 0;
                int step = 0;

                TestUtil.PlayRandomGame(state,
                    beforeApply: (_, action) =>
                    {
                        if (action is MonopolyAction or YearOfPlentyAction or RoadBuildingAction)
                            playedNonKnightDevCards++;
                    },
                    afterApply: (s, action) =>
                    {
                        step++;
                        string context = $"game {game}, step {step}, after {action}";

                        AssertBankTotals(s, playedNonKnightDevCards, context);
                        AssertBuildingStock(s, context);
                        AssertVictoryPoints(s, context);
                        AssertLongestRoad(s, context, recompute: step % 10 == 0);
                        AssertLargestArmy(s, context);

                        Assert.True(s.Turn.RoundCounter < RoundLimit, $"{context}: game exceeds {RoundLimit} rounds");
                    });

                // Players can only win on their own turn
                Assert.True(state.HasEnded);
                Assert.True(state.Players[state.Turn.PlayerIndex].VictoryPoints.Total >= state.Settings.VictoryPoints);
            }
        }

        private static void AssertBankTotals(GameState state, uint playedNonKnightDevCards, string context)
        {
            foreach (ResourceCardType type in CardSet<ResourceCardType>.Values)
            {
                if (type == ResourceCardType.Unknown) continue;

                long total = state.ResourceBank.Get(type) + state.Players.Sum(player => (long)player.ResourceCards.Get(type));
                Assert.True(total == 19, $"{context}: {type} total is {total}, expected 19");
            }

            // NewDevelopmentCards only marks which of the DevelopmentCards were bought this turn.
            // Played knights are tracked per player, other played cards are counted from the actions.
            long devTotal = state.DevelopmentBank.Count()
                + state.Players.Sum(player => (long)player.DevelopmentCards.Count() + player.PlayedKnights)
                + playedNonKnightDevCards;
            Assert.True(devTotal == 25, $"{context}: development card total is {devTotal}, expected 25");
        }

        private static void AssertBuildingStock(GameState state, string context)
        {
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                BuildingStock stock = state.Players[playerIdx].BuildingStock;

                Assert.True(TestUtil.CountBuildings(state, playerIdx, Intersection.BuildingType.Settlement) + stock.RemainingSettlements == BuildingStock.MAX_SETTLEMENTS,
                    $"{context}: settlement stock of player {playerIdx} does not match the board");
                Assert.True(TestUtil.CountBuildings(state, playerIdx, Intersection.BuildingType.City) + stock.RemainingCities == BuildingStock.MAX_CITIES,
                    $"{context}: city stock of player {playerIdx} does not match the board");
                // Free roads from Road Building are taken out of the stock until they are placed or returned at the end of the turn
                Assert.True(TestUtil.CountRoads(state, playerIdx) + stock.RemainingRoads + stock.FreeRoads == BuildingStock.MAX_ROADS,
                    $"{context}: road stock of player {playerIdx} does not match the board");
            }
        }

        private static void AssertVictoryPoints(GameState state, string context)
        {
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                PlayerState player = state.Players[playerIdx];
                VictoryPointTracker vp = player.VictoryPoints;

                Assert.True(vp.SettlementPoints == TestUtil.CountBuildings(state, playerIdx, Intersection.BuildingType.Settlement),
                    $"{context}: settlement VPs of player {playerIdx}");
                Assert.True(vp.CityPoints == 2 * TestUtil.CountBuildings(state, playerIdx, Intersection.BuildingType.City),
                    $"{context}: city VPs of player {playerIdx}");
                Assert.True(vp.DevelopmentCardPoints == player.DevelopmentCards.Get(DevelopmentCardType.VictoryPoint),
                    $"{context}: development card VPs of player {playerIdx}");
                Assert.True(player.DevelopmentCards.Contains(player.NewDevelopmentCards),
                    $"{context}: new development cards of player {playerIdx} are not part of the hand");
                Assert.True(vp.LongestRoadPoints is 0 or 2 && vp.LargestArmyPoints is 0 or 2, $"{context}: special card VPs of player {playerIdx}");
            }
        }

        private static void AssertLongestRoad(GameState state, string context, bool recompute)
        {
            uint longest = state.Players.Max(player => player.LongestRoadLength);
            int[] holders = Enumerable.Range(0, state.Players.Length).Where(idx => state.Players[idx].VictoryPoints.LongestRoadPoints > 0).ToArray();
            int leaders = state.Players.Count(player => player.LongestRoadLength == longest);

            Assert.True(holders.Length <= 1, $"{context}: {holders.Length} Longest Road holders");

            if (holders.Length == 1)
            {
                uint holderLength = state.Players[holders[0]].LongestRoadLength;
                Assert.True(holderLength >= 5 && holderLength == longest, $"{context}: holder {holders[0]} has road {holderLength}, longest is {longest}");
            }
            else
            {
                Assert.True(longest < 5 || leaders > 1, $"{context}: sole leader with road {longest} does not hold Longest Road");
            }

            if (recompute)
            {
                // The stored lengths must match a fresh calculation
                for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
                {
                    GameState copy = new GameState(state);
                    copy.CalculateLongestRoad(playerIdx);
                    Assert.True(copy.Players[playerIdx].LongestRoadLength == state.Players[playerIdx].LongestRoadLength,
                        $"{context}: stored road length of player {playerIdx} is stale");
                }
            }
        }

        private static void AssertLargestArmy(GameState state, string context)
        {
            uint largest = state.Players.Max(player => player.PlayedKnights);
            int[] holders = Enumerable.Range(0, state.Players.Length).Where(idx => state.Players[idx].VictoryPoints.LargestArmyPoints > 0).ToArray();

            Assert.True(holders.Length <= 1, $"{context}: {holders.Length} Largest Army holders");

            if (holders.Length == 1)
            {
                uint knights = state.Players[holders[0]].PlayedKnights;
                Assert.True(knights >= 3 && knights == largest, $"{context}: Largest Army holder has {knights} knights, largest army is {largest}");
            }
            else
            {
                // Played knights never decrease, so once someone reaches 3 the card is always held
                Assert.True(largest < 3, $"{context}: nobody holds Largest Army with {largest} knights played");
            }
        }
    }
}
