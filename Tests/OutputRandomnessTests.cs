using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Outcome variants of chance actions: completeness, probabilities and agreement with the real draws
    /// </summary>
    public class OutputRandomnessTests
    {
        [Fact]
        public void OutcomeProbabilitiesSumToOneAndVariantsAreValid()
        {
            TestUtil.Seed(70);
            Random sample = new Random(70);
            Dictionary<Type, int> checkedPerType = [];

            for (int game = 0; game < 20; game++)
            {
                TestUtil.PlayRandomGame(TestUtil.NewGame(), beforeApply: (state, _) =>
                {
                    if (sample.NextDouble() > 0.3) return;

                    sbyte playerIdx = TestUtil.ActingPlayer(state);
                    foreach (Action action in TestUtil.AllProvidedActions(state, playerIdx))
                    {
                        if (action is not IOutputRandomnessAction random) continue;

                        List<OutcomeVariant> variants = random.GetOutcomeVariants(state, playerIdx);
                        Assert.NotEmpty(variants);
                        Assert.Equal(1.0, variants.Sum(variant => variant.Probability), 9);
                        Assert.All(variants, variant =>
                        {
                            Assert.True(variant.Probability > 0);
                            Assert.True(variant.Action.IsValidFor(state));
                            Assert.True(variant.Action.IsSameDecision(action));
                        });

                        checkedPerType[action.GetType()] = checkedPerType.GetValueOrDefault(action.GetType()) + 1;
                    }
                });
            }

            Assert.Equal(3, checkedPerType.Count);
        }

        [Fact]
        public void RollOutcomesAreTheElevenTotalsWithDiceProbabilities()
        {
            TestUtil.Seed(71);
            GameState state = TestUtil.EmptyNormalTurn(0, mustRoll: true);
            RollAction roll = (RollAction)RollAction.GetActionsForState(state, 0).Single();
            List<OutcomeVariant> variants = roll.GetOutcomeVariants(state, 0);

            Assert.Equal(Enumerable.Range(2, 11), variants.Select(variant => (int)((RollAction)variant.Action).History!.RollResult.Total));
            foreach ((Action action, double probability) in variants)
            {
                RollResult result = ((RollAction)action).History!.RollResult;
                Assert.InRange(result.First, (byte)1, (byte)6);
                Assert.InRange(result.Second, (byte)1, (byte)6);
                Assert.Equal((6 - Math.Abs(result.Total - 7)) / 36.0, probability, 12);
            }
        }

        [Fact]
        public void EquivalentRollsLeadToTheSameStateExceptForTheDicePair()
        {
            TestUtil.Seed(72);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.MustRoll && s.Turn.TypeOfRound == TurnState.RoundType.Normal && RollAction.GetActionsForState(s, (sbyte)s.Turn.PlayerIndex).Count > 0
                && s.Players.Sum(p => (int)p.ResourceCards.Count()) > 10));
            sbyte roller = (sbyte)state.Turn.PlayerIndex;

            foreach ((Action representative, _) in ((RollAction)RollAction.GetActionsForState(state, roller)[0]).GetOutcomeVariants(state, roller))
            {
                RollResult result = ((RollAction)representative).History!.RollResult;
                GameState expected = new GameState(state);
                representative.Clone().Apply(expected);

                for (byte first = 1; first <= 6; first++)
                {
                    int second = result.Total - first;
                    if (second is < 1 or > 6) continue;

                    RollAction other = new RollAction(roller);
                    ActionTests.SetHistory(other, new RollAction.RollActionHistory(new RollResult { First = first, Second = (byte)second }, state.Turn.LastRoll));
                    GameState actual = new GameState(state);
                    other.Apply(actual);

                    Assert.Equal(result.Total, actual.Turn.LastRoll.Total);
                    actual.Turn.LastRoll = expected.Turn.LastRoll;
                    Assert.True(actual.Equals(expected), $"({first}, {second}) does not match the representative of total {result.Total}");
                }
            }
        }

        [Fact]
        public void DevelopmentCardDrawsAreWeightedByTheBankStock()
        {
            TestUtil.Seed(73);
            GameState state = TestUtil.EmptyNormalTurn(0);
            foreach (ResourceCardType type in new[] { ResourceCardType.Wool, ResourceCardType.Grain, ResourceCardType.Ore })
                TestUtil.GiveCards(state, 0, type, 1);

            BuyDevelopmentCardAction buy = (BuyDevelopmentCardAction)BuyDevelopmentCardAction.GetActionsForState(state, 0).Single();
            List<OutcomeVariant> variants = buy.GetOutcomeVariants(state, 0);
            uint bankSize = state.DevelopmentBank.Count();

            Assert.Equal(5, variants.Count);
            foreach ((Action action, double probability) in variants)
            {
                DevelopmentCardType drawn = ((BuyDevelopmentCardAction)action).History!.DrawnType;
                Assert.Equal((double)state.DevelopmentBank.Get(drawn) / bankSize, probability, 12);
            }

            // The real draw follows the same distribution
            AssertDrawFrequencies(state.DevelopmentBank, variants.ToDictionary(v => ((BuyDevelopmentCardAction)v.Action).History!.DrawnType, v => v.Probability));
        }

        [Fact]
        public void StealsAreWeightedByTheVictimsHand()
        {
            TestUtil.Seed(74);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Turn.MustMoveRobber = true;

            Tile target = state.Board.Map.Values.First(tile => tile.IsLandTile() && !state.Board.HasRobber(tile));
            int targetIdx = Array.IndexOf(state.Board.Map.Values, target);
            Intersection corner = state.Board.Adjacency.GetIntersections(target).First();
            state.Board.SetBuilding(corner.Index, Intersection.BuildingType.Settlement, 1);
            TestUtil.GiveCards(state, 1, ResourceCardType.Lumber, 4);
            TestUtil.GiveCards(state, 1, ResourceCardType.Ore, 1);

            List<OutcomeVariant> variants = new RobberAction(0, targetIdx, 1).GetOutcomeVariants(state, 0);
            Dictionary<ResourceCardType, double> probabilities = variants.ToDictionary(v => ((RobberAction)v.Action).History!.StolenCard!.Value, v => v.Probability);

            Assert.Equal(2, probabilities.Count);
            Assert.Equal(0.8, probabilities[ResourceCardType.Lumber], 12);
            Assert.Equal(0.2, probabilities[ResourceCardType.Ore], 12);
            AssertDrawFrequencies(state.Players[1].ResourceCards, probabilities);

            // A victim without cards yields a single "nothing stolen" outcome
            state.Players[1].ResourceCards = new();
            List<OutcomeVariant> empty = new RobberAction(0, targetIdx, 1).GetOutcomeVariants(state, 0);
            Assert.Single(empty);
            Assert.Null(((RobberAction)empty[0].Action).History!.StolenCard);
            Assert.Equal(1.0, empty[0].Probability, 12);
        }

        [Fact]
        public void RealDiceFollowTheTwoDiceDistribution()
        {
            TestUtil.Seed(75);
            const int Rolls = 72_000;
            int[] counts = new int[13];
            for (int i = 0; i < Rolls; i++)
                counts[RollResult.GetRandom().Total]++;

            for (int total = 2; total <= 12; total++)
                Assert.InRange(counts[total] / (double)Rolls, (6 - Math.Abs(total - 7)) / 36.0 - 0.006, (6 - Math.Abs(total - 7)) / 36.0 + 0.006);
        }

        private static void AssertDrawFrequencies<T>(CardSet<T> cards, Dictionary<T, double> probabilities) where T : Enum
        {
            const int Draws = 40_000;
            Dictionary<T, int> counts = [];
            for (int i = 0; i < Draws; i++)
            {
                T drawn = cards.Draw(consume: false);
                counts[drawn] = counts.GetValueOrDefault(drawn) + 1;
            }

            Assert.Empty(counts.Keys.Except(probabilities.Keys));
            foreach ((T type, double probability) in probabilities)
                Assert.InRange(counts.GetValueOrDefault(type) / (double)Draws, probability - 0.01, probability + 0.01);
        }
    }
}
