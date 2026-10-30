using Agents;
using Common;
using Common.Actions;
using static Tests.HiddenInformationTests;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The input of a belief-aware value net (ViewVectorizer): a seat's view in the layout of the StateVectorizer (unknown
    /// cards in the Unknown slots) and after it what the seat's tracker knows beyond the view. It must not depend on
    /// anything the seat can't see, and a full state in the search (a leaf) must be encoded as the seat would see it.
    /// </summary>
    public class ViewVectorizerTests
    {
        private static float[] Encode(GameState truth, BeliefTracker tracker)
        {
            float[] features = new float[ViewVectorizer.FeatureDim];
            ViewVectorizer.VectorizeView(truth, tracker, features);
            return features;
        }

        private static float[] EncodeLeaf(GameState full, int observer)
        {
            float[] features = new float[ViewVectorizer.FeatureDim];
            new ViewVectorizer { Observer = observer }.VectorizeInto(full, features);
            return features;
        }

        private static float[] Base(GameState state)
        {
            float[] features = new float[StateVectorizer.FeatureDim];
            StateVectorizer.Current.VectorizeInto(state, features);
            return features;
        }

        [Fact]
        public void TheViewComesFirstAndTheBeliefsAfterIt()
        {
            Assert.Equal(StateVectorizer.FeatureDim + ViewVectorizer.BeliefFeatureDim, ViewVectorizer.FeatureDim);
            Assert.Equal(ViewVectorizer.FeatureDim, ((IStateVectorizer)new ViewVectorizer()).FeatureDim);

            Followed game = new Followed(NewGame(31));
            Agent[] agents = TestUtil.RandomAgents();
            int checks = 0;
            while (!game.Truth.HasEnded && game.Played.Count < 2000)
            {
                game.Play(agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth));
                if (game.Played.Count % 11 != 0) continue;
                foreach (int observer in new[] { 0, 1, 2, 3 })
                {
                    BeliefTracker tracker = game.Trackers[observer];
                    float[] features = Encode(game.Truth, tracker);
                    Assert.Equal(Base(tracker.View(game.Truth)), features[..StateVectorizer.FeatureDim]);

                    // The belief block: per player and resource type how many unknown cards can be of it, then the hidden
                    // development cards by type
                    int at = StateVectorizer.FeatureDim;
                    for (int player = 0; player < 4; player++)
                        foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                            Assert.Equal(tracker.Hand(player).Room(type), (uint)features[at++]);
                    foreach (DevelopmentCardType type in CardSet<DevelopmentCardType>.Values.Where(t => t != DevelopmentCardType.Unknown))
                        Assert.Equal(tracker.HiddenDevelopmentCards.Get(type), (uint)features[at++]);
                    Assert.Equal(ViewVectorizer.FeatureDim, at);
                    checks++;
                }
            }
            Assert.True(checks > 100);
        }

        [Fact]
        public void TheInputDoesNotDependOnAHiddenOutcome()
        {
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action)
                && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2);
            (Followed a, Followed b) = TwoOutcomes(game, next, move => ((RobberAction)move).History!.StolenCard);
            int insider1 = next.PlayerIndex, insider2 = ((RobberAction)next).TargetPlayerIndex!.Value;
            foreach (int observer in new[] { 0, 1, 2, 3 }.Where(o => o != insider1 && o != insider2))
                Assert.Equal(Encode(a.Truth, a.Trackers[observer]), Encode(b.Truth, b.Trackers[observer]));

            (Followed bought, Action buy) = FindNext((state, action) => action is BuyDevelopmentCardAction && DistinctTypesOfDeck(state) >= 2);
            (Followed c, Followed d) = TwoOutcomes(bought, buy, move => ((BuyDevelopmentCardAction)move).History!.DrawnType);
            foreach (int observer in new[] { 0, 1, 2, 3 }.Where(o => o != buy.PlayerIndex))
                Assert.Equal(Encode(c.Truth, c.Trackers[observer]), Encode(d.Truth, d.Trackers[observer]));
        }

        [Fact]
        public void ALeafIsEncodedAsItsSeatWouldSeeIt()
        {
            // Without unknown resource cards the seat's view of the truth and the leaf encoding of the truth agree
            Followed game = new Followed(NewGame(32));
            Agent[] agents = TestUtil.RandomAgents();
            int checks = 0;
            while (!game.Truth.HasEnded && game.Played.Count < 2000)
            {
                game.Play(agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth));
                foreach (int observer in new[] { 0, 1, 2, 3 })
                {
                    BeliefTracker tracker = game.Trackers[observer];
                    if (Enumerable.Range(0, 4).Any(p => tracker.Hand(p).Unknown > 0)) continue;
                    Assert.Equal(Encode(game.Truth, tracker), EncodeLeaf(game.Truth, observer));
                    checks++;
                }
            }
            Assert.True(checks > 100);
        }

        [Fact]
        public void ALeafDoesNotShowTheOtherPlayersDevelopmentCards()
        {
            // Victory point cards are left out: swapping one would change the points
            static DevelopmentCardType[] Swappable(CardSet<DevelopmentCardType> cards) =>
                [.. CardSet<DevelopmentCardType>.Values.Where(t => t is not (DevelopmentCardType.Unknown or DevelopmentCardType.VictoryPoint) && cards.Get(t) > 0)];
            (Followed game, _) = FindNext((state, action) => Swappable(state.Players[1].DevelopmentCards).Length > 0
                && Swappable(state.DevelopmentBank).Except(Swappable(state.Players[1].DevelopmentCards)).Any());
            GameState full = new GameState(game.Truth);

            // The same position with one of player 1's cards swapped for a card of another type from the bank
            DevelopmentCardType held = Swappable(full.Players[1].DevelopmentCards)[0];
            DevelopmentCardType? other = Swappable(full.DevelopmentBank).First(t => t != held);
            GameState swapped = new GameState(full);
            swapped.Players[1].DevelopmentCards.Remove(held, 1);
            swapped.Players[1].DevelopmentCards.Add(other!.Value, 1);
            if (swapped.Players[1].NewDevelopmentCards.Get(held) > 0)
            {
                swapped.Players[1].NewDevelopmentCards.Remove(held, 1);
                swapped.Players[1].NewDevelopmentCards.Add(other.Value, 1);
            }
            swapped.DevelopmentBank.Remove(other.Value, 1);
            swapped.DevelopmentBank.Add(held, 1);

            foreach (int observer in new[] { 0, 2, 3 }) Assert.Equal(EncodeLeaf(full, observer), EncodeLeaf(swapped, observer));
            Assert.NotEqual(EncodeLeaf(full, 1), EncodeLeaf(swapped, 1));
        }

        [Fact]
        public void LeavesAreEncodedFromSeveralThreadsAtOnce()
        {
            // The search encodes leaves on all cores with one vectorizer
            GameState start = NewGame(33);
            Agent[] agents = TestUtil.RandomAgents();
            List<GameState> positions = [];
            for (int i = 0; i < 400 && !start.HasEnded; i++)
            {
                agents[TestUtil.ActingPlayer(start)].Act(start).Apply(start);
                positions.Add(new GameState(start));
            }
            ViewVectorizer vectorizer = new ViewVectorizer { Observer = 2 };
            float[][] expected = positions.Select(p => EncodeLeaf(p, 2)).ToArray();
            Parallel.For(0, positions.Count * 4, i =>
            {
                float[] features = new float[ViewVectorizer.FeatureDim];
                vectorizer.VectorizeInto(positions[i % positions.Count], features);
                Assert.Equal(expected[i % positions.Count], features);
            });
        }
    }
}
