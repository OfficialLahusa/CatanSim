using Agents;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// What each seat may know (BeliefTracker): only a steal one took no part in, another player's drawn development cards and
    /// the deck are hidden, everything else is public. Checked for every rule per observer, for the edge cases (game end,
    /// edits, replays) and for leaks: views and determinizations must not depend on anything the observer can't see.
    /// </summary>
    public class HiddenInformationTests
    {
        private static readonly int[] Observers = [BeliefTracker.Public, 0, 1, 2, 3];

        #region Helpers
        /// <summary>
        /// A game followed by a tracker per observer, checked after every move
        /// </summary>
        private sealed class Followed
        {
            public GameState Truth { get; }
            public GameState Start { get; }
            public List<Action> Played { get; } = [];
            public Dictionary<int, BeliefTracker> Trackers { get; }

            public Followed(GameState start)
            {
                Start = new GameState(start);
                Truth = new GameState(start);
                Trackers = Observers.ToDictionary(o => o, o => new BeliefTracker(o, start));
            }

            public void Play(Action action)
            {
                GameState before = new GameState(Truth);
                action.Apply(Truth);
                Played.Add(action);
                foreach (BeliefTracker tracker in Trackers.Values) tracker.Observe(before, action, Truth);
            }

            public void Edit(System.Action<GameState> change)
            {
                GameState before = new GameState(Truth);
                change(Truth);
                foreach (BeliefTracker tracker in Trackers.Values) tracker.Observe(before, null, Truth);
            }
        }

        private static GameState NewGame(int seed, bool freeTrade = false)
        {
            TestUtil.Seed(seed);
            GameState state = TestUtil.NewGame();
            state.Settings.FreeTrade = freeTrade;
            return state;
        }

        /// <summary>
        /// Plays a seeded random game until the condition holds for the next move (or the game ends); the move is not played
        /// </summary>
        private static (Followed game, Action next)? PlayUntilNext(int seed, Func<GameState, Action, bool> condition, bool freeTrade = false, int maxActions = 5000)
        {
            Followed game = new Followed(NewGame(seed, freeTrade));
            Agent[] agents = TestUtil.RandomAgents();
            while (!game.Truth.HasEnded && game.Played.Count < maxActions)
            {
                Action action = agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth);
                if (condition(game.Truth, action)) return (game, action);
                game.Play(action);
            }
            return null;
        }

        private static (Followed game, Action next) FindNext(Func<GameState, Action, bool> condition, bool freeTrade = false)
        {
            for (int seed = 1; seed < 400; seed++)
            {
                if (PlayUntilNext(seed, condition, freeTrade) is (Followed, Action) found) return found;
            }
            throw new InvalidOperationException("No random game reached the situation");
        }

        private static int DistinctTypes(CardSet<ResourceCardType> cards) => CardTypeExtensions.ResourceTypes.Count(t => cards.Get(t) > 0);

        private static bool IsSteal(GameState state, Action action) =>
            action is RobberAction { TargetPlayerIndex: sbyte victim } && victim != action.PlayerIndex && state.Players[victim].ResourceCards.Count() > 0;

        private static DevelopmentCardType? PlayedType(Action action) => action switch
        {
            KnightAction => DevelopmentCardType.Knight,
            RoadBuildingAction => DevelopmentCardType.RoadBuilding,
            YearOfPlentyAction => DevelopmentCardType.YearOfPlenty,
            MonopolyAction => DevelopmentCardType.Monopoly,
            _ => null
        };

        /// <summary>
        /// The true state lies within the observer's belief: known cards are held, the counts add up, the cards beyond the
        /// known ones are of candidate types, and the hidden development cards are exactly those the observer can't see
        /// </summary>
        private static void AssertSupports(BeliefTracker tracker, GameState truth)
        {
            for (int player = 0; player < truth.Players.Length; player++)
            {
                CardSet<ResourceCardType> hand = truth.Players[player].ResourceCards;
                HandBelief belief = tracker.Hand(player);
                if (player == tracker.Observer)
                {
                    Assert.Equal(hand, belief.Known);
                    Assert.Equal(0u, belief.Unknown);
                    continue;
                }
                Assert.Equal(hand.Count(), belief.Count);
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                {
                    Assert.True(hand.Get(type) >= belief.Known.Get(type), $"Observer {tracker.Observer} believes player {player} holds more {type} than it does");
                    if (hand.Get(type) > belief.Known.Get(type))
                        Assert.True(belief.CanBe(type), $"Observer {tracker.Observer} rules out {type} for an unknown card of player {player}");
                }
            }

            // The new cards of this turn are among the DevelopmentCards
            CardSet<DevelopmentCardType> hidden = new CardSet<DevelopmentCardType>(truth.DevelopmentBank);
            for (int player = 0; player < truth.Players.Length; player++)
            {
                if (player != tracker.Observer) hidden.Add(truth.Players[player].DevelopmentCards);
            }
            Assert.Equal(hidden, tracker.HiddenDevelopmentCards);
        }

        /// <summary>
        /// The view shows the public parts as they are and hides only what the observer may not know
        /// </summary>
        private static void AssertViewHidesOnlyHidden(GameState view, GameState truth, int observer)
        {
            if (truth.HasEnded)
            {
                Assert.Equal(truth, view);
                return;
            }
            Assert.Equal(truth.Board, view.Board);
            Assert.Equal(truth.Turn, view.Turn);
            Assert.Equal(truth.Settings, view.Settings);
            Assert.Equal(truth.ResourceBank, view.ResourceBank);
            Assert.Equal(truth.DevelopmentBank.Count(), view.DevelopmentBank.Get(DevelopmentCardType.Unknown));
            Assert.Equal(truth.DevelopmentBank.Count(), view.DevelopmentBank.Count());

            for (int player = 0; player < truth.Players.Length; player++)
            {
                PlayerState t = truth.Players[player], v = view.Players[player];
                if (player == observer)
                {
                    Assert.Equal(t, v);
                    continue;
                }
                Assert.Equal(t.ResourceCards.Count(), v.ResourceCards.Count());
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    Assert.True(v.ResourceCards.Get(type) <= t.ResourceCards.Get(type));
                Assert.Equal(t.DevelopmentCards.Count(), v.DevelopmentCards.Get(DevelopmentCardType.Unknown));
                Assert.Equal(t.DevelopmentCards.Count(), v.DevelopmentCards.Count());
                Assert.Equal(t.NewDevelopmentCards.Count(), v.NewDevelopmentCards.Get(DevelopmentCardType.Unknown));
                Assert.Equal(t.NewDevelopmentCards.Count(), v.NewDevelopmentCards.Count());
                Assert.Equal(0, v.VictoryPoints.DevelopmentCardPoints);
                Assert.Equal(t.VictoryPoints.Total - t.VictoryPoints.DevelopmentCardPoints, v.VictoryPoints.Total);
                Assert.Equal(t.PlayedKnights, v.PlayedKnights);
                Assert.Equal(t.LongestRoadLength, v.LongestRoadLength);
                Assert.Equal(t.BuildingStock, v.BuildingStock);
                Assert.Equal(t.PortPrivileges, v.PortPrivileges);
            }
        }

        /// <summary>
        /// A determinization is a full state that agrees with the view and the belief, with the true totals and nobody at the
        /// winning points
        /// </summary>
        private static void AssertDeterminization(GameState full, GameState view, BeliefTracker tracker, GameState truth)
        {
            Assert.Equal(view.Board, full.Board);
            Assert.Equal(view.Turn, full.Turn);
            Assert.Equal(view.ResourceBank, full.ResourceBank);
            Assert.Equal(0u, full.DevelopmentBank.Get(DevelopmentCardType.Unknown));
            Assert.Equal(view.DevelopmentBank.Count(), full.DevelopmentBank.Count());

            CardSet<DevelopmentCardType> hidden = new CardSet<DevelopmentCardType>(full.DevelopmentBank);
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                uint fullTotal = full.ResourceBank.Get(type) + (uint)full.Players.Sum(p => p.ResourceCards.Get(type));
                uint trueTotal = truth.ResourceBank.Get(type) + (uint)truth.Players.Sum(p => p.ResourceCards.Get(type));
                Assert.Equal(trueTotal, fullTotal);
            }

            for (int player = 0; player < full.Players.Length; player++)
            {
                PlayerState f = full.Players[player], v = view.Players[player];
                Assert.Equal(0u, f.ResourceCards.Get(ResourceCardType.Unknown));
                Assert.Equal(0u, f.DevelopmentCards.Get(DevelopmentCardType.Unknown));
                Assert.Equal(0u, f.NewDevelopmentCards.Get(DevelopmentCardType.Unknown));
                if (player == tracker.Observer)
                {
                    Assert.Equal(truth.Players[player], f);
                    continue;
                }
                HandBelief belief = tracker.Hand(player);
                Assert.Equal(v.ResourceCards.Count(), f.ResourceCards.Count());
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                {
                    Assert.True(f.ResourceCards.Get(type) >= belief.Known.Get(type));
                    if (f.ResourceCards.Get(type) > belief.Known.Get(type)) Assert.True(belief.CanBe(type));
                }
                Assert.Equal(v.DevelopmentCards.Count(), f.DevelopmentCards.Count());
                Assert.Equal(v.NewDevelopmentCards.Count(), f.NewDevelopmentCards.Count());
                hidden.Add(f.DevelopmentCards);
                Assert.True(f.DevelopmentCards.Contains(f.NewDevelopmentCards), "The new cards must be among the player's cards");
                Assert.Equal((int)f.DevelopmentCards.Get(DevelopmentCardType.VictoryPoint), f.VictoryPoints.DevelopmentCardPoints);
                Assert.True(full.HasEnded || f.VictoryPoints.Total < full.Settings.VictoryPoints, $"Player {player} would have won already");
            }
            Assert.Equal(tracker.HiddenDevelopmentCards, hidden);
        }
        #endregion

        #region Rules per observer
        [Fact]
        public void AtTheStartEveryHandIsKnownAndOnlyTheDeckIsHidden()
        {
            GameState start = NewGame(3);
            foreach (int observer in Observers)
            {
                BeliefTracker tracker = new BeliefTracker(observer, start);
                Assert.False(tracker.HasHiddenInformation);
                GameState view = tracker.View(start);
                AssertViewHidesOnlyHidden(view, start, observer);
                Assert.Equal(25u, view.DevelopmentBank.Get(DevelopmentCardType.Unknown));
                Assert.Equal(start.DevelopmentBank, tracker.HiddenDevelopmentCards);
                for (int player = 0; player < 4; player++) Assert.Equal(0u, tracker.Hand(player).Unknown);
            }
        }

        [Fact]
        public void PublicMovesKeepEveryHandKnown()
        {
            // Rolls, the starting yields, building, bank and player trades, discards and development cards change the hands in
            // public: until the first steal, every observer knows every hand exactly
            foreach (bool freeTrade in new[] { false, true })
            {
                for (int seed = 1; seed <= 6; seed++)
                {
                    Followed game = new Followed(NewGame(seed, freeTrade));
                    Agent[] agents = TestUtil.RandomAgents();
                    while (!game.Truth.HasEnded && game.Played.Count < 3000)
                    {
                        Action action = agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth);
                        if (IsSteal(game.Truth, action)) break;
                        game.Play(action);
                        foreach (BeliefTracker tracker in game.Trackers.Values)
                        {
                            for (int player = 0; player < 4; player++)
                            {
                                Assert.Equal(0u, tracker.Hand(player).Unknown);
                                Assert.Equal(game.Truth.Players[player].ResourceCards, tracker.Hand(player).Known);
                            }
                        }
                    }
                }
            }
        }

        [Fact]
        public void AStealIsKnownToTheThiefAndTheVictimOnly()
        {
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action)
                && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2);
            RobberAction steal = (RobberAction)next;
            int thief = steal.PlayerIndex, victim = steal.TargetPlayerIndex!.Value;
            game.Play(steal);
            ResourceCardType stolen = steal.History!.StolenCard!.Value;

            foreach (int observer in Observers)
            {
                BeliefTracker tracker = game.Trackers[observer];
                AssertSupports(tracker, game.Truth);
                if (observer == thief || observer == victim)
                {
                    for (int player = 0; player < 4; player++) Assert.Equal(0u, tracker.Hand(player).Unknown);
                }
                else
                {
                    HandBelief thiefBelief = tracker.Hand(thief), victimBelief = tracker.Hand(victim);
                    Assert.True(thiefBelief.Unknown >= 1);
                    Assert.True(thiefBelief.CanBe(stolen));
                    Assert.True(victimBelief.Unknown >= 1, "The victim held two types, so which one is left is unknown");
                    Assert.True(tracker.HasHiddenInformation);
                }
            }
        }

        [Fact]
        public void SpendingACardBeyondTheKnownOnesResolvesAnUnknownCard()
        {
            // A steal of a type the thief had none of: a third party then sees the thief spend that type, which must be the
            // stolen card
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action)
                && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2
                && state.Players[action.PlayerIndex].ResourceCards.Count() == 0);
            RobberAction steal = (RobberAction)next;
            int thief = steal.PlayerIndex, victim = steal.TargetPlayerIndex!.Value;
            game.Play(steal);
            ResourceCardType stolen = steal.History!.StolenCard!.Value;
            int third = Enumerable.Range(0, 4).First(p => p != thief && p != victim);
            Assert.Equal(1u, game.Trackers[third].Hand(thief).Unknown);

            // An edit is public: the thief hands the stolen card back to the bank
            game.Edit(state =>
            {
                state.Players[thief].ResourceCards.Remove(stolen, 1);
                state.ResourceBank.Add(stolen, 1);
            });
            foreach (int observer in Observers)
            {
                Assert.Equal(0u, game.Trackers[observer].Hand(thief).Unknown);
                Assert.Equal(0u, game.Trackers[observer].Hand(thief).Count);
                AssertSupports(game.Trackers[observer], game.Truth);
            }
        }

        [Fact]
        public void AMonopolyRevealsHowManyCardsOfItsTypeEveryoneHeld()
        {
            int checkedMonopolies = 0;
            for (int seed = 1; seed < 400 && checkedMonopolies < 3; seed++)
            {
                if (PlayUntilNext(seed, (state, action) => action is MonopolyAction) is not (Followed game, Action next)) continue;
                MonopolyAction monopoly = (MonopolyAction)next;
                game.Play(monopoly);
                foreach (int observer in Observers)
                {
                    BeliefTracker tracker = game.Trackers[observer];
                    AssertSupports(tracker, game.Truth);
                    for (int player = 0; player < 4; player++)
                    {
                        if (player == monopoly.PlayerIndex) continue;
                        Assert.Equal(0u, tracker.Hand(player).Known.Get(monopoly.ChosenType));
                        if (tracker.Hand(player).Unknown > 0) Assert.False(tracker.Hand(player).CanBe(monopoly.ChosenType));
                    }
                }
                checkedMonopolies++;
            }
            Assert.True(checkedMonopolies > 0, "No random game played a monopoly");
        }

        [Fact]
        public void ABoughtDevelopmentCardIsHiddenFromTheOthersUntilPlayed()
        {
            (Followed game, Action next) = FindNext((state, action) => action is BuyDevelopmentCardAction);
            int buyer = next.PlayerIndex;
            Dictionary<int, CardSet<DevelopmentCardType>> hiddenBefore = Observers.ToDictionary(o => o, o => new CardSet<DevelopmentCardType>(game.Trackers[o].HiddenDevelopmentCards));
            game.Play(next);
            DevelopmentCardType drawn = ((BuyDevelopmentCardAction)next).History!.DrawnType;

            foreach (int observer in Observers)
            {
                BeliefTracker tracker = game.Trackers[observer];
                AssertSupports(tracker, game.Truth);
                GameState view = tracker.View(game.Truth);
                AssertViewHidesOnlyHidden(view, game.Truth, observer);
                if (observer == buyer)
                {
                    CardSet<DevelopmentCardType> expected = new CardSet<DevelopmentCardType>(hiddenBefore[observer]);
                    expected.Remove(drawn, 1);
                    Assert.Equal(expected, tracker.HiddenDevelopmentCards);
                    Assert.True(view.Players[buyer].NewDevelopmentCards.Get(drawn) >= 1);
                }
                else
                {
                    // The card went from the hidden deck to a hidden hand
                    Assert.Equal(hiddenBefore[observer], tracker.HiddenDevelopmentCards);
                    Assert.True(view.Players[buyer].NewDevelopmentCards.Get(DevelopmentCardType.Unknown) >= 1);
                    Assert.True(tracker.HasHiddenInformation);
                }
            }

            // Playing a card shows its type to everyone
            (Followed played, Action play) = FindNext((state, action) => PlayedType(action) != null);
            DevelopmentCardType type = PlayedType(play)!.Value;
            Dictionary<int, CardSet<DevelopmentCardType>> beforePlay = Observers.ToDictionary(o => o, o => new CardSet<DevelopmentCardType>(played.Trackers[o].HiddenDevelopmentCards));
            played.Play(play);
            foreach (int observer in Observers)
            {
                AssertSupports(played.Trackers[observer], played.Truth);
                CardSet<DevelopmentCardType> expected = new CardSet<DevelopmentCardType>(beforePlay[observer]);
                if (observer != play.PlayerIndex) expected.Remove(type, 1);
                Assert.Equal(expected, played.Trackers[observer].HiddenDevelopmentCards);
            }
        }

        [Fact]
        public void VictoryPointCardsAreLeftOutOfTheOthersViews()
        {
            (Followed game, Action next) = FindNext((state, action) => state.Players.Any(p => p.VictoryPoints.DevelopmentCardPoints > 0));
            int holder = Enumerable.Range(0, 4).First(p => game.Truth.Players[p].VictoryPoints.DevelopmentCardPoints > 0);
            foreach (int observer in Observers)
            {
                GameState view = game.Trackers[observer].View(game.Truth);
                AssertViewHidesOnlyHidden(view, game.Truth, observer);
                int expected = observer == holder ? game.Truth.Players[holder].VictoryPoints.Total
                    : game.Truth.Players[holder].VictoryPoints.Total - game.Truth.Players[holder].VictoryPoints.DevelopmentCardPoints;
                Assert.Equal(expected, view.Players[holder].VictoryPoints.Total);
            }
        }

        [Fact]
        public void TheEndOfTheGameRevealsEverything()
        {
            Followed game = new Followed(NewGame(11));
            Agent[] agents = TestUtil.RandomAgents();
            while (!game.Truth.HasEnded) game.Play(agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth));
            foreach (int observer in Observers) Assert.Equal(game.Truth, game.Trackers[observer].View(game.Truth));
        }

        [Fact]
        public void EditsArePublic()
        {
            Followed game = new Followed(NewGame(12));
            game.Edit(state =>
            {
                state.ResourceBank.Remove(ResourceCardType.Ore, 3);
                state.Players[2].ResourceCards.Add(ResourceCardType.Ore, 3);
            });
            foreach (int observer in Observers)
            {
                Assert.Equal(3u, game.Trackers[observer].Hand(2).Known.Get(ResourceCardType.Ore));
                AssertSupports(game.Trackers[observer], game.Truth);
            }
        }

        [Fact]
        public void ReplayingTheMovesGivesTheSameTracker()
        {
            Followed game = new Followed(NewGame(13));
            TestUtil.Seed(13);
            Agent[] agents = TestUtil.RandomAgents();
            while (!game.Truth.HasEnded && game.Played.Count < 1500) game.Play(agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth));
            foreach (int observer in Observers)
            {
                BeliefTracker replayed = BeliefTracker.Replay(observer, game.Start, game.Played);
                Assert.Equal(game.Trackers[observer], replayed);
                Assert.Equal(game.Trackers[observer].GetHashCode(), replayed.GetHashCode());
                Assert.Equal(game.Trackers[observer], new BeliefTracker(game.Trackers[observer]));
            }
        }
        #endregion

        #region Support and determinizations
        [Theory]
        [InlineData(1, false)]
        [InlineData(2, false)]
        [InlineData(3, true)]
        [InlineData(4, true)]
        [InlineData(5, false)]
        public void TheTrueStateIsAlwaysWithinTheBelief(int seed, bool freeTrade)
        {
            Followed game = new Followed(NewGame(seed, freeTrade));
            Agent[] agents = TestUtil.RandomAgents();
            Random random = new Random(seed);
            while (!game.Truth.HasEnded && game.Played.Count < 4000)
            {
                game.Play(agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth));
                foreach (int observer in Observers)
                {
                    BeliefTracker tracker = game.Trackers[observer];
                    AssertSupports(tracker, game.Truth);
                    GameState view = tracker.View(game.Truth);
                    AssertViewHidesOnlyHidden(view, game.Truth, observer);
                    if (game.Played.Count % 7 == 0 && !game.Truth.HasEnded)
                        AssertDeterminization(tracker.Determinize(view, random), view, tracker, game.Truth);
                }
            }
        }

        [Fact]
        public void DeterminizationsDifferWhereSomethingIsHidden()
        {
            (Followed game, Action next) = FindNext((state, action) => state.Players.Count(p => p.DevelopmentCards.Count() > 0) >= 2);
            int observer = Enumerable.Range(0, 4).First(p => game.Truth.Players.Where((_, i) => i != p).Any(o => o.DevelopmentCards.Count() > 0));
            BeliefTracker tracker = game.Trackers[observer];
            Assert.True(tracker.HasHiddenInformation);
            GameState view = tracker.View(game.Truth);
            HashSet<string> distinct = [];
            for (int seed = 0; seed < 40; seed++)
                distinct.Add(Common.Serialization.SaveFileSerializer.Serialize(tracker.Determinize(view, new Random(seed))));
            Assert.True(distinct.Count > 1, "Every determinization dealt the same hidden cards");
        }

        [Fact]
        public void WithoutHiddenInformationTheDeterminizationIsTheTruth()
        {
            GameState start = NewGame(14);
            BeliefTracker tracker = new BeliefTracker(2, start);
            Assert.Equal(start, tracker.Determinize(tracker.View(start), new Random(1)));
        }
        #endregion

        #region Leaks
        /// <summary>
        /// The same position played on with two different hidden outcomes of the next move
        /// </summary>
        private static (Followed a, Followed b) TwoOutcomes(Followed game, Action next, Func<Action, object?> outcome)
        {
            for (int seed = 1; seed < 200; seed++)
            {
                Followed tryA = Replayed(game), tryB = Replayed(game);
                TestUtil.Seed(seed);
                Action moveA = next.Clone();
                tryA.Play(moveA);
                TestUtil.Seed(seed + 1000);
                Action moveB = next.Clone();
                tryB.Play(moveB);
                if (!Equals(outcome(moveA), outcome(moveB))) return (tryA, tryB);
            }
            throw new InvalidOperationException("The move never had another outcome");
        }

        private static Followed Replayed(Followed game)
        {
            Followed copy = new Followed(game.Start);
            foreach (Action action in game.Played) copy.Play(action.Clone());
            return copy;
        }

        private static void AssertSameForOthers(Followed a, Followed b, params int[] insiders)
        {
            foreach (int observer in Observers.Except(insiders))
            {
                Assert.Equal(a.Trackers[observer], b.Trackers[observer]);
                GameState viewA = a.Trackers[observer].View(a.Truth), viewB = b.Trackers[observer].View(b.Truth);
                Assert.Equal(viewA, viewB);
                Assert.Equal(a.Trackers[observer].Determinize(viewA, new Random(9)), b.Trackers[observer].Determinize(viewB, new Random(9)));
            }
        }

        [Fact]
        public void TheStolenCardDoesNotReachTheOthers()
        {
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action)
                && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2);
            (Followed a, Followed b) = TwoOutcomes(game, next, move => ((RobberAction)move).History!.StolenCard);
            AssertSameForOthers(a, b, next.PlayerIndex, ((RobberAction)next).TargetPlayerIndex!.Value);
        }

        [Fact]
        public void TheDrawnDevelopmentCardDoesNotReachTheOthers()
        {
            (Followed game, Action next) = FindNext((state, action) => action is BuyDevelopmentCardAction && DistinctTypesOfDeck(state) >= 2);
            (Followed a, Followed b) = TwoOutcomes(game, next, move => ((BuyDevelopmentCardAction)move).History!.DrawnType);
            AssertSameForOthers(a, b, next.PlayerIndex);
        }

        private static int DistinctTypesOfDeck(GameState state) => CardSet<DevelopmentCardType>.Values.Count(t => state.DevelopmentBank.Get(t) > 0);
        #endregion

        #region Redacted moves
        [Fact]
        public void RedactedMovesHideOnlyTheHiddenOutcome()
        {
            (Followed game, Action next) = FindNext((state, action) => IsSteal(state, action));
            game.Play(next);
            RobberAction steal = (RobberAction)next;
            int thief = steal.PlayerIndex, victim = steal.TargetPlayerIndex!.Value;
            foreach (int observer in Observers)
            {
                RobberAction redacted = (RobberAction)steal.RedactedFor(observer);
                Assert.True(redacted.IsSameDecision(steal));
                Assert.Equal(steal.History!.PrevRobberTileIndex, redacted.History!.PrevRobberTileIndex);
                Assert.Equal(observer == thief || observer == victim ? steal.History.StolenCard : ResourceCardType.Unknown, redacted.History.StolenCard);
            }
            // The original keeps its outcome
            Assert.NotEqual(ResourceCardType.Unknown, steal.History!.StolenCard);

            (Followed bought, Action buy) = FindNext((state, action) => action is BuyDevelopmentCardAction);
            bought.Play(buy);
            BuyDevelopmentCardAction purchase = (BuyDevelopmentCardAction)buy;
            foreach (int observer in Observers)
            {
                BuyDevelopmentCardAction redacted = (BuyDevelopmentCardAction)purchase.RedactedFor(observer);
                Assert.Equal(observer == purchase.PlayerIndex ? purchase.History!.DrawnType : DevelopmentCardType.Unknown, redacted.History!.DrawnType);
            }

            // Every other move is public
            foreach (Action action in game.Played.Where(a => a is not RobberAction and not BuyDevelopmentCardAction))
            {
                Action redacted = action.RedactedFor(1);
                Assert.True(redacted.IsSameDecision(action));
                Assert.NotSame(action, redacted);
            }
        }

        [Fact]
        public void RedactedMovesReplayOnTheViews()
        {
            // A client that follows a seat's view rebuilds its log by replaying the redacted moves from the view of the start:
            // the replay must not fail and must keep the public parts (turn, board, hand sizes) in step with the views
            for (int seed = 1; seed <= 4; seed++)
            {
                foreach (int observer in Observers)
                {
                    // The same seed gives every observer the same game
                    Followed follow = new Followed(NewGame(seed, freeTrade: seed % 2 == 0));
                    Agent[] agents = TestUtil.RandomAgents();
                    GameState replay = follow.Trackers[observer].View(follow.Start);
                    while (!follow.Truth.HasEnded && follow.Played.Count < 2500)
                    {
                        Action action = agents[TestUtil.ActingPlayer(follow.Truth)].Act(follow.Truth);
                        follow.Play(action);
                        action.RedactedFor(observer).Clone().Apply(replay);

                        // A win by hidden victory point cards only shows in the view itself, which reveals everything at the end
                        if (follow.Truth.HasEnded) break;
                        GameState view = follow.Trackers[observer].View(follow.Truth);
                        Assert.Equal(view.Turn, replay.Turn);
                        Assert.Equal(view.Board, replay.Board);
                        for (int player = 0; player < 4; player++)
                        {
                            Assert.Equal(view.Players[player].ResourceCards.Count(), replay.Players[player].ResourceCards.Count());
                            Assert.Equal(view.Players[player].DevelopmentCards.Count(), replay.Players[player].DevelopmentCards.Count());
                            Assert.Equal(view.Players[player].NewDevelopmentCards.Count(), replay.Players[player].NewDevelopmentCards.Count());
                        }
                    }
                }
            }
        }

        [Fact]
        public void UnknownCardsStandInForAnyTypeOnAView()
        {
            CardSet<ResourceCardType> hand = new CardSet<ResourceCardType>();
            hand.Add(ResourceCardType.Lumber, 1);
            hand.Add(ResourceCardType.Unknown, 2);

            // A type beyond the known cards comes out of the unknown ones
            Assert.True(hand.Contains(ResourceCardType.Ore, 2));
            Assert.True(hand.Contains(ResourceCardType.Lumber, 3));
            Assert.False(hand.Contains(ResourceCardType.Lumber, 4));
            hand.Remove(ResourceCardType.Lumber, 2);
            Assert.Equal(0u, hand.Get(ResourceCardType.Lumber));
            Assert.Equal(1u, hand.Get(ResourceCardType.Unknown));

            // An unknown card taken from a hand of known cards is one of them
            CardSet<ResourceCardType> known = new CardSet<ResourceCardType>();
            known.Add(ResourceCardType.Wool, 2);
            Assert.True(known.Contains(ResourceCardType.Unknown, 1));
            known.Remove(ResourceCardType.Unknown, 1);
            Assert.Equal(1u, known.Count());

            // A full hand behaves as before
            CardSet<ResourceCardType> full = new CardSet<ResourceCardType>();
            full.Add(ResourceCardType.Grain, 1);
            Assert.False(full.Contains(ResourceCardType.Ore, 1));
            Assert.Throws<InvalidOperationException>(() => full.Remove(ResourceCardType.Ore, 1));
        }
        #endregion
    }
}
