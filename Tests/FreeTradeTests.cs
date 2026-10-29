using Agents;
using Common;
using Common.Actions;
using Common.Serialization;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Trading between players by the rulebook: only the player whose turn it is trades, after rolling, with the other players
    /// (who don't trade among themselves); both sides give at least one card, no like resources, no development cards; the
    /// others may accept, reject or make a counteroffer, and the player picks one partner or none.
    /// </summary>
    public class FreeTradeTests
    {
        private const ResourceCardType Lumber = ResourceCardType.Lumber, Brick = ResourceCardType.Brick, Wool = ResourceCardType.Wool,
            Grain = ResourceCardType.Grain, Ore = ResourceCardType.Ore;

        private static CardSet<ResourceCardType> Cards(params (ResourceCardType type, uint amount)[] cards)
        {
            CardSet<ResourceCardType> set = new CardSet<ResourceCardType>();
            foreach ((ResourceCardType type, uint amount) in cards) set.Add(type, amount);
            return set;
        }

        private static byte To(params int[] players) => (byte)players.Aggregate(0, (mask, p) => mask | 1 << p);

        /// <summary>
        /// A game in the main phase right after a roll, with free trade on and known hands: every player holds two of each
        /// resource. Returns the state and the player on turn.
        /// </summary>
        private static (GameState state, sbyte active) MainPhase(int seed = 1, bool freeTrade = true)
        {
            Utils.Random.SeedCurrentThread(seed);
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            state.Settings.FreeTrade = freeTrade;
            Random random = new Random(seed);
            while (true)
            {
                sbyte acting = (sbyte)Enumerable.Range(0, 4).First(state.CanPlayerAct);
                if (state.Turn.TypeOfRound == TurnState.RoundType.Normal && !state.Turn.MustRoll && !state.Turn.MustDiscard
                    && !state.Turn.MustMoveRobber && state.Players[state.Turn.PlayerIndex].BuildingStock.FreeRoads == 0)
                    break;
                List<Action> legal = LegalActionProvider.GetActionsForState(state, acting);
                // Rolls and the forced moves after them, nothing else in the main phase, so the hands stay as the test sets them
                Action? forced = legal.FirstOrDefault(a => a is RollAction or DiscardAction or RobberAction) ?? (state.Turn.TypeOfRound != TurnState.RoundType.Normal ? legal[random.Next(legal.Count)] : legal.FirstOrDefault(a => a is EndTurnAction));
                (forced ?? legal[random.Next(legal.Count)]).Apply(state);
            }
            foreach (PlayerState player in state.Players)
                player.ResourceCards = Cards((Lumber, 2), (Brick, 2), (Wool, 2), (Grain, 2), (Ore, 2));
            return (state, (sbyte)state.Turn.PlayerIndex);
        }

        private static sbyte Seat(sbyte active, int offset) => (sbyte)((active + offset) % 4);

        private static void Play(GameState state, Action action)
        {
            Assert.True(action.IsValidFor(state), $"{action} should be valid");
            action.Apply(state);
        }

        #region Offers
        [Fact]
        public void OffersNeedTheRuleAndTheMainPhaseAfterTheRoll()
        {
            (GameState off, sbyte a) = MainPhase(freeTrade: false);
            Assert.False(new OfferTradeAction(a, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(a, 1))).IsValidFor(off));

            (GameState state, sbyte active) = MainPhase();
            OfferTradeAction offer = new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1)));
            Assert.True(offer.IsValidFor(state));

            // Not before rolling, nor while the robber has to move or someone discards
            GameState beforeRoll = new GameState(state);
            beforeRoll.Turn.MustRoll = true;
            Assert.False(offer.IsValidFor(beforeRoll));
            GameState robber = new GameState(state);
            robber.Turn.MustMoveRobber = true;
            Assert.False(offer.IsValidFor(robber));
            GameState discard = new GameState(state);
            discard.Turn.AwaitedPlayerDiscards[Seat(active, 2)] = true;
            Assert.False(offer.IsValidFor(discard));
        }

        [Fact]
        public void OnlyThePlayerOnTurnOffers()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte other = Seat(active, 1);
            Assert.False(new OfferTradeAction(other, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 2))).IsValidFor(state));
            Assert.False(new OfferTradeAction(other, Cards((Lumber, 1)), Cards((Ore, 1)), To(active)).IsValidFor(state));
        }

        [Fact]
        public void BothSidesGiveAtLeastOneCard()
        {
            (GameState state, sbyte active) = MainPhase();
            byte to = To(Seat(active, 1));
            Assert.False(new OfferTradeAction(active, Cards(), Cards((Ore, 1)), to).IsValidFor(state));
            Assert.False(new OfferTradeAction(active, Cards((Lumber, 1)), Cards(), to).IsValidFor(state));
            Assert.True(new OfferTradeAction(active, Cards((Lumber, 2)), Cards((Ore, 1)), to).IsValidFor(state));
        }

        [Fact]
        public void LikeResourcesCannotBeTraded()
        {
            (GameState state, sbyte active) = MainPhase();
            byte to = To(Seat(active, 1));
            Assert.False(new OfferTradeAction(active, Cards((Wool, 2)), Cards((Wool, 1)), to).IsValidFor(state));
            Assert.False(new OfferTradeAction(active, Cards((Wool, 1), (Ore, 1)), Cards((Wool, 1), (Grain, 1)), to).IsValidFor(state));
        }

        [Fact]
        public void OnlyResourcesTheOffererHasCanBeOffered()
        {
            (GameState state, sbyte active) = MainPhase();
            byte to = To(Seat(active, 1));
            Assert.False(new OfferTradeAction(active, Cards((Lumber, 3)), Cards((Ore, 1)), to).IsValidFor(state));
            CardSet<ResourceCardType> unknown = Cards((Lumber, 1));
            unknown.Add(ResourceCardType.Unknown, 1);
            Assert.False(new OfferTradeAction(active, unknown, Cards((Ore, 1)), to).IsValidFor(state));
        }

        [Fact]
        public void OffersGoToOtherPlayersOnly()
        {
            (GameState state, sbyte active) = MainPhase();
            Assert.False(new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), 0).IsValidFor(state));
            Assert.False(new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(active)).IsValidFor(state));
            Assert.False(new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), 1 << 6).IsValidFor(state));
            Assert.True(new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(state, active)).IsValidFor(state));
        }

        [Fact]
        public void OffersPerTurnAreLimitedAndCountedPerTurn()
        {
            (GameState state, sbyte active) = MainPhase();
            state.Settings.MaxTradeOffersPerTurn = 2;
            for (int i = 0; i < 2; i++)
            {
                Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))));
                Play(state, new RespondToTradeAction(Seat(active, 1), false));
            }
            Assert.False(new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))).IsValidFor(state));

            // The next turn starts counting again; undoing the end of the turn brings the count back
            EndTurnAction end = new EndTurnAction(active);
            Play(state, end);
            Assert.Equal(0, state.Turn.TradeOffersThisTurn);
            end.Revert(state);
            Assert.Equal(2, state.Turn.TradeOffersThisTurn);
        }

        [Fact]
        public void OffersAreNotListedAsLegalMoves()
        {
            (GameState state, sbyte active) = MainPhase();
            Assert.DoesNotContain(LegalActionProvider.GetActionsForState(state, active), a => a is OfferTradeAction);
        }
        #endregion

        #region Answers
        [Fact]
        public void TheAddressedPlayersAnswerInSeatOrderAfterTheOfferer()
        {
            (GameState state, sbyte active) = MainPhase();
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(state, active)));

            for (int offset = 1; offset <= 3; offset++)
            {
                sbyte responder = Seat(active, offset);
                Assert.Equal(responder, state.Turn.PendingTrade!.NextResponder);
                Assert.Equal([responder], Enumerable.Range(0, 4).Where(state.CanPlayerAct).Select(p => (sbyte)p));
                foreach (sbyte other in Enumerable.Range(0, 4).Select(p => (sbyte)p).Where(p => p != responder))
                    Assert.False(new RespondToTradeAction(other, false).IsValidFor(state));
                Play(state, new RespondToTradeAction(responder, offset != 3));
            }

            // Back to the offerer, who now picks
            Assert.Null(state.Turn.PendingTrade!.NextResponder);
            Assert.True(state.CanPlayerAct(active));
        }

        [Fact]
        public void OnlyTheAddressedPlayersAnswer()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte addressed = Seat(active, 2), skipped = Seat(active, 1);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(addressed)));
            Assert.False(new RespondToTradeAction(skipped, true).IsValidFor(state));
            Assert.True(state.CanPlayerAct(addressed));
            Assert.False(state.CanPlayerAct(skipped));
        }

        [Fact]
        public void AcceptingNeedsTheAskedForCards()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte responder = Seat(active, 1);
            state.Players[responder].ResourceCards = Cards((Lumber, 3));
            Play(state, new OfferTradeAction(active, Cards((Wool, 1)), Cards((Ore, 1)), To(responder)));

            Assert.False(new RespondToTradeAction(responder, true).IsValidFor(state));
            List<Action> legal = LegalActionProvider.GetActionsForState(state, responder);
            Assert.Contains(legal, a => a is RespondToTradeAction { Accept: false });
            Assert.DoesNotContain(legal, a => a is RespondToTradeAction { Accept: true });
        }

        [Fact]
        public void ConfirmingTradesTheCardsBetweenTheTwoPlayers()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte partner = Seat(active, 1);
            CardSet<ResourceCardType> bank = new CardSet<ResourceCardType>(state.ResourceBank);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 2)), Cards((Ore, 1), (Grain, 1)), To(partner)));
            Play(state, new RespondToTradeAction(partner, true));
            Play(state, new ConfirmTradeAction(active, partner));

            Assert.Equal(Cards((Brick, 2), (Wool, 2), (Grain, 3), (Ore, 3)), state.Players[active].ResourceCards);
            Assert.Equal(Cards((Lumber, 4), (Brick, 2), (Wool, 2), (Grain, 1), (Ore, 1)), state.Players[partner].ResourceCards);
            Assert.Equal(bank, state.ResourceBank);
            Assert.Null(state.Turn.PendingTrade);

            // The turn goes on: building and further trades
            Assert.True(state.CanPlayerAct(active));
            Assert.True(new OfferTradeAction(active, Cards((Ore, 1)), Cards((Lumber, 1)), To(partner)).IsValidFor(state));
        }

        [Fact]
        public void WhenEveryoneRejectsTheOfferCloses()
        {
            (GameState state, sbyte active) = MainPhase();
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1), Seat(active, 3))));
            Play(state, new RespondToTradeAction(Seat(active, 1), false));
            Play(state, new RespondToTradeAction(Seat(active, 3), false));
            Assert.Null(state.Turn.PendingTrade);
            Assert.Contains(LegalActionProvider.GetActionsForState(state, active), a => a is EndTurnAction);
        }

        [Fact]
        public void ACounterofferIsTradedOnItsOwnTerms()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte partner = Seat(active, 2);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 2)), To(partner)));

            // Terms from the offerer's side: two lumber for one ore
            CounterTradeAction counter = new CounterTradeAction(partner, Cards((Lumber, 2)), Cards((Ore, 1)));
            Play(state, counter);
            Assert.Equal(TradeResponse.Countered, state.Turn.PendingTrade!.Responses[partner]);
            Play(state, new ConfirmTradeAction(active, partner));

            Assert.Equal(0u, state.Players[active].ResourceCards.Get(Lumber));
            Assert.Equal(3u, state.Players[active].ResourceCards.Get(Ore));
            Assert.Equal(4u, state.Players[partner].ResourceCards.Get(Lumber));
            Assert.Equal(1u, state.Players[partner].ResourceCards.Get(Ore));
        }

        [Fact]
        public void CounteroffersFollowTheSameRules()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte partner = Seat(active, 1);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(partner)));
            Assert.False(new CounterTradeAction(partner, Cards(), Cards((Ore, 1))).IsValidFor(state));
            Assert.False(new CounterTradeAction(partner, Cards((Ore, 1)), Cards((Ore, 1))).IsValidFor(state));
            Assert.False(new CounterTradeAction(partner, Cards((Lumber, 1)), Cards((Ore, 5))).IsValidFor(state));
            Assert.False(new CounterTradeAction(Seat(active, 2), Cards((Lumber, 2)), Cards((Ore, 1))).IsValidFor(state));
            Assert.True(new CounterTradeAction(partner, Cards((Lumber, 2)), Cards((Ore, 1))).IsValidFor(state));
        }

        [Fact]
        public void TheOffererPicksOneOfSeveralPartners()
        {
            (GameState state, sbyte active) = MainPhase();
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(state, active)));
            Play(state, new RespondToTradeAction(Seat(active, 1), true));
            Play(state, new RespondToTradeAction(Seat(active, 2), false));
            Play(state, new RespondToTradeAction(Seat(active, 3), true));

            List<Action> legal = LegalActionProvider.GetActionsForState(state, active);
            Assert.Equal(3, legal.Count);
            Assert.Contains(legal, a => a is ConfirmTradeAction c && c.Partner == Seat(active, 1));
            Assert.Contains(legal, a => a is ConfirmTradeAction c && c.Partner == Seat(active, 3));
            Assert.Contains(legal, a => a is CancelTradeAction);
            Assert.False(new ConfirmTradeAction(active, Seat(active, 2)).IsValidFor(state));

            Play(state, new ConfirmTradeAction(active, Seat(active, 3)));
            Assert.Equal(2u, state.Players[Seat(active, 1)].ResourceCards.Get(Ore));
            Assert.Equal(1u, state.Players[Seat(active, 3)].ResourceCards.Get(Ore));
        }

        [Fact]
        public void CancellingLeavesTheHandsAlone()
        {
            (GameState state, sbyte active) = MainPhase();
            GameState before = new GameState(state);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))));
            Play(state, new RespondToTradeAction(Seat(active, 1), true));
            Play(state, new CancelTradeAction(active));
            Assert.Null(state.Turn.PendingTrade);
            for (int p = 0; p < 4; p++) Assert.Equal(before.Players[p].ResourceCards, state.Players[p].ResourceCards);
        }

        [Fact]
        public void NothingElseHappensWhileAnOfferIsOpen()
        {
            (GameState state, sbyte active) = MainPhase();
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))));

            // While the answers come in, the offerer can't move; afterwards only confirming or cancelling
            Assert.Empty(LegalActionProvider.GetActionsForState(state, active));
            Assert.False(new EndTurnAction(active).IsValidFor(state));
            Assert.False(new FourToOneTradeAction(active, Lumber, Ore).IsValidFor(state) && state.Players[active].ResourceCards.Get(Lumber) >= 4);
            Assert.False(new OfferTradeAction(active, Cards((Wool, 1)), Cards((Ore, 1)), To(Seat(active, 2))).IsValidFor(state));

            Play(state, new RespondToTradeAction(Seat(active, 1), true));
            Assert.All(LegalActionProvider.GetActionsForState(state, active), a => Assert.True(a is ConfirmTradeAction or CancelTradeAction));
            Assert.False(new EndTurnAction(active).IsValidFor(state));
        }

        [Fact]
        public void AConfirmationChecksTheHandsAgain()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte partner = Seat(active, 1);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 2)), To(partner)));
            Play(state, new RespondToTradeAction(partner, true));
            // The partner's hand changed meanwhile (an edit): the trade can't go through any more
            state.Players[partner].ResourceCards.Remove(Ore, 2);
            Assert.False(new ConfirmTradeAction(active, partner).IsValidFor(state));
            Assert.True(new CancelTradeAction(active).IsValidFor(state));
        }
        #endregion

        #region Undo, hashes and save files
        [Fact]
        public void EveryTradeStepCanBeUndone()
        {
            (GameState state, sbyte active) = MainPhase();
            GameState start = new GameState(state);
            List<Action> steps =
            [
                new OfferTradeAction(active, Cards((Lumber, 1), (Brick, 1)), Cards((Ore, 1)), OfferTradeAction.Everyone(state, active)),
                new RespondToTradeAction(Seat(active, 1), true),
                new CounterTradeAction(Seat(active, 2), Cards((Lumber, 2)), Cards((Ore, 1))),
                new RespondToTradeAction(Seat(active, 3), false),
                new ConfirmTradeAction(active, Seat(active, 2)),
                new OfferTradeAction(active, Cards((Wool, 1)), Cards((Grain, 1)), To(Seat(active, 1))),
                new RespondToTradeAction(Seat(active, 1), false),
            ];
            List<GameState> states = [new GameState(state)];
            foreach (Action step in steps)
            {
                Play(state, step);
                states.Add(new GameState(state));
            }
            for (int i = steps.Count - 1; i >= 0; i--)
            {
                steps[i].Revert(state);
                Assert.Equal(states[i], state);
                Assert.Equal(states[i].GetHashCode(), state.GetHashCode());
            }
            Assert.Equal(start, state);
        }

        [Fact]
        public void APendingOfferIsPartOfTheState()
        {
            (GameState state, sbyte active) = MainPhase();
            GameState other = new GameState(state);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))));
            Assert.NotEqual(other, state);
            GameState copy = new GameState(state);
            Assert.Equal(state, copy);
            Assert.Equal(state.GetHashCode(), copy.GetHashCode());

            // The copy is deep: answering in one doesn't change the other
            new RespondToTradeAction(Seat(active, 1), true).Apply(copy);
            Assert.Equal(TradeResponse.Pending, state.Turn.PendingTrade!.Responses[Seat(active, 1)]);
        }

        [Fact]
        public void TradesRoundTripThroughSaveFiles()
        {
            (GameState state, sbyte active) = MainPhase();
            List<Action> played =
            [
                new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 2)), OfferTradeAction.Everyone(state, active)),
                new RespondToTradeAction(Seat(active, 1), true),
                new CounterTradeAction(Seat(active, 2), Cards((Lumber, 2)), Cards((Ore, 1))),
            ];
            foreach (Action action in played) Play(state, action);

            GameState loaded = SaveFileSerializer.Deserialize<GameState>(SaveFileSerializer.Serialize(state))!;
            loaded.Board.Adjacency.Attach(loaded.Board);
            Assert.Equal(state, loaded);
            Assert.True(loaded.Settings.FreeTrade);

            List<Action> actions = SaveFileSerializer.Deserialize<List<Action>>(SaveFileSerializer.Serialize(played))!;
            for (int i = 0; i < played.Count; i++) Assert.True(played[i].IsSameDecision(actions[i]), $"{played[i]} came back as {actions[i]}");
        }

        [Fact]
        public void SaveFilesWithoutTradesStayAsTheyWere()
        {
            (GameState state, _) = MainPhase(freeTrade: false);
            string yaml = SaveFileSerializer.Serialize(state);
            Assert.DoesNotContain("FreeTrade", yaml);
            Assert.DoesNotContain("PendingTrade", yaml);
            Assert.DoesNotContain("TradeOffersThisTurn", yaml);
        }
        #endregion

        /// <summary>
        /// Random terms from a hand: one or two cards of the hand for one or two cards of other types, or null
        /// </summary>
        private static (CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)? RandomTerms(CardSet<ResourceCardType> hand, Random random)
        {
            List<ResourceCardType> held = CardTypeExtensions.ResourceTypes.Where(t => hand.Get(t) > 0).ToList();
            if (held.Count == 0) return null;
            CardSet<ResourceCardType> give = new(), get = new();
            ResourceCardType given = held[random.Next(held.Count)];
            give.Add(given, (uint)Math.Min(hand.Get(given), (uint)random.Next(1, 3)));
            List<ResourceCardType> others = CardTypeExtensions.ResourceTypes.Where(t => t != given).ToList();
            get.Add(others[random.Next(others.Count)], (uint)random.Next(1, 3));
            return (give, get);
        }

        [Fact]
        public void RandomGamesWithTradesStayConsistent()
        {
            Random random = new Random(5);
            int offers = 0, trades = 0, counters = 0;
            for (int game = 0; game < 30; game++)
            {
                Utils.Random.SeedCurrentThread(100 + game);
                GameState state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
                state.Settings.FreeTrade = true;
                for (int move = 0; move < 3000 && !state.HasEnded; move++)
                {
                    sbyte acting = (sbyte)Enumerable.Range(0, 4).First(state.CanPlayerAct);
                    List<Action> legal = LegalActionProvider.GetActionsForState(state, acting);
                    Action action = legal[random.Next(legal.Count)];

                    // Now and then an offer on turn, or a counteroffer when answering
                    if (OfferTradeAction.IsTurnValid(state, acting) && random.NextDouble() < 0.3
                        && RandomTerms(state.Players[acting].ResourceCards, random) is var (give, get))
                        action = new OfferTradeAction(acting, give, get, (byte)(random.Next(1, 16) & OfferTradeAction.Everyone(state, acting)) is byte to && to != 0 ? to : OfferTradeAction.Everyone(state, acting));
                    else if (RespondToTradeAction.IsTurnValid(state.Turn, acting) && random.NextDouble() < 0.3
                        && RandomTerms(state.Players[acting].ResourceCards, random) is var (theirs, mine))
                        action = new CounterTradeAction(acting, mine, theirs);
                    if (!action.IsValidFor(state)) action = legal[random.Next(legal.Count)];

                    GameState before = new GameState(state);
                    Action probe = action.Clone();
                    probe.Apply(state);
                    probe.Revert(state);
                    Assert.Equal(before, state);
                    Assert.Equal(before.GetHashCode(), state.GetHashCode());

                    action.Apply(state);
                    offers += action is OfferTradeAction ? 1 : 0;
                    counters += action is CounterTradeAction ? 1 : 0;
                    trades += action is ConfirmTradeAction ? 1 : 0;

                    // Cards are neither made nor lost by trading
                    foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                        Assert.Equal(19u, state.ResourceBank.Get(type) + (uint)state.Players.Sum(p => p.ResourceCards.Get(type)));
                }
            }
            Assert.True(offers > 100 && counters > 10 && trades > 10, $"{offers} offers, {counters} counteroffers, {trades} trades");
        }

        #region Agents
        public static TheoryData<string> AgentKinds => new() { "random", "builder", "greedy", "simple" };

        private static Agent CreateAgent(string kind, sbyte seat) => kind switch
        {
            "random" => new RandomAgent(seat),
            "builder" => new BuilderAgent(seat),
            "greedy" => new GreedyAgent(seat),
            _ => new SimpleAgent(seat),
        };

        [Theory]
        [MemberData(nameof(AgentKinds))]
        public void AgentsAnswerOffersWithALegalMove(string kind)
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte responder = Seat(active, 1);
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(responder)));
            Action answer = CreateAgent(kind, responder).Act(new GameState(state));
            Assert.True(answer is RespondToTradeAction or CounterTradeAction, $"{kind} answered with {answer}");
            Assert.True(answer.IsValidFor(state));
        }

        [Theory]
        [MemberData(nameof(AgentKinds))]
        public void AgentsOnTurnConfirmOrCancel(string kind)
        {
            (GameState state, sbyte active) = MainPhase();
            Play(state, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 1)), To(Seat(active, 1))));
            Play(state, new RespondToTradeAction(Seat(active, 1), true));
            Action choice = CreateAgent(kind, active).Act(new GameState(state));
            Assert.True(choice is ConfirmTradeAction or CancelTradeAction, $"{kind} chose {choice}");
            Assert.True(choice.IsValidFor(state));
        }

        [Fact]
        public void HeuristicAgentsTakeGoodTradesAndTurnDownBadOnes()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte responder = Seat(active, 1);

            // Three ore for one lumber is good for the responder; the other way round it isn't
            GameState good = new GameState(state);
            Play(good, new OfferTradeAction(active, Cards((Ore, 2), (Grain, 1)), Cards((Lumber, 1)), To(responder)));
            GameState bad = new GameState(state);
            Play(bad, new OfferTradeAction(active, Cards((Lumber, 1)), Cards((Ore, 2), (Grain, 1)), To(responder)));

            Assert.True(TradeDecisions.Decide(good, responder) is RespondToTradeAction { Accept: true });
            Assert.True(TradeDecisions.Decide(bad, responder) is RespondToTradeAction { Accept: false });
        }

        [Fact]
        public void AgentsDoNotHelpAPlayerAboutToWin()
        {
            (GameState state, sbyte active) = MainPhase();
            sbyte responder = Seat(active, 1);
            state.Players[active].VictoryPoints.SettlementPoints = 9;
            Play(state, new OfferTradeAction(active, Cards((Ore, 2), (Grain, 1)), Cards((Lumber, 1)), To(responder)));
            Assert.True(TradeDecisions.Decide(state, responder) is RespondToTradeAction { Accept: false });
        }

        [Fact]
        public void TradeDecisionsAreOnlyForPendingTrades()
        {
            (GameState state, sbyte active) = MainPhase();
            Assert.Null(TradeDecisions.Decide(state, active));
            Assert.Null(TradeDecisions.Decide(state, Seat(active, 1)));
        }
        #endregion
    }
}
