using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using Common.Actions;
using Microsoft.ML.OnnxRuntime.Tensors;
using static Tests.HiddenInformationTests;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Leak tests of the information set search: whatever the seat can't see must not change what it does. Each test gives
    /// the search or the agent two true positions the seat can't tell apart (worlds drawn from its beliefs, or the same game
    /// played on with another hidden outcome) and expects the same tree and the same move with the same seed. The full
    /// knowledge ablation is the control: it sees the true position, so the same tests must tell its searches apart. The
    /// PIMC agent (fair seats) searches with shared random numbers, so for it the worlds it searches are compared.
    /// </summary>
    public class ISMCTSLeakTests
    {
        /// <summary>
        /// A value net stand-in with deterministic utilities from the features (see ISMCTSTests)
        /// </summary>
        private sealed class FixedEvaluator : ILeafEvaluator
        {
            public bool HasPlacements => false;
            public PolicyHead? Policy => null;

            public Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements, bool withPolicy = false)
            {
                int rows = features.Dimensions[0], width = features.Dimensions[1];
                float[] probabilities = new float[rows * 4];
                ReadOnlySpan<float> buffer = features.Buffer.Span;
                for (int row = 0; row < rows; row++)
                {
                    double sum = 0;
                    for (int f = 0; f < width; f += 3) sum += buffer[row * width + f] * (f % 17);
                    for (int p = 0; p < 4; p++) probabilities[row * 4 + p] = (float)(1 + Math.Abs(Math.Sin(sum * (p + 1))));
                }
                return Task.FromResult(new LeafValues(probabilities, null));
            }
        }

        // sensitive_tiny has large random weights: its values swing with every feature, so a hidden card that reaches a
        // decision through the net almost always changes it
        private static readonly Lazy<Dictionary<string, StateValueNet>> Nets = new(() => new[] { "plain_tiny.onnx", "view_tiny.onnx", "policy_tiny.onnx", "sensitive_tiny.onnx" }
            .ToDictionary(name => name, name => new StateValueNet(TestUtil.FixturePath("ValueNet", name), provider: InferenceProvider.Cpu)));

        #region Positions
        /// <summary>
        /// Positions of random games where the seat decides between several moves and is missing something, with its beliefs
        /// </summary>
        private static List<(GameState truth, BeliefTracker beliefs)> Decisions(int seat, int count, int firstSeed, bool freeTrade = false)
        {
            List<(GameState, BeliefTracker)> found = [];
            for (int seed = firstSeed; found.Count < count && seed < firstSeed + 60; seed++)
            {
                Followed game = new Followed(NewGame(seed, freeTrade));
                Agent[] agents = TestUtil.RandomAgents();
                int sinceLast = 0;
                while (!game.Truth.HasEnded && found.Count < count && game.Played.Count < 3000)
                {
                    sbyte acting = TestUtil.ActingPlayer(game.Truth);
                    sinceLast++;
                    if (acting == seat && sinceLast > 17 && game.Trackers[seat].HasHiddenInformation && MCTSTree.GetValidActions(game.Truth).Count > 1)
                    {
                        sinceLast = 0;
                        found.Add((new GameState(game.Truth), new BeliefTracker(game.Trackers[seat])));
                    }
                    game.Play(agents[acting].Act(game.Truth));
                }
            }
            Assert.True(found.Count == count, $"only {found.Count} positions for seat {seat}");
            return found;
        }

        /// <summary>
        /// Other true positions the seat can't tell apart from the given one: worlds drawn from its beliefs, each with the
        /// same view; those that differ from the truth
        /// </summary>
        private static List<GameState> Alternatives(GameState truth, BeliefTracker beliefs, int count, int seed)
        {
            GameState view = beliefs.View(truth);
            List<GameState> alternatives = [];
            for (int i = 0; i < count; i++)
            {
                GameState alternative = beliefs.Determinize(view, new Random(seed + i));
                Assert.Equal(view, beliefs.View(alternative));
                Assert.Equal(MCTSTree.GetValidActions(truth).Select(a => a.ToString()), MCTSTree.GetValidActions(alternative).Select(a => a.ToString()));
                if (alternative.Turn.PendingTrade is TradeOffer offer)
                {
                    // The open offer stays legal: the offerer holds what it gives, who accepted holds what it would give
                    Assert.True(alternative.Players[offer.Offerer].ResourceCards.Contains(offer.Give), "the offerer lacks the offered cards");
                    for (int player = 0; player < offer.Responses.Length; player++)
                        if (offer.Responses[player] == TradeResponse.Accepted)
                            Assert.True(alternative.Players[player].ResourceCards.Contains(offer.Get), $"player {player} accepted without the cards");
                }
                if (!alternative.Equals(truth)) alternatives.Add(alternative);
            }
            return alternatives;
        }
        #endregion

        #region Comparison
        private static IEnumerable<ISNode> Nodes(ISNode node)
        {
            yield return node;
            foreach (ISNode child in node.Children)
                foreach (ISNode below in Nodes(child)) yield return below;
        }

        /// <summary>
        /// The whole tree as text: every node's move, visits and availability in tree order, and the root's values
        /// </summary>
        private static List<string> Shape(ISMCTSTree tree) => Nodes(tree.RootNode)
            .Select(node => $"{node.Move?.ToString() ?? "root"} {node.Visits} {node.Availability}")
            .Append(string.Join(" ", tree.RootValues()))
            .ToList();

        private static ISMCTSTree Search(GameState truth, BeliefTracker beliefs, bool fullKnowledge = false, bool viewFeatures = false)
        {
            ISMCTSTree tree = new ISMCTSTree(new GameState(truth), beliefs, seed: 11, fullKnowledge: fullKnowledge);
            IStateVectorizer vectorizer = viewFeatures ? new ViewVectorizer { Observer = beliefs.Observer } : new StateVectorizer();
            new MCTSSearch(new FixedEvaluator(), vectorizer, 16, parallelSelection: false).Run(tree, double.PositiveInfinity, 600);
            return tree;
        }

        /// <summary>
        /// The agent of the seat as the Arena builds it, with one-thread selection so its search repeats; its MCTS agent is
        /// given the true position too, so a leak through it would show
        /// </summary>
        private static InformationSetMCTSAgent NewAgent(GameState truth, int seat, string net, bool fullKnowledge = false)
        {
            MCTSAgent search = new MCTSAgent((sbyte)seat, truth, valueNet: Nets.Value[net], iterationLimit: 96, leavesPerBatch: 16, parallelSelection: false);
            return new InformationSetMCTSAgent(search, truth, seed: 4) { FullKnowledge = fullKnowledge };
        }

        private static (Action move, List<string>? tree) Decide(GameState truth, BeliefTracker beliefs, string net, bool fullKnowledge = false)
        {
            InformationSetMCTSAgent agent = NewAgent(truth, beliefs.Observer, net, fullKnowledge);
            agent.ResetTo(truth, beliefs);
            Action move = agent.Act(new GameState(truth));
            return (move, agent.LastTree is ISMCTSTree tree ? Shape(tree) : null);
        }

        private static void AssertSameDecision((Action move, List<string>? tree) a, (Action move, List<string>? tree) b, string context)
        {
            Assert.True(a.move.IsSameDecision(b.move), $"{context}: {a.move} against {b.move}");
            Assert.Equal(a.move.ToString(), b.move.ToString());
            Assert.Equal(a.tree, b.tree);
        }
        #endregion

        #region The tree
        [Theory]
        [InlineData(0, false)]
        [InlineData(1, false)]
        [InlineData(2, true)]
        [InlineData(3, true)]
        public void TheTreeIsTheSameInEveryWorldTheSeatCantTellApart(int seat, bool viewFeatures)
        {
            int compared = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(seat, 24, firstSeed: 100 + seat * 60))
            {
                List<string> shape = Shape(Search(truth, beliefs, viewFeatures: viewFeatures));
                foreach (GameState alternative in Alternatives(truth, beliefs, 4, seed: 50))
                {
                    Assert.Equal(shape, Shape(Search(alternative, beliefs, viewFeatures: viewFeatures)));
                    compared++;
                }
            }
            Assert.True(compared >= 60, $"only {compared} positions had a different world");
        }

        [Fact]
        public void TheTreeTellsTheWorldsApartWithFullKnowledge()
        {
            // The control: a search that sees the true position must differ in some of the same comparisons
            int differing = 0;
            foreach ((GameState truth, BeliefTracker beliefs) in Decisions(1, 8, firstSeed: 100 + 60))
            {
                List<string> shape = Shape(Search(truth, beliefs, fullKnowledge: true));
                foreach (GameState alternative in Alternatives(truth, beliefs, 3, seed: 50))
                    if (!shape.SequenceEqual(Shape(Search(alternative, beliefs, fullKnowledge: true)))) differing++;
            }
            Assert.True(differing > 0, "the comparison can't see a search that knows the true position");
        }
        #endregion

        #region The agent
        [Theory]
        [InlineData("plain_tiny.onnx")]
        [InlineData("view_tiny.onnx")]
        [InlineData("policy_tiny.onnx")]
        [InlineData("sensitive_tiny.onnx")]
        public void TheAgentDecidesTheSameInEveryWorldTheSeatCantTellApart(string net)
        {
            int compared = 0, differingWithFullKnowledge = 0;
            for (int seat = 0; seat < 4; seat++)
            {
                foreach ((GameState truth, BeliefTracker beliefs) in Decisions(seat, 6, firstSeed: 400 + seat * 60))
                {
                    var decision = Decide(truth, beliefs, net);
                    var full = Decide(truth, beliefs, net, fullKnowledge: true);
                    foreach (GameState alternative in Alternatives(truth, beliefs, 3, seed: 70))
                    {
                        AssertSameDecision(decision, Decide(alternative, beliefs, net), $"seat {seat}");
                        var fullAlternative = Decide(alternative, beliefs, net, fullKnowledge: true);
                        if (!(full.tree ?? []).SequenceEqual(fullAlternative.tree ?? [])) differingWithFullKnowledge++;
                        compared++;
                    }
                }
            }
            Assert.True(compared >= 40, $"only {compared} positions had a different world");
            Assert.True(differingWithFullKnowledge > 0, "the comparison can't see an agent that knows the true position");
        }

        [Theory]
        [InlineData("plain_tiny.onnx")]
        [InlineData("view_tiny.onnx")]
        [InlineData("sensitive_tiny.onnx")]
        public void TradeAnswersAreTheSameInEveryWorldTheSeatCantTellApart(string net)
        {
            // Every answer to an offer and the offerer's pick are decided outside the tree, on a world drawn from the view
            // (a belief-aware net sees only the view of any position, the plain net sees the whole world)
            int compared = 0, open = 0;
            for (int seed = 700; seed < 800 && open < 60; seed++)
            {
                Followed game = new Followed(NewGame(seed, freeTrade: true));
                Agent[] agents = TestUtil.RandomAgents();
                int lastOffer = 0;
                while (!game.Truth.HasEnded && game.Played.Count < 3000)
                {
                    sbyte acting = TestUtil.ActingPlayer(game.Truth);
                    if (game.Played.Count > 60 && game.Played.Count - lastOffer > 30 && Offer(game.Truth, acting) is OfferTradeAction offer)
                    {
                        lastOffer = game.Played.Count;
                        game.Play(offer);
                        // The answers in seat order, then the offerer's pick
                        while (game.Truth.Turn.PendingTrade is TradeOffer pending && !game.Truth.HasEnded)
                        {
                            sbyte deciding = TestUtil.ActingPlayer(game.Truth);
                            BeliefTracker beliefs = game.Trackers[deciding];
                            var decision = Decide(game.Truth, beliefs, net);
                            List<GameState> alternatives = Alternatives(game.Truth, beliefs, 3, seed: 90);
                            // A decision with a real choice: an answer that may accept, a pick among partners
                            bool choice = pending.NextResponder == deciding
                                ? new RespondToTradeAction(deciding, true).IsValidFor(game.Truth)
                                : ConfirmTradeAction.GetActionsForState(game.Truth, deciding).Any();
                            if (choice && alternatives.Count > 0) open++;
                            foreach (GameState alternative in alternatives)
                            {
                                AssertSameDecision(decision, Decide(alternative, beliefs, net), $"trade, seat {deciding}");
                                compared++;
                            }
                            game.Play(decision.move);
                        }
                    }
                    else game.Play(agents[acting].Act(game.Truth));
                }
            }
            Assert.True(open >= 60, $"only {open} trade decisions with a choice had a different world ({compared} compared)");
        }

        [Fact]
        public void ThePimcAgentSearchesTheSameWorldsInEveryWorldTheSeatCantTellApart()
        {
            // PIMC decides only by searching its worlds (and answers trades on them), so the worlds must depend on the view only
            int compared = 0;
            for (int seat = 0; seat < 4; seat++)
            {
                foreach ((GameState truth, BeliefTracker beliefs) in Decisions(seat, 5, firstSeed: 1000 + seat * 60))
                {
                    DeterminizedMCTSAgent Pimc(GameState position)
                    {
                        MCTSAgent search = new MCTSAgent((sbyte)seat, position, valueNet: Nets.Value["plain_tiny.onnx"], iterationLimit: 32, leavesPerBatch: 16);
                        DeterminizedMCTSAgent agent = new DeterminizedMCTSAgent(search, position, determinizations: 4, seed: 6);
                        agent.ResetTo(position, beliefs);
                        return agent;
                    }
                    IReadOnlyList<GameState> worlds = Pimc(truth).Determinize(truth);
                    foreach (GameState alternative in Alternatives(truth, beliefs, 3, seed: 30))
                    {
                        Assert.Equal(worlds, Pimc(alternative).Determinize(alternative));
                        compared++;
                    }
                }
            }
            Assert.True(compared >= 30, $"only {compared} positions had a different world");
        }

        /// <summary>
        /// An offer of one of the player's cards for one of another type to everyone, if the player may offer now
        /// </summary>
        private static OfferTradeAction? Offer(GameState state, sbyte player)
        {
            CardSet<ResourceCardType> hand = state.Players[player].ResourceCards;
            // What the next player holds, so the first answer may accept
            CardSet<ResourceCardType> next = state.Players[(player + 1) % state.Players.Length].ResourceCards;
            foreach (ResourceCardType give in CardTypeExtensions.ResourceTypes.Where(type => hand.Get(type) > 0))
            {
                ResourceCardType get = CardTypeExtensions.ResourceTypes.Where(type => type != give).OrderByDescending(next.Get).First();
                CardSet<ResourceCardType> giveSet = new(), getSet = new();
                giveSet.Add(give, 1);
                getSet.Add(get, 1);
                OfferTradeAction offer = new OfferTradeAction(player, giveSet, getSet, OfferTradeAction.Everyone(state, player));
                if (offer.IsValidFor(state)) return offer;
            }
            return null;
        }
        #endregion

        #region Whole games
        [Fact]
        public void AnAgentFollowingTheGameKnowsExactlyWhatItsSeatKnows()
        {
            // The agent learns the game from the true moves (with their hidden outcomes); its beliefs must be the seat's
            for (int seed = 800; seed < 820; seed++)
            {
                Followed game = new Followed(NewGame(seed, freeTrade: seed % 2 == 0));
                InformationSetMCTSAgent[] followers = Enumerable.Range(0, 4).Select(seat => NewAgent(game.Truth, seat, "plain_tiny.onnx")).ToArray();
                DeterminizedMCTSAgent[] pimc = Enumerable.Range(0, 4)
                    .Select(seat => new DeterminizedMCTSAgent(new MCTSAgent((sbyte)seat, game.Truth, valueNet: Nets.Value["plain_tiny.onnx"]), game.Truth)).ToArray();
                Agent[] agents = TestUtil.RandomAgents();
                while (!game.Truth.HasEnded && game.Played.Count < 3000)
                {
                    Action action = agents[TestUtil.ActingPlayer(game.Truth)].Act(game.Truth);
                    game.Play(action);
                    foreach (InformationSetMCTSAgent follower in followers)
                    {
                        follower.HandleAction(action);
                        Assert.Equal(game.Trackers[follower.PlayerIndex], follower.Beliefs);
                    }
                    foreach (DeterminizedMCTSAgent follower in pimc)
                    {
                        follower.HandleAction(action);
                        Assert.Equal(game.Trackers[follower.PlayerIndex], follower.Beliefs);
                    }
                }
            }
        }

        [Theory]
        [InlineData("steal")]
        [InlineData("draw")]
        public void AHiddenOutcomeDoesNotChangeTheLaterDecisionsOfTheOthers(string kind)
        {
            // The same game played on with two different hidden outcomes and then the same public moves: the agents of the
            // seats that didn't see the outcome followed both games move by move and must decide alike when it is their turn
            int compared = 0;
            for (int seed = 900; seed < 1300 && compared < 20; seed++)
            {
                Func<GameState, Action, bool> isEvent = kind == "steal"
                    ? (state, action) => IsSteal(state, action) && DistinctTypes(state.Players[((RobberAction)action).TargetPlayerIndex!.Value].ResourceCards) >= 2
                    : (state, action) => action is BuyDevelopmentCardAction && DistinctTypesOfDeck(state) >= 2;
                if (PlayUntilNext(seed, isEvent) is not (Followed game, Action next)) continue;

                (Followed a, Followed b) = TwoOutcomes(game, next, move => move is RobberAction robber ? robber.History!.StolenCard : ((BuyDevelopmentCardAction)move).History!.DrawnType);
                HashSet<int> insiders = [next.PlayerIndex];
                if (next is RobberAction { TargetPlayerIndex: sbyte victim }) insiders.Add(victim);

                // The same public moves on in both games, until an outsider decides between several moves
                Agent[] agents = TestUtil.RandomAgents();
                for (int step = 0; step < 400 && !a.Truth.HasEnded; step++)
                {
                    sbyte acting = TestUtil.ActingPlayer(a.Truth);
                    if (!insiders.Contains(acting) && MCTSTree.GetValidActions(a.Truth).Count > 1)
                    {
                        CompareOutsider(a, b, acting, "view_tiny.onnx");
                        CompareOutsider(a, b, acting, "plain_tiny.onnx");
                        CompareOutsider(a, b, acting, "sensitive_tiny.onnx");
                        compared++;
                        break;
                    }
                    Action action = agents[acting].Act(a.Truth);
                    a.Play(action);
                    Action replay = action.Clone();
                    if (!replay.IsValidFor(b.Truth)) break;
                    b.Play(replay);
                }
            }
            Assert.True(compared >= 20, $"only {compared} games reached an outsider's decision");
        }

        private static void CompareOutsider(Followed a, Followed b, int seat, string net)
        {
            Assert.Equal(a.Trackers[seat], b.Trackers[seat]);
            Assert.Equal(a.Trackers[seat].View(a.Truth), b.Trackers[seat].View(b.Truth));

            // Agents that followed each game from its start, as in a duel
            (Action move, List<string>? tree) Follow(Followed game)
            {
                InformationSetMCTSAgent agent = NewAgent(game.Start, seat, net);
                foreach (Action played in game.Played) agent.HandleAction(played);
                Assert.Equal(game.Trackers[seat], agent.Beliefs);
                Action move = agent.Act(new GameState(game.Truth));
                return (move, agent.LastTree is ISMCTSTree tree ? Shape(tree) : null);
            }
            AssertSameDecision(Follow(a), Follow(b), $"seat {seat}, {net}");
        }
        #endregion
    }
}
