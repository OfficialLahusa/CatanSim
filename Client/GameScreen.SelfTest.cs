using Agents;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

using Simulation;

namespace Client
{
    /// <summary>
    /// A self test of the UI logic (--selftest GAMES): whole games in which the human seat plays random legal moves through the
    /// same paths as clicks and dialogs, random edits that undo and redo to the same positions, and history navigation; one
    /// step per frame, so all panels are drawn along the way
    /// </summary>
    public partial class GameScreen
    {
        private IEnumerator<bool>? _selfTest;
        public bool SelfTestFinished { get; private set; }

        public void StartSelfTest(int games, int seed)
        {
            _selfTest = SelfTestSteps(games, seed).GetEnumerator();
        }

        private void StepSelfTest()
        {
            if (_selfTest == null) return;
            if (!_selfTest.MoveNext())
            {
                _selfTest = null;
                SelfTestFinished = true;
                Console.WriteLine("Self test passed");
            }
        }

        private static void Check(bool condition, string what)
        {
            if (!condition) throw new InvalidOperationException("Self test failed: " + what);
        }

        private IEnumerable<bool> SelfTestSteps(int games, int seed)
        {
            Random random = new Random(seed);
            Utils.Random.SeedCurrentThread(seed);

            for (int game = 0; game < games; game++)
            {
                // The match's agents play the random seats without a pause; the human seat plays through the UI's paths
                SetMode(InteractionMode.Play);
                Send(new Simulation.NewGame([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Human], false));
                Send(new SetPlayback(new PlaybackState(false, false, 0)));

                int humanMoves = 0, frames = 0, stalled = 0, lastCount = -1;
                while (!State.HasEnded && History.Count < 4000)
                {
                    if (ActingPlayer == 3)
                    {
                        foreach (bool frame in HumanSelfTestMove(random)) yield return frame;
                        humanMoves++;
                    }
                    else
                    {
                        yield return true;
                    }

                    // The agents must keep the game going
                    stalled = History.Count == lastCount ? stalled + 1 : 0;
                    lastCount = History.Count;
                    Check(stalled < 3000, "the game stopped moving");

                    // The other modes' panels now and then
                    if (++frames % 300 == 0)
                    {
                        SetMode(InteractionMode.Spectate);
                        yield return true;
                        SetMode(InteractionMode.Edit);
                        yield return true;
                        SetMode(InteractionMode.Play);
                    }
                }
                int moves = History.Count;
                Console.WriteLine($"Game {game + 1}: {moves} moves, {humanMoves} by the human seat, " + (State.HasEnded ? $"won by {UI.Theme.PlayerName(State.Turn.PlayerIndex)}" : "stopped"));
                yield return true;

                // History: back and forth through the last moves lands on the same position
                SetMode(InteractionMode.Spectate);
                GameState end = new GameState(State);
                int back = Math.Min(60, History.Count);
                for (int i = 0; i < back; i++) UndoAction();
                yield return true;
                for (int i = 0; i < back; i++) RedoAction(false);
                Check(State.Equals(end), "undo and redo of moves changed the position");
                yield return true;

                foreach (bool frame in EditSelfTest(random)) yield return frame;
            }
        }

        /// <summary>
        /// One random legal move of the human seat, taken the way the UI takes it: board moves from the clickable ghosts of
        /// their build mode, the rest through their dialogs
        /// </summary>
        private IEnumerable<bool> HumanSelfTestMove(Random random)
        {
            const int human = 3;
            List<Action> legal = Legal(human);
            Check(legal.Count > 0, "the human seat has no legal move");

            // Now and then an offer to the agents through the offer dialog, or a counteroffer through the open offer's panel
            if (CanOffer(human) && random.NextDouble() < 0.1)
            {
                foreach (bool frame in OfferSelfTest(random, human)) yield return frame;
                yield break;
            }
            if (legal.Any(a => a is RespondToTradeAction) && random.NextDouble() < 0.3)
            {
                foreach (bool frame in CounterSelfTest(random, human)) yield return frame;
                yield break;
            }

            // Ending the turn less often keeps the games going through all kinds of moves
            Action action = legal[random.Next(legal.Count)];
            if (action is EndTurnAction && legal.Count > 1 && random.NextDouble() < 0.6) action = legal[random.Next(legal.Count)];

            switch (action)
            {
                case DiscardAction discard:
                    {
                        _dialog = Dialog.Discard;
                        for (int i = 0; i < 5; i++)
                            _discardSelection[i] = (int)discard.SelectedCards.Get(CardTypeExtensions.ResourceTypes[i]);
                        yield return true;
                        Check(SelectedDiscard(human) is DiscardAction match && match.Equals(discard), "the discard dialog picks another discard");
                        PlayHumanMove(discard);
                        break;
                    }

                case FourToOneTradeAction or ThreeToOneTradeAction or TwoToOneTradeAction:
                    {
                        var trade = TradeOf(action)!.Value;
                        _dialog = Dialog.Trade;
                        _tradeGive = trade.give;
                        _tradeGet = trade.get;
                        yield return true;
                        Action? best = BestTrade(human, trade.give, trade.get);
                        Check(best != null && TradeOf(best)!.Value.ratio <= trade.ratio, "the trade dialog finds no trade at the best rate");
                        PlayHumanMove(best!);
                        break;
                    }

                case YearOfPlentyAction plenty:
                    _dialog = Dialog.YearOfPlenty;
                    _plentyFirst = plenty.FirstChoice;
                    _plentySecond = plenty.SecondChoice;
                    yield return true;
                    Check(SelectedPlenty(human) != null, "the year of plenty dialog finds no move");
                    PlayHumanMove(SelectedPlenty(human)!);
                    break;

                case MonopolyAction:
                    _dialog = Dialog.Monopoly;
                    yield return true;
                    PlayHumanMove(action);
                    break;

                case KnightAction or RoadBuildingAction:
                    _dialog = Dialog.DevelopmentCard;
                    yield return true;
                    PlayHumanMove(action);
                    break;

                default:
                    if (TargetOf(action) is BoardTarget target)
                    {
                        _buildMode = action switch
                        {
                            RoadAction when !PlacingFreeRoads(human) => BuildMode.Road,
                            SettlementAction => BuildMode.Settlement,
                            CityAction => BuildMode.City,
                            _ => BuildMode.None
                        };
                        Check(ClickableMoves(human).Any(m => m.Equals(action)), $"{action.GetType().Name} is not clickable in its build mode");
                        yield return true;

                        // Several victims on one tile go through the victim dialog
                        if (action is RobberAction && ClickableMoves(human).Count(m => TargetOf(m) == target) > 1)
                        {
                            _victimTile = target.Index;
                            _dialog = Dialog.Victim;
                            yield return true;
                        }
                    }
                    PlayHumanMove(action);
                    break;
            }
            yield return true;
        }

        /// <summary>
        /// One card of the hand for one or two of another type, picked in the offer dialog and sent to random recipients
        /// </summary>
        private IEnumerable<bool> OfferSelfTest(Random random, int human)
        {
            OpenOfferDialog(human);
            CardSet<ResourceCardType> hand = State.Players[human].ResourceCards;
            List<int> held = Enumerable.Range(0, 5).Where(i => hand.Get(CardTypeExtensions.ResourceTypes[i]) > 0).ToList();
            int give = held[random.Next(held.Count)];
            int get = (give + random.Next(1, 5)) % 5;
            _offerGive[give] = 1;
            _offerGet[get] = random.Next(1, 3);
            _offerTo = (byte)(random.Next(1, 16) & OfferTradeAction.Everyone(State, human));
            if (_offerTo == 0) _offerTo = OfferTradeAction.Everyone(State, human);
            yield return true;

            OfferTradeAction offer = new OfferTradeAction((sbyte)human, ToCards(_offerGive), ToCards(_offerGet), _offerTo);
            Check(offer.IsValidFor(State), "the offer dialog makes an invalid offer");
            int before = History.Count;
            PlayHumanMove(offer);
            yield return true;
            // The agents may already have answered and closed it
            Check(History.Skip(before).Any(e => e.Action is OfferTradeAction), "the offer was not played");
        }

        /// <summary>
        /// A counteroffer from the open offer's panel: the offer swapped to one card each way where the hand allows it
        /// </summary>
        private IEnumerable<bool> CounterSelfTest(Random random, int human)
        {
            TradeOffer offer = State.Turn.PendingTrade!;
            CardSet<ResourceCardType> hand = State.Players[human].ResourceCards;
            Array.Clear(_counterGive);
            Array.Clear(_counterGet);
            List<int> held = Enumerable.Range(0, 5).Where(i => hand.Get(CardTypeExtensions.ResourceTypes[i]) > 0).ToList();
            if (held.Count == 0)
            {
                PlayHumanMove(new RespondToTradeAction((sbyte)human, false));
                yield break;
            }
            int give = held[random.Next(held.Count)];
            _counterGive[give] = 1;
            _counterGet[(give + random.Next(1, 5)) % 5] = 1;
            _countering = true;
            yield return true;

            CounterTradeAction counter = new CounterTradeAction((sbyte)human, ToCards(_counterGet), ToCards(_counterGive));
            Check(counter.IsValidFor(State), "the counteroffer panel makes an invalid counteroffer");
            _countering = false;
            PlayHumanMove(counter);
            yield return true;
        }

        /// <summary>
        /// Random edits, then all of them undone and redone: the positions before and after must come back exactly
        /// </summary>
        private IEnumerable<bool> EditSelfTest(Random random)
        {
            SetMode(InteractionMode.Edit);
            GameState start = new GameState(State);
            int historyBefore = History.Count;

            for (int i = 0; i < 80; i++)
            {
                _editPlayer = random.Next(PLAYER_COUNT);
                if (random.NextDouble() < 0.25)
                {
                    ResourceCardType type = CardTypeExtensions.ResourceTypes[random.Next(5)];
                    EditResource(_editPlayer, type, random.NextDouble() < 0.6 ? 1 : -1);
                }
                else
                {
                    _editTool = (EditTool)random.Next(5);
                    List<BoardTarget> candidates = EditCandidates().ToList();
                    if (candidates.Count > 0) EditAt(candidates[random.Next(candidates.Count)]);
                }
                if (i % 10 == 0) yield return true;
            }

            GameState end = new GameState(State);
            int edits = History.Count - historyBefore;
            for (int i = 0; i < edits; i++) UndoAction();
            Check(State.Equals(start), "undoing the edits did not restore the position");
            yield return true;
            for (int i = 0; i < edits; i++) RedoAction(false);
            Check(State.Equals(end), "redoing the edits did not restore the edited position");
            Console.WriteLine($"Edits: {edits} edits undone and redone");
            yield return true;
        }
    }
}
