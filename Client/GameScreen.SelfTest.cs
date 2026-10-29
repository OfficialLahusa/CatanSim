using Agents;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

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
            _botPaceSeconds = float.MaxValue;
            Agent[] bots = Enumerable.Range(0, PLAYER_COUNT).Select(i => (Agent)new RandomAgent((sbyte)i)).ToArray();

            for (int game = 0; game < games; game++)
            {
                for (int seat = 0; seat < PLAYER_COUNT; seat++)
                    _seats[seat] = seat == 3 ? SeatKind.Human : SeatKind.Random;
                for (sbyte seat = 0; seat < PLAYER_COUNT; seat++)
                    _agents[seat] = CreateAgent(_seats[seat], seat);
                SetMode(InteractionMode.Play);
                NewGame();

                int humanMoves = 0, moves = 0;
                while (!_state.HasEnded && moves < 4000)
                {
                    int acting = ActingPlayer;
                    if (acting == 3)
                    {
                        foreach (bool frame in HumanSelfTestMove(random)) yield return frame;
                        humanMoves++;
                    }
                    else
                    {
                        ApplyMove(bots[acting].Act(_state), false);
                        if (moves % 25 == 0) yield return true;
                    }
                    moves++;

                    // The other modes' panels now and then
                    if (moves % 200 == 0)
                    {
                        SetMode(InteractionMode.Spectate);
                        yield return true;
                        SetMode(InteractionMode.Edit);
                        yield return true;
                        SetMode(InteractionMode.Play);
                    }
                }
                Console.WriteLine($"Game {game + 1}: {moves} moves, {humanMoves} by the human seat, " + (_state.HasEnded ? $"won by {UI.Theme.PlayerName(_state.Turn.PlayerIndex)}" : "stopped"));
                yield return true;

                // History: back and forth through the last moves lands on the same position
                SetMode(InteractionMode.Spectate);
                GameState end = new GameState(_state);
                int back = Math.Min(60, _history.Count);
                for (int i = 0; i < back; i++) UndoAction();
                yield return true;
                for (int i = 0; i < back; i++) RedoAction(false);
                Check(_state.Equals(end), "undo and redo of moves changed the position");
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
        /// Random edits, then all of them undone and redone: the positions before and after must come back exactly
        /// </summary>
        private IEnumerable<bool> EditSelfTest(Random random)
        {
            SetMode(InteractionMode.Edit);
            GameState start = new GameState(_state);
            int historyBefore = _history.Count;

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

            GameState end = new GameState(_state);
            int edits = _history.Count - historyBefore;
            for (int i = 0; i < edits; i++) UndoAction();
            Check(_state.Equals(start), "undoing the edits did not restore the position");
            yield return true;
            for (int i = 0; i < edits; i++) RedoAction(false);
            Check(_state.Equals(end), "redoing the edits did not restore the edited position");
            Console.WriteLine($"Edits: {edits} edits undone and redone");
            yield return true;
        }
    }
}
