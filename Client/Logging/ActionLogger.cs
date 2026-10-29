using Client.UI;
using Common;
using Common.Actions;
using SFML.Graphics;
using Action = Common.Actions.Action;

namespace Client.Logging
{
    /// <summary>
    /// Writes every played move to the event log as a short sentence ("Blue rolled 8 (5+3)"), one section per move so undo can
    /// take it back. Hidden information (stolen cards, drawn development cards) is only named for the players who know it.
    /// </summary>
    public class ActionLogger
    {
        private EventLog _log;
        private int _prevRound;
        private int _prevPlayer;

        public ActionLogger(EventLog log)
        {
            _log = log;
        }

        public void Init()
        {
            _log.WriteLine(new RoundEntry(0));
            _log.PushSection();
            _prevRound = 0;
            _prevPlayer = 0;
        }

        private static (string, Color) Name(int player) => (Theme.PlayerName(player), Theme.PlayerTextColor(player));

        private static (string, Color) Plain(string text) => (text, Theme.TextSoft);

        private static (string, Color) Muted(string text) => (text, Theme.TextMuted);

        private void Line(params (string, Color)[] segments) => _log.WriteLine(new SentenceEntry(segments));

        private static string Card(ResourceCardType type) => type.GetName().ToLower();

        private static string Cards(uint amount, ResourceCardType type) => amount == 1 ? $"1 {Card(type)}" : $"{amount} {Card(type)}";

        public void Log(Action action, GameState state)
        {
            int player = action.PlayerIndex;

            switch (action)
            {
                case EndTurnAction:
                    Line(Name(player), Muted(" ended the turn"));
                    break;

                case RollAction rollAction:
                    {
                        RollAction.RollActionHistory outcome = rollAction.History!;
                        Line(Name(player), Plain($" rolled {outcome.RollResult.Total}"), Muted($" ({outcome.RollResult.First}+{outcome.RollResult.Second})"));

                        if (outcome.TriggeredRobber || outcome.AwardedYields == null) break;

                        // Who received what
                        uint[,] yields = outcome.AwardedYields;
                        for (int playerIdx = 0; playerIdx < yields.GetLength(0); playerIdx++)
                        {
                            List<string> received = [];
                            for (int resourceIdx = 0; resourceIdx < yields.GetLength(1); resourceIdx++)
                            {
                                if (yields[playerIdx, resourceIdx] > 0)
                                    received.Add("+" + Cards(yields[playerIdx, resourceIdx], CardSet<ResourceCardType>.Values[resourceIdx]));
                            }
                            if (received.Count > 0)
                                Line(Plain("   "), Name(playerIdx), Plain(" " + string.Join(", ", received)));
                        }

                        if (outcome.RobbedYields > 0)
                            Line(Muted($"   The robber blocked {outcome.RobbedYields} yield" + (outcome.RobbedYields > 1 ? "s" : "")));
                        if (outcome.CappedYields > 0)
                            Line(Muted($"   The bank ran short by {outcome.CappedYields} card" + (outcome.CappedYields > 1 ? "s" : "")));
                        break;
                    }

                case DiscardAction discardAction:
                    {
                        List<string> cards = [];
                        foreach (ResourceCardType type in CardSet<ResourceCardType>.Values)
                        {
                            uint amount = discardAction.SelectedCards.Get(type);
                            if (amount > 0) cards.Add(Cards(amount, type));
                        }
                        Line(Name(player), Plain($" discarded {discardAction.SelectedCards.Count()} cards"), Muted($" ({string.Join(", ", cards)})"));
                        break;
                    }

                case RobberAction robberAction:
                    {
                        Tile tile = state.Board.Map.Values[robberAction.TargetTileIndex];
                        string where = tile.HasYield() ? $" to {tile.Number} ({tile.Type.ToString().ToLower()})" : $" to the {tile.Type.ToString().ToLower()}";
                        Line(Name(player), Plain(" moved the robber" + where));

                        RobberAction.RobberActionHistory outcome = robberAction.History!;
                        if (outcome.StolenCard.HasValue && robberAction.TargetPlayerIndex is sbyte victim)
                        {
                            _log.WriteLine(new SentenceEntry(
                                [Name(player), Plain($" stole a {Card(outcome.StolenCard.Value)} from "), Name(victim)],
                                [Name(player), Plain(" stole a card from "), Name(victim)],
                                player, victim));
                        }
                        break;
                    }

                case FirstInitialSettlementAction:
                case SecondInitialSettlementAction:
                    {
                        Line(Name(player), Plain(" placed a settlement"));

                        if (action is SecondInitialSettlementAction second && second.History!.InitialYields.Count > 0)
                        {
                            IEnumerable<string> yields = second.History.InitialYields.CountBy(x => x)
                                .OrderByDescending(x => (x.Value, -CardSet<ResourceCardType>.ToInt(x.Key)))
                                .Select(x => "+" + Cards((uint)x.Value, x.Key));
                            Line(Plain("   "), Name(player), Plain(" " + string.Join(", ", yields)));
                        }
                        break;
                    }

                case FirstInitialRoadAction:
                case SecondInitialRoadAction:
                    Line(Name(player), Plain(" placed a road"));
                    break;

                case RoadAction:
                    Line(Name(player), Plain(" built a road"));
                    break;

                case SettlementAction:
                    Line(Name(player), Plain(" built a settlement"));
                    break;

                case CityAction:
                    Line(Name(player), Plain(" upgraded a settlement to a city"));
                    break;

                case BuyDevelopmentCardAction buyAction:
                    _log.WriteLine(new SentenceEntry(
                        [Name(player), Plain(" bought a development card"), Muted($" ({buyAction.History!.DrawnType.GetName()})")],
                        [Name(player), Plain(" bought a development card")],
                        player));
                    break;

                case KnightAction:
                    Line(Name(player), Plain(" played a knight"));
                    break;

                case MonopolyAction monopolyAction:
                    {
                        Line(Name(player), Plain($" played monopoly on {Card(monopolyAction.ChosenType)}"));
                        foreach ((int playerIdx, uint amount) in monopolyAction.History!.TransferredCards)
                            Line(Plain("   took "), Plain(Cards(amount, monopolyAction.ChosenType)), Plain(" from "), Name(playerIdx));
                        break;
                    }

                case RoadBuildingAction:
                    Line(Name(player), Plain(" played road building"));
                    break;

                case YearOfPlentyAction plentyAction:
                    {
                        string cards = plentyAction.SecondChoice.HasValue ? $"{Card(plentyAction.FirstChoice)}, {Card(plentyAction.SecondChoice.Value)}" : Card(plentyAction.FirstChoice);
                        Line(Name(player), Plain($" played year of plenty: {cards}"));
                        break;
                    }

                case FourToOneTradeAction t:
                    Line(Name(player), Plain($" traded 4 {Card(t.InputType)} for 1 {Card(t.OutputType)}"));
                    break;

                case ThreeToOneTradeAction t:
                    Line(Name(player), Plain($" traded 3 {Card(t.InputType)} for 1 {Card(t.OutputType)}"));
                    break;

                case TwoToOneTradeAction t:
                    Line(Name(player), Plain($" traded 2 {Card(t.InputType)} for 1 {Card(t.OutputType)}"));
                    break;

                // TODO: Changes in Longest Road/Largest Army Leader

                default:
                    throw new InvalidOperationException();
            }

            // Round divider & header
            if (state.Turn.RoundCounter > _prevRound)
            {
                _log.WriteLine(new RoundEntry(state.Turn.RoundCounter));
            }
            // Player turn divider
            else if (state.Turn.PlayerIndex != _prevPlayer)
            {
                _log.WriteLine(new SeparatorEntry());
            }

            // Game conclusion
            if (state.HasEnded)
            {
                _log.WriteLine(new SeparatorEntry());
                Line(Name(state.Turn.PlayerIndex), Plain(" won the game"));
            }

            _log.PushSection();

            UpdateTracking(state);
        }

        public void UpdateTracking(GameState state)
        {
            _prevRound = state.Turn.RoundCounter;
            _prevPlayer = state.Turn.PlayerIndex;
        }
    }
}
