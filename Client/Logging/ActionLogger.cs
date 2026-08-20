using Common;
using Common.Actions;
using SFML.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Client.Logging
{
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

        public void Log(Action action, GameState state)
        {
            Color playerColor = ColorPalette.GetPlayerColor(action.PlayerIndex);

            // Action entry
            switch (action)
            {
                case EndTurnAction endTurnAction:
                    _log.WriteLine(new ColoredStrEntry("Ended turn", playerColor));
                    break;

                case RollAction rollAction:
                    {
                        _log.WriteLine(new ColoredStrEntry($"Rolled {rollAction.History!.RollResult.Total} ({rollAction.History!.RollResult.First}+{rollAction.History!.RollResult.Second})", playerColor));

                        RollAction.RollActionHistory outcome = rollAction.History!;

                        // Summarize yields
                        if (!outcome.TriggeredRobber)
                        {
                            uint[,] yieldSummary = outcome.AwardedYields!;

                            for (int playerIdx = 0; playerIdx < yieldSummary.GetLength(0); playerIdx++)
                            {
                                List<(uint, string)> yieldEntries = new List<(uint, string)>();

                                for (int resourceIdx = 0; resourceIdx < yieldSummary.GetLength(1); resourceIdx++)
                                {
                                    uint amount = yieldSummary[playerIdx, resourceIdx];

                                    if (amount == 1)
                                    {
                                        yieldEntries.Add((amount, CardSet<ResourceCardType>.Values[resourceIdx].GetName().ToLower()));
                                    }
                                    else if (amount > 1)
                                    {
                                        yieldEntries.Add((amount, $"{amount} {CardSet<ResourceCardType>.Values[resourceIdx].GetName().ToLower()}"));
                                    }
                                }

                                if (yieldEntries.Count > 0)
                                {
                                    List<string> yieldStrings = yieldEntries.OrderByDescending(x => x.Item1).Select(x => x.Item2).ToList();

                                    _log.WriteLine(new ColoredStrEntry("Received " + string.Join(", ", yieldStrings), ColorPalette.GetPlayerColor(playerIdx)));
                                }
                            }

                            if (outcome.RobbedYields > 0)
                            {
                                _log.WriteLine(new ColoredStrEntry($"The robber blocked {outcome.RobbedYields} yield" + (outcome.RobbedYields > 1 ? "s" : ""), playerColor));
                            }

                            if (outcome.CappedYields > 0)
                            {
                                _log.WriteLine(new ColoredStrEntry($"Bank stock limited {outcome.CappedYields} yield" + (outcome.CappedYields > 1 ? "s" : ""), playerColor));
                            }
                        }

                        break;
                    }
                    

                case DiscardAction discardAction:
                    StringBuilder sb = new StringBuilder();
                    foreach (ResourceCardType resourceType in CardSet<ResourceCardType>.Values)
                    {
                        for (int i = 0; i < discardAction.SelectedCards.Get(resourceType); i++)
                        {
                            sb.Append(resourceType.GetAbbreviation());
                        }
                    }
                    _log.WriteLine(new ColoredStrEntry($"Discarded {discardAction.SelectedCards.Count()} cards ({sb})", playerColor));
                    break;

                case RobberAction robberAction:
                    {
                        _log.WriteLine(new ColoredStrEntry("Moved robber", playerColor));

                        RobberAction.RobberActionHistory outcome = robberAction.History!;
                        
                        if (outcome.StolenCard.HasValue)
                        {
                            // TODO: Filter out unknown information
                            _log.WriteLine(new ColoredStrEntry($"Stole {outcome.StolenCard.Value.GetName().ToLower()} from Player {robberAction.TargetPlayerIndex!}", playerColor));
                        }

                        break;
                    }

                case FirstInitialSettlementAction firstInitialSettlementAction:
                    _log.WriteLine(new ColoredStrEntry("Placed 1st initial settlement", playerColor));
                    break;

                case FirstInitialRoadAction firstInitialRoadAction:
                    _log.WriteLine(new ColoredStrEntry("Placed 1st initial road", playerColor));
                    break;

                case SecondInitialSettlementAction secondInitialSettlementAction:
                    {
                        _log.WriteLine(new ColoredStrEntry("Placed 2nd initial settlement", playerColor));

                        SecondInitialSettlementAction.SecondInitialSettlementActionHistory outcome = secondInitialSettlementAction.History!;

                        // Initial yields
                        if(outcome.InitialYields.Count > 0)
                        {
                            List<KeyValuePair<ResourceCardType, int>> initialYields = outcome.InitialYields.CountBy(x => x).OrderByDescending(x => (x.Value, -CardSet<ResourceCardType>.ToInt(x.Key))).ToList();
                            List<string> initialYieldEntries = [];

                            foreach ((ResourceCardType cardType, int count) in initialYields)
                            {
                                if (count > 1)
                                {
                                    initialYieldEntries.Add($"{count} {cardType.GetName().ToLower()}");
                                }
                                else
                                {
                                    initialYieldEntries.Add(cardType.GetName().ToLower());
                                }
                            }

                            _log.WriteLine(new ColoredStrEntry("Received " + string.Join(", ", initialYieldEntries), playerColor));
                        }

                        break;
                    }

                case SecondInitialRoadAction secondInitialRoadAction:
                    _log.WriteLine(new ColoredStrEntry("Placed 2nd initial road", playerColor));
                    break;

                case RoadAction roadAction:
                    _log.WriteLine(new ColoredStrEntry("Placed road", playerColor));
                    break;

                case SettlementAction settlementAction:
                    _log.WriteLine(new ColoredStrEntry("Placed settlement", playerColor));
                    break;

                case CityAction cityAction:
                    _log.WriteLine(new ColoredStrEntry("Upgraded settlement to city", playerColor));
                    break;

                case BuyDevelopmentCardAction buyDevelopmentCardAction:
                    {
                        _log.WriteLine(new ColoredStrEntry("Bought development card", playerColor));

                        BuyDevelopmentCardAction.BuyDevelopmentCardActionHistory outcome = buyDevelopmentCardAction.History!;

                        // TODO: Filter out unknown information
                        _log.WriteLine(new ColoredStrEntry($"Received {outcome.DrawnType.GetName().ToLower()}", playerColor));
                    }
                    break;

                case KnightAction knightAction:
                    _log.WriteLine(new ColoredStrEntry("Activated knight", playerColor));
                    break;

                case MonopolyAction monopolyAction:
                    {
                        _log.WriteLine(new ColoredStrEntry($"Activated monopoly on {monopolyAction.ChosenType.GetName().ToLower()}", playerColor));

                        MonopolyAction.MonopolyActionHistory outcome = monopolyAction.History!;

                        // Transferred cards
                        foreach ((int playerIdx, uint amount) in outcome.TransferredCards)
                        {
                            _log.WriteLine(new ColoredStrEntry($"Received {amount} {monopolyAction.ChosenType.GetName().ToLower()} from Player {playerIdx}", playerColor));
                        }

                        break;
                    }

                case RoadBuildingAction roadBuildingAction:
                    _log.WriteLine(new ColoredStrEntry("Activated road building", playerColor));
                    break;

                case YearOfPlentyAction yearOfPlentyAction:
                    _log.WriteLine(new ColoredStrEntry("Activated year of plenty", playerColor));
                    if (yearOfPlentyAction.SecondChoice.HasValue)
                    {
                        _log.WriteLine(new ColoredStrEntry($"Received {yearOfPlentyAction.FirstChoice.GetName().ToLower()}, {yearOfPlentyAction.SecondChoice.Value.GetName().ToLower()}", playerColor));
                    }
                    else
                    {
                        _log.WriteLine(new ColoredStrEntry($"Received {yearOfPlentyAction.FirstChoice.GetName().ToLower()}", playerColor));
                    }
                    break;

                case FourToOneTradeAction fourToOneTradeAction:
                    _log.WriteLine(new ColoredStrEntry($"Traded 4 {fourToOneTradeAction.InputType.GetName().ToLower()} for 1 {fourToOneTradeAction.OutputType.GetName().ToLower()}", playerColor));
                    break;

                case ThreeToOneTradeAction threeToOneTradeAction:
                    _log.WriteLine(new ColoredStrEntry($"Traded 3 {threeToOneTradeAction.InputType.GetName().ToLower()} for 1 {threeToOneTradeAction.OutputType.GetName().ToLower()}", playerColor));
                    break;

                case TwoToOneTradeAction twoToOneTradeAction:
                    _log.WriteLine(new ColoredStrEntry($"Traded 2 {twoToOneTradeAction.InputType.GetName().ToLower()} for 1 {twoToOneTradeAction.OutputType.GetName().ToLower()}", playerColor));
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
                _log.WriteLine(new ColoredStrEntry($"Player {state.Turn.PlayerIndex} won", ColorPalette.GetPlayerColor(state.Turn.PlayerIndex)));
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
