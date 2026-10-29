using Client.Logging;
using Client.UI;
using Common;
using Common.Actions;
using ImGuiNET;
using SFML.Graphics;
using System.Numerics;
using Action = Common.Actions.Action;

namespace Client
{
    /// <summary>
    /// Trading between players: the offer dialog of the player on turn, and the panel of an open offer, where the addressed
    /// humans accept, reject or counter in turn and the offerer then picks a partner or withdraws. Agents answer by themselves.
    /// </summary>
    public partial class GameScreen
    {
        // The offer being put together, per resource in CardTypeExtensions.ResourceTypes order, and its recipients
        private readonly int[] _offerGive = new int[5], _offerGet = new int[5];
        private byte _offerTo;

        // A counteroffer being put together by the player who answers: what that player gives and gets
        private bool _countering;
        private readonly int[] _counterGive = new int[5], _counterGet = new int[5];

        private bool CanOffer(int player) => OfferTradeAction.IsTurnValid(State, player) && State.Players[player].ResourceCards.Count() > 0;

        private static CardSet<ResourceCardType> ToCards(int[] counts)
        {
            CardSet<ResourceCardType> cards = new CardSet<ResourceCardType>();
            for (int i = 0; i < counts.Length; i++)
                if (counts[i] > 0) cards.Add(CardTypeExtensions.ResourceTypes[i], (uint)counts[i]);
            return cards;
        }

        private static void FromCards(CardSet<ResourceCardType> cards, int[] counts)
        {
            for (int i = 0; i < counts.Length; i++) counts[i] = (int)cards.Get(CardTypeExtensions.ResourceTypes[i]);
        }

        /// <summary>
        /// The trade dialog for the player on turn: with the bank if possible, otherwise the offer to the other players
        /// </summary>
        private void OpenTradeDialog(int player)
        {
            if (HasTrades(player)) _dialog = Dialog.Trade;
            else if (CanOffer(player)) OpenOfferDialog(player);
        }

        /// <summary>
        /// Opens the offer dialog with everyone else addressed (the first time this turn) and nothing chosen yet
        /// </summary>
        private void OpenOfferDialog(int player)
        {
            Array.Clear(_offerGive);
            Array.Clear(_offerGet);
            _offerTo = OfferTradeAction.Everyone(State, player);
            _dialog = Dialog.Offer;
        }

        /// <summary>
        /// One row of cards to pick amounts from: a click adds a card, the minus under it takes one back. Types the other row
        /// uses are disabled, since like resources can't be traded.
        /// </summary>
        private static void DrawCardRow(ImDrawListPtr dl, string id, Vector2 pos, int[] counts, int[] other, Func<int, int> max, Color accent, Color accentText)
        {
            for (int i = 0; i < counts.Length; i++)
            {
                ResourceCardType type = CardTypeExtensions.ResourceTypes[i];
                Vector2 card = pos + new Vector2(i * 108, 0);
                bool enabled = other[i] == 0 && counts[i] < max(i);
                if (CardButton(dl, $"{id}{type}", type, card, new Vector2(64, 96), enabled, counts[i] > 0)) counts[i]++;
                if (counts[i] > 0)
                {
                    Vector2 pill = card + new Vector2(6, 104);
                    Theme.Pill(dl, pill, pill + new Vector2(52, 24), accent, Theme.Bold15, accentText, $"{counts[i]}  -");
                    if (Theme.HitArea($"{id}less{type}", pill, new Vector2(52, 24), out _)) counts[i]--;
                }
            }
        }

        private void DrawOfferDialog()
        {
            if (HumanToAct is not int player || !CanOffer(player)) { _dialog = Dialog.None; return; }

            Vector2 size = new Vector2(600, 520);
            ImDrawListPtr dl = BeginDialog("##offer", size, "Offer a trade", "Both sides give at least one card, and no resource on both. The others answer in turn.", player, out Vector2 min);
            CardSet<ResourceCardType> hand = State.Players[player].ResourceCards;

            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, 84), Theme.TextSoft, "You give");
            DrawCardRow(dl, "give", min + new Vector2(28, 110), _offerGive, _offerGet, i => (int)hand.Get(CardTypeExtensions.ResourceTypes[i]), Theme.PlayerColor(player), Theme.TextOnPlayer(player));
            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, 248), Theme.TextSoft, "You get");
            DrawCardRow(dl, "get", min + new Vector2(28, 274), _offerGet, _offerGive, _ => 9, Theme.PanelActive, Theme.Text);

            // Who it goes to
            Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, 412), Theme.TextSoft, "To");
            float x = 70;
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
            {
                if (seat == player) continue;
                bool on = (_offerTo & 1 << seat) != 0;
                Vector2 chip = min + new Vector2(x, 404);
                Vector2 chipSize = new Vector2(Theme.TextSize(Theme.Bold15, Theme.PlayerName(seat)).X + 34, 32);
                Theme.Pill(dl, chip, chip + chipSize, on ? Theme.PlayerColor(seat) : Theme.Panel, Theme.Bold15, on ? Theme.TextOnPlayer(seat) : Theme.TextMuted,
                    Theme.PlayerName(seat), on ? null : Theme.Border);
                if (Theme.HitArea($"to{seat}", chip, chipSize, out _)) _offerTo ^= (byte)(1 << seat);
                x += chipSize.X + 10;
            }

            OfferTradeAction offer = new OfferTradeAction((sbyte)player, ToCards(_offerGive), ToCards(_offerGet), _offerTo);
            bool valid = offer.IsValidFor(State);
            int left = State.Settings.TradeOfferLimit - State.Turn.TradeOffersThisTurn;
            Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 24, 420), Theme.TextMuted, $"{left} offer{(left == 1 ? "" : "s")} left this turn");

            if (DrawButton(dl, "offerok", min + new Vector2(size.X - 24 - 180, size.Y - 20 - 44), new Vector2(180, 44), "Offer", valid, Theme.Accent, Theme.AccentBorder, Theme.Bold17))
                PlayHumanMove(offer);
            if (HasTrades(player) && DrawButton(dl, "offerbank", min + new Vector2(size.X / 2 - 80, size.Y - 20 - 44), new Vector2(160, 44), "Bank trade", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                _dialog = Dialog.Trade;
            if (DrawButton(dl, "offercancel", min + new Vector2(24, size.Y - 20 - 44), new Vector2(120, 44), "Cancel", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                ClearHumanInput();
            ImGui.End();
        }

        /// <summary>
        /// The open offer for everyone to see; the addressed human answers here, and the offerer picks a partner once all
        /// answered
        /// </summary>
        private void DrawPendingTrade(TradeOffer offer)
        {
            int offerer = offer.Offerer;
            sbyte? next = offer.NextResponder;
            bool answering = next is sbyte n && HumanToAct == n;
            bool picking = next == null && HumanToAct == offerer;
            if (!answering) _countering = false;

            List<int> addressed = Enumerable.Range(0, PLAYER_COUNT).Where(p => offer.Responses[p] != TradeResponse.NotAsked).ToList();
            Vector2 size = new Vector2(600, (_countering ? 470 : 190) + addressed.Count * 40 + (answering || picking ? 70 : 20));
            string title = answering ? $"{Theme.PlayerName(offerer)} offers you a trade" : $"{Theme.PlayerName(offerer)} offers a trade";
            ImDrawListPtr dl = BeginDialog("##pendingtrade", size, title, null, offerer, out Vector2 min);

            // The terms, from the side of whoever looks: the one answering gets what the offerer gives
            DrawTerms(dl, min + new Vector2(24, 60), answering ? "You get" : $"{Theme.PlayerName(offerer)} gives", offer.Give);
            DrawTerms(dl, min + new Vector2(310, 60), answering ? "You give" : $"{Theme.PlayerName(offerer)} wants", offer.Get);

            // The answers so far
            float y = 160;
            foreach (int seat in addressed)
            {
                Vector2 row = min + new Vector2(24, y);
                Theme.DrawText(dl, Theme.Bold15, row, Theme.PlayerTextColor(seat), Theme.PlayerName(seat));
                (string text, Color color) = offer.Responses[seat] switch
                {
                    TradeResponse.Accepted => ("accepted", new Color(120, 220, 140)),
                    TradeResponse.Rejected => ("rejected", Theme.TextFaint),
                    TradeResponse.Countered when offer.Counters.FirstOrDefault(c => c.Player == seat) is TradeCounter counter =>
                        ($"wants {ActionLogger.CardList(counter.Give)}, gives {ActionLogger.CardList(counter.Get)}", Theme.Highlight),
                    _ when seat == next => (IsHuman(seat) ? "deciding..." : "thinking...", Theme.TextSoft),
                    _ => ("waiting", Theme.TextFaint),
                };
                Theme.DrawText(dl, Theme.Regular15, row + new Vector2(110, 0), color, text);

                if (picking && offer.TermsWith(seat) != null)
                {
                    ConfirmTradeAction confirm = new ConfirmTradeAction((sbyte)offerer, (sbyte)seat);
                    if (DrawButton(dl, $"confirm{seat}", row + new Vector2(size.X - 48 - 120, -6), new Vector2(120, 32), "Trade", confirm.IsValidFor(State), Theme.Accent, Theme.AccentBorder, Theme.Bold15))
                        PlayHumanMove(confirm);
                }
                y += 40;
            }

            float bottom = size.Y - 20 - 44;
            if (answering && next is sbyte responder)
            {
                if (_countering)
                {
                    // The counteroffer from the answering player's side; the action takes it in the offerer's terms
                    CardSet<ResourceCardType> hand = State.Players[responder].ResourceCards;
                    Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, y + 6), Theme.TextSoft, "You give");
                    DrawCardRow(dl, "cgive", min + new Vector2(28, y + 30), _counterGive, _counterGet, i => (int)hand.Get(CardTypeExtensions.ResourceTypes[i]), Theme.PlayerColor(responder), Theme.TextOnPlayer(responder));
                    Theme.DrawText(dl, Theme.SemiBold15, min + new Vector2(24, y + 166), Theme.TextSoft, "You get");
                    DrawCardRow(dl, "cget", min + new Vector2(28, y + 190), _counterGet, _counterGive, _ => 9, Theme.PanelActive, Theme.Text);

                    CounterTradeAction counter = new CounterTradeAction(responder, ToCards(_counterGet), ToCards(_counterGive));
                    if (DrawButton(dl, "countersend", min + new Vector2(size.X - 24 - 200, bottom), new Vector2(200, 44), "Send counteroffer", counter.IsValidFor(State), Theme.Accent, Theme.AccentBorder, Theme.Bold17))
                    {
                        _countering = false;
                        PlayHumanMove(counter);
                    }
                    if (DrawButton(dl, "counterback", min + new Vector2(24, bottom), new Vector2(120, 44), "Back", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                        _countering = false;
                }
                else
                {
                    RespondToTradeAction accept = new RespondToTradeAction(responder, true);
                    if (DrawButton(dl, "tradeaccept", min + new Vector2(size.X - 24 - 160, bottom), new Vector2(160, 44), "Accept", accept.IsValidFor(State), Theme.Accent, Theme.AccentBorder, Theme.Bold17))
                        PlayHumanMove(accept);
                    if (DrawButton(dl, "tradecounter", min + new Vector2(size.X / 2 - 80, bottom), new Vector2(160, 44), "Counteroffer", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                    {
                        // Starts from the offer, seen from this side
                        FromCards(offer.Get, _counterGive);
                        FromCards(offer.Give, _counterGet);
                        _countering = true;
                    }
                    if (DrawButton(dl, "tradereject", min + new Vector2(24, bottom), new Vector2(120, 44), "Reject", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                        PlayHumanMove(new RespondToTradeAction(responder, false));
                }
            }
            else if (picking)
            {
                if (DrawButton(dl, "tradewithdraw", min + new Vector2(24, bottom), new Vector2(160, 44), "Withdraw offer", true, Theme.PanelRaised, Theme.BorderLight, Theme.Bold15))
                    PlayHumanMove(new CancelTradeAction((sbyte)offerer));
                Theme.TextRight(dl, Theme.Regular13, min + new Vector2(size.X - 24, bottom + 14), Theme.TextMuted, offer.HasPartner ? "Pick whom to trade with" : "");
            }
            ImGui.End();
        }

        private static void DrawTerms(ImDrawListPtr dl, Vector2 pos, string label, CardSet<ResourceCardType> cards)
        {
            Theme.DrawText(dl, Theme.SemiBold15, pos, Theme.TextMuted, label);
            float x = 0;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                for (int i = 0; i < cards.Get(type); i++)
                {
                    Theme.ResourceCard(dl, type, pos + new Vector2(x, 26), new Vector2(40, 60));
                    x += i + 1 < cards.Get(type) ? 16 : 48;
                }
            }
        }
    }
}
