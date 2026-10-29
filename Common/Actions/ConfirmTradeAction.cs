namespace Common.Actions
{
    /// <summary>
    /// Once everyone answered, the offerer trades with one partner who accepted (on the offer's terms) or countered (on the
    /// counteroffer's terms), if both still hold the cards. The offer closes.
    /// </summary>
    public class ConfirmTradeAction : Action, IReplayAction, IActionProvider
    {
        public sbyte Partner { get; init; }

        /// <summary>
        /// The closed offer, restored by Revert
        /// </summary>
        public TradeOffer? History { get; private set; }

        public ConfirmTradeAction(sbyte playerIdx, sbyte partner)
            : base(playerIdx)
        {
            Partner = partner;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private ConfirmTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            TradeOffer offer = state.Turn.PendingTrade!;
            (CardSet<ResourceCardType> give, CardSet<ResourceCardType> get) = offer.TermsWith(Partner)!.Value;
            Transfer(state, give, get);
            History = new TradeOffer(offer);
            state.Turn.PendingTrade = null;
        }

        public override void Revert(GameState state)
        {
            TradeOffer offer = new TradeOffer(History!);
            (CardSet<ResourceCardType> give, CardSet<ResourceCardType> get) = offer.TermsWith(Partner)!.Value;
            Transfer(state, get, give);
            state.Turn.PendingTrade = offer;
        }

        /// <summary>
        /// The offerer gives one set to the partner and gets the other from them
        /// </summary>
        private void Transfer(GameState state, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
        {
            CardSet<ResourceCardType> offerer = state.Players[PlayerIndex].ResourceCards, partner = state.Players[Partner].ResourceCards;
            offerer.Remove(give);
            partner.Add(give);
            partner.Remove(get);
            offerer.Add(get);
        }

        public override bool IsValidFor(GameState state)
        {
            if (!IsTurnValid(state.Turn, PlayerIndex) || Partner < 0 || Partner >= state.Players.Length) return false;
            if (state.Turn.PendingTrade!.TermsWith(Partner) is not { } terms) return false;
            return state.Players[PlayerIndex].ResourceCards.Contains(terms.give) && state.Players[Partner].ResourceCards.Contains(terms.get);
        }

        /// <summary>
        /// Whether the player offered the open trade and everyone answered
        /// </summary>
        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PendingTrade is TradeOffer offer && offer.Offerer == playerIdx && offer.NextResponder == null;
        }

        public bool HasHistory() => History != null;

        public void ClearHistory() => History = null;

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            if (!IsTurnValid(state.Turn, playerIdx)) return [];
            List<Action> actions = [];
            for (int partner = 0; partner < state.Players.Length; partner++)
            {
                ConfirmTradeAction confirm = new ConfirmTradeAction(playerIdx, (sbyte)partner);
                if (confirm.IsValidFor(state)) actions.Add(confirm);
            }
            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", trades with P{Partner}";
        }
    }
}
