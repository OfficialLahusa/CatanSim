namespace Common.Actions
{
    /// <summary>
    /// An addressed player answers the open offer with other terms, in the offerer's view: what the offerer would give and get.
    /// The same rules as for offers hold, and the player needs the cards the offerer would get. The offerer may then trade on
    /// these terms. Like offers, counteroffers are not listed as legal moves.
    /// </summary>
    public class CounterTradeAction : Action
    {
        public CardSet<ResourceCardType> Give { get; init; } = new();
        public CardSet<ResourceCardType> Get { get; init; } = new();

        public CounterTradeAction(sbyte playerIdx, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
            : base(playerIdx)
        {
            Give = new CardSet<ResourceCardType>(give);
            Get = new CardSet<ResourceCardType>(get);
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private CounterTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            TradeOffer offer = state.Turn.PendingTrade!;
            offer.Responses[PlayerIndex] = TradeResponse.Countered;
            offer.Counters.Add(new TradeCounter(PlayerIndex, Give, Get));
        }

        public override void Revert(GameState state)
        {
            TradeOffer offer = state.Turn.PendingTrade!;
            offer.Responses[PlayerIndex] = TradeResponse.Pending;
            offer.Counters.RemoveAll(c => c.Player == PlayerIndex);
        }

        public override bool IsValidFor(GameState state)
        {
            return RespondToTradeAction.IsTurnValid(state.Turn, PlayerIndex)
                && TradeOffer.AreValidTerms(Give, Get)
                && state.Players[PlayerIndex].ResourceCards.Contains(Get);
        }

        public override string ToString()
        {
            return base.ToString() + $", counters: {OfferTradeAction.Describe(Give)} for {OfferTradeAction.Describe(Get)}";
        }
    }
}
