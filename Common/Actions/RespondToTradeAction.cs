namespace Common.Actions
{
    /// <summary>
    /// An addressed player accepts or rejects the open offer, in turn. Accepting needs the cards asked for. When the last
    /// answer leaves nobody to trade with, the offer closes.
    /// </summary>
    public class RespondToTradeAction : Action, IReplayAction, IActionProvider
    {
        public bool Accept { get; init; }

        /// <summary>
        /// The offer this answer closed (when nobody is left to trade with), restored by Revert
        /// </summary>
        public TradeOffer? History { get; private set; }

        public RespondToTradeAction(sbyte playerIdx, bool accept)
            : base(playerIdx)
        {
            Accept = accept;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private RespondToTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            TradeOffer offer = state.Turn.PendingTrade!;
            offer.Responses[PlayerIndex] = Accept ? TradeResponse.Accepted : TradeResponse.Rejected;
            History = null;
            if (offer.NextResponder == null && !offer.HasPartner)
            {
                History = new TradeOffer(offer);
                state.Turn.PendingTrade = null;
            }
        }

        public override void Revert(GameState state)
        {
            if (History != null) state.Turn.PendingTrade = new TradeOffer(History);
            state.Turn.PendingTrade!.Responses[PlayerIndex] = TradeResponse.Pending;
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex)
                && (!Accept || state.Players[PlayerIndex].ResourceCards.Contains(state.Turn.PendingTrade!.Get));
        }

        /// <summary>
        /// Whether the player is the one to answer the open offer now
        /// </summary>
        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PendingTrade?.NextResponder == playerIdx;
        }

        public bool HasHistory() => History != null;

        public void ClearHistory() => History = null;

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            if (!IsTurnValid(state.Turn, playerIdx)) return [];
            List<Action> actions = [new RespondToTradeAction(playerIdx, false)];
            if (state.Players[playerIdx].ResourceCards.Contains(state.Turn.PendingTrade!.Get)) actions.Add(new RespondToTradeAction(playerIdx, true));
            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + (Accept ? ", accepts the offer" : ", rejects the offer");
        }
    }
}
