namespace Common.Actions
{
    /// <summary>
    /// The player on turn offers a trade to some of the other players: cards to give for cards to get. Only with free trade,
    /// after the roll, while nothing else is pending. The addressed players answer in seat order after the offerer.
    /// Offers can be any combination of cards, so they are not listed as legal moves: the client or an agent makes them.
    /// </summary>
    public class OfferTradeAction : Action
    {
        public CardSet<ResourceCardType> Give { get; init; } = new();
        public CardSet<ResourceCardType> Get { get; init; } = new();

        /// <summary>
        /// The addressed players, one bit per seat
        /// </summary>
        public byte Recipients { get; init; }

        public OfferTradeAction(sbyte playerIdx, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get, byte recipients)
            : base(playerIdx)
        {
            Give = new CardSet<ResourceCardType>(give);
            Get = new CardSet<ResourceCardType>(get);
            Recipients = recipients;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private OfferTradeAction()
            : base(-1)
        { }

        /// <summary>
        /// The recipients mask for all players but the offerer
        /// </summary>
        public static byte Everyone(GameState state, int offerer) => (byte)(((1 << state.Players.Length) - 1) & ~(1 << offerer));

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Road Building roads that were not placed yet are given up, as with any other move
            ForfeitFreeRoads(state);

            state.Turn.PendingTrade = new TradeOffer(PlayerIndex, Give, Get, Recipients, state.Players.Length);
            state.Turn.TradeOffersThisTurn++;
        }

        public override void Revert(GameState state)
        {
            state.Turn.PendingTrade = null;
            state.Turn.TradeOffersThisTurn--;
            ReturnForfeitedFreeRoads(state);
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state, PlayerIndex)
                && TradeOffer.AreValidTerms(Give, Get)
                && state.Players[PlayerIndex].ResourceCards.Contains(Give)
                && AreValidRecipients(state);
        }

        /// <summary>
        /// Whether the player may offer a trade now: free trade, on turn, after the roll, nothing pending, offers left
        /// </summary>
        public static bool IsTurnValid(GameState state, int playerIdx)
        {
            TurnState turn = state.Turn;
            return state.Settings.FreeTrade
                && turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && turn.PendingTrade == null
                && !turn.MustRoll
                && !turn.MustDiscard
                && !turn.MustMoveRobber
                && turn.TradeOffersThisTurn < state.Settings.TradeOfferLimit;
        }

        private bool AreValidRecipients(GameState state)
        {
            int all = (1 << state.Players.Length) - 1;
            return Recipients != 0 && (Recipients & ~all) == 0 && (Recipients & 1 << PlayerIndex) == 0;
        }

        public override string ToString()
        {
            return base.ToString() + $", offers {Describe(Give)} for {Describe(Get)} to {Recipients:b4}";
        }

        internal static string Describe(CardSet<ResourceCardType> cards) =>
            string.Concat(CardTypeExtensions.ResourceTypes.SelectMany(type => Enumerable.Repeat(type.GetAbbreviation(), (int)cards.Get(type))));
    }
}
