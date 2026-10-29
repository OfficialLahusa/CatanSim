namespace Common.Actions
{
    /// <summary>
    /// Once everyone answered, the offerer lets the offer go without trading
    /// </summary>
    public class CancelTradeAction : Action, IReplayAction, IActionProvider
    {
        /// <summary>
        /// The closed offer, restored by Revert
        /// </summary>
        public TradeOffer? History { get; private set; }

        public CancelTradeAction(sbyte playerIdx)
            : base(playerIdx)
        { }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private CancelTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);
            History = new TradeOffer(state.Turn.PendingTrade!);
            state.Turn.PendingTrade = null;
        }

        public override void Revert(GameState state)
        {
            state.Turn.PendingTrade = new TradeOffer(History!);
        }

        public override bool IsValidFor(GameState state)
        {
            return ConfirmTradeAction.IsTurnValid(state.Turn, PlayerIndex);
        }

        public bool HasHistory() => History != null;

        public void ClearHistory() => History = null;

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            return ConfirmTradeAction.IsTurnValid(state.Turn, playerIdx) ? [new CancelTradeAction(playerIdx)] : [];
        }

        public override string ToString()
        {
            return base.ToString() + ", cancels the offer";
        }
    }
}
