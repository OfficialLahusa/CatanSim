using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents
{
    /// <summary>
    /// How agents take part in trades between players: they answer offers addressed to them and, as the player on turn (a
    /// stand-in for a human), pick a partner or let an offer go. They don't make offers or counteroffers themselves. A trade is
    /// judged by comparing the position after it with the position without it, by a value function if the agent has one
    /// (the value net's utility) or else by how close the hand comes to the next buildings.
    /// </summary>
    public static class TradeDecisions
    {
        // A trade has to be better than this to be worth it, so agents don't trade for noise in the value
        private const float ValueMargin = 0.002f, HandMargin = 0.15f;

        // Victory points short of the win from which agents stop helping a player
        private const int LeaderDistance = 2;

        /// <summary>
        /// The agent's move in an open trade, or null if there is no open trade it has to act in
        /// </summary>
        /// <param name="value">The agent's utility of a position for a player, or null for the hand heuristic</param>
        public static Action? Decide(GameState state, sbyte player, Func<GameState, int, float>? value = null)
        {
            if (state.Turn.PendingTrade is not TradeOffer offer) return null;

            if (offer.NextResponder == player)
            {
                RespondToTradeAction reject = new RespondToTradeAction(player, false), accept = new RespondToTradeAction(player, true);
                if (!accept.IsValidFor(state)) return reject;

                // No help for a player who is about to win
                if (state.Players[offer.Offerer].VictoryPoints.Total >= state.Settings.VictoryPoints - LeaderDistance) return reject;

                GameState traded = AfterTrade(state, offer.Offerer, player, offer.Give, offer.Get);
                return IsBetter(traded, state, player, value) ? accept : reject;
            }

            if (offer.Offerer == player && offer.NextResponder == null)
            {
                Action best = new CancelTradeAction(player);
                GameState bestState = state;
                foreach (Action confirm in ConfirmTradeAction.GetActionsForState(state, player))
                {
                    GameState after = new GameState(state);
                    confirm.Apply(after);
                    if (IsBetter(after, bestState, player, value))
                    {
                        best = confirm;
                        bestState = after;
                    }
                }
                return best;
            }

            return null;
        }

        /// <summary>
        /// The position after the offerer gives one set of cards to the partner and gets the other, with the offer closed
        /// </summary>
        private static GameState AfterTrade(GameState state, int offerer, int partner, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
        {
            GameState after = new GameState(state);
            after.Turn.PendingTrade = null;
            after.Players[offerer].ResourceCards.Remove(give);
            after.Players[partner].ResourceCards.Add(give);
            after.Players[partner].ResourceCards.Remove(get);
            after.Players[offerer].ResourceCards.Add(get);
            return after;
        }

        private static bool IsBetter(GameState candidate, GameState current, int player, Func<GameState, int, float>? value)
        {
            if (value != null) return value(candidate, player) > value(current, player) + ValueMargin;
            return HandValue(candidate.Players[player].ResourceCards) > HandValue(current.Players[player].ResourceCards) + HandMargin;
        }

        private static readonly (float weight, (ResourceCardType type, uint amount)[] cost)[] Buildings =
        [
            (1f, [(ResourceCardType.Lumber, 1), (ResourceCardType.Brick, 1)]),
            (3f, [(ResourceCardType.Lumber, 1), (ResourceCardType.Brick, 1), (ResourceCardType.Wool, 1), (ResourceCardType.Grain, 1)]),
            (3f, [(ResourceCardType.Grain, 2), (ResourceCardType.Ore, 3)]),
            (1.5f, [(ResourceCardType.Wool, 1), (ResourceCardType.Grain, 1), (ResourceCardType.Ore, 1)]),
        ];

        /// <summary>
        /// How close a hand comes to the buildings: per building the share of its cost covered, squared so that completing one
        /// counts most, weighted by what the building is worth; a little for every card on top
        /// </summary>
        public static float HandValue(CardSet<ResourceCardType> hand)
        {
            float value = 0.02f * hand.Count();
            foreach ((float weight, (ResourceCardType type, uint amount)[] cost) in Buildings)
            {
                uint covered = 0, total = 0;
                foreach ((ResourceCardType type, uint amount) in cost)
                {
                    covered += Math.Min(hand.Get(type), amount);
                    total += amount;
                }
                float share = (float)covered / total;
                value += weight * share * share;
            }
            return value;
        }
    }
}
