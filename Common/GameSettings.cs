using YamlDotNet.Serialization;

namespace Common
{
    public class GameSettings
    {
        // Maximum allowed number of resource cards before discarding is required
        public int RobberCardLimit { get; set; }
        // Required number of victory points to win the match
        public int VictoryPoints { get; set; }

        /// <summary>
        /// Whether players trade with each other (offers, counteroffers) besides with the bank. Off for agent games, self-play
        /// and datasets, whose agents and nets don't trade; on for games with humans.
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public bool FreeTrade { get; set; }

        /// <summary>
        /// Offers a player may make per turn, 0 for the default: the rulebook sets no limit, but endless offers would stall the game
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public int MaxTradeOffersPerTurn { get; set; }

        public const int DefaultMaxTradeOffersPerTurn = 10;

        /// <summary>
        /// Whether the players' screens show the other hands as each player can know them (the cards confirmed by what
        /// happened in public, see BeliefTracker) instead of only their size. A table rule of the display, not of the game:
        /// all of it is public, a player could count along.
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public bool ShowTrackedCards { get; set; }

        [YamlIgnore]
        public int TradeOfferLimit => MaxTradeOffersPerTurn > 0 ? MaxTradeOffersPerTurn : DefaultMaxTradeOffersPerTurn;

        public GameSettings()
        {
            RobberCardLimit = 7;
            VictoryPoints = 10;
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public GameSettings(GameSettings copy)
        {
            RobberCardLimit = copy.RobberCardLimit;
            VictoryPoints = copy.VictoryPoints;
            FreeTrade = copy.FreeTrade;
            MaxTradeOffersPerTurn = copy.MaxTradeOffersPerTurn;
            ShowTrackedCards = copy.ShowTrackedCards;
        }

        public override bool Equals(object? obj)
        {
            return obj is GameSettings settings &&
                   RobberCardLimit == settings.RobberCardLimit &&
                   VictoryPoints == settings.VictoryPoints &&
                   FreeTrade == settings.FreeTrade &&
                   MaxTradeOffersPerTurn == settings.MaxTradeOffersPerTurn &&
                   ShowTrackedCards == settings.ShowTrackedCards;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(RobberCardLimit, VictoryPoints, FreeTrade, MaxTradeOffersPerTurn, ShowTrackedCards);
        }
    }
}