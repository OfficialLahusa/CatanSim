namespace Common
{
    /// <summary>
    /// How an addressed player answered an offer
    /// </summary>
    public enum TradeResponse : byte
    {
        NotAsked,
        Pending,
        Accepted,
        Rejected,
        Countered
    }

    /// <summary>
    /// A counteroffer, in the terms of the player on turn: what that player would give and get
    /// </summary>
    public class TradeCounter
    {
        public sbyte Player { get; set; }
        public CardSet<ResourceCardType> Give { get; set; } = new();
        public CardSet<ResourceCardType> Get { get; set; } = new();

        public TradeCounter(sbyte player, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
        {
            Player = player;
            Give = new CardSet<ResourceCardType>(give);
            Get = new CardSet<ResourceCardType>(get);
        }

        public TradeCounter(TradeCounter copy) : this(copy.Player, copy.Give, copy.Get) { }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private TradeCounter() { }

        public override bool Equals(object? obj) => obj is TradeCounter other && Player == other.Player && Give.Equals(other.Give) && Get.Equals(other.Get);

        public override int GetHashCode() => HashCode.Combine(Player, Give, Get);
    }

    /// <summary>
    /// A trade the player on turn offered to some of the others: what the offerer gives and gets, and the answers so far.
    /// The addressed players answer one after another in seat order after the offerer; then the offerer trades with one
    /// who accepted (or on the terms of one's counteroffer) or lets the offer go.
    /// </summary>
    public class TradeOffer
    {
        public sbyte Offerer { get; set; }
        public CardSet<ResourceCardType> Give { get; set; } = new();
        public CardSet<ResourceCardType> Get { get; set; } = new();
        public TradeResponse[] Responses { get; set; } = [];
        public List<TradeCounter> Counters { get; set; } = [];

        public TradeOffer(sbyte offerer, CardSet<ResourceCardType> give, CardSet<ResourceCardType> get, byte recipients, int playerCount)
        {
            Offerer = offerer;
            Give = new CardSet<ResourceCardType>(give);
            Get = new CardSet<ResourceCardType>(get);
            Responses = new TradeResponse[playerCount];
            for (int player = 0; player < playerCount; player++)
                Responses[player] = (recipients & 1 << player) != 0 ? TradeResponse.Pending : TradeResponse.NotAsked;
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        public TradeOffer(TradeOffer copy)
        {
            Offerer = copy.Offerer;
            Give = new CardSet<ResourceCardType>(copy.Give);
            Get = new CardSet<ResourceCardType>(copy.Get);
            Responses = (TradeResponse[])copy.Responses.Clone();
            Counters = copy.Counters.Select(c => new TradeCounter(c)).ToList();
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private TradeOffer() { }

        /// <summary>
        /// The addressed player who answers next (seat order after the offerer), or null once everyone answered
        /// </summary>
        public sbyte? NextResponder
        {
            get
            {
                for (int i = 1; i < Responses.Length; i++)
                {
                    int player = (Offerer + i) % Responses.Length;
                    if (Responses[player] == TradeResponse.Pending) return (sbyte)player;
                }
                return null;
            }
        }

        /// <summary>
        /// Whether anyone accepted or countered, so there is still something to trade
        /// </summary>
        public bool HasPartner => Responses.Any(r => r is TradeResponse.Accepted or TradeResponse.Countered);

        /// <summary>
        /// What the offerer gives and gets when trading with a partner: the offer, or the partner's counteroffer
        /// </summary>
        public (CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)? TermsWith(int partner)
        {
            if (partner < 0 || partner >= Responses.Length) return null;
            return Responses[partner] switch
            {
                TradeResponse.Accepted => (Give, Get),
                TradeResponse.Countered when Counters.FirstOrDefault(c => c.Player == partner) is TradeCounter counter => (counter.Give, counter.Get),
                _ => null
            };
        }

        /// <summary>
        /// Whether one side's cards are a valid half of a trade: at least one card, resources only, and no type the other side
        /// also gives (the rulebook forbids gifts and trading like resources)
        /// </summary>
        public static bool AreValidTerms(CardSet<ResourceCardType> give, CardSet<ResourceCardType> get)
        {
            if (give.Count() == 0 || get.Count() == 0) return false;
            if (give.Get(ResourceCardType.Unknown) > 0 || get.Get(ResourceCardType.Unknown) > 0) return false;
            return CardTypeExtensions.ResourceTypes.All(type => give.Get(type) == 0 || get.Get(type) == 0);
        }

        public override bool Equals(object? obj) => obj is TradeOffer other && Offerer == other.Offerer && Give.Equals(other.Give) && Get.Equals(other.Get)
            && Responses.SequenceEqual(other.Responses) && Counters.SequenceEqual(other.Counters);

        public override int GetHashCode()
        {
            HashCode hash = new HashCode();
            hash.Add(Offerer);
            hash.Add(Give);
            hash.Add(Get);
            foreach (TradeResponse response in Responses) hash.Add(response);
            foreach (TradeCounter counter in Counters) hash.Add(counter);
            return hash.ToHashCode();
        }
    }
}
