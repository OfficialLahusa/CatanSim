using Common.Actions;
using Action = Common.Actions.Action;

namespace Common
{
    /// <summary>
    /// What an observer knows about another player's resource cards: per type the fewest (Known) and the most (Maximum) cards
    /// the player can hold, and how many of its cards are of unknown type. Only a steal the observer didn't take part in makes
    /// cards unknown; every other move changes the hands in public.
    /// </summary>
    public sealed class HandBelief
    {
        /// <summary>
        /// Cards the player certainly holds (it may hold more of a type among the unknown cards)
        /// </summary>
        public CardSet<ResourceCardType> Known { get; }

        /// <summary>
        /// The most cards of each type the player can hold: after a steal of one of five different cards, at most one of each
        /// </summary>
        public CardSet<ResourceCardType> Maximum { get; }

        /// <summary>
        /// Cards whose type the observer doesn't know
        /// </summary>
        public uint Unknown { get; internal set; }

        public HandBelief()
        {
            Known = new CardSet<ResourceCardType>();
            Maximum = new CardSet<ResourceCardType>();
        }

        public HandBelief(CardSet<ResourceCardType> known)
        {
            Known = new CardSet<ResourceCardType>(known);
            Maximum = new CardSet<ResourceCardType>(known);
        }

        public HandBelief(HandBelief copy)
        {
            Known = new CardSet<ResourceCardType>(copy.Known);
            Maximum = new CardSet<ResourceCardType>(copy.Maximum);
            Unknown = copy.Unknown;
        }

        /// <summary>
        /// How many of the unknown cards can be of the type at most
        /// </summary>
        public uint Room(ResourceCardType type) => Math.Min(Unknown, Maximum.Get(type) - Known.Get(type));

        /// <summary>
        /// Whether an unknown card can be of the type
        /// </summary>
        public bool CanBe(ResourceCardType type) => Room(type) > 0;

        /// <summary>
        /// A bit (1 &lt;&lt; type) for each type an unknown card can be
        /// </summary>
        public byte Candidates => (byte)CardTypeExtensions.ResourceTypes.Where(CanBe).Aggregate(0, (mask, type) => mask | 1 << (int)type);

        public uint Count => Known.Count() + Unknown;

        internal void Gain(ResourceCardType type, uint amount)
        {
            Known.Add(type, amount);
            Maximum.Add(type, amount);
        }

        public override bool Equals(object? obj) => obj is HandBelief other && Known.Equals(other.Known) && Maximum.Equals(other.Maximum) && Unknown == other.Unknown;

        public override int GetHashCode() => HashCode.Combine(Known, Maximum, Unknown);
    }

    /// <summary>
    /// Follows a game from one player's seat (or from nobody's, Public) and keeps what that observer can know: the other
    /// players' resource cards (see HandBelief) and the development cards it can't see. From the true state it makes the
    /// observer's view (StateView), in which unknown cards are of type Unknown, and from the view random full states that
    /// agree with everything the observer saw (determinizations).
    ///
    /// The hands change in public with every move, except for a steal between two other players: the observer then knows
    /// that the victim lost one of its cards and the thief got it, not which. Spending more of a type than known resolves
    /// unknown cards, a monopoly shows how many of its type everyone held, and the cards of each type in all hands are known
    /// (the bank is public), which narrows the candidates further. The development cards in the other hands and in the bank
    /// are known as a whole (the deck minus the observer's own and the played ones), not who holds which.
    /// </summary>
    public sealed class BeliefTracker
    {
        /// <summary>
        /// The observer of a tracker that holds no seat: it sees no hand
        /// </summary>
        public const int Public = -1;

        public int Observer { get; }

        private readonly HandBelief[] _hands;

        // All resource cards of each type in the game, in the bank and the hands
        private CardSet<ResourceCardType> _totals;

        // The development cards the observer can't see, and how many of them the other players hold
        private CardSet<DevelopmentCardType> _hiddenDevelopment;
        private uint _othersDevelopment;

        // A determinization that gives someone the winning points is drawn again, at most this often
        private const int MaxDeals = 200;

        /// <summary>
        /// Starts following a game at the given state, whose hands count as known (a new game starts with empty hands)
        /// </summary>
        public BeliefTracker(int observer, GameState start)
        {
            Observer = observer;
            _hands = start.Players.Select(p => new HandBelief(p.ResourceCards)).ToArray();
            _totals = Totals(start);
            (_hiddenDevelopment, _othersDevelopment) = HiddenDevelopment(start);
        }

        public BeliefTracker(BeliefTracker copy)
        {
            Observer = copy.Observer;
            _hands = copy._hands.Select(h => new HandBelief(h)).ToArray();
            _totals = new CardSet<ResourceCardType>(copy._totals);
            _hiddenDevelopment = new CardSet<DevelopmentCardType>(copy._hiddenDevelopment);
            _othersDevelopment = copy._othersDevelopment;
        }

        /// <summary>
        /// What the observer knows about a player's resource cards; its own hand is known exactly
        /// </summary>
        public HandBelief Hand(int player) => _hands[player];

        /// <summary>
        /// The development cards the observer can't see: those in the other players' hands and in the bank
        /// </summary>
        public CardSet<DevelopmentCardType> HiddenDevelopmentCards => _hiddenDevelopment;

        /// <summary>
        /// Whether the observer is missing anything: unknown resource cards or development cards it can't see in another
        /// player's hand (without them the bank's cards are exactly the hidden ones)
        /// </summary>
        public bool HasHiddenInformation => _othersDevelopment > 0 || _hands.Any(h => h.Unknown > 0);

        private bool IsOther(int player) => player != Observer;

        private static CardSet<ResourceCardType> Totals(GameState state)
        {
            CardSet<ResourceCardType> totals = new CardSet<ResourceCardType>(state.ResourceBank);
            foreach (PlayerState player in state.Players) totals.Add(player.ResourceCards);
            return totals;
        }

        private (CardSet<DevelopmentCardType> hidden, uint others) HiddenDevelopment(GameState state)
        {
            CardSet<DevelopmentCardType> hidden = new CardSet<DevelopmentCardType>(state.DevelopmentBank);
            uint others = 0;
            // The new cards of this turn are among the DevelopmentCards
            for (int player = 0; player < state.Players.Length; player++)
            {
                if (!IsOther(player)) continue;
                hidden.Add(state.Players[player].DevelopmentCards);
                others += state.Players[player].DevelopmentCards.Count();
            }
            return (hidden, others);
        }

        #region Observing
        /// <summary>
        /// Takes in a move (with its resolved outcome) or an edit (action null) as the observer sees it, given the true states
        /// before and after it
        /// </summary>
        public void Observe(GameState before, Action? action, GameState after)
        {
            // A steal between two others: the card's move is taken out of the public changes
            (int thief, int victim, ResourceCardType card)? hiddenSteal = action is RobberAction { History.StolenCard: ResourceCardType stolen, TargetPlayerIndex: sbyte target } robber
                && IsOther(robber.PlayerIndex) && IsOther(target) ? (robber.PlayerIndex, target, stolen) : null;

            int types = CardSet<ResourceCardType>.Values.Count;
            for (int player = 0; player < _hands.Length; player++)
            {
                int[] change = new int[types];
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    change[(int)type] = (int)after.Players[player].ResourceCards.Get(type) - (int)before.Players[player].ResourceCards.Get(type);
                if (hiddenSteal is (int thief, int victim, ResourceCardType card))
                {
                    if (player == thief) change[(int)card]--;
                    if (player == victim) change[(int)card]++;
                }

                // Spending first: a card that changes hands within the move is never both spent and gained of one type
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    if (change[(int)type] < 0) Spend(player, type, (uint)-change[(int)type]);
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    if (change[(int)type] > 0) _hands[player].Gain(type, (uint)change[(int)type]);
            }

            if (hiddenSteal is (int thiefIdx, int victimIdx, _)) StealUnseen(thiefIdx, victimIdx);

            // After a monopoly nobody else holds a card of its type
            if (action is MonopolyAction monopoly)
            {
                for (int player = 0; player < _hands.Length; player++)
                {
                    if (player != monopoly.PlayerIndex) RuleOut(player, monopoly.ChosenType);
                }
            }

            if (action == null)
            {
                // An edit is public: the totals and the development cards are taken from it
                _totals = Totals(after);
                (_hiddenDevelopment, _othersDevelopment) = HiddenDevelopment(after);
            }
            else
            {
                ObserveDevelopment(action);
                _othersDevelopment = 0;
                for (int player = 0; player < after.Players.Length; player++)
                    if (IsOther(player)) _othersDevelopment += after.Players[player].DevelopmentCards.Count();
            }

            Propagate(after.ResourceBank);
        }

        /// <summary>
        /// A drawn card leaves the hidden ones for its buyer, a played card for everyone else
        /// </summary>
        private void ObserveDevelopment(Action action)
        {
            DevelopmentCardType? played = action switch
            {
                KnightAction => DevelopmentCardType.Knight,
                RoadBuildingAction => DevelopmentCardType.RoadBuilding,
                YearOfPlentyAction => DevelopmentCardType.YearOfPlenty,
                MonopolyAction => DevelopmentCardType.Monopoly,
                _ => null
            };
            if (played is DevelopmentCardType type && IsOther(action.PlayerIndex)) _hiddenDevelopment.Remove(type, 1);
            if (action is BuyDevelopmentCardAction { History.DrawnType: DevelopmentCardType drawn } && !IsOther(action.PlayerIndex))
                _hiddenDevelopment.Remove(drawn, 1);
        }

        /// <summary>
        /// The player gave up cards of a type in public: known ones first, the rest were unknown cards of that type
        /// </summary>
        private void Spend(int player, ResourceCardType type, uint amount)
        {
            HandBelief hand = _hands[player];
            uint fromKnown = Math.Min(hand.Known.Get(type), amount);
            uint fromUnknown = amount - fromKnown;
            if (fromUnknown > hand.Room(type))
                throw new InvalidOperationException($"Observer {Observer} lost track of player {player}: it spent {type} it can't hold");

            hand.Known.Remove(type, fromKnown);
            hand.Maximum.Remove(type, amount);
            hand.Unknown -= fromUnknown;
        }

        /// <summary>
        /// The thief took one of the victim's cards without the observer seeing which
        /// </summary>
        private void StealUnseen(int thief, int victim)
        {
            HandBelief from = _hands[victim];
            List<ResourceCardType> possible = CardTypeExtensions.ResourceTypes.Where(type => from.Known.Get(type) > 0 || from.CanBe(type)).ToList();

            // A single possible type is a known card after all
            if (possible.Count == 1)
            {
                Spend(victim, possible[0], 1);
                _hands[thief].Gain(possible[0], 1);
                return;
            }

            // Every known type may have lost a card: one of each becomes unknown, less the stolen card; the victim gained
            // nothing, so its maximum stays, and the thief may hold one more of each possible type
            uint count = from.Count;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                if (from.Known.Get(type) > 0) from.Known.Remove(type, 1);
            from.Unknown = count - 1 - from.Known.Count();

            HandBelief to = _hands[thief];
            to.Unknown++;
            foreach (ResourceCardType type in possible) to.Maximum.Add(type, 1);
        }

        /// <summary>
        /// The player holds no card of the type (after a monopoly)
        /// </summary>
        private void RuleOut(int player, ResourceCardType type)
        {
            HandBelief hand = _hands[player];
            if (hand.Known.Get(type) > 0) throw new InvalidOperationException($"Observer {Observer} lost track of player {player}: it kept {type} through a monopoly");
            hand.Maximum.Remove(type, hand.Maximum.Get(type));
        }

        /// <summary>
        /// The cards no hand is known to hold are exactly the unknown ones, which narrows the bounds until nothing changes: a
        /// hand holds no more of a type than remain or than it has unknown cards; unknown cards the other types can't fill are
        /// of the one left; where the room of all hands for a type is just what remains of it, every hand holds its most; a
        /// hand without unknown cards is known exactly
        /// </summary>
        private void Propagate(CardSet<ResourceCardType> bank)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                CardSet<ResourceCardType> remaining = Remaining(bank);

                for (int player = 0; player < _hands.Length; player++)
                {
                    HandBelief hand = _hands[player];
                    foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    {
                        uint most = hand.Known.Get(type) + Math.Min(hand.Unknown, Math.Min(remaining.Get(type), hand.Maximum.Get(type) - hand.Known.Get(type)));
                        if (most < hand.Maximum.Get(type))
                        {
                            hand.Maximum.Remove(type, hand.Maximum.Get(type) - most);
                            changed = true;
                        }
                    }
                    if (hand.Unknown == 0) continue;

                    uint room = (uint)CardTypeExtensions.ResourceTypes.Sum(type => (long)hand.Room(type));
                    if (room < hand.Unknown) throw new InvalidOperationException($"Observer {Observer} lost track of player {player}: no room is left for its unknown cards");
                    foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    {
                        uint others = room - hand.Room(type);
                        if (hand.Unknown > others)
                        {
                            uint certain = hand.Unknown - others;
                            hand.Known.Add(type, certain);
                            hand.Unknown -= certain;
                            changed = true;
                            break;
                        }
                    }
                }
                if (changed) continue;

                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                {
                    uint left = remaining.Get(type);
                    long room = _hands.Sum(hand => (long)hand.Room(type));
                    if (room < left) throw new InvalidOperationException($"Observer {Observer} lost track of the unknown {type}");
                    if (left == 0 || room != left) continue;
                    foreach (HandBelief hand in _hands)
                    {
                        uint certain = hand.Room(type);
                        if (certain == 0) continue;
                        hand.Known.Add(type, certain);
                        hand.Unknown -= certain;
                    }
                    changed = true;
                    break;
                }
            }
        }

        /// <summary>
        /// The cards of each type that are in a hand but not known to be: the unknown cards of all hands together
        /// </summary>
        private CardSet<ResourceCardType> Remaining(CardSet<ResourceCardType> bank)
        {
            CardSet<ResourceCardType> remaining = new CardSet<ResourceCardType>();
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
            {
                long left = (long)_totals.Get(type) - bank.Get(type) - _hands.Sum(h => (long)h.Known.Get(type));
                if (left < 0) throw new InvalidOperationException($"Observer {Observer} believes in more {type} than there are");
                remaining.Add(type, (uint)left);
            }
            return remaining;
        }
        #endregion

        #region Views and determinizations
        /// <summary>
        /// The observer's view of the true state: the other players' unknown resource cards and all their development cards
        /// as Unknown, their victory point cards left out of their points, the development bank as Unknown cards. Once the
        /// game has ended, everything is shown.
        /// </summary>
        public GameState View(GameState truth)
        {
            GameState view = new GameState(truth);
            if (truth.HasEnded) return view;

            for (int player = 0; player < view.Players.Length; player++)
            {
                if (!IsOther(player)) continue;
                PlayerState state = view.Players[player];
                HandBelief hand = _hands[player];
                state.ResourceCards = new CardSet<ResourceCardType>(hand.Known);
                state.ResourceCards.Add(ResourceCardType.Unknown, hand.Unknown);
                state.DevelopmentCards = AllUnknown(state.DevelopmentCards.Count());
                state.NewDevelopmentCards = AllUnknown(state.NewDevelopmentCards.Count());
                state.VictoryPoints.DevelopmentCardPoints = 0;
            }
            view.DevelopmentBank = AllUnknown(view.DevelopmentBank.Count());
            return view;
        }

        private static CardSet<DevelopmentCardType> AllUnknown(uint count)
        {
            CardSet<DevelopmentCardType> cards = new CardSet<DevelopmentCardType>();
            cards.Add(DevelopmentCardType.Unknown, count);
            return cards;
        }

        /// <summary>
        /// A full state that agrees with the view and with everything the observer saw: the unknown resource cards drawn
        /// from the cards that remain unexplained, the hidden development cards dealt at random, nobody at the winning
        /// points
        /// </summary>
        public GameState Determinize(GameState view, Random random)
        {
            GameState full = new GameState(view);
            if (view.HasEnded) return full;

            // Resources: the remaining cards are exactly the unknown ones, dealt so every card is of one of its hand's candidates
            List<int> slots = [];
            for (int player = 0; player < _hands.Length; player++)
                for (int i = 0; i < _hands[player].Unknown; i++) slots.Add(player);
            ResourceCardType[] dealt = DealResources(slots, Remaining(view.ResourceBank), random);
            for (int player = 0; player < _hands.Length; player++)
            {
                if (!IsOther(player)) continue;
                full.Players[player].ResourceCards = new CardSet<ResourceCardType>(_hands[player].Known);
            }
            for (int i = 0; i < slots.Count; i++) full.Players[slots[i]].ResourceCards.Add(dealt[i], 1);

            // Development cards: the hidden ones shuffled and dealt to the hands as large as in the view, the rest in the bank;
            // the cards bought this turn are some of a hand's cards
            List<DevelopmentCardType> deck = [];
            foreach (DevelopmentCardType type in CardSet<DevelopmentCardType>.Values)
                for (int i = 0; i < _hiddenDevelopment.Get(type); i++) deck.Add(type);
            for (int deal = 0; deal < MaxDeals; deal++)
            {
                Shuffle(deck, random);
                int next = 0;
                CardSet<DevelopmentCardType> Take(uint count)
                {
                    CardSet<DevelopmentCardType> cards = new CardSet<DevelopmentCardType>();
                    for (int i = 0; i < count; i++) cards.Add(deck[next++], 1);
                    return cards;
                }

                bool someoneWon = false;
                for (int player = 0; player < full.Players.Length; player++)
                {
                    if (!IsOther(player)) continue;
                    PlayerState state = full.Players[player];
                    int first = next;
                    state.DevelopmentCards = Take(view.Players[player].DevelopmentCards.Count());
                    state.NewDevelopmentCards = new CardSet<DevelopmentCardType>();
                    for (int i = 0; i < view.Players[player].NewDevelopmentCards.Count(); i++) state.NewDevelopmentCards.Add(deck[first + i], 1);
                    state.VictoryPoints.DevelopmentCardPoints = (byte)state.DevelopmentCards.Get(DevelopmentCardType.VictoryPoint);
                    someoneWon |= state.VictoryPoints.Total >= full.Settings.VictoryPoints;
                }
                full.DevelopmentBank = Take((uint)(deck.Count - next));

                // Victory point cards count at once: a player at the winning points would have won already
                if (!someoneWon) break;
            }
            return full;
        }

        /// <summary>
        /// One type per unknown card, together exactly the remaining cards, each a candidate of its hand: drawn card by card
        /// in proportion to what remains, backtracking where a choice leaves the rest without a fit
        /// </summary>
        private ResourceCardType[] DealResources(List<int> slots, CardSet<ResourceCardType> remaining, Random random)
        {
            ResourceCardType[] dealt = new ResourceCardType[slots.Count];
            uint[] left = new uint[CardSet<ResourceCardType>.Values.Count];
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes) left[(int)type] = remaining.Get(type);
            uint[,] room = new uint[_hands.Length, left.Length];
            for (int player = 0; player < _hands.Length; player++)
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes) room[player, (int)type] = _hands[player].Room(type);

            bool Deal(int slot)
            {
                if (slot == slots.Count) return true;
                int player = slots[slot];
                List<ResourceCardType> options = CardTypeExtensions.ResourceTypes.Where(t => left[(int)t] > 0 && room[player, (int)t] > 0).ToList();
                while (options.Count > 0)
                {
                    long total = options.Sum(t => (long)left[(int)t]);
                    long pick = random.NextInt64(total);
                    ResourceCardType chosen = options[^1];
                    foreach (ResourceCardType type in options)
                    {
                        if (pick < left[(int)type]) { chosen = type; break; }
                        pick -= left[(int)type];
                    }
                    left[(int)chosen]--;
                    room[player, (int)chosen]--;
                    dealt[slot] = chosen;
                    if (Deal(slot + 1)) return true;
                    left[(int)chosen]++;
                    room[player, (int)chosen]++;
                    options.Remove(chosen);
                }
                return false;
            }

            if (!Deal(0)) throw new InvalidOperationException($"Observer {Observer} has no way to deal the unknown cards");
            return dealt;
        }

        private static void Shuffle<T>(List<T> list, Random random)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }
        #endregion

        /// <summary>
        /// A tracker that followed the given moves from the start state
        /// </summary>
        public static BeliefTracker Replay(int observer, GameState start, IEnumerable<Action> actions)
        {
            BeliefTracker tracker = new BeliefTracker(observer, start);
            GameState state = new GameState(start);
            foreach (Action action in actions)
            {
                GameState before = new GameState(state);
                Action played = action.Clone();
                played.Apply(state);
                tracker.Observe(before, played, state);
            }
            return tracker;
        }

        public override bool Equals(object? obj) => obj is BeliefTracker other && Observer == other.Observer && _hands.SequenceEqual(other._hands)
            && _totals.Equals(other._totals) && _hiddenDevelopment.Equals(other._hiddenDevelopment) && _othersDevelopment == other._othersDevelopment;

        public override int GetHashCode()
        {
            HashCode hash = new HashCode();
            hash.Add(Observer);
            foreach (HandBelief hand in _hands) hash.Add(hand);
            hash.Add(_totals);
            hash.Add(_hiddenDevelopment);
            hash.Add(_othersDevelopment);
            return hash.ToHashCode();
        }
    }
}
