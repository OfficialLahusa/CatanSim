using Common;

namespace Agents
{
    /// <summary>
    /// The encoding a net was trained on, for an agent of a seat: a net of the view layout gets the seat's view
    /// (ViewVectorizer), the others the StateVectorizer of their layout
    /// </summary>
    public static class Vectorizers
    {
        public static IStateVectorizer For(Inference.StateValueNet net, int seat) =>
            net.InputDim == ViewVectorizer.FeatureDim ? new ViewVectorizer { Observer = seat } : StateVectorizer.For(net);
    }

    /// <summary>
    /// The input of a belief-aware value net: a seat's view (see BeliefTracker) in the layout of the StateVectorizer, whose
    /// Unknown slots hold the cards the seat can't see, followed by what the seat's tracker knows beyond the view: per player
    /// and resource type how many of the unknown cards can be of it, and the hidden development cards by type. The net so
    /// needn't count cards from the history itself.
    ///
    /// A full state in the search (a leaf, see VectorizeInto) is encoded as the Observer would see it: its own cards, the
    /// others' resource cards (a determinization deals them), the others' development cards and the deck only as counts.
    /// </summary>
    public sealed class ViewVectorizer : IStateVectorizer
    {
        public const int BeliefFeatureDim = 4 * 5 + 5;
        public const int FeatureDim = StateVectorizer.FeatureDim + BeliefFeatureDim;

        private static readonly DevelopmentCardType[] DevelopmentTypes = CardSet<DevelopmentCardType>.Values.Where(t => t != DevelopmentCardType.Unknown).ToArray();

        int IStateVectorizer.FeatureDim => FeatureDim;

        /// <summary>
        /// The seat whose view a leaf is encoded as
        /// </summary>
        public int Observer { get; set; }

        /// <summary>
        /// A seat's view of the true state with what its tracker knows
        /// </summary>
        public static void VectorizeView(GameState truth, BeliefTracker tracker, Span<float> destination)
        {
            destination.Clear();
            StateVectorizer.Current.VectorizeInto(tracker.View(truth), destination[..StateVectorizer.FeatureDim]);

            Span<float> beliefs = destination[StateVectorizer.FeatureDim..];
            int at = 0;
            for (int player = 0; player < 4; player++)
            {
                HandBelief hand = tracker.Hand(player);
                foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes) beliefs[at++] = hand.Room(type);
            }
            foreach (DevelopmentCardType type in DevelopmentTypes) beliefs[at++] = tracker.HiddenDevelopmentCards.Get(type);
        }

        // A view of the leaf without copying it: its public parts are shared, the others' development cards and the deck are
        // counts in scratch sets. One per thread, as the search encodes leaves on all cores.
        [ThreadStatic] private static Scratch? _scratch;

        private sealed class Scratch
        {
            public readonly GameState View;
            public readonly PlayerState[] Players;
            public readonly PlayerState[] Masked;
            public readonly CardSet<DevelopmentCardType> Bank = new();
            public readonly CardSet<DevelopmentCardType> Hidden = new();

            public Scratch(GameState template)
            {
                View = new GameState(template);
                Players = new PlayerState[template.Players.Length];
                Masked = template.Players.Select(_ => new PlayerState()).ToArray();
            }
        }

        /// <summary>
        /// A full state as the Observer would see it, with nothing unknown about the resource cards
        /// </summary>
        public void VectorizeInto(GameState state, Span<float> destination)
        {
            destination.Clear();
            Span<float> beliefs = destination[StateVectorizer.FeatureDim..];

            // The end of the game shows everything
            if (state.HasEnded)
            {
                StateVectorizer.Current.VectorizeInto(state, destination[..StateVectorizer.FeatureDim]);
                CardSet<DevelopmentCardType> hidden = new CardSet<DevelopmentCardType>(state.DevelopmentBank);
                for (int player = 0; player < state.Players.Length; player++)
                    if (player != Observer) hidden.Add(state.Players[player].DevelopmentCards);
                WriteHiddenDevelopment(hidden, beliefs);
                return;
            }

            Scratch scratch = _scratch ??= new Scratch(state);
            GameState view = scratch.View;
            view.Settings = state.Settings;
            view.Turn = state.Turn;
            view.Board = state.Board;
            view.ResourceBank = state.ResourceBank;
            scratch.Bank.Clear();
            scratch.Bank.Add(DevelopmentCardType.Unknown, state.DevelopmentBank.Count());
            view.DevelopmentBank = scratch.Bank;
            scratch.Hidden.Clear();
            scratch.Hidden.Add(state.DevelopmentBank);

            for (int player = 0; player < state.Players.Length; player++)
            {
                PlayerState truth = state.Players[player];
                if (player == Observer)
                {
                    scratch.Players[player] = truth;
                    continue;
                }

                // The same as BeliefTracker.View for a hand without unknown cards
                PlayerState masked = scratch.Masked[player];
                masked.VictoryPoints.SettlementPoints = truth.VictoryPoints.SettlementPoints;
                masked.VictoryPoints.CityPoints = truth.VictoryPoints.CityPoints;
                masked.VictoryPoints.DevelopmentCardPoints = 0;
                masked.VictoryPoints.LongestRoadPoints = truth.VictoryPoints.LongestRoadPoints;
                masked.VictoryPoints.LargestArmyPoints = truth.VictoryPoints.LargestArmyPoints;
                masked.PlayedKnights = truth.PlayedKnights;
                masked.LongestRoadLength = truth.LongestRoadLength;
                masked.ResourceCards = truth.ResourceCards;
                masked.DevelopmentCards.Clear();
                masked.DevelopmentCards.Add(DevelopmentCardType.Unknown, truth.DevelopmentCards.Count());
                masked.NewDevelopmentCards.Clear();
                masked.NewDevelopmentCards.Add(DevelopmentCardType.Unknown, truth.NewDevelopmentCards.Count());
                masked.BuildingStock = truth.BuildingStock;
                masked.PortPrivileges = truth.PortPrivileges;
                scratch.Players[player] = masked;
                scratch.Hidden.Add(truth.DevelopmentCards);
            }
            view.Players = scratch.Players;

            StateVectorizer.Current.VectorizeInto(view, destination[..StateVectorizer.FeatureDim]);
            WriteHiddenDevelopment(scratch.Hidden, beliefs);
        }

        /// <summary>
        /// The belief block of a leaf: no unknown resource cards (the room is 0), then the hidden development cards
        /// </summary>
        private static void WriteHiddenDevelopment(CardSet<DevelopmentCardType> hidden, Span<float> beliefs)
        {
            int at = 4 * 5;
            foreach (DevelopmentCardType type in DevelopmentTypes) beliefs[at++] = hidden.Get(type);
        }
    }
}
