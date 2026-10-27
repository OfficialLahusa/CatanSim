using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents.Inference
{
    /// <summary>
    /// Encodes a move as a few tokens of a shared vocabulary for the policy head: one token for the move type plus tokens
    /// for its parameters, shared between types where they mean the same (a settlement and a city at intersection 17 share
    /// the token of intersection 17). A token can carry a weight (the count of a discarded resource), everything else has
    /// weight 1. Moves with random outcomes (rolling, buying a development card, the robber's steal) are one move each.
    /// The robbed player is encoded relative to the mover, so the encoding stays the same under player rotation.
    ///
    /// The policy logit of a move is the query of its type (from the value net, MoveTypeCount x Dim numbers per state)
    /// times the weighted sum of its token embeddings (a constant table of VocabularySize x Dim numbers): attention with
    /// one query per move type, scored on the CPU for the legal moves of a node.
    /// </summary>
    public static class ActionEncoding
    {
        public enum MoveType
        {
            EndTurn, Roll, BuyDevelopmentCard, Knight, RoadBuilding,
            Road, Settlement, City, FirstInitialSettlement, SecondInitialSettlement, FirstInitialRoad, SecondInitialRoad,
            Robber, Monopoly, YearOfPlenty, FourToOneTrade, ThreeToOneTrade, TwoToOneTrade, Discard
        }

        public const int MoveTypeCount = 19;
        public const int IntersectionCount = 54, EdgeCount = 72, TileCount = 49, ResourceCount = 5;
        // None, then the next, second and third player after the mover
        public const int VictimCount = 4;
        public const int MaxTokens = 1 + ResourceCount;

        public const int IntersectionStart = MoveTypeCount;
        public const int EdgeStart = IntersectionStart + IntersectionCount;
        public const int TileStart = EdgeStart + EdgeCount;
        public const int VictimStart = TileStart + TileCount;
        public const int GivenResourceStart = VictimStart + VictimCount;
        public const int ReceivedResourceStart = GivenResourceStart + ResourceCount;
        public const int DiscardedResourceStart = ReceivedResourceStart + ResourceCount;
        public const int VocabularySize = DiscardedResourceStart + ResourceCount;

        /// <summary>
        /// Token ranges by name, written into the metadata of self-play runs so the training reads the layout from the data
        /// </summary>
        public static IReadOnlyDictionary<string, int[]> Layout => new Dictionary<string, int[]>
        {
            ["move_types"] = [0, MoveTypeCount],
            ["intersections"] = [IntersectionStart, IntersectionCount],
            ["edges"] = [EdgeStart, EdgeCount],
            ["tiles"] = [TileStart, TileCount],
            ["victims"] = [VictimStart, VictimCount],
            ["given_resources"] = [GivenResourceStart, ResourceCount],
            ["received_resources"] = [ReceivedResourceStart, ResourceCount],
            ["discarded_resources"] = [DiscardedResourceStart, ResourceCount],
            ["vocabulary"] = [0, VocabularySize],
            ["max_tokens"] = [MaxTokens],
        };

        /// <summary>
        /// Writes the move's tokens and weights (the move type first) and returns how many there are
        /// </summary>
        public static int Encode(Action action, Span<short> tokens, Span<float> weights)
        {
            int count = 0;

            switch (action)
            {
                case EndTurnAction: Put(tokens, weights, ref count, (int)MoveType.EndTurn); break;
                case RollAction: Put(tokens, weights, ref count, (int)MoveType.Roll); break;
                case BuyDevelopmentCardAction: Put(tokens, weights, ref count, (int)MoveType.BuyDevelopmentCard); break;
                case KnightAction: Put(tokens, weights, ref count, (int)MoveType.Knight); break;
                case RoadBuildingAction: Put(tokens, weights, ref count, (int)MoveType.RoadBuilding); break;
                case RoadAction road: Put(tokens, weights, ref count, (int)MoveType.Road); Put(tokens, weights, ref count, EdgeStart + road.EdgeIndex); break;
                case SettlementAction settlement: Put(tokens, weights, ref count, (int)MoveType.Settlement); Put(tokens, weights, ref count, IntersectionStart + settlement.IntersectionIndex); break;
                case CityAction city: Put(tokens, weights, ref count, (int)MoveType.City); Put(tokens, weights, ref count, IntersectionStart + city.IntersectionIndex); break;
                case FirstInitialSettlementAction first: Put(tokens, weights, ref count, (int)MoveType.FirstInitialSettlement); Put(tokens, weights, ref count, IntersectionStart + first.IntersectionIndex); break;
                case SecondInitialSettlementAction second: Put(tokens, weights, ref count, (int)MoveType.SecondInitialSettlement); Put(tokens, weights, ref count, IntersectionStart + second.IntersectionIndex); break;
                case FirstInitialRoadAction first: Put(tokens, weights, ref count, (int)MoveType.FirstInitialRoad); Put(tokens, weights, ref count, EdgeStart + first.EdgeIndex); break;
                case SecondInitialRoadAction second: Put(tokens, weights, ref count, (int)MoveType.SecondInitialRoad); Put(tokens, weights, ref count, EdgeStart + second.EdgeIndex); break;
                case RobberAction robber:
                    Put(tokens, weights, ref count, (int)MoveType.Robber);
                    Put(tokens, weights, ref count, TileStart + robber.TargetTileIndex);
                    Put(tokens, weights, ref count, VictimStart + (robber.TargetPlayerIndex is sbyte victim ? (victim - robber.PlayerIndex + 4) % 4 : 0));
                    break;
                case MonopolyAction monopoly: Put(tokens, weights, ref count, (int)MoveType.Monopoly); Put(tokens, weights, ref count, Received(monopoly.ChosenType)); break;
                case YearOfPlentyAction plenty:
                    Put(tokens, weights, ref count, (int)MoveType.YearOfPlenty);
                    if (plenty.SecondChoice == plenty.FirstChoice) Put(tokens, weights, ref count, Received(plenty.FirstChoice), 2);
                    else
                    {
                        Put(tokens, weights, ref count, Received(plenty.FirstChoice));
                        if (plenty.SecondChoice is ResourceCardType second) Put(tokens, weights, ref count, Received(second));
                    }
                    break;
                case FourToOneTradeAction trade: Put(tokens, weights, ref count, (int)MoveType.FourToOneTrade); Put(tokens, weights, ref count, Given(trade.InputType)); Put(tokens, weights, ref count, Received(trade.OutputType)); break;
                case ThreeToOneTradeAction trade: Put(tokens, weights, ref count, (int)MoveType.ThreeToOneTrade); Put(tokens, weights, ref count, Given(trade.InputType)); Put(tokens, weights, ref count, Received(trade.OutputType)); break;
                case TwoToOneTradeAction trade: Put(tokens, weights, ref count, (int)MoveType.TwoToOneTrade); Put(tokens, weights, ref count, Given(trade.InputType)); Put(tokens, weights, ref count, Received(trade.OutputType)); break;
                case DiscardAction discard:
                    Put(tokens, weights, ref count, (int)MoveType.Discard);
                    foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                    {
                        uint discarded = discard.SelectedCards.Get(type);
                        if (discarded > 0) Put(tokens, weights, ref count, DiscardedResourceStart + ResourceIndex(type), discarded);
                    }
                    break;
                default: throw new ArgumentException($"No policy encoding for {action.GetType().Name}", nameof(action));
            }
            return count;
        }

        private static void Put(Span<short> tokens, Span<float> weights, ref int count, int token, float weight = 1)
        {
            tokens[count] = (short)token;
            weights[count] = weight;
            count++;
        }

        private static int ResourceIndex(ResourceCardType type) => (int)type - (int)ResourceCardType.Lumber;
        private static int Given(ResourceCardType type) => GivenResourceStart + ResourceIndex(type);
        private static int Received(ResourceCardType type) => ReceivedResourceStart + ResourceIndex(type);

        /// <summary>
        /// Policy logits of the moves: the query of each move's type (queries: MoveTypeCount x dim) times the weighted sum of
        /// its token embeddings (embeddings: VocabularySize x dim)
        /// </summary>
        public static void Logits(ReadOnlySpan<float> queries, ReadOnlySpan<float> embeddings, int dim, IReadOnlyList<Action> actions, Span<float> logits)
        {
            Span<short> tokens = stackalloc short[MaxTokens];
            Span<float> weights = stackalloc float[MaxTokens];
            for (int a = 0; a < actions.Count; a++)
            {
                int count = Encode(actions[a], tokens, weights);
                ReadOnlySpan<float> query = queries.Slice(tokens[0] * dim, dim);
                float logit = 0;
                for (int t = 0; t < count; t++)
                {
                    ReadOnlySpan<float> embedding = embeddings.Slice(tokens[t] * dim, dim);
                    float dot = 0;
                    for (int i = 0; i < dim; i++) dot += query[i] * embedding[i];
                    logit += weights[t] * dot;
                }
                logits[a] = logit;
            }
        }

        /// <summary>
        /// Softmax of the logits in place
        /// </summary>
        public static void Softmax(Span<float> values)
        {
            if (values.IsEmpty) return;
            float max = float.NegativeInfinity;
            foreach (float value in values) max = Math.Max(max, value);
            float sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Exp(values[i] - max);
                sum += values[i];
            }
            for (int i = 0; i < values.Length; i++) values[i] /= sum;
        }
    }
}
