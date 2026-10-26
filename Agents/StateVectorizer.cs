using Common;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Agents
{
    public class StateVectorizer : IStateVectorizer
    {
        // Length of the feature vector produced by Vectorize for the classic 4-player board: the encoded state, then the
        // derived features, the ports and the endgame features (shared with DatasetProcessing/state_to_vector.py, see
        // EncodeDerived, EncodePorts and EncodeEndgame)
        public const int BaseFeatureDim = 1608;
        public const int DerivedFeatureDim = 156;
        public const int PortFeatureDim = 180;
        public const int EndgameFeatureDim = 21;

        // Layout of the value nets up to zjahe050: the base and derived features. The later blocks are appended, so it is the
        // start of the current layout.
        public const int LegacyFeatureDim = BaseFeatureDim + DerivedFeatureDim;
        public const int FeatureDim = LegacyFeatureDim + PortFeatureDim + EndgameFeatureDim;

        // Shared instances per layout (vectorizers hold no state while vectorizing)
        public static readonly StateVectorizer Current = new(FeatureDim);
        public static readonly StateVectorizer Legacy = new(LegacyFeatureDim);

        // The setup rounds count the round counter up to 2, so the first normal round is 2
        private const int FirstNormalRound = 2;

        // Enum values are cached, Enum.GetValues allocates a new array on every call
        private static readonly ResourceCardType[] ResourceTypes = Enum.GetValues<ResourceCardType>();
        private static readonly DevelopmentCardType[] DevelopmentTypes = Enum.GetValues<DevelopmentCardType>();
        private static readonly PortPrivileges[] PortPrivilegeFlags = Enum.GetValues<PortPrivileges>().Where(privilege => privilege != PortPrivileges.None).ToArray();

        protected EnumVocab<TurnState.RoundType> _roundTypeVocab;
        protected EnumVocab<Tile.TileType> _tileTypeVocab;
        protected EnumVocab<Intersection.BuildingType> _intersectionBuildingTypeVocab;
        protected EnumVocab<Edge.BuildingType> _edgeBuildingTypeVocab;

        private readonly int _featureDim;

        /// <summary>
        /// Number of features this vectorizer writes: FeatureDim, or LegacyFeatureDim for the older value nets
        /// </summary>
        public int Dimension => _featureDim;

        public StateVectorizer() : this(FeatureDim)
        {
        }

        public StateVectorizer(int featureDim)
        {
            if (featureDim != FeatureDim && featureDim != LegacyFeatureDim)
                throw new ArgumentException($"Supported layouts have {FeatureDim} or {LegacyFeatureDim} features, not {featureDim}.", nameof(featureDim));
            _featureDim = featureDim;
            _roundTypeVocab = new EnumVocab<TurnState.RoundType>(nameof(TurnState.RoundType));
            _tileTypeVocab = new EnumVocab<Tile.TileType>(nameof(Tile.TileType));
            _intersectionBuildingTypeVocab = new EnumVocab<Intersection.BuildingType>(nameof(Intersection.BuildingType));
            _edgeBuildingTypeVocab = new EnumVocab<Edge.BuildingType>(nameof(Edge.BuildingType));
        }

        /// <summary>
        /// The vectorizer for the layout the net was trained on (its input size)
        /// </summary>
        public static StateVectorizer For(Inference.StateValueNet net) => net.InputDim switch
        {
            FeatureDim => Current,
            LegacyFeatureDim => Legacy,
            _ => throw new ArgumentException($"The value net takes {net.InputDim} features, which no StateVectorizer layout has.", nameof(net))
        };

        int IStateVectorizer.FeatureDim => _featureDim;

        public DenseTensor<float> Vectorize(GameState state)
        {
            float[] features = new float[_featureDim];
            VectorizeInto(state, features);

            return new DenseTensor<float>(features, [1, _featureDim]);
        }

        /// <summary>
        /// Writes the feature vector of the state into the destination (e.g. a row of a batch tensor) without allocating.
        /// The destination must hold exactly Dimension values.
        /// </summary>
        public void VectorizeInto(GameState state, Span<float> destination)
        {
            if (state.Players.Length != 4)
                throw new NotImplementedException("Only 4 players are supported for now.");
            if (destination.Length != _featureDim)
                throw new ArgumentException($"The destination must hold exactly {_featureDim} values.", nameof(destination));

            // One-hot and multi-hot features only set their active entries
            destination.Clear();
            FeatureWriter writer = new(destination);

            EncodeSettings(state, ref writer);
            EncodeTurn(state, ref writer);
            EncodeBoard(state, ref writer);
            EncodeResourceCardSet(state.ResourceBank, ref writer);
            EncodeDevelopmentCardSet(state.DevelopmentBank, ref writer);
            EncodePlayers(state, ref writer);
            EncodeDerived(state, ref writer);
            if (_featureDim == FeatureDim)
            {
                EncodePorts(state, ref writer);
                EncodeEndgame(state, ref writer);
            }

            if (writer.Position != _featureDim)
                throw new InvalidOperationException($"The board produced {writer.Position} features instead of {_featureDim}.");
        }

        private static void EncodeSettings(GameState state, ref FeatureWriter writer)
        {
            writer.Add(state.Settings.RobberCardLimit);
            writer.Add(state.Settings.VictoryPoints);
        }

        private void EncodeTurn(GameState state, ref FeatureWriter writer)
        {
            // The round counter is not encoded by design: no rule depends on it, and a net trained with it used it as a
            // shortcut for game progress that doesn't carry over to stronger play (the board, hands and VP show progress)
            // Only the total of the last roll affects the game, so equivalent rolls get identical features.
            // Before anyone rolled, the last roll is a random placeholder and encoded as 0.
            writer.Add(HasRolled(state.Turn) ? state.Turn.LastRoll.Total : 0);

            writer.OneHot(state.Turn.PlayerIndex, state.Players.Length);
            writer.OneHot(_roundTypeVocab, state.Turn.TypeOfRound);
            writer.Add(state.Turn.MustRoll);

            for (int i = 0; i < state.Players.Length; i++)
            {
                writer.Add(state.Turn.AwaitedPlayerDiscards[i]);
            }

            writer.Add(state.Turn.MustMoveRobber);
            writer.Add(state.Turn.HasPlayedDevelopmentCard);
        }

        /// <summary>
        /// Whether the last roll is a real roll: not during the setup rounds and not in the first normal turn before its roll
        /// </summary>
        private static bool HasRolled(TurnState turn) => turn.TypeOfRound switch
        {
            TurnState.RoundType.FirstInitial or TurnState.RoundType.SecondInitial => false,
            TurnState.RoundType.Normal => !(turn.RoundCounter == FirstNormalRound && turn.PlayerIndex == 0 && turn.MustRoll),
            _ => true
        };

        private void EncodeBoard(GameState state, ref FeatureWriter writer)
        {
            if (state.Board.Robber == null)
                throw new InvalidOperationException("Board robber is null.");

            writer.Add(state.Board.Robber.X);
            writer.Add(state.Board.Robber.Y);
            writer.OneHot(_tileTypeVocab, state.Board.Robber.Type);
            writer.HexNumber(state.Board.Robber.Number);

            // Tiles
            // Row-major (y outer, x inner) to match HexMap.Values storage order and the Python vectorizer
            for (int y = 0; y < state.Board.Map.Height; y++)
            {
                for (int x = 0; x < state.Board.Map.Width; x++)
                {
                    var tile = state.Board.Map.GetTile(x, y);
                    if (tile == null)
                        throw new InvalidOperationException("Tile is null at position (" + x + ", " + y + ").");
                    // Coordinates not necessary, since position is already encoded in the order of the tiles
                    writer.OneHot(_tileTypeVocab, tile.Type);
                    writer.HexNumber(tile.Number);
                }
            }

            // Intersections
            for (int i = 0; i < state.Board.IntersectionCount; i++)
            {
                writer.OneHot(_intersectionBuildingTypeVocab, state.Board.GetBuilding(i));
                writer.NullableOwner(state.Board.GetBuildingOwner(i), state.Players.Length);
            }

            // Edges
            for (int i = 0; i < state.Board.EdgeCount; i++)
            {
                writer.OneHot(_edgeBuildingTypeVocab, state.Board.GetRoad(i));
                writer.NullableOwner(state.Board.GetRoadOwner(i), state.Players.Length);
            }
        }

        private static void EncodePlayers(GameState state, ref FeatureWriter writer)
        {
            foreach (var player in state.Players)
            {
                // Victory point trackers
                writer.Add(player.VictoryPoints.SettlementPoints);
                writer.Add(player.VictoryPoints.CityPoints);
                writer.Add(player.VictoryPoints.DevelopmentCardPoints);
                writer.Add(player.VictoryPoints.LongestRoadPoints);
                writer.Add(player.VictoryPoints.LargestArmyPoints);

                // Contested objective trackers
                writer.Add(player.PlayedKnights);
                writer.Add(player.LongestRoadLength);

                // Cards
                EncodeResourceCardSet(player.ResourceCards, ref writer);
                EncodeDevelopmentCardSet(player.DevelopmentCards, ref writer);
                EncodeDevelopmentCardSet(player.NewDevelopmentCards, ref writer);

                // Building stock
                writer.Add(player.BuildingStock.RemainingRoads);
                writer.Add(player.BuildingStock.RemainingSettlements);
                writer.Add(player.BuildingStock.RemainingCities);
                writer.Add(player.BuildingStock.FreeRoads);

                // Port privileges (multi-hot encoding, every flag except None has a feature indexed by its bit)
                Span<float> portPrivileges = writer.Reserve(PortPrivilegeFlags.Length);
                foreach (PortPrivileges privilege in PortPrivilegeFlags)
                {
                    if (player.PortPrivileges.HasFlag(privilege))
                        portPrivileges[BitOperations.TrailingZeroCount((int)privilege)] = 1.0f;
                }
            }
        }

        // Costs in resource order (Lumber, Brick, Wool, Grain, Ore) of a road, a settlement, a city and a development card
        private static readonly int[,] BuildCosts = { { 1, 1, 0, 0, 0 }, { 1, 1, 1, 1, 0 }, { 0, 0, 0, 2, 3 }, { 0, 0, 1, 1, 1 } };

        /// <summary>
        /// Compound quantities that an MLP finds hard to compute from the one-hot board (production needs ownership, building,
        /// tile type and number through the geometry), all per player. Integer arithmetic and float32 in the same order as
        /// derived_features in state_to_vector.py, which the parity fixtures compare exactly:
        /// production per resource without the robber's tile (expected cards per roll, cities count twice), production blocked
        /// by the robber, total production, trade rates, per build (road, settlement, city, development card) whether it can be
        /// built now, missing cards, missing cards after trading surplus and rounds until affordable, and race margins
        /// </summary>
        private static void EncodeDerived(GameState state, ref FeatureWriter writer)
        {
            const int players = 4, resources = 5, builds = 4;
            Board board = state.Board;
            int robber = board.GetRobberTileIndex() ?? -1;

            Span<int> productionPips = stackalloc int[players * resources];
            Span<int> blockedPips = stackalloc int[players * resources];
            Span<int> settlements = stackalloc int[players];
            Span<int> cornerTiles = stackalloc int[6];
            for (int i = 0; i < board.IntersectionCount; i++)
            {
                Intersection.BuildingType building = board.GetBuilding(i);
                if (building == Intersection.BuildingType.None) continue;

                int owner = board.GetBuildingOwner(i);
                int multiplier = building == Intersection.BuildingType.City ? 2 : 1;
                if (building == Intersection.BuildingType.Settlement) settlements[owner]++;

                int tileCount = board.Adjacency.GetTileIndices(i, cornerTiles);
                for (int t = 0; t < tileCount; t++)
                {
                    Tile tile = board.Map.Values[cornerTiles[t]];
                    int resource = ResourceIndex(tile.Type);
                    if (resource < 0) continue;

                    Span<int> target = cornerTiles[t] == robber ? blockedPips : productionPips;
                    target[owner * resources + resource] += multiplier * tile.YieldPoints;
                }
            }

            Span<float> total = stackalloc float[players];
            for (int p = 0; p < players; p++)
            {
                int pips = 0;
                for (int r = 0; r < resources; r++) pips += productionPips[p * resources + r];
                total[p] = pips / 36f;
            }

            foreach (int pips in productionPips) writer.Add(pips / 36f);
            foreach (int pips in blockedPips) writer.Add(pips / 36f);
            foreach (float production in total) writer.Add(production);

            // Trade rates: 2 with the resource's port, 3 with a generic port, otherwise 4
            Span<int> rates = stackalloc int[players * resources];
            for (int p = 0; p < players; p++)
            {
                PortPrivileges ports = state.Players[p].PortPrivileges;
                for (int r = 0; r < resources; r++)
                {
                    PortPrivileges resourcePort = (PortPrivileges)(1 << (r + 1));
                    rates[p * resources + r] = ports.HasFlag(resourcePort) ? 2 : ports.HasFlag(PortPrivileges.GenericThreeToOne) ? 3 : 4;
                    writer.Add(rates[p * resources + r]);
                }
            }

            // Affordability per player and build
            uint developmentLeft = 0;
            foreach (DevelopmentCardType development in DevelopmentTypes)
                if (development != DevelopmentCardType.Unknown) developmentLeft += state.DevelopmentBank.Get(development);

            Span<int> missing = stackalloc int[players * builds];
            Span<int> missingTraded = stackalloc int[players * builds];
            for (int p = 0; p < players; p++)
            {
                PlayerState player = state.Players[p];
                for (int b = 0; b < builds; b++)
                {
                    int lacking = 0, tradeable = 0;
                    for (int r = 0; r < resources; r++)
                    {
                        int held = (int)player.ResourceCards.Get((ResourceCardType)(r + 1));
                        lacking += Math.Max(BuildCosts[b, r] - held, 0);
                        tradeable += Math.Max(held - BuildCosts[b, r], 0) / rates[p * resources + r];
                    }
                    missing[p * builds + b] = lacking;
                    missingTraded[p * builds + b] = Math.Max(lacking - tradeable, 0);
                }
            }

            for (int p = 0; p < players; p++)
            {
                BuildingStock stock = state.Players[p].BuildingStock;
                writer.Add(missingTraded[p * builds + 0] == 0 && stock.RemainingRoads > 0);
                writer.Add(missingTraded[p * builds + 1] == 0 && stock.RemainingSettlements > 0);
                writer.Add(missingTraded[p * builds + 2] == 0 && stock.RemainingCities > 0 && settlements[p] > 0);
                writer.Add(missingTraded[p * builds + 3] == 0 && developmentLeft > 0);
            }
            foreach (int lacking in missing) writer.Add(lacking);
            foreach (int lacking in missingTraded) writer.Add(lacking);
            for (int p = 0; p < players; p++)
                for (int b = 0; b < builds; b++)
                    writer.Add(Math.Min(missingTraded[p * builds + b] / Math.Max(4f * total[p], 0.05f), 20f));

            // Race margins against the best other player
            Span<int> victoryPoints = stackalloc int[players];
            Span<int> roads = stackalloc int[players];
            Span<int> knights = stackalloc int[players];
            for (int p = 0; p < players; p++)
            {
                var points = state.Players[p].VictoryPoints;
                victoryPoints[p] = (int)(points.SettlementPoints + points.CityPoints + points.DevelopmentCardPoints + points.LongestRoadPoints + points.LargestArmyPoints);
                roads[p] = (int)state.Players[p].LongestRoadLength;
                knights[p] = (int)state.Players[p].PlayedKnights;
            }

            for (int p = 0; p < players; p++)
            {
                int handSize = 0;
                for (int r = 0; r < resources; r++) handSize += (int)state.Players[p].ResourceCards.Get((ResourceCardType)(r + 1));

                writer.Add(victoryPoints[p]);
                writer.Add(Lead(victoryPoints, p));
                writer.Add(state.Settings.VictoryPoints - victoryPoints[p]);
                writer.Add(Lead(roads, p));
                writer.Add(Lead(knights, p));
                writer.Add(handSize);
                writer.Add(handSize > state.Settings.RobberCardLimit);
            }
        }

        private static readonly int PortTypeCount = Enum.GetValues<Port.TradeType>().Length;

        /// <summary>
        /// Per coastal intersection (fewer than three land tiles, in index order) the one-hot type of the port serving it
        /// (Port.TradeType), all 0 without a port. Same as port_features in state_to_vector.py.
        /// </summary>
        private static void EncodePorts(GameState state, ref FeatureWriter writer)
        {
            Span<float> features = writer.Reserve(PortFeatureDim);
            foreach (int position in PortFeaturePositions.GetValue(state.Board.Ports, _ => ComputePortFeaturePositions(state.Board)))
                features[position] = 1f;
        }

        // Ports never change and are shared by all copies of a board, so their feature positions are computed once per map
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<List<Port>, int[]> PortFeaturePositions = new();

        private static int[] ComputePortFeaturePositions(Board board)
        {
            Span<int> coastalIndex = stackalloc int[board.IntersectionCount];
            Span<int> tiles = stackalloc int[6];
            int coastal = 0;
            for (int i = 0; i < board.IntersectionCount; i++)
                coastalIndex[i] = board.Adjacency.GetTileIndices(i, tiles) < 3 ? coastal++ : -1;
            if (coastal * PortTypeCount != PortFeatureDim)
                throw new InvalidOperationException($"The board has {coastal} coastal intersections instead of {PortFeatureDim / PortTypeCount}.");

            List<int> positions = new(2 * board.Ports.Count);
            foreach (Port port in board.Ports)
            {
                // The port's edge between its water tile and the land, as for the port privileges (see SettlementAction)
                Tile sea = board.Map.GetTile(port.AnchorTileX, port.AnchorTileY);
                Tile land = board.Adjacency.GetTile(sea, port.AnchorDirection)!;
                Edge edge = board.Adjacency.GetEdge(land, port.AnchorDirection.Mirror())!;
                (int top, int bottom) = board.Adjacency.GetIntersectionIndices(edge.Index);
                positions.Add(coastalIndex[top] * PortTypeCount + (int)port.Type);
                positions.Add(coastalIndex[bottom] * PortTypeCount + (int)port.Type);
            }
            return positions.ToArray();
        }

        /// <summary>
        /// Per player: settlement spots and road spots (Board.GetSettlementSites, GetRoadSites), the victory points the player
        /// can gain this turn at most with settlements and cities paid from its cards (trading surplus at its rates), whether
        /// that wins, and the knight lead counting the knight cards in the hand; then the share of victory point cards among
        /// the development cards in the bank. Integer arithmetic and float32 as in endgame_features in state_to_vector.py.
        /// </summary>
        private static void EncodeEndgame(GameState state, ref FeatureWriter writer)
        {
            const int players = 4, resources = 5;
            Board board = state.Board;

            Span<int> settlements = stackalloc int[players];
            for (int i = 0; i < board.IntersectionCount; i++)
                if (board.GetBuilding(i) == Intersection.BuildingType.Settlement) settlements[board.GetBuildingOwner(i)]++;

            UInt128 freeSites = board.GetFreeBuildingSites();
            Span<int> hand = stackalloc int[resources];
            Span<int> rates = stackalloc int[resources];
            for (int p = 0; p < players; p++)
            {
                PlayerState player = state.Players[p];
                int settlementSpots = (int)UInt128.PopCount(board.GetRoadEndMask(p) & freeSites);
                int roadSpots = (int)UInt128.PopCount(board.GetRoadSites(p));

                int handSize = 0;
                for (int r = 0; r < resources; r++)
                {
                    hand[r] = (int)player.ResourceCards.Get((ResourceCardType)(r + 1));
                    handSize += hand[r];
                    PortPrivileges resourcePort = (PortPrivileges)(1 << (r + 1));
                    rates[r] = player.PortPrivileges.HasFlag(resourcePort) ? 2 : player.PortPrivileges.HasFlag(PortPrivileges.GenericThreeToOne) ? 3 : 4;
                }

                // Settlements s and cities c the cards pay for, trading surplus as for "missing after trading". A build costs
                // at least 4 cards, and what more of one build doesn't pay for, more of it never pays for either.
                int maxSettlements = Math.Min(settlementSpots, (int)player.BuildingStock.RemainingSettlements);
                int maxCitiesStock = (int)player.BuildingStock.RemainingCities;
                int gain = 0;
                for (int s = 0; s <= maxSettlements && handSize >= 4; s++)
                {
                    if (!Affordable(hand, rates, s, 0)) break;
                    gain = Math.Max(gain, s);
                    for (int c = 1; c <= Math.Min(maxCitiesStock, settlements[p] + s) && Affordable(hand, rates, s, c); c++)
                        gain = Math.Max(gain, s + c);
                }

                var points = player.VictoryPoints;
                int victoryPoints = (int)(points.SettlementPoints + points.CityPoints + points.DevelopmentCardPoints + points.LongestRoadPoints + points.LargestArmyPoints);
                int bestOtherKnights = int.MinValue;
                for (int other = 0; other < players; other++)
                    if (other != p) bestOtherKnights = Math.Max(bestOtherKnights, (int)state.Players[other].PlayedKnights);

                writer.Add(settlementSpots);
                writer.Add(roadSpots);
                writer.Add(gain);
                writer.Add(victoryPoints + gain >= state.Settings.VictoryPoints);
                writer.Add((int)player.PlayedKnights + (int)player.DevelopmentCards.Get(DevelopmentCardType.Knight) - bestOtherKnights);
            }

            uint developmentLeft = 0;
            foreach (DevelopmentCardType development in DevelopmentTypes)
                if (development != DevelopmentCardType.Unknown) developmentLeft += state.DevelopmentBank.Get(development);
            writer.Add(developmentLeft > 0 ? (float)state.DevelopmentBank.Get(DevelopmentCardType.VictoryPoint) / developmentLeft : 0f);
        }

        /// <summary>
        /// Whether the cards pay for s settlements and c cities, trading surplus at the rates
        /// </summary>
        private static bool Affordable(ReadOnlySpan<int> hand, ReadOnlySpan<int> rates, int settlements, int cities)
        {
            int lacking = 0, tradeable = 0;
            for (int r = 0; r < hand.Length; r++)
            {
                int cost = settlements * BuildCosts[1, r] + cities * BuildCosts[2, r];
                lacking += Math.Max(cost - hand[r], 0);
                tradeable += Math.Max(hand[r] - cost, 0) / rates[r];
            }
            return lacking <= tradeable;
        }

        private static int Lead(ReadOnlySpan<int> values, int player)
        {
            int bestOther = int.MinValue;
            for (int p = 0; p < values.Length; p++)
                if (p != player) bestOther = Math.Max(bestOther, values[p]);
            return values[player] - bestOther;
        }

        private static int ResourceIndex(Tile.TileType type) => type switch
        {
            Tile.TileType.Lumber => 0,
            Tile.TileType.Brick => 1,
            Tile.TileType.Wool => 2,
            Tile.TileType.Grain => 3,
            Tile.TileType.Ore => 4,
            _ => -1
        };

        private static void EncodeResourceCardSet(CardSet<ResourceCardType> cardSet, ref FeatureWriter writer)
        {
            foreach (var resource in ResourceTypes)
            {
                writer.Add(cardSet.Get(resource));
            }
        }

        private static void EncodeDevelopmentCardSet(CardSet<DevelopmentCardType> cardSet, ref FeatureWriter writer)
        {
            foreach (var development in DevelopmentTypes)
            {
                writer.Add(cardSet.Get(development));
            }
        }

        /// <summary>
        /// Appends features to a zero-initialized span
        /// </summary>
        private ref struct FeatureWriter
        {
            private readonly Span<float> _features;
            public int Position { get; private set; }

            public FeatureWriter(Span<float> features)
            {
                _features = features;
                Position = 0;
            }

            public void Add(float value)
            {
                _features[Position++] = value;
            }

            public void Add(bool value)
            {
                Add(value ? 1.0f : 0.0f);
            }

            /// <summary>
            /// Reserves the next features (left at zero) and returns them for writing
            /// </summary>
            public Span<float> Reserve(int count)
            {
                Span<float> reserved = _features.Slice(Position, count);
                Position += count;
                return reserved;
            }

            public void OneHot(int index, int size)
            {
                Reserve(size)[index] = 1.0f;
            }

            public void OneHot<TEnum>(EnumVocab<TEnum> vocab, TEnum value) where TEnum : struct, Enum
            {
                vocab.WriteOneHot(value, Reserve(vocab.Size));
            }

            /// <summary>
            /// Owner one-hot with an extra leading slot for "no owner" (-1)
            /// </summary>
            public void NullableOwner(sbyte owner, int playerCount)
            {
                OneHot(owner + 1, playerCount + 1);
            }

            public void HexNumber(byte? number)
            {
                if (!number.HasValue)
                {
                    Reserve(2);
                    return;
                }

                // 2..12 auf 0..1 abbilden
                Add(((float)number.Value - 2f) / 10f);
                Add(1f);
            }
        }
    }
}
