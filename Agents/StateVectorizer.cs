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
    public class StateVectorizer
    {
        // Length of the feature vector produced by Vectorize for the classic 4-player board
        public const int FeatureDim = 1609;

        // Enum values are cached, Enum.GetValues allocates a new array on every call
        private static readonly ResourceCardType[] ResourceTypes = Enum.GetValues<ResourceCardType>();
        private static readonly DevelopmentCardType[] DevelopmentTypes = Enum.GetValues<DevelopmentCardType>();
        private static readonly PortPrivileges[] PortPrivilegeFlags = Enum.GetValues<PortPrivileges>().Where(privilege => privilege != PortPrivileges.None).ToArray();

        protected EnumVocab<TurnState.RoundType> _roundTypeVocab;
        protected EnumVocab<Tile.TileType> _tileTypeVocab;
        protected EnumVocab<Intersection.BuildingType> _intersectionBuildingTypeVocab;
        protected EnumVocab<Edge.BuildingType> _edgeBuildingTypeVocab;

        public StateVectorizer()
        {
            _roundTypeVocab = new EnumVocab<TurnState.RoundType>(nameof(TurnState.RoundType));
            _tileTypeVocab = new EnumVocab<Tile.TileType>(nameof(Tile.TileType));
            _intersectionBuildingTypeVocab = new EnumVocab<Intersection.BuildingType>(nameof(Intersection.BuildingType));
            _edgeBuildingTypeVocab = new EnumVocab<Edge.BuildingType>(nameof(Edge.BuildingType));
        }

        public DenseTensor<float> Vectorize(GameState state)
        {
            float[] features = new float[FeatureDim];
            VectorizeInto(state, features);

            return new DenseTensor<float>(features, [1, FeatureDim]);
        }

        /// <summary>
        /// Writes the feature vector of the state into the destination (e.g. a row of a batch tensor) without allocating.
        /// The destination must hold exactly FeatureDim values.
        /// </summary>
        public void VectorizeInto(GameState state, Span<float> destination)
        {
            if (state.Players.Length != 4)
                throw new NotImplementedException("Only 4 players are supported for now.");
            if (destination.Length != FeatureDim)
                throw new ArgumentException($"The destination must hold exactly {FeatureDim} values.", nameof(destination));

            // One-hot and multi-hot features only set their active entries
            destination.Clear();
            FeatureWriter writer = new(destination);

            EncodeSettings(state, ref writer);
            EncodeTurn(state, ref writer);
            EncodeBoard(state, ref writer);
            EncodeResourceCardSet(state.ResourceBank, ref writer);
            EncodeDevelopmentCardSet(state.DevelopmentBank, ref writer);
            EncodePlayers(state, ref writer);

            if (writer.Position != FeatureDim)
                throw new InvalidOperationException($"The board produced {writer.Position} features instead of {FeatureDim}.");
        }

        private static void EncodeSettings(GameState state, ref FeatureWriter writer)
        {
            writer.Add(state.Settings.RobberCardLimit);
            writer.Add(state.Settings.VictoryPoints);
        }

        private void EncodeTurn(GameState state, ref FeatureWriter writer)
        {
            writer.Add(state.Turn.RoundCounter);

            // Only the total of the last roll affects the game, so equivalent rolls get identical features
            writer.Add(state.Turn.LastRoll.Total);

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
