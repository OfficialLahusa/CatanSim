using Common;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualBasic;
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

        public DenseTensor<float> Vectorize(GameState state, uint playedActionsCount)
        {
            List<float> features = new List<float>();
            
            if(state.Players.Length != 4)
                throw new NotImplementedException("Only 4 players are supported for now.");

            features.AddRange(EncodeSettings(state));
            features.AddRange(EncodeTurn(state));
            features.AddRange(EncodeBoard(state));
            features.AddRange(EncodeResourceCardSet(state.ResourceBank));
            features.AddRange(EncodeDevelopmentCardSet(state.DevelopmentBank));
            features.AddRange(EncodePlayers(state));
            features.Add((float)playedActionsCount);

            DenseTensor<float> tensor = new DenseTensor<float>(features.ToArray(), new int[] { 1, features.Count });

            //Console.WriteLine(tensor);
            //Console.WriteLine("Tensor shape: " + tensor.Dimensions[0] + " x " + tensor.Dimensions[1]);

            return tensor;
        }

        protected List<float> EncodeSettings(GameState state)
        {
            List<float> features = new List<float>();

            features.Add((float)state.Settings.RobberCardLimit);
            features.Add((float)state.Settings.VictoryPoints);

            return features;
        }

        protected List<float> EncodeTurn(GameState state)
        {
            List<float> features = new List<float>();

            features.Add((float)state.Turn.RoundCounter);

            features.Add((float)state.Turn.LastRoll.First);
            features.Add((float)state.Turn.LastRoll.Second);

            float[] playerIdx = new float[state.Players.Length];
            playerIdx[state.Turn.PlayerIndex] = 1.0f;
            features.AddRange(playerIdx);

            features.AddRange(_roundTypeVocab.OneHot(state.Turn.TypeOfRound));
            features.Add(BoolFeature(state.Turn.MustRoll));

            float[] awaitedDiscards = new float[state.Players.Length];
            for(int i = 0; i < state.Players.Length; i++)
            {
                awaitedDiscards[i] = BoolFeature(state.Turn.AwaitedPlayerDiscards[i]);
            }
            features.AddRange(awaitedDiscards);

            features.Add(BoolFeature(state.Turn.MustMoveRobber));
            features.Add(BoolFeature(state.Turn.HasPlayedDevelopmentCard));

            return features;
        }

        protected List<float> EncodeBoard(GameState state)
        {
            List<float> features = new List<float>();

            if (state.Board.Robber == null)
                throw new InvalidOperationException("Board robber is null.");

            features.Add((float)state.Board.Robber.X);
            features.Add((float)state.Board.Robber.Y);
            features.AddRange(_tileTypeVocab.OneHot(state.Board.Robber.Type));
            features.AddRange(HexNumberFeature(state.Board.Robber.Number));

            // Tiles
            for (int x = 0; x < state.Board.Map.Width; x++)
            {
                for (int y = 0; y < state.Board.Map.Height; y++)
                {
                    var tile = state.Board.Map.GetTile(x, y);
                    if (tile == null)
                        throw new InvalidOperationException("Tile is null at position (" + x + ", " + y + ").");
                    // Coordinates not necessary, since position is already encoded in the order of the tiles
                    features.AddRange(_tileTypeVocab.OneHot(tile.Type));
                    features.AddRange(HexNumberFeature(tile.Number));
                }
            }

            // Intersections
            for (int i = 0; i < state.Board.Intersections.Count; i++)
            {
                Intersection intersection = state.Board.Intersections[i];

                features.AddRange(_intersectionBuildingTypeVocab.OneHot(intersection.Building));
                features.AddRange(NullableOwnerOneHot(intersection.Owner, (uint)state.Players.Length));
            }

            // Edges
            for (int i = 0; i < state.Board.Edges.Count; i++)
            {
                Edge edge = state.Board.Edges[i];

                features.AddRange(_edgeBuildingTypeVocab.OneHot(edge.Building));
                features.AddRange(NullableOwnerOneHot(edge.Owner, (uint)state.Players.Length));
            }

            return features;
        }

        protected List<float> EncodePlayers(GameState state)
        {
            List<float> features = new List<float>();
            foreach (var player in state.Players)
            {
                // Victory point trackers
                features.Add((float)player.VictoryPoints.SettlementPoints);
                features.Add((float)player.VictoryPoints.CityPoints);
                features.Add((float)player.VictoryPoints.DevelopmentCardPoints);
                features.Add((float)player.VictoryPoints.LongestRoadPoints);
                features.Add((float)player.VictoryPoints.LargestArmyPoints);

                // Contested objective trackers
                features.Add((float)player.PlayedKnights);
                features.Add((float)player.LongestRoadLength);

                // Cards
                features.AddRange(EncodeResourceCardSet(player.ResourceCards));
                features.AddRange(EncodeDevelopmentCardSet(player.DevelopmentCards));
                features.AddRange(EncodeDevelopmentCardSet(player.NewDevelopmentCards));

                // Building stock
                features.Add((float)player.BuildingStock.RemainingRoads);
                features.Add((float)player.BuildingStock.RemainingSettlements);
                features.Add((float)player.BuildingStock.RemainingCities);
                features.Add((float)player.BuildingStock.FreeRoads);

                // Port privileges (multi-hot encoding)
                float[] portPrivileges = new float[Enum.GetValues<PortPrivileges>().Length - 1]; // Every flag except None has a feature
                foreach (PortPrivileges privilege in Enum.GetValues<PortPrivileges>())
                {
                    if (privilege == PortPrivileges.None)
                        continue;

                    portPrivileges[BitOperations.TrailingZeroCount((int)privilege)] = player.PortPrivileges.HasFlag(privilege) ? 1.0f : 0.0f;
                }
                features.AddRange(portPrivileges);
            }
            return features;
        }

        protected List<float> EncodeResourceCardSet(CardSet<ResourceCardType> cardSet)
        {
            List<float> features = new List<float>();
            foreach (var resource in Enum.GetValues<ResourceCardType>())
            {
                features.Add((float)cardSet.Get(resource));
            }
            return features;
        }

        protected List<float> EncodeDevelopmentCardSet(CardSet<DevelopmentCardType> cardSet)
        {
            List<float> features = new List<float>();
            foreach (var development in Enum.GetValues<DevelopmentCardType>())
            {
                features.Add((float)cardSet.Get(development));
            }
            return features;
        }

        protected float[] NullableOwnerOneHot(sbyte owner, uint playerCount)
        {
            float[] ownerOneHot = new float[playerCount + 1];
            ownerOneHot[owner+1] = 1.0f;
            return ownerOneHot;
        }

        protected float BoolFeature(bool val)
        {
            return val ? 1.0f : 0.0f;
        }

        protected float[] HexNumberFeature(byte? number)
        {
            if (!number.HasValue)
                return [0f, 0f];

            // 2..12 auf 0..1 abbilden
            return [((float)number.Value - 2f) / 10f, 1f];
        }
    }
}
