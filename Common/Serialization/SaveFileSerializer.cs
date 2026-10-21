using Common.Actions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.BufferedDeserialization;
using YamlDotNet.Serialization.NamingConventions;
using YamlDotNet.Serialization.NodeDeserializers;
using Action = Common.Actions.Action;

namespace Common.Serialization
{
    public class SaveFileSerializer
    {
        protected static readonly IDeserializer Deserializer;
        protected static readonly ISerializer Serializer;
        static SaveFileSerializer()
        {
            List<Type> actionTypes = [
                typeof(BuyDevelopmentCardAction),
                typeof(CityAction),
                typeof(DiscardAction),
                typeof(EndTurnAction),
                typeof(FirstInitialRoadAction),
                typeof(FirstInitialSettlementAction),
                typeof(FourToOneTradeAction),
                typeof(KnightAction),
                typeof(MonopolyAction),
                typeof(RoadAction),
                typeof(RoadBuildingAction),
                typeof(RobberAction),
                typeof(RollAction),
                typeof(SecondInitialRoadAction),
                typeof(SecondInitialSettlementAction),
                typeof(SettlementAction),
                typeof(ThreeToOneTradeAction),
                typeof(TwoToOneTradeAction),
                typeof(YearOfPlentyAction)
            ];
            Dictionary<string, Type> actionTypeMapping = new Dictionary<string, Type>(StringComparer.Ordinal);
            foreach (Type actionType in actionTypes)
            {
                actionTypeMapping[actionType.Name] = actionType;
            }

            ActionTypeDiscriminator actionTypeDiscriminator = new ActionTypeDiscriminator();

            Deserializer = new DeserializerBuilder()
                    .WithTypeConverter(new AdjacencyMatrix.Converter())
                    .WithTypeConverter(new CardSet<ResourceCardType>.Converter())
                    .WithTypeConverter(new CardSet<DevelopmentCardType>.Converter())
                    /*.WithTypeDiscriminatingNodeDeserializer(options =>
                    {
                        options.AddKeyValueTypeDiscriminator<Action>("ActionType", actionTypeMapping);
                    })*/
                    .WithNodeDeserializer(
                        inner => new AbstractNodeNodeTypeResolver(inner, actionTypeDiscriminator),
                        s => s.InsteadOf<ObjectNodeDeserializer>())
                    .IgnoreUnmatchedProperties()
                    .EnablePrivateConstructors()
                    .Build();

            Serializer = new SerializerBuilder()
                    .WithTypeConverter(new AdjacencyMatrix.Converter())
                    .WithTypeConverter(new CardSet<ResourceCardType>.Converter())
                    .WithTypeConverter(new CardSet<DevelopmentCardType>.Converter())
                    .EnablePrivateConstructors()
                    .DisableAliases()
                    .Build();
        }

        public static string Serialize(SaveFile saveFile)
        {
            return Serializer.Serialize(saveFile);
        }

        public static SaveFile Deserialize(string yaml)
        {
            SaveFile save = Deserializer.Deserialize<SaveFile>(yaml);

            // Reattach AdjacencyMatrix to board after deserialization
            save.GameState.Board.Adjacency.Attach(save.GameState.Board);

            // Re-anchor the robber to the tile instance inside the map
            // (aliases are disabled, so it was deserialized as a separate copy of that tile)
            Board board = save.GameState.Board;
            if (board.Robber != null)
                board.Robber = board.Map.GetTile(board.Robber.X, board.Robber.Y);

            return save;
        }

        public static string Serialize<T>(T obj)
        {
            if (typeof(T) == typeof(SaveFile))
                throw new InvalidOperationException("Use Serialize(SaveFile) method for SaveFile serialization.");

            return Serializer.Serialize(obj);
        }

        public static T Deserialize<T>(string yaml)
        {
            if (typeof(T) == typeof(SaveFile))
                throw new InvalidOperationException("Use Deserialize(string) method for SaveFile deserialization.");

            return Deserializer.Deserialize<T>(yaml);
        }
    }
}
