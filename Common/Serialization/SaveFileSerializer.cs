using Common.Actions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
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
                typeof(CancelTradeAction),
                typeof(CityAction),
                typeof(ConfirmTradeAction),
                typeof(CounterTradeAction),
                typeof(DiscardAction),
                typeof(EndTurnAction),
                typeof(FirstInitialRoadAction),
                typeof(FirstInitialSettlementAction),
                typeof(FourToOneTradeAction),
                typeof(KnightAction),
                typeof(MonopolyAction),
                typeof(OfferTradeAction),
                typeof(RespondToTradeAction),
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
            return Deserialize(yaml, checkVersion: true);
        }

        /// <summary>
        /// Without the version check only for SaveFileUpgrader, on older documents with the current layout
        /// </summary>
        internal static SaveFile Deserialize(string yaml, bool checkVersion)
        {
            SaveFile save = Deserializer.Deserialize<SaveFile>(yaml);

            // Older files are rejected instead of being replayed with silently wrong action histories
            // (SaveFileUpgrader can convert them first)
            if (checkVersion && save.Version != SaveFile.CurrentVersion)
                throw new UnsupportedSaveFileVersionException(save.Version);

            // Reattach AdjacencyMatrix to board after deserialization
            save.GameState.Board.Adjacency.Attach(save.GameState.Board);

            return save;
        }

        /// <summary>
        /// Reads a save file and resolves its HistoryReference: the referenced file's played actions (in the same directory,
        /// resolved recursively) are put before the file's own, and the reference is cleared, so the result has the whole
        /// history. If the referenced file is missing or doesn't lead to this state, only the file's own actions are kept and
        /// LoadWarning explains why.
        /// </summary>
        public static SaveFile Load(string path)
        {
            return Load(Path.GetFullPath(path), new HashSet<string>(StringComparer.OrdinalIgnoreCase));
        }

        private static SaveFile Load(string path, HashSet<string> visited)
        {
            SaveFile save = Deserialize(File.ReadAllText(path));
            if (save.HistoryReference == null) return save;

            visited.Add(path);
            string referencePath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(path)!, save.HistoryReference));
            string reference = save.HistoryReference;
            string fileName = Path.GetFileName(path);

            if (!File.Exists(referencePath))
            {
                save.LoadWarning = $"\"{fileName}\" is a sibling state: it stores only its own last move and takes the actions before it from "
                    + $"\"{reference}\", the state it branches off, which has to be in the same directory. That file was not found, so the state "
                    + "is loaded with only its last move as history and can't be undone further. Keep sibling files (named like r1_s2_c3.yaml) "
                    + "together with the file they reference (r1_s2.yaml) when copying samples.";
                return save;
            }
            if (visited.Contains(referencePath))
                throw new InvalidDataException($"The history references of \"{fileName}\" form a cycle through \"{reference}\".");

            SaveFile referenced = Load(referencePath, visited);

            // The file's own actions, undone from its state, have to lead back to the referenced state
            GameState undone = new GameState(save.GameState);
            for (int i = save.PlayedActions.Count - 1; i >= 0; i--)
                save.PlayedActions[i].Clone().Revert(undone);
            if (!undone.Equals(referenced.GameState))
            {
                save.LoadWarning = $"\"{fileName}\" takes the actions before its last move from \"{reference}\", but that file holds a different "
                    + "state than the one this move was played from (was it replaced, or copied from another dataset?). The state is loaded with "
                    + "only its last move as history.";
                return save;
            }

            save.PlayedActions = [.. referenced.PlayedActions, .. save.PlayedActions];
            save.HistoryReference = null;
            save.LoadWarning = referenced.LoadWarning;
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
