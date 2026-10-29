using Common.Actions;
using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;
using Action = Common.Actions.Action;

namespace Common.Serialization
{
    public class ActionTypeDiscriminator : ITypeDiscriminator
    {
        public const string TargetKey = "ActionType";
        private readonly string targetKey;
        private readonly Dictionary<string, Type> typeLookup;

        public ActionTypeDiscriminator()
        {
            targetKey = TargetKey;

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

            typeLookup = new Dictionary<string, Type>();

            foreach (Type actionType in actionTypes)
            {
                typeLookup.Add(actionType.Name, actionType);
            }
        }
        public Type BaseType => typeof(Action);

        public bool TryResolve(ParsingEventBuffer buffer, out Type suggestedType)
        {
            if (buffer.TryFindMappingEntry(
                scalar => targetKey == scalar.Value,
                out Scalar key,
                out ParsingEvent value))
            {
                // read the value of the kind key
                if (value is Scalar valueScalar)
                {
                    suggestedType = CheckName(valueScalar.Value);

                    return true;
                }
                else
                {
                    FailEmpty();
                }
            }

            // we could not find our key, thus we could not determine correct child type
            suggestedType = null;
            return false;
        }


        private void FailEmpty()
        {
            throw new Exception($"Could not determin expectation type, {targetKey} has an empty value");
        }

        private Type CheckName(string value)
        {
            if (typeLookup.TryGetValue(value, out var childType))
            {
                return childType;
            }

            var known = string.Join(", ", typeLookup.Keys);
            throw new Exception($"Could not match `{targetKey}: {value} to a known expectation. Expecting one of: {known}");
        }
    }
}
