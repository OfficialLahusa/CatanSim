using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;

namespace Common.Actions
{
    public abstract class Action
    {
        // Public parameters of each action type that define its decision (history and serialization helpers excluded)
        private static readonly ConcurrentDictionary<Type, MemberInfo[]> _decisionMembers = new();

        // Index of the player performing the action
        public sbyte PlayerIndex { get; init; }

        // For serialization
        public string ActionType { get => GetType().Name; }

        /// <summary>
        /// Road Building roads the player had not placed yet when taking this action. They are placed right after the card
        /// or not at all, so any other action returns them to the stock (only written to save files when non-zero).
        /// </summary>
        [YamlMember(DefaultValuesHandling = DefaultValuesHandling.OmitDefaults)]
        public uint ForfeitedFreeRoads { get; private set; }

        public Action(sbyte playerIndex)
        {
            PlayerIndex = playerIndex;
        }

        /// <summary>
        /// Shallow copy of this action including its stored history (which holds the outcome of output randomness actions).
        /// Applying the copy only replaces the copy's history, so the original can be shared, e.g. by a search tree.
        /// </summary>
        public Action Clone()
        {
            return (Action)MemberwiseClone();
        }

        /// <summary>
        /// Whether the other action is the same decision: same type, same player and same parameters.
        /// Stored history, and with it any resolved output randomness, is ignored.
        /// </summary>
        public bool IsSameDecision(Action other)
        {
            if (other.GetType() != GetType()) return false;

            foreach (MemberInfo member in _decisionMembers.GetOrAdd(GetType(), GetDecisionMembers))
            {
                object? ownValue = member is PropertyInfo property ? property.GetValue(this) : ((FieldInfo)member).GetValue(this);
                object? otherValue = member is PropertyInfo otherProperty ? otherProperty.GetValue(other) : ((FieldInfo)member).GetValue(other);

                if (!Equals(ownValue, otherValue)) return false;
            }

            return true;
        }

        private static MemberInfo[] GetDecisionMembers(Type type)
        {
            IEnumerable<MemberInfo> properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(property => property.Name != "History" && property.Name != nameof(ActionType) && property.Name != nameof(ForfeitedFreeRoads)
                    && property.GetIndexParameters().Length == 0);
            IEnumerable<MemberInfo> fields = type.GetFields(BindingFlags.Public | BindingFlags.Instance);

            return properties.Concat(fields).ToArray();
        }

        public virtual void Apply(GameState state)
        {
            if (!IsValidFor(state)) throw new InvalidOperationException();
        }

        public abstract void Revert(GameState state);

        /// <summary>
        /// Ends the placement of Road Building roads: unplaced ones go back to the player's stock.
        /// Called by every action that can be taken while free roads are left, right after validation.
        /// </summary>
        protected void ForfeitFreeRoads(GameState state)
        {
            BuildingStock stock = state.Players[PlayerIndex].BuildingStock;
            ForfeitedFreeRoads = stock.FreeRoads;
            stock.RemainingRoads += stock.FreeRoads;
            stock.FreeRoads = 0;
        }

        /// <summary>
        /// Undoes ForfeitFreeRoads, at the end of Revert
        /// </summary>
        protected void ReturnForfeitedFreeRoads(GameState state)
        {
            BuildingStock stock = state.Players[PlayerIndex].BuildingStock;
            stock.RemainingRoads -= ForfeitedFreeRoads;
            stock.FreeRoads += ForfeitedFreeRoads;
        }

        public abstract bool IsValidFor(GameState state);

        public override string ToString()
        {
            return $"{GetType().Name}: P{PlayerIndex}";
        }
    }
}
