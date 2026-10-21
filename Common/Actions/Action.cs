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
                .Where(property => property.Name != "History" && property.Name != nameof(ActionType) && property.GetIndexParameters().Length == 0);
            IEnumerable<MemberInfo> fields = type.GetFields(BindingFlags.Public | BindingFlags.Instance);

            return properties.Concat(fields).ToArray();
        }

        public virtual void Apply(GameState state)
        {
            if (!IsValidFor(state)) throw new InvalidOperationException();
        }

        public abstract void Revert(GameState state);

        public abstract bool IsValidFor(GameState state);

        public override string ToString()
        {
            return $"{GetType().Name}: P{PlayerIndex}";
        }
    }
}
