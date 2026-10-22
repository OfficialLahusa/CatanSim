using Microsoft.VisualBasic;
using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Windows.Markup;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;

namespace Common
{
    public enum ResourceCardType : byte
    {
        Unknown = 0,
        Lumber = 1,
        Brick = 2,
        Wool = 3,
        Grain = 4,
        Ore = 5
    };
    public enum DevelopmentCardType : byte
    {
        Unknown = 0,
        Knight = 1,
        RoadBuilding = 2,
        YearOfPlenty = 3,
        Monopoly = 4,
        VictoryPoint = 5
    };

    public class CardSet<T> where T : Enum
    {
        // Card count indexed by type
        private uint[] _cards;

        public static readonly IReadOnlyList<T> Values;

        static CardSet() {
            Values = (IReadOnlyList<T>)Enum.GetValues(typeof(T));
        }

        public CardSet()
        {
            _cards = new uint[Enum.GetValues(typeof(T)).Length];
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public CardSet(CardSet<T> copy)
        {
            _cards = (uint[])copy._cards.Clone();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt(T val)
        {
            // Performance comparison https://stackoverflow.com/questions/16960555
            //return Convert.ToInt32(Convert.ChangeType(val, val.GetTypeCode()));
            //return (byte)(ValueType)val;
            //return Unsafe.As<T, byte>(ref val);
            return val.GetHashCode();
        }

        public T Draw(bool consume = true)
        {
            // Weighted random draw
            uint totalWeight = Count();

            // No cards in set
            if(totalWeight == 0)
            {
                throw new InvalidOperationException();
            }

            uint draw = (uint)Utils.Random.Next((int)totalWeight);

            foreach(T type in Values)
            {
                uint typeWeight = _cards[ToInt(type)];

                if (draw < typeWeight)
                {
                    // Remove drawn card from stock
                    if(consume)
                    {
                        _cards[ToInt(type)] = typeWeight - 1;
                    }

                    return type;
                }

                draw -= typeWeight;
            }

            throw new InvalidOperationException();
        }

        public uint Get(T type)
        {
            return _cards[ToInt(type)];
        }

        public uint Count()
        {
            uint sum = 0;

            foreach(uint amount in _cards)
            {
                sum += amount;
            }

            return sum;
        }

        public void Add(T type, uint amount)
        {
            _cards[ToInt(type)] += amount;
        }

        public void Remove(T type, uint amount)
        {
            if (!Contains(type, amount)) throw new InvalidOperationException("Removal amount exceeds available card count");

            _cards[ToInt(type)] -= amount;
        }

        public bool Contains(T type, uint amount = 1)
        {
            return _cards[ToInt(type)] >= amount;
        }

        public bool Contains(CardSet<T> subset)
        {
            foreach(T cardType in Values)
            {
                if (subset.Get(cardType) > Get(cardType)) return false;
            }

            return true;
        }

        public void Add(CardSet<T> subset)
        {
            foreach (T cardType in Values)
            {
                _cards[ToInt(cardType)] += subset.Get(cardType);
            }
        }

        public void Remove(CardSet<T> subset)
        {
            if (!Contains(subset)) throw new InvalidOperationException("Removal amount exceeds available card count");

            foreach (T cardType in Values)
            {
                _cards[ToInt(cardType)] -= subset.Get(cardType);
            }
        }

        public override bool Equals(object? obj)
        {
            return obj is CardSet<T> set
                && _cards.SequenceEqual(set._cards);
        }

        public override int GetHashCode()
        {
            HashCode hash = new HashCode();

            foreach(uint cardAmount in _cards)
            {
                hash.Add(cardAmount);
            }

            return hash.ToHashCode();
        }

        public sealed class Converter : IYamlTypeConverter
        {
            public bool Accepts(Type type)
            {
                return type == typeof(CardSet<T>);
            }

            public object? ReadYaml(IParser parser, Type type, ObjectDeserializer rootDeserializer)
            {
                parser.Expect<MappingStart>();

                var result = new CardSet<T>();

                while (!parser.Accept<MappingEnd>())
                {
                    var keyScalar = parser.Expect<Scalar>().Value ?? string.Empty;

                    // Determine enum value from key (allow name or numeric)
                    T enumKey;
                    object? enumObj = null;
                    if (!Enum.TryParse(typeof(T), keyScalar, ignoreCase: true, out enumObj) || enumObj == null)
                    {
                        if (int.TryParse(keyScalar, out int numericKey))
                        {
                            enumKey = (T)Enum.ToObject(typeof(T), numericKey);
                        }
                        else
                        {
                            throw new YamlException($"Invalid enum key '{keyScalar}' for type {typeof(T).Name}");
                        }
                    }
                    else
                    {
                        enumKey = (T)enumObj;
                    }

                    // Deserialize the value (use rootDeserializer to consume next node)
                    object? deserializedValue = rootDeserializer(typeof(object));
                    uint amount = 0;
                    if (deserializedValue != null)
                    {
                        try
                        {
                            amount = Convert.ToUInt32(deserializedValue);
                        }
                        catch (Exception ex)
                        {
                            throw new YamlException($"Invalid amount for '{keyScalar}': {ex.Message}");
                        }
                    }

                    result._cards[ToInt(enumKey)] = amount;
                }

                parser.Expect<MappingEnd>();
                return result;
            }


            public void WriteYaml(IEmitter emitter, object? value, Type type, ObjectSerializer serializer)
            {
                emitter.Emit(new MappingStart());

                var set = (CardSet<T>)value!;

                foreach (T cardType in Values)
                {
                    uint amount = set._cards[ToInt(cardType)];
                    // Write all entries (including zero) to be explicit; adjust if you prefer sparse output.
                    emitter.Emit(new Scalar(cardType.ToString()));
                    serializer(amount);
                }

                emitter.Emit(new MappingEnd());
            }
        }
    }

    public static class CardTypeExtensions
    {
        /// <summary>
        /// The five resource types, without Unknown
        /// </summary>
        public static readonly ResourceCardType[] ResourceTypes =
            [ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain, ResourceCardType.Ore];

        public static string GetName(this ResourceCardType type)
        {
            return type switch
            {
                ResourceCardType.Unknown => "Unknown",
                ResourceCardType.Lumber => "Lumber",
                ResourceCardType.Brick => "Brick",
                ResourceCardType.Wool => "Wool",
                ResourceCardType.Grain => "Grain",
                ResourceCardType.Ore => "Ore",
                _ => throw new InvalidOperationException(),
            };
        }

        public static string GetName(this DevelopmentCardType type)
        {
            return type switch
            {
                DevelopmentCardType.Unknown => "Unknown",
                DevelopmentCardType.Knight => "Knight",
                DevelopmentCardType.RoadBuilding => "Road Building",
                DevelopmentCardType.YearOfPlenty => "Year Of Plenty",
                DevelopmentCardType.Monopoly => "Monopoly",
                DevelopmentCardType.VictoryPoint => "Victory Point",
                _ => throw new InvalidOperationException(),
            };
        }

        public static string GetAbbreviation(this ResourceCardType type)
        {
            return type switch
            {
                ResourceCardType.Unknown => "?",
                ResourceCardType.Lumber => "L",
                ResourceCardType.Brick => "B",
                ResourceCardType.Wool => "W",
                ResourceCardType.Grain => "G",
                ResourceCardType.Ore => "O",
                _ => throw new InvalidOperationException()
            };
        }

        public static string GetAbbreviation(this DevelopmentCardType type)
        {
            return type switch
            {
                DevelopmentCardType.Unknown => "?",
                DevelopmentCardType.Knight => "K",
                DevelopmentCardType.RoadBuilding => "R",
                DevelopmentCardType.YearOfPlenty => "Y",
                DevelopmentCardType.Monopoly => "M",
                DevelopmentCardType.VictoryPoint => "V",
                _ => throw new InvalidOperationException()
            };
        }
    }
}
