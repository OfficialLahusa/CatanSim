using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Agents
{
    public class EnumVocab<TEnum> where TEnum : struct, Enum
    {
        public string Name { get; }
        public IReadOnlyList<TEnum> Values { get; }
        public int Size => Values.Count;

        private readonly Dictionary<TEnum, int> _index;

        public EnumVocab(string name)
        {
            Name = name;

            // Cache all enum values and their 0-indexed vector positions
            Values = Enum.GetValues<TEnum>();
            _index = Values
                .Select((val, i) => (val, i))
                .ToDictionary(x => x.val, x => x.i);
        }

        public float[] OneHot(TEnum? value)
        {
            var vec = new float[Size];
            WriteOneHot(value, vec);
            return vec;
        }

        /// <summary>
        /// Sets the entry of the value in a zero-initialized destination of length Size
        /// </summary>
        public void WriteOneHot(TEnum? value, Span<float> destination)
        {
            // Handles null or unmapped enum values safely
            if (!value.HasValue || !_index.TryGetValue(value.Value, out int idx))
            {
                string keyName = value.HasValue ? value.Value.ToString() : "None";

                Console.WriteLine(
                    $"[state_yaml_to_tensor] WARNING: unseen value '{keyName}' for " +
                    $"field '{Name}'; encoding as all-zero. Consider adding it to the vocabulary."
                );
                return;
            }

            destination[idx] = 1.0f;
        }
    }
}
