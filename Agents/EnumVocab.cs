using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models
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

            // Handles null or unmapped enum values safely
            if (!value.HasValue || !_index.TryGetValue(value.Value, out int idx))
            {
                string keyName = value.HasValue ? value.Value.ToString() : "None";

                Console.WriteLine(
                    $"[state_yaml_to_tensor] WARNING: unseen value '{keyName}' for " +
                    $"field '{Name}'; encoding as all-zero. Consider adding it to the vocabulary."
                );
                return vec;
            }

            vec[idx] = 1.0f;
            return vec;
        }
    }
}
