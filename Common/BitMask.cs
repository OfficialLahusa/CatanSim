using System.Numerics;

namespace Common
{
    /// <summary>
    /// Sets of intersection or edge indices as 128-bit masks (bit i = index i)
    /// </summary>
    public static class BitMask
    {
        public const int Capacity = 128;

        public static UInt128 Bit(int idx) => UInt128.One << idx;

        /// <summary>
        /// Mask with the lowest count bits set
        /// </summary>
        public static UInt128 Lowest(int count) => count >= Capacity ? UInt128.MaxValue : (UInt128.One << count) - 1;

        public static bool Contains(UInt128 mask, int idx) => (mask & Bit(idx)) != UInt128.Zero;

        /// <summary>
        /// Indices of the set bits in ascending order
        /// </summary>
        public static Enumerator Of(UInt128 mask) => new(mask);

        public struct Enumerator
        {
            private ulong _low;
            private ulong _high;

            public Enumerator(UInt128 mask)
            {
                _low = (ulong)mask;
                _high = (ulong)(mask >> 64);
                Current = -1;
            }

            public int Current { get; private set; }

            public readonly Enumerator GetEnumerator() => this;

            public bool MoveNext()
            {
                if (_low != 0)
                {
                    Current = BitOperations.TrailingZeroCount(_low);
                    _low &= _low - 1;
                    return true;
                }

                if (_high != 0)
                {
                    Current = 64 + BitOperations.TrailingZeroCount(_high);
                    _high &= _high - 1;
                    return true;
                }

                return false;
            }
        }
    }
}
