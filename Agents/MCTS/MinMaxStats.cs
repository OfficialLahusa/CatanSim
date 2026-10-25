using System;

namespace Agents.MCTS
{
    /// <summary>
    /// Running minimum and maximum of the node values seen in a tree, used to normalize Q values to [0, 1]
    /// before the UCT exploration term is added (as in MuZero). This keeps selection scale-invariant:
    /// a player whose win probability is ~0.01 across all moves still gets a usable exploitation signal.
    /// Lock-free: every backpropagation updates it for each node on its path, from all threads, but values rarely
    /// widen the range, so most updates only read.
    /// </summary>
    public class MinMaxStats
    {
        private long _minBits = BitConverter.DoubleToInt64Bits(double.PositiveInfinity);
        private long _maxBits = BitConverter.DoubleToInt64Bits(double.NegativeInfinity);

        public void Update(double value)
        {
            UpdateBound(ref _minBits, value, lower: true);
            UpdateBound(ref _maxBits, value, lower: false);
        }

        private static void UpdateBound(ref long bits, double value, bool lower)
        {
            long current = Volatile.Read(ref bits);
            while (lower ? value < BitConverter.Int64BitsToDouble(current) : value > BitConverter.Int64BitsToDouble(current))
            {
                long seen = Interlocked.CompareExchange(ref bits, BitConverter.DoubleToInt64Bits(value), current);
                if (seen == current) return;
                current = seen;
            }
        }

        public double Normalize(double value)
        {
            return GetRange().Normalize(value);
        }

        /// <summary>
        /// The current range, to normalize several values with one read
        /// </summary>
        public Range GetRange()
        {
            return new Range(BitConverter.Int64BitsToDouble(Volatile.Read(ref _minBits)), BitConverter.Int64BitsToDouble(Volatile.Read(ref _maxBits)));
        }

        public readonly record struct Range(double Min, double Max)
        {
            public double Normalize(double value)
            {
                // Without a meaningful range yet (fewer than two distinct values), leave the value as is
                if (Max <= Min) return value;

                return (value - Min) / (Max - Min);
            }
        }
    }
}
