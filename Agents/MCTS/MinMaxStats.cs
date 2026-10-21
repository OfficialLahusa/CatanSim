using System;

namespace Agents.MCTS
{
    /// <summary>
    /// Running minimum and maximum of the node values seen in a tree, used to normalize Q values to [0, 1]
    /// before the UCT exploration term is added (as in MuZero). This keeps selection scale-invariant:
    /// a player whose win probability is ~0.01 across all moves still gets a usable exploitation signal.
    /// </summary>
    public class MinMaxStats
    {
        private readonly object _lock = new object();
        private double _min = double.PositiveInfinity;
        private double _max = double.NegativeInfinity;

        public void Update(double value)
        {
            lock (_lock)
            {
                if (value < _min) _min = value;
                if (value > _max) _max = value;
            }
        }

        public double Normalize(double value)
        {
            double min, max;
            lock (_lock)
            {
                min = _min;
                max = _max;
            }

            // Without a meaningful range yet (fewer than two distinct values), leave the value as is
            if (max <= min) return value;

            return (value - min) / (max - min);
        }
    }
}
