using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Common
{
    /// <summary>
    /// Random source that can be used from many threads at once. Each thread draws from its own Random, seeded from a
    /// master generator on first use, so parallel playouts don't contend on a shared lock.
    /// Single-threaded use is fully reproducible for a given seed. With several threads, each thread's sequence depends
    /// on the order in which the threads first draw.
    /// Dice have a stream of their own, so two games from the same seed roll the same dice even after their decisions
    /// differ (common random numbers for comparing moves).
    /// </summary>
    public class ThreadSafeRandom
    {
        private readonly object _seedLock = new object();
        private readonly Random _seedSource;
        private readonly Random _diceSeedSource;
        private readonly ThreadLocal<Random> _random;
        private readonly ThreadLocal<Random> _dice;

        public ThreadSafeRandom(int seed)
        {
            _seedSource = new Random(seed);
            _diceSeedSource = new Random(DiceSeed(seed));
            _random = new ThreadLocal<Random>(CreateThreadRandom);
            _dice = new ThreadLocal<Random>(CreateThreadDice);
        }

        public ThreadSafeRandom()
            : this(Guid.NewGuid().GetHashCode())
        { }

        private Random CreateThreadRandom()
        {
            lock (_seedLock) return new Random(_seedSource.Next());
        }

        private Random CreateThreadDice()
        {
            lock (_seedLock) return new Random(_diceSeedSource.Next());
        }

        // The dice seed is derived from the seed, so that one seed determines both streams
        private static int DiceSeed(int seed)
        {
            return unchecked((int)((uint)seed * 2654435761u) ^ 0x6A09E667);
        }

        /// <summary>
        /// Replaces the random source of the calling thread with one from the seed, so everything this thread draws from
        /// now on is reproducible (e.g. a whole match played on one thread)
        /// </summary>
        public void SeedCurrentThread(int seed)
        {
            _random.Value = new Random(seed);
            _dice.Value = new Random(DiceSeed(seed));
        }

        /// <summary>
        /// Makes the calling thread draw from the given random source (except dice) until it is replaced again
        /// </summary>
        public void UseOnCurrentThread(Random random)
        {
            _random.Value = random;
        }

        /// <summary>
        /// A die roll from 1 to 6, from the dice stream
        /// </summary>
        public int NextDie()
        {
            return _dice.Value!.Next(6) + 1;
        }

        public int Next(int min, int max)
        {
            return _random.Value!.Next(min, max);
        }

        public int Next(int max)
        {
            return _random.Value!.Next(max);
        }

        public int Next()
        {
            return _random.Value!.Next();
        }

        public double NextDouble()
        {
            return _random.Value!.NextDouble();
        }
    }
}
