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
    /// </summary>
    public class ThreadSafeRandom
    {
        private readonly object _seedLock = new object();
        private readonly Random _seedSource;
        private readonly ThreadLocal<Random> _random;

        public ThreadSafeRandom(int seed)
        {
            _seedSource = new Random(seed);
            _random = new ThreadLocal<Random>(CreateThreadRandom);
        }

        public ThreadSafeRandom()
            : this(Guid.NewGuid().GetHashCode())
        { }

        private Random CreateThreadRandom()
        {
            lock (_seedLock) return new Random(_seedSource.Next());
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
    }
}
