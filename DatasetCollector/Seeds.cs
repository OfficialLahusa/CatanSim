namespace DatasetCollector
{
    /// <summary>
    /// Seeds derived from the run seed: one per match (it reproduces the match and is the sample's group ID), one per sample
    /// and one per playout. Derived with SplitMix64 (Steele, Lea, Flood), whose outputs are well mixed even for consecutive
    /// indices, so runs with different run seeds don't share matches.
    /// </summary>
    public static class Seeds
    {
        public static ulong Derive(ulong parent, ulong index)
        {
            ulong z = parent + (index + 1) * 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        /// <summary>
        /// The seed for System.Random, folded from all 64 bits
        /// </summary>
        public static int ToInt32(ulong seed)
        {
            return (int)(seed ^ (seed >> 32));
        }

        public static ulong NewRunSeed()
        {
            return (ulong)Random.Shared.NextInt64(long.MinValue, long.MaxValue);
        }
    }
}
