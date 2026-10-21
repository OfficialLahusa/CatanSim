using Common;

namespace DatasetCollector
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("CettlersOfSatan Dataset Collector");
            Console.WriteLine("=================================");

            // Playouts keep scaling up to all logical cores (per-thread random sources, server GC)
            uint threadCount = ReadDefaultedUInt32("Thread count", (uint)Environment.ProcessorCount);
            uint matchCount = ReadDefaultedUInt32("Match count", 100u);
            uint samplesPerMatch = ReadDefaultedUInt32("Samples per match", 5u);
            uint playoutsPerSample = ReadDefaultedUInt32("Playouts per sample", 1000u);
            int seed = ReadDefaultedInt32("Seed", Guid.NewGuid().GetHashCode());

            Utils.Random = new ThreadSafeRandom(seed);

            DatasetCollector collector = new DatasetCollector(threadCount, matchCount, samplesPerMatch, playoutsPerSample);
            collector.Collect();
        }

        private static uint ReadDefaultedUInt32(string prompt, uint defaultValue)
        {
            Console.Write($"{prompt} (default {defaultValue}): ");
            string? input = Console.ReadLine();
            if (string.IsNullOrEmpty(input))
            {
                return defaultValue;
            }
            return Convert.ToUInt32(input);
        }

        private static int ReadDefaultedInt32(string prompt, int defaultValue)
        {
            Console.Write($"{prompt} (default {defaultValue}): ");
            string? input = Console.ReadLine();
            if (string.IsNullOrEmpty(input))
            {
                return defaultValue;
            }
            return Convert.ToInt32(input);
        }
    }
}