namespace DatasetCollector
{
    /// <summary>
    /// Finds the fastest thread count for the real settings: short time-limited runs into a temporary directory with all,
    /// 3/4, 1/2 and 1/4 of the logical cores, measured in playouts per second (matches still in progress at the end are
    /// abandoned). A warm-up run comes first, as the first run measured about 25% too slow (JIT, the server GC growing its
    /// heaps). Every candidate runs twice, in order and then in reverse order, so drift (heat, background load) affects all
    /// of them alike; runs on a desktop vary by about 8-15%.
    /// </summary>
    public static class ThreadCalibration
    {
        public static int[] Candidates(int logicalCores)
        {
            return new[] { logicalCores, logicalCores * 3 / 4, logicalCores / 2, logicalCores / 4 }
                .Where(threads => threads >= 1).Distinct().ToArray();
        }

        public static int FindBestThreadCount(CollectorSettings settings, TimeSpan timePerRun, TextWriter log)
        {
            int[] candidates = Candidates(Environment.ProcessorCount);
            Dictionary<int, List<double>> rates = candidates.ToDictionary(threads => threads, _ => new List<double>());

            log.WriteLine($"\nCalibrating the thread count: {string.Join(", ", candidates)} threads, {timePerRun.TotalSeconds:F0}s per run, each twice");
            log.WriteLine($"  warm-up: {Measure(settings with { Threads = candidates[0] }, timePerRun),8:F0} playouts/s (not counted)");
            foreach (int threads in candidates.Concat(candidates.Reverse()))
            {
                double rate = Measure(settings with { Threads = threads }, timePerRun);
                rates[threads].Add(rate);
                log.WriteLine($"  {threads,3} threads: {rate,8:F0} playouts/s");
            }

            int best = candidates.MaxBy(threads => rates[threads].Average());
            foreach (int threads in candidates)
                log.WriteLine($"  {threads,3} threads: mean {rates[threads].Average(),8:F0} playouts/s{(threads == best ? "  <= fastest" : "")}");
            return best;
        }

        private static double Measure(CollectorSettings settings, TimeSpan duration)
        {
            string directory = Path.Combine(Path.GetTempPath(), "catansim-calibration-" + Guid.NewGuid().ToString("N"));
            try
            {
                // Enough matches to last the whole run; the same run seed gives every candidate the same matches
                CollectorSettings runSettings = settings with { OutputDirectory = directory, Name = "calibration", Matches = int.MaxValue };
                using CancellationTokenSource cancellation = new(duration);
                return new DatasetCollector(runSettings, TextWriter.Null).Collect(cancellation.Token).PlayoutsPerSecond;
            }
            finally
            {
                if (Directory.Exists(directory))
                    Directory.Delete(directory, recursive: true);
            }
        }
    }
}
