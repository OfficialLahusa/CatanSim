using Common;
using System.Globalization;

namespace DatasetCollector
{
    /// <summary>
    /// Regenerates the maps of collected matches from their seeds, to extend datasets of an older feature layout with
    /// features of the map that their vectors lack (the ports). A match seed always gives the same match with the run's
    /// agent population and desert option, also when the match was started over on a new map.
    /// </summary>
    public static class MapRegeneration
    {
        /// <summary>
        /// Reads one match seed per line and writes one line per seed: the seed, the port types in the order of the
        /// generator's slots and the tiles in map storage order as type:number (0 without a number), so the caller can
        /// check the map against the stored vectors (see DatasetProcessing/migrate_states_1764_to_1965.py)
        /// </summary>
        public static int Run(string seedsPath, string outputPath, CollectorSettings settings, TextWriter log)
        {
            ulong[] seeds = File.ReadLines(seedsPath)
                .Where(line => line.Length > 0)
                .Select(line => ulong.Parse(line, CultureInfo.InvariantCulture))
                .ToArray();
            log.WriteLine($"Regenerating the maps of {seeds.Length} matches with {settings.Threads} threads");

            string[] lines = new string[seeds.Length];
            int done = 0;
            Parallel.For(0, seeds.Length, new ParallelOptions { MaxDegreeOfParallelism = settings.Threads }, i =>
            {
                (GameState baseState, _, _) = DatasetCollector.PlayMatch(seeds[i], i, settings.Agents, settings.CenterDesert);
                Board board = baseState.Board;
                string ports = string.Join(" ", board.Ports.Select(port => port.Type.ToString()));
                string tiles = string.Join(" ", board.Map.Values.Select(tile => $"{tile.Type}:{tile.Number ?? 0}"));
                lines[i] = $"{seeds[i]},{ports},{tiles}";

                int finished = Interlocked.Increment(ref done);
                if (finished % 50_000 == 0) log.WriteLine($"{finished}/{seeds.Length}");
            });

            File.WriteAllLines(outputPath, lines);
            log.WriteLine($"Saved the maps to \"{outputPath}\".");
            return 0;
        }
    }
}
