using Agents;
using Common;
using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;
using Action = Common.Actions.Action;

namespace SelfPlayCollector
{
    /// <summary>
    /// The input of a belief-aware value net (ViewVectorizer) for every recorded position of a run: views.f16 holds one row of
    /// little-endian float16 features per line of samples.csv, in the same order, each the position as its actor saw it.
    /// Written afterwards by replaying the games with a BeliefTracker per seat.
    /// </summary>
    internal static class ViewFeatureWriter
    {
        public const string FileName = "views.f16";

        /// <summary>
        /// Writes the view features of a run. Returns the number of positions.
        /// </summary>
        public static long ConvertRun(string directory, TextWriter log)
        {
            string path = Path.Combine(directory, FileName);
            if (File.Exists(path)) throw new InvalidOperationException($"{directory} already has view features.");

            using JsonDocument metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "metadata.json")));
            bool centerDesert = metadata.RootElement.GetProperty("settings").GetProperty("center_desert").GetBoolean();

            var records = new Dictionary<long, (ulong seed, int chunk, long offset, int length)>();
            foreach (string line in File.ReadLines(Path.Combine(directory, "games.csv")).Skip(1))
            {
                string[] f = line.Split(',');
                records[long.Parse(f[0], CultureInfo.InvariantCulture)] = (ulong.Parse(f[1], CultureInfo.InvariantCulture),
                    int.Parse(f[13], CultureInfo.InvariantCulture), long.Parse(f[14], CultureInfo.InvariantCulture), int.Parse(f[15], CultureInfo.InvariantCulture));
            }

            // The positions of a game are consecutive lines of samples.csv: (game, action index and actor per line)
            var games = new List<(long game, List<(int action, int actor)> positions)>();
            foreach (string line in File.ReadLines(Path.Combine(directory, "samples.csv")).Skip(1))
            {
                string[] f = line.Split(',', 6);
                long game = long.Parse(f[0], CultureInfo.InvariantCulture);
                if (games.Count == 0 || games[^1].game != game) games.Add((game, []));
                games[^1].positions.Add((int.Parse(f[2], CultureInfo.InvariantCulture), int.Parse(f[4], CultureInfo.InvariantCulture)));
            }

            using FileStream file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            long positions = 0;
            const int GamesPerBatch = 2000;
            for (int start = 0; start < games.Count; start += GamesPerBatch)
            {
                var batch = games.Skip(start).Take(GamesPerBatch).ToArray();
                var rows = new byte[batch.Length][];
                Parallel.For(0, batch.Length, i =>
                {
                    (long game, var gamePositions) = batch[i];
                    if (!records.TryGetValue(game, out var record))
                        throw new InvalidOperationException($"Game {game} has positions but no line in games.csv.");
                    rows[i] = ViewsOf(directory, record, centerDesert, gamePositions);
                });
                foreach (byte[] gameRows in rows)
                {
                    file.Write(gameRows);
                    positions += gameRows.Length / (ViewVectorizer.FeatureDim * sizeof(ushort));
                }
                log.Write($"\r{Math.Min(start + GamesPerBatch, games.Count)}/{games.Count} games, {positions} positions   ");
            }
            log.WriteLine();
            return positions;
        }

        /// <summary>
        /// The float16 rows of a game's positions, replaying its moves with a tracker per seat
        /// </summary>
        private static byte[] ViewsOf(string directory, (ulong seed, int chunk, long offset, int length) record, bool centerDesert, List<(int action, int actor)> positions)
        {
            SaveFile saved = SelfPlayWriter.ReadRecord(directory, record.chunk, record.offset, record.length);
            GameState state = SelfPlayGame.InitialState(record.seed, centerDesert);
            BeliefTracker[] trackers = Enumerable.Range(0, state.Players.Length).Select(seat => new BeliefTracker(seat, state)).ToArray();

            byte[] rows = new byte[positions.Count * ViewVectorizer.FeatureDim * sizeof(ushort)];
            float[] features = new float[ViewVectorizer.FeatureDim];
            int next = 0;
            for (int i = 0; i <= saved.PlayedActions.Count && next < positions.Count; i++)
            {
                while (next < positions.Count && positions[next].action == i)
                {
                    ViewVectorizer.VectorizeView(state, trackers[positions[next].actor], features);
                    Span<byte> row = rows.AsSpan(next * features.Length * sizeof(ushort));
                    for (int f = 0; f < features.Length; f++) BinaryPrimitives.WriteHalfLittleEndian(row[(f * sizeof(ushort))..], (Half)features[f]);
                    next++;
                }
                if (i == saved.PlayedActions.Count) break;

                GameState before = new GameState(state);
                Action action = saved.PlayedActions[i];
                action.Apply(state);
                foreach (BeliefTracker tracker in trackers) tracker.Observe(before, action, state);
            }
            if (next < positions.Count) throw new InvalidOperationException($"A position's action index lies beyond the game's {saved.PlayedActions.Count} actions.");
            return rows;
        }
    }
}
