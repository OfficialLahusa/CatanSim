using Agents.Inference;
using Agents.MCTS;
using Common;
using System.Buffers.Binary;
using System.Globalization;
using System.Text.Json;
using Action = Common.Actions.Action;

namespace SelfPlayCollector
{
    /// <summary>
    /// The legal moves of every recorded position, encoded for the policy head (ActionEncoding), in the order of the
    /// position's root children (the move indices of the visits in samples.csv). One entry per line of samples.csv, in the
    /// same order: policy_counts.i32 holds the number of moves, policy_tokens.i16 MaxTokens tokens per move (-1 for unused)
    /// and policy_weights.u8 their weights, all little-endian; policy_layout.json the token ranges.
    /// </summary>
    internal sealed class PolicyTargetWriter : IDisposable
    {
        public const string CountsFile = "policy_counts.i32", TokensFile = "policy_tokens.i16", WeightsFile = "policy_weights.u8", LayoutFile = "policy_layout.json";

        private readonly FileStream _counts, _tokens, _weights;
        private readonly short[] _moveTokens = new short[ActionEncoding.MaxTokens];
        private readonly float[] _moveWeights = new float[ActionEncoding.MaxTokens];
        private readonly byte[] _tokenBytes = new byte[ActionEncoding.MaxTokens * sizeof(short)];
        private readonly byte[] _weightBytes = new byte[ActionEncoding.MaxTokens];

        public PolicyTargetWriter(string directory)
        {
            _counts = new FileStream(Path.Combine(directory, CountsFile), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            _tokens = new FileStream(Path.Combine(directory, TokensFile), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            _weights = new FileStream(Path.Combine(directory, WeightsFile), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            File.WriteAllText(Path.Combine(directory, LayoutFile), JsonSerializer.Serialize(ActionEncoding.Layout, new JsonSerializerOptions { WriteIndented = true }));
        }

        public void Write(IReadOnlyList<Action> moves)
        {
            Span<byte> count = stackalloc byte[sizeof(int)];
            BinaryPrimitives.WriteInt32LittleEndian(count, moves.Count);
            _counts.Write(count);

            foreach (Action move in moves)
            {
                int tokens = ActionEncoding.Encode(move, _moveTokens, _moveWeights);
                for (int t = 0; t < ActionEncoding.MaxTokens; t++)
                {
                    BinaryPrimitives.WriteInt16LittleEndian(_tokenBytes.AsSpan(t * sizeof(short)), t < tokens ? _moveTokens[t] : (short)-1);
                    _weightBytes[t] = t < tokens ? checked((byte)_moveWeights[t]) : (byte)0;
                }
                _tokens.Write(_tokenBytes);
                _weights.Write(_weightBytes);
            }
        }

        public void Flush()
        {
            _counts.Flush();
            _tokens.Flush();
            _weights.Flush();
        }

        public void Dispose()
        {
            _counts.Dispose();
            _tokens.Dispose();
            _weights.Dispose();
        }

        /// <summary>
        /// Writes the policy files for a run collected without them, by replaying every game from its save file. Returns the
        /// number of positions.
        /// </summary>
        public static long ConvertRun(string directory, TextWriter log)
        {
            if (File.Exists(Path.Combine(directory, CountsFile)))
                throw new InvalidOperationException($"{directory} already has policy targets.");

            using JsonDocument metadata = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "metadata.json")));
            bool centerDesert = metadata.RootElement.GetProperty("settings").GetProperty("center_desert").GetBoolean();

            var records = new Dictionary<long, (ulong seed, int chunk, long offset, int length)>();
            foreach (string line in File.ReadLines(Path.Combine(directory, "games.csv")).Skip(1))
            {
                string[] f = line.Split(',');
                records[long.Parse(f[0], CultureInfo.InvariantCulture)] = (ulong.Parse(f[1], CultureInfo.InvariantCulture),
                    int.Parse(f[13], CultureInfo.InvariantCulture), long.Parse(f[14], CultureInfo.InvariantCulture), int.Parse(f[15], CultureInfo.InvariantCulture));
            }

            // The positions of a game are consecutive lines of samples.csv: (game, action indices in line order)
            var games = new List<(long game, List<int> actionIndices)>();
            foreach (string line in File.ReadLines(Path.Combine(directory, "samples.csv")).Skip(1))
            {
                int first = line.IndexOf(','), second = line.IndexOf(',', first + 1), third = line.IndexOf(',', second + 1);
                long game = long.Parse(line.AsSpan(0, first), CultureInfo.InvariantCulture);
                int actionIndex = int.Parse(line.AsSpan(second + 1, third - second - 1), CultureInfo.InvariantCulture);
                if (games.Count == 0 || games[^1].game != game) games.Add((game, []));
                games[^1].actionIndices.Add(actionIndex);
            }

            using PolicyTargetWriter writer = new(directory);
            long positions = 0;
            const int GamesPerBatch = 2000;
            for (int start = 0; start < games.Count; start += GamesPerBatch)
            {
                var batch = games.Skip(start).Take(GamesPerBatch).ToArray();
                var moves = new List<Action>[batch.Length][];
                Parallel.For(0, batch.Length, i =>
                {
                    (long game, List<int> actionIndices) = batch[i];
                    // Samples of games without a games.csv line (a crashed run) still need their entries
                    if (!records.TryGetValue(game, out var record))
                        throw new InvalidOperationException($"Game {game} has positions but no line in games.csv.");
                    moves[i] = LegalMovesAt(directory, record, centerDesert, actionIndices);
                });
                foreach (var gameMoves in moves)
                    foreach (var positionMoves in gameMoves)
                    {
                        writer.Write(positionMoves);
                        positions++;
                    }
                log.Write($"\r{Math.Min(start + GamesPerBatch, games.Count)}/{games.Count} games, {positions} positions   ");
            }
            writer.Flush();
            log.WriteLine();
            return positions;
        }

        private static List<Action>[] LegalMovesAt(string directory, (ulong seed, int chunk, long offset, int length) record, bool centerDesert, List<int> actionIndices)
        {
            SaveFile saved = SelfPlayWriter.ReadRecord(directory, record.chunk, record.offset, record.length);
            GameState state = SelfPlayGame.InitialState(record.seed, centerDesert);
            var result = new List<Action>[actionIndices.Count];
            int next = 0;
            for (int i = 0; i <= saved.PlayedActions.Count && next < actionIndices.Count; i++)
            {
                while (next < actionIndices.Count && actionIndices[next] == i)
                    result[next++] = MCTSTree.GetValidActions(state);
                if (i < saved.PlayedActions.Count) saved.PlayedActions[i].Apply(state);
            }
            if (next < actionIndices.Count) throw new InvalidOperationException($"A position's action index lies beyond the game's {saved.PlayedActions.Count} actions.");
            return result;
        }
    }
}
