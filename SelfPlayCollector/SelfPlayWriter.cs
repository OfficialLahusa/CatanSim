using Common;
using Common.Serialization;
using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;

namespace SelfPlayCollector
{
    /// <summary>
    /// Writes the finished games of a run (from one thread):
    /// states_c{chunk}.f16 holds one row of little-endian float16 features per recorded position;
    /// samples.csv one line per position with its search results, the row of its features (chunk, row) and the visits of
    /// the root moves as "move:visits" pairs (move = index in MCTSTree.GetValidActions of the position; with target pruning
    /// without the visits only the root noise caused, so they can add up to less than root_visits);
    /// games.csv one line per game with its seed and result, and where its save file is: games_c{chunk}.yaml.gz holds one
    /// gzip member per game (at record_offset, record_length bytes), the final state with all played actions.
    /// A game's line is written last, after its rows, positions and save file are flushed, so a crash never leaves a game
    /// line without its data. The legal moves of each position go to the policy files of PolicyTargetWriter.
    /// DatasetProcessing/selfplay_to_dataset.py reads the run.
    /// </summary>
    internal sealed class SelfPlayWriter : IDisposable
    {
        // 50,000 rows of 1965 features are about 200 MB
        public const int DefaultRowsPerChunk = 50_000;
        public const int DefaultGamesPerChunk = 1000;

        public const string SampleHeader = "game,position,action,round,actor,iterations,root_visits,q0,q1,q2,q3,chunk,row,visits";
        public const string GameHeader = "game,seed,finished,winner,vp0,vp1,vp2,vp3,rounds,actions,positions,searched_moves,seconds,record_chunk,record_offset,record_length";

        private readonly string _directory;
        private readonly int _featureDim;
        private readonly int _rowsPerChunk, _gamesPerChunk;
        private readonly StreamWriter _samples, _games;
        private readonly PolicyTargetWriter _policy;
        private readonly byte[] _row;

        private FileStream? _states, _records;
        private int _stateChunk = -1, _rowsInChunk;
        private int _recordChunk = -1, _gamesInChunk;

        public SelfPlayWriter(string directory, int featureDim, int rowsPerChunk = DefaultRowsPerChunk, int gamesPerChunk = DefaultGamesPerChunk)
        {
            _directory = directory;
            _featureDim = featureDim;
            _rowsPerChunk = rowsPerChunk;
            _gamesPerChunk = gamesPerChunk;
            _row = new byte[featureDim * sizeof(ushort)];

            _samples = new StreamWriter(Path.Combine(directory, "samples.csv"), append: false);
            _samples.WriteLine(SampleHeader);
            _games = new StreamWriter(Path.Combine(directory, "games.csv"), append: false);
            _games.WriteLine(GameHeader);
            _policy = new PolicyTargetWriter(directory);
        }

        public void Write(GameRecord game)
        {
            // Feature rows
            var rows = new (int chunk, int row)[game.Positions.Count];
            for (int i = 0; i < game.Positions.Count; i++)
            {
                if (_states == null || _rowsInChunk == _rowsPerChunk)
                {
                    _states?.Dispose();
                    _stateChunk++;
                    _rowsInChunk = 0;
                    _states = new FileStream(Path.Combine(_directory, $"states_c{_stateChunk:D4}.f16"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                }

                Half[] features = game.Positions[i].Features;
                if (features.Length != _featureDim) throw new ArgumentException($"A position has {features.Length} features instead of {_featureDim}.");
                for (int f = 0; f < features.Length; f++)
                    BinaryPrimitives.WriteHalfLittleEndian(_row.AsSpan(f * sizeof(ushort)), features[f]);
                _states.Write(_row);
                rows[i] = (_stateChunk, _rowsInChunk++);
            }
            _states?.Flush();

            // Positions
            for (int i = 0; i < game.Positions.Count; i++)
            {
                RecordedPosition position = game.Positions[i];
                StringBuilder line = new();
                line.Append(CultureInfo.InvariantCulture, $"{game.Game},{i},{position.ActionIndex},{position.Round},{position.Actor},{position.Iterations},{position.RootVisits}");
                foreach (double value in position.RootValues)
                    line.Append(',').Append(value.ToString("R", CultureInfo.InvariantCulture));
                line.Append(CultureInfo.InvariantCulture, $",{rows[i].chunk},{rows[i].row},");
                line.AppendJoin(' ', position.Visits.Select(v => $"{v.move}:{v.visits}"));
                _samples.WriteLine(line.ToString());
                _policy.Write(position.Moves);
            }
            _samples.Flush();
            _policy.Flush();

            // Save file
            if (_records == null || _gamesInChunk == _gamesPerChunk)
            {
                _records?.Dispose();
                _recordChunk++;
                _gamesInChunk = 0;
                _records = new FileStream(Path.Combine(_directory, $"games_c{_recordChunk:D4}.yaml.gz"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            }
            byte[] record = Compress(SaveFileSerializer.Serialize(new SaveFile(game.FinalState, game.Actions, [])));
            long offset = _records.Position;
            _records.Write(record);
            _records.Flush();
            _gamesInChunk++;

            // Game line, last
            int[] vp = game.FinalState.Players.Select(player => player.VictoryPoints.Total).ToArray();
            _games.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{game.Game},{game.Seed},{(game.Finished ? 1 : 0)},{game.Winner},{vp[0]},{vp[1]},{vp[2]},{vp[3]},{game.FinalState.Turn.RoundCounter},"
                + $"{game.Actions.Count},{game.Positions.Count},{game.SearchedMoves},{game.Seconds:F1},{_recordChunk},{offset},{record.Length}"));
            _games.Flush();
        }

        private static byte[] Compress(string text)
        {
            using MemoryStream compressed = new();
            using (GZipStream gzip = new(compressed, CompressionLevel.Optimal, leaveOpen: true))
                gzip.Write(Encoding.UTF8.GetBytes(text));
            return compressed.ToArray();
        }

        /// <summary>
        /// The save file of a game from its run directory and its games.csv entry
        /// </summary>
        public static SaveFile ReadRecord(string directory, int chunk, long offset, int length)
        {
            using FileStream file = File.OpenRead(Path.Combine(directory, $"games_c{chunk:D4}.yaml.gz"));
            file.Seek(offset, SeekOrigin.Begin);
            byte[] compressed = new byte[length];
            file.ReadExactly(compressed);

            using GZipStream gzip = new(new MemoryStream(compressed), CompressionMode.Decompress);
            using StreamReader reader = new(gzip, Encoding.UTF8);
            return SaveFileSerializer.Deserialize(reader.ReadToEnd());
        }

        public void Dispose()
        {
            _states?.Dispose();
            _records?.Dispose();
            _samples.Dispose();
            _games.Dispose();
            _policy.Dispose();
        }
    }
}
