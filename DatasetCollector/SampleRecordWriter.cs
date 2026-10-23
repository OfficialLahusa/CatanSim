using Common;
using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace DatasetCollector
{
    /// <summary>
    /// Writes the raw results of one worker, so labels can be derived later without a new run:
    /// playouts/w{worker}.bin holds one 12-byte record per playout (winner as sbyte, -1 if the playout didn't finish; final VP
    /// of the 4 players as bytes; agent kind of the 4 seats as bytes; round counter at the end as little-endian ushort;
    /// one unused byte), samples/w{worker}.csv one line per sample with its playouts' offset and count in that file.
    /// DatasetProcessing/playout_results.py reads them.
    /// </summary>
    internal sealed class SampleRecordWriter : IDisposable
    {
        public const int PlayoutRecordSize = 12;
        public const string Header = "name,group,match,sample,position,match_length,round,terminal,playouts,finished,playout_offset,match_winner,match_vp,match_rounds,match_agents";

        private readonly FileStream _playouts;
        private readonly StreamWriter _samples;
        private readonly MemoryStream _pending = new();
        private readonly byte[] _record = new byte[PlayoutRecordSize];

        private long _writtenPlayouts;
        private int _pendingPlayouts;
        private int _pendingFinished;

        public SampleRecordWriter(string datasetDirectory, int workerIdx)
        {
            string playoutDirectory = Path.Combine(datasetDirectory, "playouts");
            string sampleDirectory = Path.Combine(datasetDirectory, "samples");
            Directory.CreateDirectory(playoutDirectory);
            Directory.CreateDirectory(sampleDirectory);

            _playouts = new FileStream(Path.Combine(playoutDirectory, $"w{workerIdx:D2}.bin"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            _samples = new StreamWriter(Path.Combine(sampleDirectory, $"w{workerIdx:D2}.csv"), append: false, new UTF8Encoding(false));
            _samples.WriteLine(Header);
            _samples.Flush();
        }

        /// <summary>
        /// Adds a playout of the next sample, from the state it ended in (or was aborted in)
        /// </summary>
        public void AddPlayout(GameState end, ReadOnlySpan<AgentKind> agents)
        {
            Span<byte> record = _record;
            record[0] = (byte)(end.HasEnded ? end.Turn.PlayerIndex : -1);
            for (int p = 0; p < 4; p++)
            {
                record[1 + p] = p < end.Players.Length ? (byte)Math.Clamp(end.Players[p].VictoryPoints.Total, 0, byte.MaxValue) : (byte)0;
                record[5 + p] = p < agents.Length ? (byte)agents[p] : (byte)0;
            }
            BinaryPrimitives.WriteUInt16LittleEndian(record[9..], (ushort)Math.Clamp(end.Turn.RoundCounter, 0, ushort.MaxValue));
            record[11] = 0;

            _pending.Write(record);
            _pendingPlayouts++;
            if (end.HasEnded) _pendingFinished++;
        }

        /// <summary>
        /// Drops the playouts added since the last sample (the sample is not written)
        /// </summary>
        public void DiscardPlayouts()
        {
            _pending.SetLength(0);
            _pendingPlayouts = _pendingFinished = 0;
        }

        public int PendingFinished => _pendingFinished;

        /// <summary>
        /// Writes the sample with the playouts added since the last sample. The playouts are flushed before the sample's line,
        /// so a crash can leave playouts without a sample but never a sample without its playouts.
        /// </summary>
        public void WriteSample(in SampleInfo sample)
        {
            long offset = _writtenPlayouts;
            _pending.WriteTo(_playouts);
            _playouts.Flush();
            _writtenPlayouts += _pendingPlayouts;

            MatchInfo match = sample.Match;
            _samples.WriteLine(string.Join(",",
                sample.Name,
                match.Seed.ToString(CultureInfo.InvariantCulture),
                match.Index.ToString(CultureInfo.InvariantCulture),
                sample.Index.ToString(CultureInfo.InvariantCulture),
                sample.Position.ToString(CultureInfo.InvariantCulture),
                match.Length.ToString(CultureInfo.InvariantCulture),
                sample.Round.ToString(CultureInfo.InvariantCulture),
                sample.Position == match.Length ? "1" : "0",
                _pendingPlayouts.ToString(CultureInfo.InvariantCulture),
                _pendingFinished.ToString(CultureInfo.InvariantCulture),
                offset.ToString(CultureInfo.InvariantCulture),
                match.Winner.ToString(CultureInfo.InvariantCulture),
                string.Join("/", match.VictoryPoints),
                match.Rounds.ToString(CultureInfo.InvariantCulture),
                match.Agents));
            _samples.Flush();

            DiscardPlayouts();
        }

        public void Dispose()
        {
            _playouts.Dispose();
            _samples.Dispose();
        }
    }

    /// <summary>
    /// A played match: its seed (also the group ID of its samples), length in actions, winner, final VP, rounds and seat agents
    /// </summary>
    internal readonly record struct MatchInfo(ulong Seed, int Index, int Length, int Winner, int[] VictoryPoints, int Rounds, string Agents);

    /// <summary>
    /// A sampled state of a match: after the first Position actions, in the given round
    /// </summary>
    internal readonly record struct SampleInfo(string Name, MatchInfo Match, int Index, int Position, int Round);
}
