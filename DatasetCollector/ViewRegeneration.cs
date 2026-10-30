using Agents;
using Common;
using Microsoft.Win32.SafeHandles;
using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using Action = Common.Actions.Action;

namespace DatasetCollector
{
    /// <summary>
    /// The input of a belief-aware value net (ViewVectorizer) for a prepared dataset collected before it existed: every
    /// match is played again from its seed with a BeliefTracker per seat, its samples and sibling groups drawn again, and
    /// each entry's state checked against its row of the prepared states.npy before its view is written. A sample is seen
    /// by the seat that moves there, a sibling by the seat that chose its move.
    /// </summary>
    public static class ViewRegeneration
    {
        private const int GroupsPerBatch = 2000;

        /// <summary>
        /// Writes views.npy (float16, one row per entry of the index, in its order) for the prepared set with the index and
        /// states. Needs the run's --seed, --agents, --exploration, --samples, --sibling-groups, --siblings and
        /// --center-desert. Returns the number of rows.
        /// </summary>
        public static long Run(string indexPath, string statesPath, string outputPath, CollectorSettings settings, TextWriter log)
        {
            if (File.Exists(outputPath)) throw new InvalidOperationException($"{outputPath} already exists.");

            // Entries in row order, grouped into runs of the same match
            List<(int match, List<(long row, string name)> entries)> groups = [];
            long rows = 0;
            foreach (string line in File.ReadLines(indexPath).Skip(1))
            {
                string[] fields = line.Split(',');
                long row = long.Parse(fields[0], CultureInfo.InvariantCulture);
                if (row != rows) throw new InvalidOperationException($"Entry {row} of the index is not in row order.");
                string name = fields[3];
                int match = int.Parse(name.AsSpan(1, name.IndexOf('_') - 1), CultureInfo.InvariantCulture);
                if (groups.Count == 0 || groups[^1].match != match) groups.Add((match, []));
                groups[^1].entries.Add((row, name));
                rows++;
            }

            using SafeFileHandle states = File.OpenHandle(statesPath);
            long statesData = ReadNpyHeader(states, rows, StateVectorizer.FeatureDim);
            log.WriteLine($"Regenerating the views of {rows} entries in {groups.Count} match runs with {settings.Threads} threads");

            // Written under another name until complete, so an aborted run leaves no views behind
            string partial = outputPath + ".partial";
            try
            {
                using FileStream output = new FileStream(partial, FileMode.Create, FileAccess.Write);
                WriteNpyHeader(output, rows, ViewVectorizer.FeatureDim);
                for (int start = 0; start < groups.Count; start += GroupsPerBatch)
                {
                    var batch = groups.Skip(start).Take(GroupsPerBatch).ToArray();
                    byte[][] written = new byte[batch.Length][];
                    try
                    {
                        Parallel.For(0, batch.Length, new ParallelOptions { MaxDegreeOfParallelism = settings.Threads },
                            i => written[i] = ViewsOf(batch[i].match, batch[i].entries, settings, states, statesData));
                    }
                    catch (AggregateException e) when (e.InnerExceptions.Count > 0)
                    {
                        // The first entry that failed, as if it had been checked alone
                        System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
                    }
                    foreach (byte[] bytes in written) output.Write(bytes);
                    log.Write($"\r{Math.Min(start + GroupsPerBatch, groups.Count)}/{groups.Count} match runs   ");
                }
            }
            catch
            {
                File.Delete(partial);
                throw;
            }
            log.WriteLine();
            File.Move(partial, outputPath);
            return rows;
        }

        /// <summary>
        /// The float16 views of a match's entries, each checked against its stored row first
        /// </summary>
        private static byte[] ViewsOf(int matchIdx, List<(long row, string name)> entries, CollectorSettings settings, SafeFileHandle states, long statesData)
        {
            DatasetCollector.MatchSamples samples = DatasetCollector.SampleMatch(matchIdx, settings);

            // The trackers of every seat at each sampled position
            Dictionary<int, BeliefTracker[]> trackersAt = [];
            GameState state = new GameState(samples.BaseState);
            BeliefTracker[] trackers = Enumerable.Range(0, state.Players.Length).Select(seat => new BeliefTracker(seat, state)).ToArray();
            int lastPosition = samples.States.Max(sample => sample.position);
            for (int position = 0; ; position++)
            {
                if (samples.States.Any(sample => sample.position == position)) trackersAt[position] = trackers.Select(t => new BeliefTracker(t)).ToArray();
                if (position == lastPosition) break;
                GameState before = new GameState(state);
                Action action = samples.PlayedActions[position];
                action.Clone().Apply(state);
                foreach (BeliefTracker tracker in trackers) tracker.Observe(before, action, state);
            }

            byte[] result = new byte[entries.Count * ViewVectorizer.FeatureDim * sizeof(ushort)];
            float[] full = new float[StateVectorizer.FeatureDim], view = new float[ViewVectorizer.FeatureDim];
            byte[] stored = new byte[StateVectorizer.FeatureDim * sizeof(ushort)];
            for (int e = 0; e < entries.Count; e++)
            {
                (long row, string name) = entries[e];
                (GameState entryState, BeliefTracker tracker) = Entry(name, samples, trackersAt);

                StateVectorizer.Current.VectorizeInto(entryState, full);
                RandomAccess.Read(states, stored, statesData + row * stored.Length);
                for (int f = 0; f < full.Length; f++)
                {
                    if (BinaryPrimitives.ReadHalfLittleEndian(stored.AsSpan(f * sizeof(ushort))) != (Half)full[f])
                        throw new InvalidOperationException($"{name}: the replayed state differs from its stored row {row} in feature {f}; are the collection settings the run's?");
                }

                ViewVectorizer.VectorizeView(entryState, tracker, view);
                Span<byte> target = result.AsSpan(e * view.Length * sizeof(ushort));
                for (int f = 0; f < view.Length; f++) BinaryPrimitives.WriteHalfLittleEndian(target[(f * sizeof(ushort))..], (Half)view[f]);
            }
            return result;
        }

        /// <summary>
        /// An entry's state and the tracker of the seat that sees it: r{match}_s{sample} or r{match}_s{sample}_c{sibling}
        /// </summary>
        private static (GameState state, BeliefTracker tracker) Entry(string name, DatasetCollector.MatchSamples samples, Dictionary<int, BeliefTracker[]> trackersAt)
        {
            string[] parts = name.Split('_');
            int sampleIdx = int.Parse(parts[1].AsSpan(1), CultureInfo.InvariantCulture);
            (GameState sampleState, int position) = samples.States[sampleIdx];
            BeliefTracker[] trackers = trackersAt[position];

            if (parts.Length == 2)
            {
                int seat = sampleState.HasEnded ? 0 : Enumerable.Range(0, sampleState.Players.Length).First(sampleState.CanPlayerAct);
                return (sampleState, trackers[seat]);
            }

            int siblingIdx = int.Parse(parts[2].AsSpan(1), CultureInfo.InvariantCulture);
            DatasetCollector.Sibling sibling = samples.Groups[sampleIdx][siblingIdx];
            BeliefTracker chooser = new BeliefTracker(trackers[sibling.Move.PlayerIndex]);
            chooser.Observe(sampleState, sibling.Move, sibling.State);
            return (sibling.State, chooser);
        }

        /// <summary>
        /// The header of a C-ordered float16 .npy array of rows x columns
        /// </summary>
        public static void WriteNpyHeader(Stream stream, long rows, int columns)
        {
            string dictionary = $"{{'descr': '<f2', 'fortran_order': False, 'shape': ({rows}, {columns}), }}";
            int unpadded = 10 + dictionary.Length + 1;
            string header = dictionary + new string(' ', (64 - unpadded % 64) % 64) + "\n";
            stream.Write([0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0]);
            Span<byte> length = stackalloc byte[2];
            BinaryPrimitives.WriteUInt16LittleEndian(length, (ushort)header.Length);
            stream.Write(length);
            stream.Write(Encoding.ASCII.GetBytes(header));
        }

        /// <summary>
        /// Checks a float16 .npy array's header against the expected shape; returns where its data starts
        /// </summary>
        private static long ReadNpyHeader(SafeFileHandle file, long rows, int columns)
        {
            byte[] start = new byte[10];
            RandomAccess.Read(file, start, 0);
            if (start[0] != 0x93 || Encoding.ASCII.GetString(start, 1, 5) != "NUMPY" || start[6] != 1)
                throw new InvalidOperationException("The states are not a version 1 .npy file.");
            int length = BinaryPrimitives.ReadUInt16LittleEndian(start.AsSpan(8));
            byte[] header = new byte[length];
            RandomAccess.Read(file, header, 10);
            string text = Encoding.ASCII.GetString(header);
            if (!text.Contains("'<f2'") || !text.Contains("False") || !text.Contains($"({rows}, {columns})"))
                throw new InvalidOperationException($"The states are not a float16 array of {rows} x {columns}: {text.Trim()}");
            return 10 + length;
        }
    }
}
