using Agents;
using Common;
using System.Buffers.Binary;

namespace DatasetCollector
{
    /// <summary>
    /// Appends the state vectors of one worker to chunk files: vectors/w{worker}_c{chunk}.f16 holds one row of
    /// StateVectorizer.FeatureDim little-endian float16 values per sample, the .txt file of the same name lists the sample
    /// name of each row. DatasetProcessing/vector_chunks.py reads them, so the YAML samples don't need to be vectorized again.
    /// </summary>
    internal sealed class VectorChunkWriter : IDisposable
    {
        // A chunk of 50,000 rows is about 160 MB
        public const int DefaultRowsPerChunk = 50_000;

        private static readonly StateVectorizer Vectorizer = new();

        private readonly string _directory;
        private readonly int _workerIdx;
        private readonly int _rowsPerChunk;
        private readonly byte[] _row = new byte[StateVectorizer.FeatureDim * sizeof(ushort)];

        private FileStream? _data;
        private StreamWriter? _index;
        private int _chunkIdx = -1;
        private int _rowsInChunk;

        public VectorChunkWriter(string directory, int workerIdx, int rowsPerChunk = DefaultRowsPerChunk)
        {
            _directory = directory;
            _workerIdx = workerIdx;
            _rowsPerChunk = rowsPerChunk;
        }

        public void Write(string sampleName, GameState state)
        {
            if (_data == null || _rowsInChunk == _rowsPerChunk)
                OpenNextChunk();

            Span<float> features = stackalloc float[StateVectorizer.FeatureDim];
            Vectorizer.VectorizeInto(state, features);
            for (int i = 0; i < features.Length; i++)
                BinaryPrimitives.WriteHalfLittleEndian(_row.AsSpan(i * sizeof(ushort)), (Half)features[i]);

            // The row is flushed before its name, so a crash can leave a row without a name but never a name without a row
            _data!.Write(_row);
            _data.Flush();
            _index!.WriteLine(sampleName);
            _index.Flush();
            _rowsInChunk++;
        }

        private void OpenNextChunk()
        {
            CloseChunk();

            _chunkIdx++;
            string name = $"w{_workerIdx:D2}_c{_chunkIdx:D4}";
            _data = new FileStream(Path.Combine(_directory, name + ".f16"), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            _index = new StreamWriter(Path.Combine(_directory, name + ".txt"), append: false);
            _rowsInChunk = 0;
        }

        private void CloseChunk()
        {
            _data?.Dispose();
            _index?.Dispose();
            _data = null;
            _index = null;
        }

        public void Dispose()
        {
            CloseChunk();
        }
    }
}
