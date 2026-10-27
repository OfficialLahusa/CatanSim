using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Agents.Inference
{
    /// <summary>
    /// Collects the leaf batches of many searches (e.g. games played at once) into large batches for the value net. Each net
    /// gets its own worker thread that takes waiting requests until the batch is full, runs them together and hands every
    /// search its rows back; with two nets one batch runs while the next one is collected. DirectML doesn't allow parallel
    /// Run calls on one session, so parallel workers need nets of their own (loaded from the same model).
    /// </summary>
    public sealed class BatchingEvaluator : ILeafEvaluator, IDisposable
    {
        private sealed record Request(DenseTensor<float> Features, int Rows, bool WithPlacements, TaskCompletionSource<LeafValues> Completion);

        private readonly BlockingCollection<Request> _queue = new(new ConcurrentQueue<Request>());
        private readonly Thread[] _workers;
        private readonly int _maxRows;
        private readonly int _featureDim;
        private long _batches, _rows;

        /// <param name="nets">One net per worker, all of the same model</param>
        /// <param name="maxRows">Rows per net batch at most; nets with a fixed batch size (DirectML) need it at least this large</param>
        public BatchingEvaluator(IReadOnlyList<StateValueNet> nets, int maxRows)
        {
            if (nets.Count == 0) throw new ArgumentException("At least one net is needed.", nameof(nets));
            if (nets.Any(net => net.FixedBatchSize < maxRows))
                throw new ArgumentException($"The nets' fixed batch size is below {maxRows} rows.", nameof(nets));
            if (nets.Any(net => net.InputDim != nets[0].InputDim || net.HasPlacements != nets[0].HasPlacements))
                throw new ArgumentException("The nets must be of the same model.", nameof(nets));

            _maxRows = maxRows;
            _featureDim = nets[0].InputDim;
            HasPlacements = nets[0].HasPlacements;
            _workers = nets.Select((net, i) => new Thread(() => Work(net)) { IsBackground = true, Name = $"Value net worker {i}" }).ToArray();
            foreach (Thread worker in _workers)
                worker.Start();
        }

        public bool HasPlacements { get; }

        /// <summary>
        /// Net batches run so far
        /// </summary>
        public long Batches => Interlocked.Read(ref _batches);

        /// <summary>
        /// Rows evaluated so far (Rows / Batches is the mean batch size)
        /// </summary>
        public long Rows => Interlocked.Read(ref _rows);

        public Task<LeafValues> EvaluateAsync(DenseTensor<float> features, bool withPlacements)
        {
            int rows = features.Dimensions[0];
            if (rows > _maxRows) throw new ArgumentException($"{rows} rows don't fit into a batch of {_maxRows}.", nameof(features));
            if (features.Dimensions[1] != _featureDim) throw new ArgumentException($"The net takes {_featureDim} features, not {features.Dimensions[1]}.", nameof(features));
            if (withPlacements && !HasPlacements)
                throw new InvalidOperationException("The value net has no placements output (it was trained without auxiliary heads).");

            // The search continues on the thread pool, not on the worker, which goes on with the next batch
            var completion = new TaskCompletionSource<LeafValues>(TaskCreationOptions.RunContinuationsAsynchronously);
            _queue.Add(new Request(features, rows, withPlacements, completion));
            return completion.Task;
        }

        private void Work(StateValueNet net)
        {
            // A fixed batch size runs the whole buffer (unused rows are stale and ignored), otherwise only the rows in use
            int bufferRows = net.FixedBatchSize ?? _maxRows;
            float[] buffer = new float[bufferRows * _featureDim];
            var batch = new List<Request>();
            Request? carried = null;

            while (true)
            {
                batch.Clear();
                Request first;
                if (carried != null)
                    first = carried;
                else if (!_queue.TryTake(out first!, Timeout.Infinite))
                    return; // Disposed
                carried = null;

                // Take what is waiting until the batch is full; a request that no longer fits starts the next batch
                int rows = 0;
                Request? next = first;
                do
                {
                    if (rows + next.Rows > _maxRows)
                    {
                        carried = next;
                        break;
                    }
                    next.Features.Buffer.Span[..(next.Rows * _featureDim)].CopyTo(buffer.AsSpan(rows * _featureDim));
                    rows += next.Rows;
                    batch.Add(next);
                }
                while (_queue.TryTake(out next));

                try
                {
                    int runRows = net.FixedBatchSize ?? rows;
                    var input = new DenseTensor<float>(buffer.AsMemory(0, runRows * _featureDim), [runRows, _featureDim]);
                    float[] winProbabilities, placements = [];
                    if (batch.Any(request => request.WithPlacements)) (winProbabilities, placements) = net.RunWithPlacements(input);
                    else winProbabilities = net.Run(input);

                    int row = 0;
                    foreach (Request request in batch)
                    {
                        float[] wins = winProbabilities.AsSpan(row * StateValueNet.PlayerCount, request.Rows * StateValueNet.PlayerCount).ToArray();
                        float[]? places = request.WithPlacements
                            ? placements.AsSpan(row * StateValueNet.PlayerCount * 4, request.Rows * StateValueNet.PlayerCount * 4).ToArray()
                            : null;
                        request.Completion.SetResult(new LeafValues(wins, places));
                        row += request.Rows;
                    }

                    Interlocked.Increment(ref _batches);
                    Interlocked.Add(ref _rows, rows);
                }
                catch (Exception e)
                {
                    foreach (Request request in batch)
                        request.Completion.TrySetException(e);
                }
            }
        }

        /// <summary>
        /// Stops the workers after the requests already waiting
        /// </summary>
        public void Dispose()
        {
            _queue.CompleteAdding();
            foreach (Thread worker in _workers)
                worker.Join();
            _queue.Dispose();
        }
    }
}
