using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading.Channels;

namespace SelfPlayCollector
{
    /// <summary>
    /// Result of a run: games written (finished with a result, stopped at the action limit), recorded positions, errors
    /// </summary>
    public sealed record SelfPlayResult(int Games, int FinishedGames, int Positions, int Errors, TimeSpan Elapsed, bool Cancelled);

    /// <summary>
    /// Plays self-play games, ConcurrentGames at once, whose searches share one BatchingEvaluator, and writes every finished
    /// game with SelfPlayWriter into the run directory, plus metadata.json. A cancelled run stops soon and abandons the games
    /// in progress; the games written so far stay complete.
    /// </summary>
    public sealed class SelfPlayCollector
    {
        private readonly SelfPlaySettings _settings;
        private readonly TextWriter _log;

        private int _nextGame = -1;
        private int _games, _finishedGames, _positions, _errors;
        private long _iterations, _actions;
        private Exception? _firstError;

        public SelfPlayCollector(SelfPlaySettings settings, TextWriter? log = null)
        {
            settings.Validate();
            _settings = settings;
            _log = log ?? Console.Out;
            RunDirectory = Path.Combine(settings.OutputDirectory, settings.RunName);
        }

        public string RunDirectory { get; }

        public SelfPlayResult Collect(CancellationToken cancellation = default)
        {
            // Never write into an earlier run (its files would clash)
            if (Directory.Exists(RunDirectory) && Directory.EnumerateFileSystemEntries(RunDirectory).Any())
                throw new InvalidOperationException($"The run directory \"{Path.GetFullPath(RunDirectory)}\" already holds a run. Delete it or choose another --name.");
            Directory.CreateDirectory(RunDirectory);

            // Relative paths from the command line are meant from the working directory, not from the executable
            string modelPath = _settings.ModelPath != null ? Path.GetFullPath(_settings.ModelPath) : StateValueNet.DefaultModelPath;
            StateValueNet[] nets = Enumerable.Range(0, _settings.Nets)
                .Select(_ => new StateValueNet(modelPath, provider: _settings.Provider, fixedBatchSize: _settings.BatchRows)).ToArray();
            IStateVectorizer vectorizer = StateVectorizer.For(nets[0]);

            DateTime started = DateTime.Now;
            WriteMetadata(modelPath, nets[0], started, null, null);
            _log.WriteLine("\nRun directory:\n" + Path.GetFullPath(RunDirectory));
            _log.WriteLine($"Seed {_settings.Seed}, {_settings.Games} games ({_settings.ConcurrentGames} at once), {_settings.FullIterations}/{_settings.FastIterations} iterations "
                + $"(full with p = {_settings.FullSearchProbability}), net {Path.GetFileName(modelPath)} on {nets[0].Provider}");

            var stopwatch = Stopwatch.StartNew();
            using (BatchingEvaluator evaluator = new(nets, _settings.BatchRows))
            using (SelfPlayWriter writer = new(RunDirectory, vectorizer.FeatureDim))
            {
                // Finished games go to one writer thread; the bound holds the players back if writing falls behind
                Channel<GameRecord> finished = Channel.CreateBounded<GameRecord>(_settings.ConcurrentGames);
                Task writing = Task.Run(async () =>
                {
                    await foreach (GameRecord game in finished.Reader.ReadAllAsync())
                    {
                        writer.Write(game);
                        Interlocked.Increment(ref _games);
                        if (game.Finished) Interlocked.Increment(ref _finishedGames);
                        Interlocked.Add(ref _positions, game.Positions.Count);
                        Interlocked.Add(ref _actions, game.Actions.Count);
                    }
                });

                Task[] players = Enumerable.Range(0, Math.Min(_settings.ConcurrentGames, _settings.Games))
                    .Select(_ => Task.Run(() => PlayGamesAsync(evaluator, vectorizer, finished.Writer, cancellation))).ToArray();
                while (!Task.WaitAll(players, TimeSpan.FromSeconds(10)))
                    ReportProgress(stopwatch, evaluator);

                finished.Writer.Complete();
                writing.Wait();
                ReportProgress(stopwatch, evaluator);
            }
            stopwatch.Stop();

            SelfPlayResult result = new(_games, _finishedGames, _positions, _errors, stopwatch.Elapsed, cancellation.IsCancellationRequested);
            WriteMetadata(modelPath, nets[0], started, DateTime.Now, result);

            if (_errors > 0)
            {
                _log.WriteLine($"\n\nWarning: {_errors} game(s) failed, the first error:");
                _log.WriteLine(_firstError);
            }
            _log.WriteLine(result.Cancelled ? "\nCancelled." : "\nDone!");
            return result;
        }

        private async Task PlayGamesAsync(BatchingEvaluator evaluator, IStateVectorizer vectorizer, ChannelWriter<GameRecord> finished, CancellationToken cancellation)
        {
            MCTSSearch search = new(evaluator, vectorizer, _settings.LeavesPerBatch, parallelSelection: false);
            int localIdx;
            while (!cancellation.IsCancellationRequested && (localIdx = Interlocked.Increment(ref _nextGame)) < _settings.Games)
            {
                long game = (long)_settings.FirstGame + localIdx;
                try
                {
                    GameRecord? record = await SelfPlayGame.PlayAsync(game, Seeds.Derive(_settings.Seed, (ulong)game), _settings, search, vectorizer,
                        iterations => Interlocked.Add(ref _iterations, iterations), cancellation);
                    if (record != null)
                        await finished.WriteAsync(record);
                }
                catch (Exception e)
                {
                    if (Interlocked.Increment(ref _errors) == 1) _firstError = e;
                }
            }
        }

        private void ReportProgress(Stopwatch stopwatch, BatchingEvaluator evaluator)
        {
            double seconds = Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9);
            int games = Volatile.Read(ref _games);
            double gamesPerSecond = games / seconds;
            string eta = gamesPerSecond > 0 ? TimeSpan.FromSeconds((_settings.Games - games) / gamesPerSecond).ToString(@"hh\:mm\:ss") : "?";
            _log.Write($"\r{games}/{_settings.Games} games ({Volatile.Read(ref _finishedGames)} ended), {Volatile.Read(ref _positions)} positions, "
                + $"{gamesPerSecond * 3600:N0} games/h, {Interlocked.Read(ref _iterations) / seconds:N0} iterations/s, "
                + $"{Interlocked.Read(ref _actions) / Math.Max(games, 1):F0} actions/game, batch {(double)evaluator.Rows / Math.Max(evaluator.Batches, 1):F0}, ETA {eta}   ");
        }

        private void WriteMetadata(string modelPath, StateValueNet net, DateTime started, DateTime? finished, SelfPlayResult? result)
        {
            var metadata = new
            {
                format = 1,
                seed = _settings.Seed.ToString(CultureInfo.InvariantCulture),
                commit = CommitHash(),
                started = started.ToString("o", CultureInfo.InvariantCulture),
                finished = finished?.ToString("o", CultureInfo.InvariantCulture),
                status = result == null ? "running" : result.Cancelled ? "cancelled" : "finished",
                model = new
                {
                    file = Path.GetFileName(modelPath),
                    sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath))),
                    feature_dim = net.InputDim,
                    provider = net.Provider.ToString(),
                },
                settings = new
                {
                    games = _settings.Games,
                    first_game = _settings.FirstGame,
                    concurrent_games = _settings.ConcurrentGames,
                    leaves_per_batch = _settings.LeavesPerBatch,
                    full_iterations = _settings.FullIterations,
                    fast_iterations = _settings.FastIterations,
                    full_search_probability = _settings.FullSearchProbability,
                    temperature = _settings.Temperature,
                    temperature_moves = _settings.TemperatureMoves,
                    exploration = _settings.Exploration,
                    max_actions = _settings.MaxActions,
                    opponent_model = "MaxN",
                    nets = _settings.Nets,
                    batch_rows = _settings.BatchRows,
                    center_desert = _settings.CenterDesert,
                },
                result = result == null ? null : new
                {
                    games = result.Games,
                    finished_games = result.FinishedGames,
                    positions = result.Positions,
                    errors = result.Errors,
                    seconds = Math.Round(result.Elapsed.TotalSeconds, 1),
                },
            };
            File.WriteAllText(Path.Combine(RunDirectory, "metadata.json"), JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
        }

        private static string? CommitHash()
        {
            string? version = typeof(SelfPlayCollector).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            int plus = version?.IndexOf('+') ?? -1;
            return plus >= 0 ? version![(plus + 1)..] : null;
        }
    }
}
