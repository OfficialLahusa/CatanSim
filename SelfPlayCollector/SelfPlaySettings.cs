using Agents.Inference;
using Common;
using System.Globalization;

namespace SelfPlayCollector
{
    /// <summary>
    /// Settings of a self-play run, from the command line
    /// </summary>
    public sealed record SelfPlaySettings
    {
        public int Games { get; init; } = 1000;

        /// <summary>
        /// Index of the first game: games are numbered from it (seeds and IDs), so separate runs with one seed don't overlap
        /// </summary>
        public int FirstGame { get; init; }

        /// <summary>
        /// Games played at once; their searches share the value net batches
        /// </summary>
        public int ConcurrentGames { get; init; } = 256;

        /// <summary>
        /// Leaves a search selects before their evaluation
        /// </summary>
        public int LeavesPerBatch { get; init; } = 16;

        /// <summary>
        /// Iterations of a full search, whose position is recorded
        /// </summary>
        public int FullIterations { get; init; } = 800;

        /// <summary>
        /// Iterations of a fast search, which only moves the game on (playout cap randomization)
        /// </summary>
        public int FastIterations { get; init; } = 150;

        /// <summary>
        /// Probability that a decision gets a full search
        /// </summary>
        public double FullSearchProbability { get; init; } = 0.25;

        /// <summary>
        /// The first searched decisions of a game sample their move from the visits with this temperature instead of
        /// taking the most visited move
        /// </summary>
        public double Temperature { get; init; } = 1.0;
        public int TemperatureMoves { get; init; } = 30;

        public double Exploration { get; init; } = 1.414;

        /// <summary>
        /// Dirichlet noise mixed into the root's priors of full searches (only with a policy head): its weight and its total
        /// concentration (alpha per move = concentration / moves)
        /// </summary>
        public double RootNoise { get; init; } = 0.25;
        public double RootNoiseConcentration { get; init; } = 10;

        /// <summary>
        /// Record the root visits of full searches without those only the root noise caused (policy target pruning, see
        /// MCTSTree.PrunedRootVisits); only with a policy head and root noise
        /// </summary>
        public bool PruneTargets { get; init; } = true;

        /// <summary>
        /// Games still running after this many actions are stopped and written without a result
        /// </summary>
        public int MaxActions { get; init; } = 3000;

        /// <summary>
        /// Value net, the deployed default model if null
        /// </summary>
        public string? ModelPath { get; init; }

        /// <summary>
        /// Net sessions evaluating batches in turn, and the most rows per batch
        /// </summary>
        public int Nets { get; init; } = 2;
        public int BatchRows { get; init; } = 1024;
        public InferenceProvider Provider { get; init; } = InferenceProvider.Auto;

        public ulong Seed { get; init; } = Seeds.NewRunSeed();
        public bool CenterDesert { get; init; }

        /// <summary>
        /// Directory the run directory is created in
        /// </summary>
        public string OutputDirectory { get; init; } = "selfplay";

        /// <summary>
        /// Name of the run directory, generated from the settings and the time if null
        /// </summary>
        public string? Name { get; init; }

        public string RunName => Name ?? $"g{Games}_i{FullIterations}-{FastIterations}_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}";

        public void Validate()
        {
            if (Games < 1) throw new ArgumentException("--games must be at least 1.");
            if (FirstGame < 0) throw new ArgumentException("--first-game must not be negative.");
            if (ConcurrentGames < 1) throw new ArgumentException("--concurrent must be at least 1.");
            if (LeavesPerBatch < 1) throw new ArgumentException("--leaves must be at least 1.");
            if (FullIterations < 1 || FastIterations < 1) throw new ArgumentException("--full-iterations and --fast-iterations must be at least 1.");
            if (FullSearchProbability is < 0 or > 1) throw new ArgumentException("--full-probability must be between 0 and 1.");
            if (Temperature < 0) throw new ArgumentException("--temperature must not be negative.");
            if (TemperatureMoves < 0) throw new ArgumentException("--temperature-moves must not be negative.");
            if (Exploration < 0) throw new ArgumentException("--exploration must not be negative.");
            if (RootNoise is < 0 or > 1) throw new ArgumentException("--root-noise must be between 0 and 1.");
            if (RootNoiseConcentration <= 0) throw new ArgumentException("--noise-concentration must be positive.");
            if (MaxActions < 1) throw new ArgumentException("--max-actions must be at least 1.");
            if (Nets < 1) throw new ArgumentException("--nets must be at least 1.");
            if (BatchRows < LeavesPerBatch) throw new ArgumentException("--batch must be at least --leaves.");
        }
    }

    /// <summary>
    /// What the command line asks for
    /// </summary>
    public sealed record CommandLine(SelfPlaySettings Settings, bool ShowHelp)
    {
        public const string Usage = """
            Usage: SelfPlayCollector [options]
                   SelfPlayCollector policy-targets RUN_DIRECTORY

            Plays games with all four seats searching with MCTS (max^n) on one value net and records the positions of
            full searches with the search's values and visits, plus every game's result and save file.
            policy-targets writes the encoded legal moves of every position for a run collected without them.

              --games N                Games to play (default 1000)
              --first-game I           Number the games from I (default 0), to extend a run with the same seed
              --concurrent N           Games played at once (default 256)
              --leaves L               Leaves a search selects before their evaluation (default 16)
              --full-iterations N      Iterations of a recorded full search (default 800)
              --fast-iterations N      Iterations of a fast search, not recorded (default 150)
              --full-probability P     Probability of a full search per decision (default 0.25)
              --temperature T          Temperature of the sampled moves at the start of a game (default 1, 0 for none)
              --temperature-moves M    Searched decisions per game with sampled moves (default 30)
              --exploration C          UCT exploration constant, PUCT with a policy head (default 1.414)
              --root-noise W           Weight of the Dirichlet noise on the root priors of full searches, with a policy
                                       head (default 0.25, 0 for none)
              --noise-concentration A  Total concentration of the root noise, alpha per move = A / moves (default 10)
              --no-target-pruning      Record all root visits, also those only the root noise caused
              --max-actions N          Stop games after N actions, written without a result (default 3000)
              --model PATH             Value net (default: the deployed model)
              --nets N                 Net sessions evaluating batches in turn (default 2)
              --batch N                Most rows per net batch (default 1024)
              --provider cpu|dml|cuda  Where the net runs (default: the GPU if available, DirectML on Windows, CUDA on Linux)
              --seed S                 Run seed (default: random), every game seed is derived from it
              --center-desert          Generate maps with the desert in the center
              --output DIR             Directory to create the run in (default selfplay)
              --name NAME              Name of the run directory (default from the settings and the time)
              --help                   Show this help
            """;

        public static CommandLine Parse(string[] args)
        {
            SelfPlaySettings settings = new();
            bool showHelp = false;

            for (int i = 0; i < args.Length; i++)
            {
                string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
                int IntValue() => int.Parse(Value(), CultureInfo.InvariantCulture);
                double DoubleValue() => double.Parse(Value(), CultureInfo.InvariantCulture);

                switch (args[i])
                {
                    case "--games": settings = settings with { Games = IntValue() }; break;
                    case "--first-game": settings = settings with { FirstGame = IntValue() }; break;
                    case "--concurrent": settings = settings with { ConcurrentGames = IntValue() }; break;
                    case "--leaves": settings = settings with { LeavesPerBatch = IntValue() }; break;
                    case "--full-iterations": settings = settings with { FullIterations = IntValue() }; break;
                    case "--fast-iterations": settings = settings with { FastIterations = IntValue() }; break;
                    case "--full-probability": settings = settings with { FullSearchProbability = DoubleValue() }; break;
                    case "--temperature": settings = settings with { Temperature = DoubleValue() }; break;
                    case "--temperature-moves": settings = settings with { TemperatureMoves = IntValue() }; break;
                    case "--exploration": settings = settings with { Exploration = DoubleValue() }; break;
                    case "--root-noise": settings = settings with { RootNoise = DoubleValue() }; break;
                    case "--noise-concentration": settings = settings with { RootNoiseConcentration = DoubleValue() }; break;
                    case "--no-target-pruning": settings = settings with { PruneTargets = false }; break;
                    case "--max-actions": settings = settings with { MaxActions = IntValue() }; break;
                    case "--model": settings = settings with { ModelPath = Value() }; break;
                    case "--nets": settings = settings with { Nets = IntValue() }; break;
                    case "--batch": settings = settings with { BatchRows = IntValue() }; break;
                    case "--provider":
                        settings = settings with
                        {
                            Provider = Value() switch
                            {
                                "cpu" => InferenceProvider.Cpu,
                                "dml" => InferenceProvider.DirectML,
                                "cuda" => InferenceProvider.Cuda,
                                string other => throw new ArgumentException($"Unknown provider \"{other}\" (cpu, dml or cuda)."),
                            }
                        };
                        break;
                    case "--seed": settings = settings with { Seed = ulong.Parse(Value(), CultureInfo.InvariantCulture) }; break;
                    case "--center-desert": settings = settings with { CenterDesert = true }; break;
                    case "--output": settings = settings with { OutputDirectory = Value() }; break;
                    case "--name": settings = settings with { Name = Value() }; break;
                    case "--help" or "-h": showHelp = true; break;
                    default: throw new ArgumentException($"Unknown option \"{args[i]}\".");
                }
            }

            settings.Validate();
            return new CommandLine(settings, showHelp);
        }
    }
}
