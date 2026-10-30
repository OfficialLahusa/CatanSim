using Common;
using System.Globalization;

namespace DatasetCollector
{
    /// <summary>
    /// Settings of a collection run, from the command line
    /// </summary>
    public sealed record CollectorSettings
    {
        public int Threads { get; init; } = Environment.ProcessorCount;
        public int Matches { get; init; } = 100;

        /// <summary>
        /// Index of the first match: matches are numbered from it (seeds and sample names), so separately collected parts
        /// of one run don't overlap
        /// </summary>
        public int FirstMatch { get; init; }

        /// <summary>
        /// Number of partial runs (subdirectories part_00, part_01, ...) the matches are split into; see PartialRuns
        /// </summary>
        public int Parts { get; init; } = 1;
        public int SamplesPerMatch { get; init; } = 5;
        public int PlayoutsPerSample { get; init; } = 300;

        /// <summary>
        /// Sampled states per match that also get a sibling group: the states after several of the acting player's moves,
        /// labeled with the same playout seeds as the sampled state
        /// </summary>
        public int SiblingGroupsPerMatch { get; init; } = 1;

        /// <summary>
        /// Most siblings per group (fewer if the player has fewer distinct moves)
        /// </summary>
        public int SiblingsPerGroup { get; init; } = 6;
        public ulong Seed { get; init; } = Seeds.NewRunSeed();
        public AgentMix Agents { get; init; } = AgentMix.Default;
        public bool CenterDesert { get; init; }

        /// <summary>
        /// Directory the dataset directory is created in
        /// </summary>
        public string OutputDirectory { get; init; } = "datasets";

        /// <summary>
        /// Name of the dataset directory, generated from the settings and the time if null
        /// </summary>
        public string? Name { get; init; }

        public string DatasetName => Name ?? $"m{Matches}_s{SamplesPerMatch}_p{PlayoutsPerSample}_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}";

        public void Validate()
        {
            if (Threads < 1) throw new ArgumentException("--threads must be at least 1.");
            if (Matches < 0) throw new ArgumentException("--matches must not be negative.");
            if (FirstMatch < 0) throw new ArgumentException("--first-match must not be negative.");
            if (Parts < 1 || (Matches > 0 && Parts > Matches)) throw new ArgumentException("--parts must be between 1 and the number of matches.");
            if (SamplesPerMatch < 1) throw new ArgumentException("--samples must be at least 1.");
            if (PlayoutsPerSample < 1) throw new ArgumentException("--playouts must be at least 1.");
            if (SiblingGroupsPerMatch < 0) throw new ArgumentException("--sibling-groups must not be negative.");
            if (SiblingGroupsPerMatch > 0 && SiblingsPerGroup < 2) throw new ArgumentException("--siblings must be at least 2.");
            Agents.Validate();
        }
    }

    /// <summary>
    /// What the command line asks for: a run with the settings, after an optional thread calibration
    /// </summary>
    public sealed record CommandLine(CollectorSettings Settings, bool CalibrateThreads, bool CalibrateOnly, TimeSpan CalibrationTime, bool ShowHelp,
        (string Seeds, string Output)? RegenerateMaps = null, (string Index, string States, string Output)? RegenerateViews = null)
    {
        public const string Usage = """
            Usage: DatasetCollector [options]

              --matches N            Matches to play (default 100)
              --samples K            States sampled per match, uniformly over all positions including the empty
                                     board and the ended match (default 5)
              --playouts P           Playouts per sampled state (default 300), ended matches need none
              --sibling-groups G     Sampled states per match that get a sibling group (default 1, 0 for none):
                                     the states after several moves of the acting player, labeled with the
                                     same playout seeds, so moves can be compared with little noise
              --siblings S           Most siblings per group (default 6)
              --parts P              Split the matches into P partial runs <name>/part_00, ... (default 1). Running
                                     the same command again skips finished parts and redoes an unfinished one; the
                                     result equals a single run with the same seed. Needs --name to resume.
              --first-match I        Number the matches from I (default 0), e.g. to split one run with one seed
                                     between machines without overlap
              --threads N|auto       Worker threads (default: all logical cores); auto calibrates first
              --seed S               Run seed (default: random), every match seed is derived from it
              --agents SPEC          Agent population for matches and playouts, drawn per seat
                                     (default random=0.3,simple=0.35,builder=0.35)
              --exploration MIN-MAX  Share of random actions of SimpleAgents and BuilderAgents, drawn per seat
                                     (default 0.1-0.2)
              --center-desert        Generate maps with the desert in the center
              --output DIR           Directory to create the dataset in (default datasets)
              --name NAME            Name of the dataset directory (default from the settings and the time)
              --calibrate            Only calibrate the thread count with the other settings and report it
                                     (tries all, 3/4, 1/2 and 1/4 of the logical cores and 32 and 24 threads)
              --calibration-seconds  Seconds per calibration run (default 20, every thread count runs twice)
              --regenerate-maps SEEDS OUT
                                     Instead of collecting: regenerate the maps of the matches with the seeds in
                                     SEEDS (one per line) and write their ports and tiles to OUT, to extend a
                                     dataset of an older feature layout; needs the run's --agents, --exploration
                                     and --center-desert
              --regenerate-views INDEX STATES OUT
                                     Instead of collecting: write the belief-aware inputs (ViewVectorizer) of a
                                     prepared dataset (its dataset_index.csv and states.npy) to OUT (.npy), playing
                                     every match again and checking each state against its stored row; needs the
                                     run's --seed, --agents, --exploration, --samples, --sibling-groups, --siblings
                                     and --center-desert
              --help                 Show this help
            """;

        public static CommandLine Parse(string[] args)
        {
            CollectorSettings settings = new();
            bool calibrateThreads = false, calibrateOnly = false, showHelp = false;
            TimeSpan calibrationTime = TimeSpan.FromSeconds(20);
            (string, string)? regenerateMaps = null;
            (string, string, string)? regenerateViews = null;
            string agents = "random=0.3,simple=0.35,builder=0.35", exploration = "0.1-0.2";

            for (int i = 0; i < args.Length; i++)
            {
                string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
                int IntValue() => int.Parse(Value(), CultureInfo.InvariantCulture);

                switch (args[i])
                {
                    case "--matches": settings = settings with { Matches = IntValue() }; break;
                    case "--samples": settings = settings with { SamplesPerMatch = IntValue() }; break;
                    case "--playouts": settings = settings with { PlayoutsPerSample = IntValue() }; break;
                    case "--sibling-groups": settings = settings with { SiblingGroupsPerMatch = IntValue() }; break;
                    case "--siblings": settings = settings with { SiblingsPerGroup = IntValue() }; break;
                    case "--parts": settings = settings with { Parts = IntValue() }; break;
                    case "--first-match": settings = settings with { FirstMatch = IntValue() }; break;
                    case "--threads":
                        string threads = Value();
                        if (threads == "auto") calibrateThreads = true;
                        else settings = settings with { Threads = int.Parse(threads, CultureInfo.InvariantCulture) };
                        break;
                    case "--seed": settings = settings with { Seed = ulong.Parse(Value(), CultureInfo.InvariantCulture) }; break;
                    case "--agents": agents = Value(); break;
                    case "--exploration": exploration = Value(); break;
                    case "--center-desert": settings = settings with { CenterDesert = true }; break;
                    case "--output": settings = settings with { OutputDirectory = Value() }; break;
                    case "--name": settings = settings with { Name = Value() }; break;
                    case "--calibrate": calibrateThreads = calibrateOnly = true; break;
                    case "--calibration-seconds": calibrationTime = TimeSpan.FromSeconds(double.Parse(Value(), CultureInfo.InvariantCulture)); break;
                    case "--regenerate-maps": regenerateMaps = (Value(), Value()); break;
                    case "--regenerate-views": regenerateViews = (Value(), Value(), Value()); break;
                    case "--help" or "-h": showHelp = true; break;
                    default: throw new ArgumentException($"Unknown option \"{args[i]}\".");
                }
            }

            settings = settings with { Agents = AgentMix.Parse(agents, exploration) };
            settings.Validate();
            return new CommandLine(settings, calibrateThreads, calibrateOnly, calibrationTime, showHelp, regenerateMaps, regenerateViews);
        }
    }
}
