using Agents.Inference;
using System.Globalization;

namespace Arena
{
    public class Program
    {
        public static int Main(string[] args)
        {
            CommandLine commandLine;
            try
            {
                commandLine = CommandLine.Parse(args);
            }
            catch (Exception e) when (e is ArgumentException or FormatException or OverflowException)
            {
                Console.Error.WriteLine(e.Message);
                Console.Error.WriteLine();
                Console.Error.WriteLine(CommandLine.Usage);
                return 1;
            }

            if (commandLine.Settings == null)
            {
                Console.WriteLine(CommandLine.Usage);
                return 0;
            }

            // Ctrl+C stops the duel soon and prints the result so far
            using CancellationTokenSource cancellation = new();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellation.Cancel();
            };

            PairStatistics result = new MirrorDuel(commandLine.Settings, Console.Out).Run(cancellation.Token);
            // 10 if H1 is accepted, 11 if H0 is accepted, 0 if the test stayed open
            return result.Verdict switch { 1 => 10, -1 => 11, _ => 0 };
        }
    }

    /// <summary>
    /// What the command line asks for; no settings for the help
    /// </summary>
    internal sealed record CommandLine(MirrorSettings? Settings)
    {
        public const string Usage = """
            Usage: Arena mirror --challenger AGENT --baseline AGENT [options]

            Mirror duel of two MCTS agents with two seats each: every map is played twice with the sides swapped. Stops when
            the SPRT on the map pairs decides, after --max-games or after --max-minutes. Exit code 10 if H1 is accepted (the
            challenger is stronger by about ELO1), 11 if H0 is accepted, 0 if the test is still open.

            An AGENT is NAME=MODEL.onnx[=OPTION...] with the options MaxN, Paranoid or Random (opponent model, default MaxN),
            cX (exploration constant, default 1.414), bN (leaves per batch, default 256), fX (first play reduction, default
            0.1), pX (placement weight, default 0) and u (uniform priors instead of the policy head's).

              --challenger AGENT     The agent tested
              --baseline AGENT       The agent it is compared with
              --iterations N         MCTS iterations per decision (default 5000)
              --seconds S            Seconds per decision instead of iterations; games then run one at a time
              --parallel N           Games played at once with an iteration budget (default 16)
              --max-games N          Most games (default 4000)
              --name NAME            Names the duel; the maps are drawn from a seed derived from it
              --seed-base N          First map seed instead (default 30000)
              --sprt ELO0 ELO1       Hypotheses of the SPRT (default 0 40)
              --alpha A              False positive rate of the SPRT (default 0.05)
              --beta B               False negative rate of the SPRT (default 0.05)
              --max-minutes M        Stop after M minutes even if the SPRT is open (default 120)
              --provider cpu|dml|cuda  Where the nets run (default: the GPU if available)
              --help                 Show this help
            """;

        public static CommandLine Parse(string[] args)
        {
            if (args.Length == 0 || args.Any(a => a is "--help" or "-h"))
                return new CommandLine((MirrorSettings?)null);
            if (args[0] != "mirror")
                throw new ArgumentException($"Unknown mode \"{args[0]}\".");

            AgentSpec? challenger = null, baseline = null;
            string? name = null;
            int? seedBase = null;
            long iterations = 5000;
            double? seconds = null;
            var settings = new MirrorSettings { Challenger = null!, Baseline = null! };

            for (int i = 1; i < args.Length; i++)
            {
                string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
                int IntValue() => int.Parse(Value(), CultureInfo.InvariantCulture);
                double DoubleValue() => double.Parse(Value(), CultureInfo.InvariantCulture);

                switch (args[i])
                {
                    case "--challenger": challenger = AgentSpec.Parse(Value()); break;
                    case "--baseline": baseline = AgentSpec.Parse(Value()); break;
                    case "--iterations": iterations = long.Parse(Value(), CultureInfo.InvariantCulture); break;
                    case "--seconds": seconds = DoubleValue(); break;
                    case "--parallel": settings = settings with { Parallel = IntValue() }; break;
                    case "--max-games": settings = settings with { MaxGames = IntValue() }; break;
                    case "--name": name = Value(); break;
                    case "--seed-base": seedBase = IntValue(); break;
                    case "--sprt": settings = settings with { Elo0 = DoubleValue(), Elo1 = DoubleValue() }; break;
                    case "--alpha": settings = settings with { Alpha = DoubleValue() }; break;
                    case "--beta": settings = settings with { Beta = DoubleValue() }; break;
                    case "--max-minutes": settings = settings with { MaxDuration = TimeSpan.FromMinutes(DoubleValue()) }; break;
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
                    default: throw new ArgumentException($"Unknown option \"{args[i]}\".");
                }
            }

            if (challenger == null || baseline == null)
                throw new ArgumentException("A mirror duel needs --challenger and --baseline.");
            if (challenger.Name == baseline.Name)
                throw new ArgumentException("The challenger and the baseline need different names.");
            if (settings.Elo1 <= settings.Elo0)
                throw new ArgumentException("--sprt needs ELO0 < ELO1.");
            if (settings.MaxGames < 2 || settings.Parallel < 1)
                throw new ArgumentException("--max-games needs at least 2 and --parallel at least 1.");

            return new CommandLine(settings with
            {
                Challenger = challenger,
                Baseline = baseline,
                Iterations = seconds == null ? iterations : 0,
                Seconds = seconds ?? settings.Seconds,
                SeedBase = seedBase ?? (name == null ? settings.SeedBase : SeedOf(name)),
            });
        }

        /// <summary>
        /// A map seed from 100,000 on derived from the duel's name (FNV-1a), the same on every run and platform, so duels
        /// with different names draw different maps
        /// </summary>
        public static int SeedOf(string name)
        {
            uint hash = 2166136261;
            foreach (char c in name)
                hash = (hash ^ c) * 16777619;
            return 100_000 + (int)(hash % 1_000_000_000);
        }
    }
}
