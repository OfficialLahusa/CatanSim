namespace SelfPlayCollector
{
    public class Program
    {
        public static int Main(string[] args)
        {
            Console.WriteLine("CettlersOfSatan Self-Play Collector");
            Console.WriteLine("===================================");

            // Policy files for a run collected before they were written: SelfPlayCollector policy-targets RUN_DIRECTORY
            if (args.Length == 2 && args[0] == "policy-targets")
            {
                try
                {
                    long positions = PolicyTargetWriter.ConvertRun(args[1], Console.Out);
                    Console.WriteLine($"Policy targets of {positions} positions written to {args[1]}");
                    return 0;
                }
                catch (Exception e) when (e is InvalidOperationException or IOException)
                {
                    Console.Error.WriteLine(e.Message);
                    return 1;
                }
            }

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

            if (commandLine.ShowHelp)
            {
                Console.WriteLine(CommandLine.Usage);
                return 0;
            }

            // Ctrl+C stops the run soon; the games written so far stay complete
            using CancellationTokenSource cancellation = new();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellation.Cancel();
            };

            try
            {
                SelfPlayResult result = new SelfPlayCollector(commandLine.Settings).Collect(cancellation.Token);
                return result.Errors > 0 ? 2 : 0;
            }
            catch (InvalidOperationException e)
            {
                Console.Error.WriteLine(e.Message);
                return 1;
            }
        }
    }
}
