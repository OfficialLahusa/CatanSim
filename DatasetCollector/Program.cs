namespace DatasetCollector
{
    public class Program
    {
        public static int Main(string[] args)
        {
            Console.WriteLine("CettlersOfSatan Dataset Collector");
            Console.WriteLine("=================================");

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

            CollectorSettings settings = commandLine.Settings;
            if (commandLine.CalibrateThreads)
            {
                int threads = ThreadCalibration.FindBestThreadCount(settings, commandLine.CalibrationTime, Console.Out);
                if (commandLine.CalibrateOnly)
                {
                    Console.WriteLine($"\nFastest: --threads {threads}");
                    return 0;
                }
                settings = settings with { Threads = threads };
            }

            CollectionResult result = new DatasetCollector(settings).Collect();
            return result.Errors > 0 ? 2 : 0;
        }
    }
}
