using Common;
using Simulation;

namespace Server
{
    /// <summary>
    /// Dedicated simulation server: hosts one match over WebSockets, with an admin console on standard input.
    /// Usage: Server [--config server.json] [--port N]
    /// </summary>
    public static class Program
    {
        public static async Task<int> Main(string[] args)
        {
            // The status lines use characters outside the console's code page (seat names like "MCTS · Medium")
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            string? Option(string name) => Array.IndexOf(args, name) is int i && i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
            string path = Option("--config") ?? "server.json";

            // A first start writes a configuration with an admin token, shown once
            ServerConfig config;
            if (File.Exists(path))
            {
                config = ServerConfig.Load(path);
            }
            else
            {
                string adminToken = ServerConfig.NewToken();
                config = new ServerConfig { BindAddress = "0.0.0.0", Tokens = [new TokenEntry(ServerConfig.HashToken(adminToken), Role.Admin, null)] };
                config.Save(path);
                Console.WriteLine($"Wrote {Path.GetFullPath(path)} with a new admin token (shown once): {adminToken}");
            }
            if (Option("--port") is string port) config = config with { Port = int.Parse(port) };

            using Match match = new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human]);
            match.Start();
            ServerConsole console = new ServerConsole(match, config, path);
            await using SimulationServer server = await SimulationServer.StartAsync(match, config);
            console.Server = server;

            Console.WriteLine($"CatanSim server listening on ws://{config.BindAddress}:{server.Port}{Protocol.Path}. Type help for the commands.");
            using CancellationTokenSource stop = new();
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                stop.Cancel();
            };

            while (!stop.IsCancellationRequested)
            {
                string? line = await Task.Run(Console.ReadLine, stop.Token).WaitAsync(stop.Token).ContinueWith(t => t.IsCompletedSuccessfully ? t.Result : null);
                if (line == null || line.Trim().Equals("quit", StringComparison.OrdinalIgnoreCase)) break;
                string output = console.Execute(line);
                if (output.Length > 0) Console.WriteLine(output);
            }

            Console.WriteLine("Stopping.");
            return 0;
        }
    }
}
