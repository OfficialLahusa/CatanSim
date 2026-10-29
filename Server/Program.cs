using Common;
using Simulation;

namespace Server
{
    /// <summary>
    /// Dedicated simulation server: hosts one match over WebSockets, with an admin console on standard input.
    /// Usage: Server [--config server.json] [--port N] [--tls]
    /// --tls serves wss:// with a certificate next to the configuration (self-signed and created on first use); clients pin
    /// its fingerprint, which the server prints
    /// </summary>
    public static class Program
    {
        public static async Task<int> Main(string[] args)
        {
            // The status lines may show names outside the console's code page (players name themselves)
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
                // Watching shows every hand until hidden information is filtered per session, so it needs a token by default
                config = new ServerConfig { BindAddress = "0.0.0.0", AllowAnonymousSpectators = false };
                string adminToken = config.AddToken(Role.Admin, null, "First admin").token;
                config.Save(path);
                Console.WriteLine($"Wrote {Path.GetFullPath(path)} with a new admin token (shown once): {adminToken}");
            }
            if (Option("--port") is string port) config = config with { Port = int.Parse(port) };
            if (args.Contains("--tls") && config.CertificatePath == null)
            {
                config = config with { CertificatePath = Path.ChangeExtension(Path.GetFullPath(path), ".pfx") };
                config.Save(path);
            }

            using Match match = new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human]);
            match.Start();
            ServerConsole console = new ServerConsole(match, config, path);
            await using SimulationServer server = await SimulationServer.StartAsync(match, config, path);
            console.Server = server;

            Console.WriteLine($"Hexfarer server listening on {(server.Secure ? "wss" : "ws")}://{config.BindAddress}:{server.Port}{Protocol.Path}. Type help for the commands.");
            if (server.CertificateFingerprint is string fingerprint)
                Console.WriteLine($"Certificate fingerprint (clients enter it to trust this server): {fingerprint}");
            if (config.AllowRemoteAdmins && !server.Secure)
                Console.WriteLine("Warning: admins may log in from other computers without TLS, so their token travels in plain text. Start with --tls.");
            else if (!config.AllowRemoteAdmins)
                Console.WriteLine("Admins log in from this computer only (AllowRemoteAdmins in the configuration, best together with --tls).");
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
