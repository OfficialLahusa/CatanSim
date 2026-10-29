using Common;
using Simulation;

namespace Tests
{
    /// <summary>
    /// The dedicated server's configuration and admin console
    /// </summary>
    public class ServerConsoleTests
    {
        [Fact]
        public void ConfigsKeepOnlyTokenHashesAndRoundTrip()
        {
            string path = Path.Combine(Path.GetTempPath(), $"catansim-server-{Guid.NewGuid():N}.json");
            try
            {
                ServerConfig config = new() { Port = 18018, BindAddress = "0.0.0.0", AllowAnonymousSpectators = false,
                    Tokens = [new TokenEntry(ServerConfig.HashToken("secret"), Role.Player, 3)] };
                config.Save(path);

                Assert.DoesNotContain("\"secret\"", File.ReadAllText(path));
                ServerConfig loaded = ServerConfig.Load(path);
                Assert.Equal(18018, loaded.Port);
                Assert.Equal("0.0.0.0", loaded.BindAddress);
                Assert.False(loaded.AllowAnonymousSpectators);
                TokenEntry entry = Assert.Single(loaded.Tokens);
                Assert.Equal(Role.Player, entry.Role);
                Assert.Equal(3, entry.Seat);
                Assert.Same(loaded.Tokens[0], loaded.Find("secret"));
                Assert.Null(loaded.Find("Secret"));
                Assert.Null(loaded.Find(null));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void NewTokensAreRandom()
        {
            Assert.NotEqual(ServerConfig.NewToken(), ServerConfig.NewToken());
            Assert.True(ServerConfig.NewToken().Length >= 12);
        }

        private static (ServerConsole console, Match match) NewConsole()
        {
            Utils.Random.SeedCurrentThread(5);
            Match match = new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human], valueNetAvailable: false);
            return (new ServerConsole(match, new ServerConfig { Port = 0 }, path: null), match);
        }

        [Fact]
        public void TheConsoleAddsTokensThatAdmitTheirRole()
        {
            (ServerConsole console, Match match) = NewConsole();
            using (match)
            {
                string output = console.Execute("token player 2");
                string token = output.Split(' ', StringSplitOptions.RemoveEmptyEntries).Last();
                TokenEntry? entry = console.Config.Find(token);
                Assert.NotNull(entry);
                Assert.Equal(Role.Player, entry!.Role);
                Assert.Equal(1, entry.Seat);

                Assert.Contains("seat", console.Execute("token player"));
                Assert.Contains("Unknown", console.Execute("token wizard"));
                Assert.Equal(Role.Admin, console.Config.Find(console.Execute("token admin").Split(' ').Last())!.Role);
            }
        }

        [Fact]
        public void TheConsoleReportsAndControlsTheMatch()
        {
            (ServerConsole console, Match match) = NewConsole();
            using (match)
            {
                Assert.Contains("Human", console.Execute("status"));
                Assert.Contains("No connections", console.Execute("sessions"));

                console.Execute("seat 1 builder");
                Assert.Equal(SeatKind.Builder, match.Seats[0]);
                Assert.Contains("Unknown seat kind", console.Execute("seat 1 dragon"));

                console.Execute("newgame random random random human");
                Assert.Equal([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Human], match.Seats);

                Assert.Contains("pause", console.Execute("pause").ToLower());
                Assert.True(match.Playback.Paused);
                console.Execute("resume");
                Assert.False(match.Playback.Paused);

                Assert.Contains("help", console.Execute("nonsense").ToLower());
                Assert.Contains("token", console.Execute("help"));
            }
        }

        [Fact]
        public async Task TheConsoleListsAndKicksConnections()
        {
            (ServerConsole console, Match match) = NewConsole();
            using (match)
            {
                await using SimulationServer server = await SimulationServer.StartAsync(match, console.Config);
                console.Server = server;
                using RemoteSimulation spectator = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Sam", null, Role.Spectator, null, null));

                string sessions = console.Execute("sessions");
                Assert.Contains("Sam", sessions);
                Assert.Contains(spectator.Session!.Id, sessions);

                console.Execute($"kick {spectator.Session.Id}");
                System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
                while (spectator.IsConnected && clock.ElapsedMilliseconds < 10000) await Task.Delay(20);
                Assert.False(spectator.IsConnected);
            }
        }
    }
}
