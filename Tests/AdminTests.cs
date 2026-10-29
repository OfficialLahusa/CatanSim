using Common;
using Simulation;
using System.Net;
using System.Net.WebSockets;
using System.Text;

namespace Tests
{
    /// <summary>
    /// The admin side of a hosted match (tokens, server options, what admins see) and the hardening of the login
    /// </summary>
    public class AdminTests
    {
        private static readonly SeatKind[] AllHuman = [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human];

        private static Match NewMatch()
        {
            Utils.Random.SeedCurrentThread(1);
            return new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), AllHuman);
        }

        private static ServerConfig Config() => new()
        {
            Port = 0,
            Tokens =
            [
                new TokenEntry(ServerConfig.HashToken("admin-secret"), Role.Admin, null),
                new TokenEntry(ServerConfig.HashToken("admin2-secret"), Role.Admin, null),
                new TokenEntry(ServerConfig.HashToken("seat2-secret"), Role.Player, 2),
            ],
            AllowAnonymousSpectators = true
        };

        private static bool WaitFor(Func<bool> condition, int milliseconds = 10000)
        {
            DateTime end = DateTime.UtcNow.AddMilliseconds(milliseconds);
            while (DateTime.UtcNow < end)
            {
                if (condition()) return true;
                Thread.Sleep(10);
            }
            return condition();
        }

        private static bool Pump(ISimulation client, Func<bool> condition, int milliseconds = 10000)
        {
            return WaitFor(() => { client.ProcessEvents(); return condition(); }, milliseconds);
        }

        private static Task<RemoteSimulation> Connect(SimulationServer server, string? token, Role role, int? seat = null, string name = "Client")
        {
            return RemoteSimulation.ConnectAsync(server.Uri, new Hello(name, token, role, seat, null));
        }

        #region Tokens
        [Fact]
        public async Task AdminsCreateTokensThatAdmitTheirRole()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation admin = await Connect(server, "admin-secret", Role.Admin);

            admin.Send(new CreateToken(Role.Player, 1, "Ann"));
            Assert.True(Pump(admin, () => admin.Game.CreatedToken != null));
            TokenCreatedEvent created = admin.Game.CreatedToken!;
            Assert.Equal(Role.Player, created.Info.Role);
            Assert.Equal(1, created.Info.Seat);
            Assert.Equal("Ann", created.Info.Label);
            Assert.True(created.Token.Length >= 43);

            // The list shows the token without its secret
            Assert.True(Pump(admin, () => admin.Game.Tokens.Any(t => t.Id == created.Info.Id)));
            Assert.DoesNotContain(admin.Game.Tokens, t => t.Label.Contains(created.Token));

            using RemoteSimulation ann = await Connect(server, created.Token, Role.Player, 1, "Ann");
            Assert.Equal(Role.Player, ann.Session!.Role);
            Assert.Equal(1, ann.Session.Seat);
        }

        [Fact]
        public async Task OnlyTheAdminWhoAskedSeesTheNewToken()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation first = await Connect(server, "admin-secret", Role.Admin, name: "First");
            using RemoteSimulation second = await Connect(server, "admin2-secret", Role.Admin, name: "Second");
            using RemoteSimulation spectator = await Connect(server, null, Role.Spectator);

            first.Send(new CreateToken(Role.Spectator, null, "Guests"));
            Assert.True(Pump(first, () => first.Game.CreatedToken != null));
            string id = first.Game.CreatedToken!.Info.Id;

            // The other admin sees the new entry in the list, but not the token; the spectator sees neither
            Assert.True(Pump(second, () => second.Game.Tokens.Any(t => t.Id == id)));
            Assert.Null(second.Game.CreatedToken);
            Thread.Sleep(200);
            spectator.ProcessEvents();
            Assert.Empty(spectator.Game.Tokens);
            Assert.Null(spectator.Game.CreatedToken);
            Assert.Null(spectator.Game.Server);
        }

        [Fact]
        public async Task RevokedTokensNoLongerAdmitAndTheirConnectionsClose()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation admin = await Connect(server, "admin-secret", Role.Admin);

            admin.Send(new CreateToken(Role.Player, 0, "Bob"));
            Assert.True(Pump(admin, () => admin.Game.CreatedToken != null));
            TokenCreatedEvent created = admin.Game.CreatedToken!;
            using RemoteSimulation bob = await Connect(server, created.Token, Role.Player, 0, "Bob");

            admin.Send(new RevokeToken(created.Info.Id));
            Assert.True(Pump(admin, () => admin.Game.Tokens.All(t => t.Id != created.Info.Id)));
            Assert.True(Pump(bob, () => !bob.IsConnected));
            await Assert.ThrowsAnyAsync<Exception>(() => Connect(server, created.Token, Role.Player, 0, "Bob"));
        }

        [Fact]
        public async Task OnlyAdminsManageTokensAndOptions()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation player = await Connect(server, "seat2-secret", Role.Player, 2);
            int before = server.Tokens.Count;

            player.Send(new CreateToken(Role.Admin, null, "Mine"));
            player.Send(new RevokeToken(server.Tokens[0].Id));
            player.Send(new SetServerOptions(false));
            Assert.True(Pump(player, () => player.Errors.Count >= 3));
            Assert.Equal(before, server.Tokens.Count);
            Assert.True(server.Config.AllowAnonymousSpectators);
            Assert.Null(player.Game.CreatedToken);
        }

        [Fact]
        public async Task TokenChangesAreSavedToTheConfig()
        {
            string path = Path.Combine(Path.GetTempPath(), $"catansim-admin-{Guid.NewGuid():N}.json");
            try
            {
                using Match match = NewMatch();
                await using (SimulationServer server = await SimulationServer.StartAsync(match, Config(), path))
                {
                    (TokenInfo info, string token) = server.CreateToken(Role.Agent, 3, "Bridge");
                    ServerConfig saved = ServerConfig.Load(path);
                    TokenEntry entry = Assert.Single(saved.Tokens, t => t.Id == info.Id);
                    Assert.Equal(ServerConfig.HashToken(token), entry.Hash);
                    Assert.Equal("Bridge", entry.Label);
                    Assert.DoesNotContain(token, File.ReadAllText(path));

                    Assert.True(server.RevokeToken(info.Id));
                    Assert.DoesNotContain(ServerConfig.Load(path).Tokens, t => t.Id == info.Id);
                }
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void OldConfigsGetTokenIds()
        {
            string path = Path.Combine(Path.GetTempPath(), $"catansim-old-{Guid.NewGuid():N}.json");
            try
            {
                File.WriteAllText(path, """
                    { "Port": 17017, "BindAddress": "0.0.0.0", "AllowAnonymousSpectators": true,
                      "Tokens": [ { "Hash": "ab12", "Role": "Admin", "Seat": null } ] }
                    """);
                ServerConfig config = ServerConfig.Load(path);
                TokenEntry entry = Assert.Single(config.Tokens);
                Assert.False(string.IsNullOrEmpty(entry.Id));
                Assert.Equal(entry.Id, ServerConfig.Load(path).Tokens[0].Id);
                Assert.False(config.AllowRemoteAdmins);
            }
            finally
            {
                File.Delete(path);
            }
        }
        #endregion

        #region What admins see
        [Fact]
        public async Task AdminsSeeTheServerAndItsConnections()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation admin = await Connect(server, "admin-secret", Role.Admin, name: "Root");
            using RemoteSimulation spectator = await Connect(server, null, Role.Spectator, name: "Sam");

            Assert.True(Pump(admin, () => admin.Game.Server != null && admin.Game.Sessions.Count == 2 && admin.Game.Tokens.Count == 3));
            ServerInfo info = admin.Game.Server!;
            Assert.Equal(server.Port, info.Port);
            Assert.False(info.Secure);
            Assert.True(info.AllowAnonymousSpectators);
            Assert.Contains(admin.Game.Sessions, s => s.Name == "Sam" && s.Role == Role.Spectator);
        }

        [Fact]
        public async Task AdminsSwitchAnonymousSpectatorsOff()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation admin = await Connect(server, "admin-secret", Role.Admin);

            admin.Send(new SetServerOptions(false));
            Assert.True(Pump(admin, () => admin.Game.Server is { AllowAnonymousSpectators: false }));
            await Assert.ThrowsAnyAsync<Exception>(() => Connect(server, null, Role.Spectator));
        }

        [Fact]
        public async Task TheHostingClientGetsTheAdminEventsLocally()
        {
            using Match match = NewMatch();
            using LocalSimulation host = new LocalSimulation(match);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation guest = await Connect(server, null, Role.Spectator, name: "Guest");

            Assert.True(Pump(host, () => host.Game.Server != null && host.Game.Sessions.Any(s => s.Name == "Guest")));
            host.Send(new CreateToken(Role.Player, 3, "Dora"));
            Assert.True(Pump(host, () => host.Game.CreatedToken != null && host.Game.Tokens.Any(t => t.Label == "Dora")));
            using RemoteSimulation dora = await Connect(server, host.Game.CreatedToken!.Token, Role.Player, 3, "Dora");
            Assert.Equal(3, dora.Session!.Seat);
        }

        [Fact]
        public void TokenCommandsWithoutAServerAreRefused()
        {
            using Match match = NewMatch();
            using LocalSimulation local = new LocalSimulation(match);
            local.Send(new CreateToken(Role.Player, 0, "Nobody"));
            List<MatchEvent> events = local.ProcessEvents();
            Assert.Contains(events, e => e is ErrorEvent);
            Assert.Null(local.Game.CreatedToken);
        }

        [Fact]
        public void TheAdminMessagesRoundTrip()
        {
            DateTimeOffset created = new DateTimeOffset(2026, 10, 29, 12, 0, 0, TimeSpan.Zero);
            TokenInfo info = new TokenInfo("t1", Role.Player, 2, "Ann", created);
            List<Command> commands = [new CreateToken(Role.Player, 2, "Ann"), new CreateToken(Role.Admin, null, ""), new RevokeToken("t1"), new SetServerOptions(false)];
            foreach (Command c in commands)
                Assert.Equal(Protocol.Serialize(c), Protocol.Serialize(Protocol.DeserializeCommand(Protocol.Serialize(c))));

            List<MatchEvent> events =
            [
                new TokenCreatedEvent(info, "secret"),
                new TokensEvent([info, info with { Id = "t2", Role = Role.Admin, Seat = null }]),
                new ServerInfoEvent(new ServerInfo(17017, "0.0.0.0", true, "ab:cd", false, true)),
                new ServerInfoEvent(null),
            ];
            foreach (MatchEvent e in events)
                Assert.Equal(Protocol.Serialize(e), Protocol.Serialize(Protocol.DeserializeEvent(Protocol.Serialize(e))));
        }
        #endregion

        #region Login hardening
        [Fact]
        public void NewTokensHave256RandomBits()
        {
            HashSet<string> tokens = Enumerable.Range(0, 200).Select(_ => ServerConfig.NewToken()).ToHashSet();
            Assert.Equal(200, tokens.Count);
            Assert.All(tokens, t => Assert.True(t.Length >= 43 && t.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_')));
        }

        [Fact]
        public async Task RepeatedWrongTokensLockTheAddressOut()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config() with { MaxFailedLogins = 3, LockoutSeconds = 6 });
            for (int i = 0; i < 3; i++)
                await Assert.ThrowsAnyAsync<Exception>(() => Connect(server, $"guess-{i}", Role.Admin));

            // Locked out: even the right token is refused for a while
            Exception locked = await Assert.ThrowsAnyAsync<Exception>(() => Connect(server, "admin-secret", Role.Admin));
            Assert.Contains("Too many", locked.Message);

            Thread.Sleep(6500);
            using RemoteSimulation admin = await Connect(server, "admin-secret", Role.Admin);
            Assert.Equal(Role.Admin, admin.Session!.Role);
        }

        [Fact]
        public async Task ClientsOfAnotherProtocolVersionAreRefused()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());

            Exception older = await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri,
                new Hello("Old", "admin-secret", Role.Admin, null, null, Protocol.Version - 1)));
            Assert.Contains("protocol version", older.Message);
            Assert.Contains("update the client", older.Message);

            Exception newer = await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri,
                new Hello("New", "admin-secret", Role.Admin, null, null, Protocol.Version + 1)));
            Assert.Contains("update the server", newer.Message);

            using RemoteSimulation current = await Connect(server, "admin-secret", Role.Admin);
            Assert.Equal(Role.Admin, current.Session!.Role);
        }

        [Fact]
        public void MessagesWithoutAVersionAreVersionZero()
        {
            // Clients and servers from before the protocol version sent none
            Assert.Equal(0, Protocol.DeserializeHello("""{"type":"Hello","name":"Old","role":"Spectator"}""").Version);
            WelcomeEvent welcome = (WelcomeEvent)Protocol.DeserializeEvent("""{"type":"WelcomeEvent","session":{"id":"s","name":"S","role":"Spectator"}}""");
            Assert.Equal(0, welcome.Version);

            Assert.Equal(Protocol.Version, Protocol.DeserializeHello(Protocol.Serialize(new Hello("Now", null, Role.Spectator, null, null))).Version);
            Assert.Null(Protocol.VersionMismatch(Protocol.Version, otherIsServer: true));
            Assert.Contains("update the server", Protocol.VersionMismatch(0, otherIsServer: true));
        }

        [Fact]
        public async Task AdminsFromOtherComputersNeedThePermission()
        {
            using Match match = NewMatch();
            await using SimulationServer closed = await SimulationServer.StartAsync(match, Config());
            Hello hello = new Hello("Root", "admin-secret", Role.Admin, null, null);
            IPAddress lan = IPAddress.Parse("192.168.1.20");

            Assert.NotNull(closed.Authenticate(hello, IPAddress.Loopback).session);
            Assert.NotNull(closed.Authenticate(hello, IPAddress.IPv6Loopback).session);
            var refused = closed.Authenticate(hello, lan);
            Assert.Null(refused.session);
            Assert.Contains("other computers", refused.refusal);

            // Players from the network are fine either way
            Assert.NotNull(closed.Authenticate(new Hello("Ann", "seat2-secret", Role.Player, 2, null), lan).session);

            using Match other = NewMatch();
            await using SimulationServer open = await SimulationServer.StartAsync(other, Config() with { AllowRemoteAdmins = true });
            Assert.NotNull(open.Authenticate(hello, lan).session);
        }

        [Fact]
        public async Task BrowserPagesCannotConnect()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using ClientWebSocket socket = new ClientWebSocket();
            socket.Options.SetRequestHeader("Origin", "https://evil.example");
            await Assert.ThrowsAnyAsync<WebSocketException>(() => socket.ConnectAsync(server.Uri, CancellationToken.None));
        }

        [Fact]
        public async Task OversizedMessagesCloseTheConnection()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config() with { MaxMessageBytes = 64 * 1024 });
            using ClientWebSocket socket = new ClientWebSocket();
            await socket.ConnectAsync(server.Uri, CancellationToken.None);
            byte[] huge = Encoding.UTF8.GetBytes(new string('x', 200 * 1024));
            try
            {
                await socket.SendAsync(huge, WebSocketMessageType.Text, true, CancellationToken.None);
            }
            catch (WebSocketException)
            {
                // The server may close while the message is still on its way
            }

            byte[] buffer = new byte[4096];
            Assert.True(WaitFor(() =>
            {
                try
                {
                    WebSocketReceiveResult result = socket.ReceiveAsync(buffer, new CancellationTokenSource(500).Token).GetAwaiter().GetResult();
                    return result.MessageType == WebSocketMessageType.Close || socket.State != WebSocketState.Open;
                }
                catch (Exception)
                {
                    return socket.State != WebSocketState.Open;
                }
            }));
        }

        [Fact]
        public async Task NamesAreTrimmedAndCleaned()
        {
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            string name = "Evil\u001b[31m\r\nAdmin" + new string('a', 500);
            SessionInfo session = server.Authenticate(new Hello(name, null, Role.Spectator, null, null), IPAddress.Loopback).session!;
            Assert.True(session.Name.Length <= 40);
            Assert.DoesNotContain(session.Name, char.IsControl);
        }

        [Fact]
        public void UnknownRolesAndSeatKindsAreRejected()
        {
            Assert.ThrowsAny<Exception>(() => Protocol.DeserializeHello("""{"type":"Hello","name":"x","role":"7"}"""));
            Assert.ThrowsAny<Exception>(() => Protocol.DeserializeCommand("""{"type":"SetSeat","seat":1,"kind":"99"}"""));
            Assert.ThrowsAny<Exception>(() => Protocol.DeserializeCommand("""{"type":"NewGame","seats":["Human","Human"],"centerDesert":false}"""));
        }

        [Fact]
        public async Task InvalidAdminCommandsDoNotBreakTheMatch()
        {
            using Match match = NewMatch();
            using LocalSimulation admin = new LocalSimulation(match);
            admin.ProcessEvents();
            admin.Send(new Simulation.SetSeat(7, SeatKind.Random));
            admin.Send(new Simulation.NewGame([SeatKind.Human], false));
            List<MatchEvent> events = admin.ProcessEvents();
            Assert.Equal(2, events.Count(e => e is ErrorEvent));
            Assert.Equal(4, match.Seats.Length);
            await Task.CompletedTask;
        }

        [Fact]
        public void ClientsCannotChooseTheOutcomesOfTheirMoves()
        {
            using Match match = NewMatch();
            SessionInfo admin = new SessionInfo("admin", "Admin", Role.Admin, null);
            Random random = new Random(3);

            // Up to the first roll of the main game
            while (!match.Record.State.HasEnded)
            {
                GameState state = match.Record.State;
                int acting = Match.ActingPlayerOf(state);
                List<Common.Actions.Action> legal = Common.Actions.LegalActionProvider.GetActionsForState(state, (sbyte)acting);
                if (legal.Any(a => a is Common.Actions.RollAction)) break;
                match.Handle(admin, new PlayMove(legal[random.Next(legal.Count)]));
            }

            // The player sends a roll whose outcome it decided itself (applied on its own copy); the match rolls on its own
            int player = Match.ActingPlayerOf(match.Record.State);
            SessionInfo session = new SessionInfo("p", "P", Role.Player, player);
            int same = 0;
            for (int trial = 0; trial < 12; trial++)
            {
                Common.Actions.RollAction forged = new Common.Actions.RollAction((sbyte)player);
                forged.Apply(new GameState(match.Record.State));
                RollResult chosen = forged.History!.RollResult;

                match.Handle(session, new PlayMove(forged));
                Common.Actions.RollAction played = Assert.IsType<Common.Actions.RollAction>(match.Record.History[^1].Action);
                if (played.History!.RollResult.Equals(chosen)) same++;
                match.Handle(admin, new UndoCommand());
            }
            Assert.True(same < 6, $"{same} of 12 rolls came out as the client chose");
        }

        [Fact]
        public async Task SecureConnectionsCheckThePinnedCertificate()
        {
            using var certificate = ServerCertificates.CreateSelfSigned();
            string pin = ServerCertificates.Fingerprint(certificate);
            using Match match = NewMatch();
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config() with { Certificate = certificate });
            Assert.Equal("wss", server.Uri.Scheme);

            Hello hello = new Hello("Root", "admin-secret", Role.Admin, null, null);
            using (RemoteSimulation admin = await RemoteSimulation.ConnectAsync(server.Uri, hello, pin))
            {
                Assert.True(Pump(admin, () => admin.Game.Server != null));
                Assert.True(admin.Game.Server!.Secure);
                Assert.Equal(pin, admin.Game.Server.CertificateFingerprint);
            }

            // A self-signed certificate without the pin, or with another one, is not trusted
            await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri, hello));
            await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri, hello, new string('0', 64)));
        }

        [Fact]
        public void CertificatesAreCreatedOnceAndReloaded()
        {
            string path = Path.Combine(Path.GetTempPath(), $"catansim-cert-{Guid.NewGuid():N}.pfx");
            try
            {
                using var created = ServerCertificates.LoadOrCreate(path);
                using var loaded = ServerCertificates.LoadOrCreate(path);
                Assert.Equal(ServerCertificates.Fingerprint(created), ServerCertificates.Fingerprint(loaded));
                Assert.True(loaded.HasPrivateKey);
            }
            finally
            {
                File.Delete(path);
            }
        }
        #endregion
    }
}
