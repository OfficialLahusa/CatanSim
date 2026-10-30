using Agents;
using Common;
using Common.Actions;
using Simulation;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The simulation: the authoritative match (rules, roles, undo, edits), clients that follow it by its events, the agents
    /// that play their seats, the protocol, and the server that hosts a match over WebSockets
    /// </summary>
    public class SimulationTests
    {
        private static readonly SessionInfo Admin = new("admin", "Admin", Role.Admin, null);
        private static readonly SessionInfo Spectator = new("spectator", "Spectator", Role.Spectator, null);
        private static SessionInfo PlayerAt(int seat) => new($"player{seat}", $"Player {seat}", Role.Player, seat);

        private static readonly SeatKind[] AllHuman = [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human];

        private static Match NewMatch(SeatKind[] seats, int seed = 1)
        {
            Utils.Random.SeedCurrentThread(seed);
            return new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), seats, valueNetAvailable: false);
        }

        /// <summary>
        /// The events of a match, by recipient (null for everyone)
        /// </summary>
        private static List<(MatchEvent e, string? to)> Record(Match match)
        {
            List<(MatchEvent, string?)> events = [];
            match.Published += (e, to) => { lock (events) events.Add((e, to)); };
            return events;
        }

        private static int Acting(GameState state)
        {
            for (int p = 0; p < 4; p++) if (state.CanPlayerAct(p)) return p;
            return -1;
        }

        /// <summary>
        /// A random legal move of the acting player
        /// </summary>
        private static Action RandomMove(Match match, Random random)
        {
            GameState state = match.Record.State;
            List<Action> legal = LegalActionProvider.GetActionsForState(state, (sbyte)Acting(state));
            return legal[random.Next(legal.Count)];
        }

        #region Match
        [Fact]
        public void PlayersMoveOnlyForTheirOwnSeat()
        {
            using Match match = NewMatch(AllHuman);
            var events = Record(match);
            int acting = Acting(match.Record.State);
            Action move = RandomMove(match, new Random(1));

            match.Handle(PlayerAt((acting + 1) % 4), new PlayMove(move));
            Assert.Empty(match.Record.History);
            Assert.Contains(events, x => x.e is ErrorEvent && x.to == PlayerAt((acting + 1) % 4).Id);

            match.Handle(PlayerAt(acting), new PlayMove(move));
            Assert.Single(match.Record.History);
            Assert.Contains(events, x => x.e is AppliedEvent && x.to == Match.Full);
        }

        [Fact]
        public void IllegalMovesAreRefused()
        {
            using Match match = NewMatch(AllHuman);
            var events = Record(match);
            int acting = Acting(match.Record.State);

            // Ending the turn is not legal during the setup
            match.Handle(Admin, new PlayMove(new EndTurnAction((sbyte)acting)));
            Assert.Empty(match.Record.History);
            Assert.Contains(events, x => x.e is ErrorEvent);
        }

        [Fact]
        public void MovesForAgentSeatsAreRefusedFromPlayers()
        {
            using Match match = NewMatch([SeatKind.Random, SeatKind.Human, SeatKind.Human, SeatKind.Human]);
            Action move = RandomMove(match, new Random(1));
            Assert.Equal(0, move.PlayerIndex);

            match.Handle(PlayerAt(0), new PlayMove(move));
            Assert.Empty(match.Record.History);
        }

        [Fact]
        public void UndoAndRedoRestoreThePositions()
        {
            using Match match = NewMatch(AllHuman, 3);
            Random random = new Random(3);
            GameState start = new GameState(match.Record.State);
            for (int i = 0; i < 300 && !match.Record.State.HasEnded; i++)
                match.Handle(Admin, new PlayMove(RandomMove(match, random)));
            GameState end = new GameState(match.Record.State);
            int moves = match.Record.History.Count;

            for (int i = 0; i < moves; i++) match.Handle(Admin, new UndoCommand());
            Assert.Equal(start, match.Record.State);
            Assert.Equal(moves, match.Record.Redo.Count);

            for (int i = 0; i < moves; i++) match.Handle(Admin, new RedoCommand());
            Assert.Equal(end, match.Record.State);
            Assert.Empty(match.Record.Redo);
        }

        [Fact]
        public void EditsCanBeUndoneAndRedone()
        {
            using Match match = NewMatch(AllHuman);
            GameState before = new GameState(match.Record.State);
            GameState after = new GameState(before);
            after.Board.SetBuilding(10, Intersection.BuildingType.Settlement, 2);
            BoardEdit.RecomputeFromBoard(after);

            match.Handle(Admin, new ApplyEdit(new BoardEdit("placed a settlement for Green", 2, before, after)));
            Assert.Equal(after, match.Record.State);
            Assert.Equal(1, match.Record.State.Players[2].VictoryPoints.SettlementPoints);

            match.Handle(Admin, new UndoCommand());
            Assert.Equal(before, match.Record.State);
            match.Handle(Admin, new RedoCommand());
            Assert.Equal(after, match.Record.State);
        }

        [Fact]
        public void EditsOfAnOutdatedPositionAreRefused()
        {
            using Match match = NewMatch(AllHuman);
            GameState stale = new GameState(match.Record.State);
            match.Handle(Admin, new PlayMove(RandomMove(match, new Random(1))));

            GameState after = new GameState(stale);
            after.Board.SetBuilding(10, Intersection.BuildingType.Settlement, 2);
            match.Handle(Admin, new ApplyEdit(new BoardEdit("placed a settlement for Green", 2, stale, after)));
            Assert.Single(match.Record.History);
            Assert.Null(match.Record.History[0].Edit);
        }

        [Fact]
        public void OnlyAdminsControlTheMatch()
        {
            using Match match = NewMatch(AllHuman);
            var events = Record(match);
            match.Handle(Admin, new PlayMove(RandomMove(match, new Random(1))));

            foreach (SessionInfo session in new[] { Spectator, PlayerAt(0) })
            {
                Command[] controls = [new UndoCommand(), new NewGame(AllHuman, false), new SetSeat(1, SeatKind.Random),
                    new SetPlayback(new PlaybackState(true, false, 0)), new KickSession("x")];
                foreach (Command command in controls) match.Handle(session, command);
                Assert.Equal(controls.Length, events.Count(x => x.e is ErrorEvent && x.to == session.Id));
            }
            Assert.Single(match.Record.History);
            Assert.All(match.Seats, kind => Assert.Equal(SeatKind.Human, kind));
        }

        [Fact]
        public void SpectatorsCannotMove()
        {
            using Match match = NewMatch(AllHuman);
            match.Handle(Spectator, new PlayMove(RandomMove(match, new Random(1))));
            Assert.Empty(match.Record.History);
        }

        [Fact]
        public void ClientRecordsFollowTheMatch()
        {
            using Match match = NewMatch(AllHuman, 5);
            ClientGame client = new ClientGame();
            match.Published += (e, to) => { if (Match.IsFor(to, Admin)) client.Apply(e); };
            client.Apply(match.Snapshot());

            Random random = new Random(5);
            for (int i = 0; i < 400 && !match.Record.State.HasEnded; i++)
            {
                double r = random.NextDouble();
                if (r < 0.1) match.Handle(Admin, new UndoCommand());
                else if (r < 0.15) match.Handle(Admin, new RedoCommand());
                else match.Handle(Admin, new PlayMove(RandomMove(match, random)));
                Assert.Equal(match.Record.State, client.Record.State);
            }

            GameState before = new GameState(match.Record.State);
            GameState after = new GameState(before);
            after.Players[1].ResourceCards.Add(ResourceCardType.Ore, 1);
            after.ResourceBank.Remove(ResourceCardType.Ore, 1);
            match.Handle(Admin, new ApplyEdit(new BoardEdit("gave Red an ore card", 1, before, after)));
            match.Handle(Admin, new SetSeat(2, SeatKind.Builder));

            Assert.Equal(match.Record.State, client.Record.State);
            Assert.Equal(match.Record.History.Count, client.Record.History.Count);
            Assert.Equal(match.Record.Redo.Count, client.Record.Redo.Count);
            Assert.Equal(SeatKind.Builder, client.Seats[2]);
        }

        [Fact]
        public void NewGameResetsTheMatchForEveryone()
        {
            using Match match = NewMatch(AllHuman);
            var events = Record(match);
            match.Handle(Admin, new PlayMove(RandomMove(match, new Random(1))));

            match.Handle(Admin, new NewGame([SeatKind.Builder, SeatKind.Human, SeatKind.Human, SeatKind.Human], false));
            Assert.Empty(match.Record.History);
            Assert.Equal(SeatKind.Builder, match.Seats[0]);
            Assert.Contains(events, x => x.e is SnapshotEvent && x.to == Match.Full);
        }
        #endregion

        #region Agents
        private static bool WaitFor(Func<bool> condition, int milliseconds = 20000)
        {
            System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
            while (clock.ElapsedMilliseconds < milliseconds)
            {
                if (condition()) return true;
                Thread.Sleep(10);
            }
            return condition();
        }

        [Fact]
        public void AgentSeatsPlayUntilAHumanMustAct()
        {
            using Match match = NewMatch([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Human], 7);
            match.Handle(Admin, new SetPlayback(new PlaybackState(false, false, 0)));
            match.Start();

            Assert.True(WaitFor(() => match.ActingPlayer == 3 && !match.IsAgentThinking));
            int moves = match.Record.History.Count;
            Assert.True(moves >= 3);
            Thread.Sleep(300);
            Assert.Equal(moves, match.Record.History.Count);
        }

        [Fact]
        public void PausedAgentsWaitAndStepPlaysOneMove()
        {
            using Match match = NewMatch([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Random], 8);
            match.Handle(Admin, new SetPlayback(new PlaybackState(true, false, 0)));
            match.Start();
            Thread.Sleep(300);
            Assert.Empty(match.Record.History);

            match.Handle(Admin, new StepCommand());
            Assert.True(WaitFor(() => match.Record.History.Count == 1));
            Thread.Sleep(300);
            Assert.Single(match.Record.History);
        }

        [Fact]
        public void StandInsPlayHumanSeats()
        {
            using Match match = NewMatch(AllHuman, 9);
            match.Handle(Admin, new ConfigureAgents(AgentSettings.Default));
            match.Handle(Admin, new SetPlayback(new PlaybackState(false, true, 0)));
            match.Start();
            Assert.True(WaitFor(() => match.Record.History.Count >= 8));
        }

        [Fact]
        public void AgentGamesRunToTheEnd()
        {
            using Match match = NewMatch([SeatKind.Random, SeatKind.Random, SeatKind.Random, SeatKind.Random], 10);
            match.Handle(Admin, new SetPlayback(new PlaybackState(false, false, 0)));
            match.Start();
            Assert.True(WaitFor(() => match.Record.State.HasEnded, 60000));
        }
        #endregion

        #region Protocol
        [Fact]
        public void TheProtocolRoundTripsEveryMessage()
        {
            using Match match = NewMatch(AllHuman, 11);
            Random random = new Random(11);
            for (int i = 0; i < 120 && !match.Record.State.HasEnded; i++)
                match.Handle(Admin, new PlayMove(RandomMove(match, random)));
            for (int i = 0; i < 5; i++) match.Handle(Admin, new UndoCommand());
            GameState before = new GameState(match.Record.State);
            GameState after = new GameState(before);
            after.Board.SetBuilding(20, Intersection.BuildingType.City, 1);
            BoardEdit.RecomputeFromBoard(after);
            BoardEdit edit = new BoardEdit("placed a city for Red", 1, before, after);

            // Events: a snapshot with moves (with their rolls) and an edit, then each kind on its own
            SnapshotEvent snapshot = match.Snapshot();
            SnapshotEvent snapshotBack = Assert.IsType<SnapshotEvent>(Protocol.DeserializeEvent(Protocol.Serialize(snapshot)));
            Assert.Equal(snapshot.State, snapshotBack.State);
            Assert.Equal(snapshot.History.Count, snapshotBack.History.Count);
            Assert.Equal(snapshot.Redo.Count, snapshotBack.Redo.Count);
            Assert.Equal(snapshot.Seats, snapshotBack.Seats);

            ClientGame replay = new ClientGame();
            replay.Apply(snapshotBack);
            for (int i = 0; i < 5; i++) replay.Apply(Protocol.DeserializeEvent(Protocol.Serialize(new AppliedEvent(snapshot.Redo[^(i + 1)], true))));
            ClientGame original = new ClientGame();
            original.Apply(snapshot);
            for (int i = 0; i < 5; i++) original.Apply(new AppliedEvent(snapshot.Redo[^(i + 1)], true));
            Assert.Equal(original.Record.State, replay.Record.State);

            AppliedEvent editEvent = Assert.IsType<AppliedEvent>(Protocol.DeserializeEvent(Protocol.Serialize(new AppliedEvent(HistoryEntry.Of(edit), false))));
            Assert.Equal(after, editEvent.Entry.Edit!.After);
            Assert.Equal(before, editEvent.Entry.Edit.Before);
            Assert.Equal("placed a city for Red", editEvent.Entry.Edit.Description);

            MatchEvent[] simple = [new UndoneEvent(), new SeatsEvent(AllHuman), new PlaybackEvent(new PlaybackState(true, true, 1.5f)),
                new ThinkingEvent(2, true), new ErrorEvent("no"), new ClosedEvent("bye"), new AgentsEvent(new AgentSettings(2, 0, Agents.MCTS.OpponentModel.Paranoid)),
                new WelcomeEvent(new SessionInfo("s1", "Ann", Role.Player, 2, "1.2.3.4")), new SessionsEvent([new SessionInfo("s1", "Ann", Role.Player, 2)])];
            foreach (MatchEvent e in simple)
            {
                MatchEvent back = Protocol.DeserializeEvent(Protocol.Serialize(e));
                Assert.Equal(Protocol.Serialize(e), Protocol.Serialize(back));
                Assert.Equal(e.GetType(), back.GetType());
            }

            // Commands
            Action move = RandomMove(match, random);
            PlayMove moveBack = Assert.IsType<PlayMove>(Protocol.DeserializeCommand(Protocol.Serialize(new PlayMove(move))));
            Assert.True(move.IsSameDecision(moveBack.Action));
            LoadGame load = new LoadGame(match.Record.State, match.Record.ActionsSinceLastEdit(), match.Record.RedoActionsUntilEdit());
            LoadGame loadBack = Assert.IsType<LoadGame>(Protocol.DeserializeCommand(Protocol.Serialize(load)));
            Assert.Equal(load.State, loadBack.State);
            Assert.Equal(load.Played.Count, loadBack.Played.Count);
            Command[] commands = [new UndoCommand(), new RedoCommand(), new NewGame(AllHuman, true, true), new SetSeat(3, SeatKind.MctsHard),
                new SetPlayback(new PlaybackState(false, true, 0.2f)), new StepCommand(), new ConfigureAgents(AgentSettings.Default), new KickSession("s9"), new ApplyEdit(edit)];
            foreach (Command c in commands)
                Assert.Equal(Protocol.Serialize(c), Protocol.Serialize(Protocol.DeserializeCommand(Protocol.Serialize(c))));
        }
        #endregion

        #region Server
        private static ServerConfig Config() => new()
        {
            Port = 0,
            Tokens =
            [
                new TokenEntry(ServerConfig.HashToken("admin-secret"), Role.Admin, null),
                new TokenEntry(ServerConfig.HashToken("seat2-secret"), Role.Player, 2),
                new TokenEntry(ServerConfig.HashToken("bridge-secret"), Role.Agent, 1),
            ],
            AllowAnonymousSpectators = true
        };

        private static bool Pump(RemoteSimulation client, Func<bool> condition, int milliseconds = 10000)
        {
            return WaitFor(() => { client.ProcessEvents(); return condition(); }, milliseconds);
        }

        [Fact]
        public async Task RemoteClientsFollowTheHostedMatch()
        {
            using Match match = NewMatch(AllHuman, 12);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());

            using RemoteSimulation admin = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Admin", "admin-secret", Role.Admin, null, null));
            using RemoteSimulation player = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Ann", "seat2-secret", Role.Player, 2, null));
            using RemoteSimulation spectator = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Sam", null, Role.Spectator, null, null));
            Assert.Equal(Role.Admin, admin.Session!.Role);
            Assert.Equal(2, player.Session!.Seat);
            Assert.Equal(Role.Spectator, spectator.Session!.Role);

            // The admin plays the first moves; the seat 3 player's moves come from its own connection
            Random random = new Random(12);
            for (int i = 0; i < 40; i++)
            {
                GameState state = admin.Game.Record.State;
                int acting = Acting(state);
                List<Action> legal = LegalActionProvider.GetActionsForState(state, (sbyte)acting);
                Action move = legal[random.Next(legal.Count)];
                int before = match.Record.History.Count;
                (acting == 2 ? player : admin).Send(new PlayMove(move));
                Assert.True(Pump(admin, () => admin.Game.Record.History.Count == before + 1));
            }
            Assert.True(Pump(player, () => player.Game.Record.History.Count == 40));
            Assert.True(Pump(spectator, () => spectator.Game.Record.History.Count == 40));
            Assert.Equal(match.Record.State, admin.Game.Record.State);
            // The player follows its seat's view, not the full game
            Assert.Equal(match.Beliefs.View(match.Record.State, 2), player.Game.Record.State);
            Assert.Equal(match.Record.State, spectator.Game.Record.State);

            // Spectators can't move, players can't undo; both get an error and the match stays
            spectator.Send(new UndoCommand());
            player.Send(new UndoCommand());
            Assert.True(Pump(spectator, () => spectator.Errors.Count == 1));
            Assert.True(Pump(player, () => player.Errors.Count == 1));
            Assert.Equal(40, match.Record.History.Count);

            // The admin undoes for everyone
            admin.Send(new UndoCommand());
            Assert.True(Pump(spectator, () => spectator.Game.Record.History.Count == 39));
            Assert.Equal(match.Record.State, spectator.Game.Record.State);

            // Only admins learn who is connected
            Assert.True(Pump(admin, () => admin.Game.Sessions.Count == 3));
            spectator.ProcessEvents();
            Assert.Empty(spectator.Game.Sessions);
        }

        [Fact]
        public async Task WrongTokensAreRejected()
        {
            using Match match = NewMatch(AllHuman);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config() with { AllowAnonymousSpectators = false });

            await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri, new Hello("Eve", "guess", Role.Admin, null, null)));
            await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri, new Hello("Eve", "seat2-secret", Role.Admin, null, null)));
            await Assert.ThrowsAnyAsync<Exception>(() => RemoteSimulation.ConnectAsync(server.Uri, new Hello("Eve", null, Role.Spectator, null, null)));
        }

        [Fact]
        public async Task AgentBridgesMoveForTheirSeat()
        {
            using Match match = NewMatch(AllHuman, 13);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation admin = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Admin", "admin-secret", Role.Admin, null, null));
            using RemoteSimulation bridge = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Bot", "bridge-secret", Role.Agent, 1, null));
            Assert.Equal(Role.Agent, bridge.Session!.Role);

            // The admin plays seat 0's setup, then the bridge places for seat 1
            Random random = new Random(13);
            while (Acting(admin.Game.Record.State) == 0)
            {
                int before = admin.Game.Record.History.Count;
                admin.Send(new PlayMove(RandomMove(match, random)));
                Assert.True(Pump(admin, () => admin.Game.Record.History.Count == before + 1));
            }
            Assert.True(Pump(bridge, () => bridge.Game.Record.History.Count == admin.Game.Record.History.Count));
            int count = bridge.Game.Record.History.Count;
            bridge.Send(new PlayMove(RandomMove(match, random)));
            Assert.True(Pump(bridge, () => bridge.Game.Record.History.Count == count + 1));
        }

        [Fact]
        public async Task ReconnectingKeepsTheRole()
        {
            using Match match = NewMatch(AllHuman);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            string sessionId;
            using (RemoteSimulation player = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Ann", "seat2-secret", Role.Player, 2, null)))
                sessionId = player.Session!.Id;

            using RemoteSimulation again = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Ann", "seat2-secret", Role.Player, 2, sessionId));
            Assert.Equal(sessionId, again.Session!.Id);
            Assert.Equal(2, again.Session.Seat);
        }

        [Fact]
        public async Task KickedClientsAreClosed()
        {
            using Match match = NewMatch(AllHuman);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation admin = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Admin", "admin-secret", Role.Admin, null, null));
            using RemoteSimulation spectator = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Sam", null, Role.Spectator, null, null));

            admin.Send(new KickSession(spectator.Session!.Id));
            Assert.True(Pump(spectator, () => !spectator.IsConnected));
            Assert.True(Pump(admin, () => admin.Game.Sessions.Count == 1));
        }

        [Fact]
        public async Task RemotePlayersAndAgentsPlayAWholeGame()
        {
            // Seats 1 and 3 play from their own connections, the match's agents play the random seats
            using Match match = NewMatch([SeatKind.Random, SeatKind.Human, SeatKind.Random, SeatKind.Human], 15);
            match.Handle(Admin, new SetPlayback(new PlaybackState(false, false, 0)));
            match.Start();
            ServerConfig config = Config() with
            {
                Tokens = [.. Config().Tokens, new TokenEntry(ServerConfig.HashToken("seat4-secret"), Role.Player, 3), new TokenEntry(ServerConfig.HashToken("seat2b-secret"), Role.Player, 1)]
            };
            await using SimulationServer server = await SimulationServer.StartAsync(match, config);
            using RemoteSimulation red = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Red", "seat2b-secret", Role.Player, 1, null));
            using RemoteSimulation orange = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Orange", "seat4-secret", Role.Player, 3, null));

            Random random = new Random(15);
            int sentAt = -1;
            System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
            while (!orange.Game.Record.State.HasEnded && clock.Elapsed.TotalSeconds < 120)
            {
                red.ProcessEvents();
                orange.ProcessEvents();

                // A player moves once its copy shows its turn, one move per position
                foreach (RemoteSimulation player in new[] { red, orange })
                {
                    GameState state = player.Game.Record.State;
                    int seat = player.Session!.Seat!.Value;
                    if (state.HasEnded || !state.CanPlayerAct(seat) || Acting(state) != seat || player.Game.Record.Version == sentAt) continue;
                    List<Action> legal = LegalActionProvider.GetActionsForState(state, (sbyte)seat);
                    player.Send(new PlayMove(legal[random.Next(legal.Count)]));
                    sentAt = player.Game.Record.Version;
                }
                Thread.Sleep(2);
            }

            Assert.True(orange.Game.Record.State.HasEnded, "the game did not end in time");
            Assert.True(Pump(red, () => red.Game.Record.History.Count == match.Record.History.Count));
            Assert.Equal(match.Record.State, red.Game.Record.State);
            Assert.Equal(match.Record.State, orange.Game.Record.State);
            Assert.Empty(red.Errors.Where(e => !e.Contains("not a legal move")));
        }

        [Fact]
        public async Task TheHostPlaysThroughTheSameMatch()
        {
            // A client hosting a game: its local simulation and the remote players share one match
            using Match match = NewMatch(AllHuman, 14);
            using LocalSimulation host = new LocalSimulation(match);
            await using SimulationServer server = await SimulationServer.StartAsync(match, Config());
            using RemoteSimulation guest = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Sam", null, Role.Spectator, null, null));

            host.ProcessEvents();
            Random random = new Random(14);
            for (int i = 0; i < 10; i++)
            {
                host.Send(new PlayMove(RandomMove(match, random)));
                host.ProcessEvents();
            }
            Assert.Equal(10, host.Game.Record.History.Count);
            Assert.True(Pump(guest, () => guest.Game.Record.History.Count == 10));
            Assert.Equal(host.Game.Record.State, guest.Game.Record.State);
            Assert.Equal(Role.Admin, host.Session!.Role);
        }
        #endregion
    }
}
