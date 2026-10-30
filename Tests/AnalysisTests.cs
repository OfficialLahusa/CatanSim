using Agents.MCTS;
using Common;
using Simulation;
using System.Collections.Concurrent;
using System.Text.Json;

namespace Tests
{
    /// <summary>
    /// The match's analysis (its agents' win chances, priors and live search): what it sends, to whom, and over the wire
    /// </summary>
    public class AnalysisTests
    {
        private static readonly SeatKind[] AllHuman = [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human];

        private static bool WaitFor(Func<bool> condition, int milliseconds = 20000)
        {
            DateTime end = DateTime.UtcNow.AddMilliseconds(milliseconds);
            while (DateTime.UtcNow < end)
            {
                if (condition()) return true;
                Thread.Sleep(10);
            }
            return condition();
        }

        /// <summary>
        /// Collects what the match sends to one session
        /// </summary>
        private static ConcurrentQueue<MatchEvent> Listen(Match match, SessionInfo session)
        {
            ConcurrentQueue<MatchEvent> events = new();
            match.Subscribe((e, to) => { if (Match.IsFor(to, session)) events.Enqueue(e); }, _ => { });
            return events;
        }

        [Fact]
        public void OnlySpectatorsAndAdminsGetTheAnalysis()
        {
            Assert.True(Match.IsFor(Match.Watchers, new SessionInfo("a", "A", Role.Admin, null)));
            Assert.True(Match.IsFor(Match.Watchers, new SessionInfo("s", "S", Role.Spectator, null)));
            Assert.False(Match.IsFor(Match.Watchers, new SessionInfo("p", "P", Role.Player, 0)));
            Assert.False(Match.IsFor(Match.Watchers, new SessionInfo("b", "B", Role.Agent, 1)));
        }

        [Fact]
        public void AnalysisEventsSurviveTheWire()
        {
            EvaluationEvent evaluation = new([0.1f, 0.2f, 0.3f, 0.4f], [0.5f, 0.25f, float.NaN]);
            EvaluationEvent evaluationBack = (EvaluationEvent)Protocol.DeserializeEvent(Protocol.Serialize(evaluation));
            Assert.Equal([0.1f, 0.2f, 0.3f, 0.4f], evaluationBack.WinChances);
            Assert.Equal([0.5f, 0.25f, 0f], evaluationBack.Priors);

            SearchEvent search = new(2, 15, [10, 0, 5], [0.6f, 0.1f, 0.3f], [[0.1f, 0.2f, 0.3f, 0.4f], [0, 0, 0, 0], [0.25f, 0.25f, 0.25f, 0.25f]]);
            SearchEvent searchBack = (SearchEvent)Protocol.DeserializeEvent(Protocol.Serialize(search));
            Assert.Equal(2, searchBack.Seat);
            Assert.Equal(15, searchBack.Iterations);
            Assert.Equal(search.Visits, searchBack.Visits);
            Assert.Equal(search.Priors, searchBack.Priors);
            for (int i = 0; i < search.MeanValues.Length; i++) Assert.Equal(search.MeanValues[i], searchBack.MeanValues[i]);
        }

        [Fact]
        public void ASearchWithListsOfDifferentLengthsIsRefused()
        {
            string text = """{"type":"SearchEvent","seat":0,"iterations":1,"visits":[1,2],"priors":[0.5],"means":[[0,0,0,0],[0,0,0,0]]}""";
            Assert.ThrowsAny<JsonException>(() => Protocol.DeserializeEvent(text));
        }

        [Fact]
        public async Task SpectatorsGetTheWinChancesOverTheNetworkAndPlayersDont()
        {
            Utils.Random.SeedCurrentThread(5);
            using Match match = new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), AllHuman, valueNetAvailable: false);
            ServerConfig config = new()
            {
                Port = 0,
                Tokens = [new TokenEntry(ServerConfig.HashToken("seat1-secret"), Role.Player, 1)],
                AllowAnonymousSpectators = true
            };
            await using SimulationServer server = await SimulationServer.StartAsync(match, config);
            match.Start();

            using RemoteSimulation spectator = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Watcher", null, Role.Spectator, null, null));
            using RemoteSimulation player = await RemoteSimulation.ConnectAsync(server.Uri, new Hello("Ann", "seat1-secret", Role.Player, 1, null));

            Assert.True(WaitFor(() => { spectator.ProcessEvents(); return spectator.Game.Evaluation != null; }));
            float[] chances = spectator.Game.Evaluation!.WinChances;
            Assert.Equal(4, chances.Length);
            Assert.All(chances, c => Assert.InRange(c, 0f, 1f));

            // The player had the same time to get it
            Assert.False(WaitFor(() => { player.ProcessEvents(); return player.Game.Evaluation != null; }, 1000));
        }

        [Fact]
        public void ALateWatcherGetsTheCurrentAnalysisRightAway()
        {
            Utils.Random.SeedCurrentThread(6);
            using Match match = new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4), AllHuman, valueNetAvailable: false);
            ConcurrentQueue<MatchEvent> first = Listen(match, new SessionInfo("s1", "S1", Role.Spectator, null));
            match.Start();
            Assert.True(WaitFor(() => first.Any(e => e is EvaluationEvent)));

            ConcurrentQueue<MatchEvent> late = Listen(match, new SessionInfo("s2", "S2", Role.Spectator, null));
            Assert.Contains(late, e => e is EvaluationEvent);
        }

        [SkippableFact]
        public void ShortSearchesStillReachTheWatchers()
        {
            // Searches far shorter than the analysis' interval: the finished search is sent when the agent is done, and the
            // pause between moves comes after it, so every kind of searching seat shows its visits
            Skip.IfNot(Seats.ValueNetAvailable, "The value net is missing.");
            Utils.Random.SeedCurrentThread(8);
            using Match match = new Match(new GameState(MapGenerator.GenerateRandomClassic(), 4),
                [SeatKind.PimcEasy, SeatKind.MctsEasy, SeatKind.PimcEasy, SeatKind.MctsEasy]);
            SessionInfo admin = new("admin", "Admin", Role.Admin, null);
            match.Handle(admin, new ConfigureAgents(new AgentSettings(null, 256, Agents.MCTS.OpponentModel.MaxN)));
            match.Handle(admin, new SetPlayback(new PlaybackState(false, false, 0.05f)));
            ConcurrentQueue<MatchEvent> watcher = Listen(match, new SessionInfo("s", "S", Role.Spectator, null));
            match.Start();

            Assert.True(WaitFor(() => watcher.OfType<SearchEvent>().Select(s => s.Seat).Distinct().Count() == 4, 60000),
                $"searches only from seats {string.Join(", ", watcher.OfType<SearchEvent>().Select(s => s.Seat).Distinct())}");
            Assert.All(watcher.OfType<SearchEvent>(), search => Assert.True(search.Visits.Sum(v => (long)v) > 0));
        }

        [SkippableFact]
        public void TheThinkingAgentsSearchGoesToWatchersOnly()
        {
            Skip.IfNot(Seats.ValueNetAvailable, "The value net is missing.");
            Utils.Random.SeedCurrentThread(7);
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            int moves = MCTSTree.GetValidActions(new GameState(state)).Count;
            using Match match = new Match(state, [SeatKind.MctsEasy, SeatKind.Human, SeatKind.Human, SeatKind.Human]);

            // A long search for the first settlement, so the analysis reads it several times
            SessionInfo admin = new("admin", "Admin", Role.Admin, null);
            match.Handle(admin, new ConfigureAgents(new AgentSettings(2, 0, Agents.MCTS.OpponentModel.MaxN)));
            ConcurrentQueue<MatchEvent> watcher = Listen(match, new SessionInfo("s", "S", Role.Spectator, null));
            ConcurrentQueue<MatchEvent> player = Listen(match, new SessionInfo("p", "P", Role.Player, 1));
            match.Start();

            Assert.True(WaitFor(() => watcher.Any(e => e is SearchEvent)));
            EvaluationEvent evaluation = watcher.OfType<EvaluationEvent>().First();
            Assert.Equal(4, evaluation.WinChances.Length);
            Assert.True(evaluation.Priors.Length == 0 || evaluation.Priors.Length == moves);
            if (evaluation.Priors.Length > 0) Assert.Equal(1f, evaluation.Priors.Sum(), 3);

            SearchEvent search = watcher.OfType<SearchEvent>().First();
            Assert.Equal(0, search.Seat);
            Assert.Equal(moves, search.Visits.Length);
            Assert.True(search.Iterations > 0);
            Assert.Equal(search.Iterations, search.Visits.Sum(v => (long)v));

            Assert.DoesNotContain(player, e => e is EvaluationEvent or SearchEvent);
        }
    }
}
