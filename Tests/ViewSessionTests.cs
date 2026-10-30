using Common;
using Common.Actions;
using Simulation;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Hidden information over the network: a player or agent bridge only ever receives its seat's view (positions as
    /// BeliefTracker.View, moves redacted, edits as views), spectators and admins the full game. Undo, redo, edits, loading
    /// and joining late keep every session in step.
    /// </summary>
    public class ViewSessionTests
    {
        private static readonly SessionInfo Admin = new("admin", "Admin", Role.Admin, null);
        private static readonly SessionInfo Spectator = new("spectator", "Spectator", Role.Spectator, null);
        private static SessionInfo PlayerAt(int seat) => new($"player{seat}", $"Player {seat}", Role.Player, seat);
        private static SessionInfo BridgeAt(int seat) => new($"bridge{seat}", $"Bridge {seat}", Role.Agent, seat);
        private static readonly SeatKind[] AllHuman = [SeatKind.Human, SeatKind.Human, SeatKind.Human, SeatKind.Human];

        private static Match NewMatch(int seed, bool freeTrade = false)
        {
            Utils.Random.SeedCurrentThread(seed);
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), 4);
            state.Settings.FreeTrade = freeTrade;
            return new Match(state, AllHuman, valueNetAvailable: false);
        }

        /// <summary>
        /// A client that follows the match like a connection: its snapshot and every event it may get, through the protocol
        /// </summary>
        private sealed class Follower
        {
            public SessionInfo Session { get; }
            public ClientGame Game { get; } = new ClientGame();
            public List<MatchEvent> Received { get; } = [];

            public Follower(SessionInfo session) => Session = session;

            public void Receive(MatchEvent e)
            {
                MatchEvent wire = Protocol.DeserializeEvent(Protocol.Serialize(e));
                Received.Add(wire);
                Game.Apply(wire);
            }
        }

        private static Follower Follow(Match match, SessionInfo session)
        {
            Follower follower = new Follower(session);
            match.Subscribe(session, (e, to) => { if (Match.IsFor(to, session)) follower.Receive(e); }, follower.Receive);
            return follower;
        }

        private static List<Follower> FollowAll(Match match) =>
            [Follow(match, Admin), Follow(match, Spectator), .. Enumerable.Range(0, 4).Select(seat => Follow(match, PlayerAt(seat))), Follow(match, BridgeAt(2))];

        private static int Acting(GameState state)
        {
            for (int p = 0; p < 4; p++) if (state.CanPlayerAct(p)) return p;
            return -1;
        }

        private static Action RandomMove(Match match, Random random)
        {
            GameState state = match.Record.State;
            List<Action> legal = LegalActionProvider.GetActionsForState(state, (sbyte)Acting(state));
            return legal[random.Next(legal.Count)];
        }

        /// <summary>
        /// Two positions part by part, so a failure says which part differs
        /// </summary>
        private static void AssertSamePosition(GameState expected, GameState actual, string who)
        {
            Assert.True(expected.Board.Equals(actual.Board), $"{who}: the board differs");
            Assert.True(expected.Turn.Equals(actual.Turn), $"{who}: the turn differs");
            Assert.True(expected.Settings.Equals(actual.Settings), $"{who}: the settings differ");
            Assert.True(expected.ResourceBank.Equals(actual.ResourceBank), $"{who}: the resource bank differs");
            Assert.True(expected.DevelopmentBank.Equals(actual.DevelopmentBank), $"{who}: the development bank differs");
            for (int player = 0; player < expected.Players.Length; player++)
            {
                PlayerState e = expected.Players[player], a = actual.Players[player];
                Assert.True(e.ResourceCards.Equals(a.ResourceCards), $"{who}: player {player}'s resource cards differ");
                Assert.True(e.DevelopmentCards.Equals(a.DevelopmentCards), $"{who}: player {player}'s development cards differ");
                Assert.True(e.NewDevelopmentCards.Equals(a.NewDevelopmentCards), $"{who}: player {player}'s new development cards differ");
                Assert.True(e.VictoryPoints.Equals(a.VictoryPoints), $"{who}: player {player}'s victory points differ");
                Assert.True(e.Equals(a), $"{who}: player {player} differs");
            }
            Assert.Equal(expected, actual);
        }

        /// <summary>
        /// Every follower has what it may see of the match's current position, history and redo list
        /// </summary>
        private static void AssertInStep(Match match, IEnumerable<Follower> followers)
        {
            GameState truth = match.Record.State;
            foreach (Follower follower in followers)
            {
                GameRecord record = follower.Game.Record;
                Assert.Equal(match.Record.History.Count, record.History.Count);
                if (Match.SeesView(follower.Session))
                {
                    int seat = follower.Session.Seat!.Value;
                    AssertSamePosition(match.Beliefs.View(truth, seat), record.State, $"seat {seat}");
                    Assert.NotNull(record.Start);
                    AssertSamePosition(match.Beliefs.StartView(seat), record.Start!, $"start of seat {seat}");
                }
                else
                {
                    AssertSamePosition(truth, record.State, follower.Session.Name);
                    Assert.Equal(match.Record.Redo.Count, record.Redo.Count);
                }
            }
        }

        /// <summary>
        /// No move a seat received shows a hidden outcome it may not know
        /// </summary>
        private static void AssertNoHiddenOutcome(Follower follower)
        {
            if (!Match.SeesView(follower.Session)) return;
            int seat = follower.Session.Seat!.Value;
            IEnumerable<HistoryEntry> entries = follower.Received.SelectMany(e => e switch
            {
                AppliedEvent applied => [applied.Entry],
                SnapshotEvent snapshot => snapshot.History.Concat(snapshot.Redo),
                _ => []
            });
            foreach (HistoryEntry entry in entries)
            {
                switch (entry.Action)
                {
                    case RobberAction { History.StolenCard: ResourceCardType stolen } robber when stolen != ResourceCardType.Unknown:
                        Assert.True(seat == robber.PlayerIndex || seat == robber.TargetPlayerIndex, $"Seat {seat} learned the stolen card");
                        break;
                    case BuyDevelopmentCardAction { History.DrawnType: DevelopmentCardType drawn } buy when drawn != DevelopmentCardType.Unknown:
                        Assert.Equal(seat, buy.PlayerIndex);
                        break;
                }
            }
        }

        [Fact]
        public void TheRecipientsAreTheRightSessions()
        {
            Assert.True(Match.IsFor(null, PlayerAt(1)));
            Assert.True(Match.IsFor(Match.Full, Admin));
            Assert.True(Match.IsFor(Match.Full, Spectator));
            Assert.False(Match.IsFor(Match.Full, PlayerAt(1)));
            Assert.False(Match.IsFor(Match.Full, BridgeAt(1)));
            Assert.True(Match.IsFor(Match.SeatView(1), PlayerAt(1)));
            Assert.True(Match.IsFor(Match.SeatView(1), BridgeAt(1)));
            Assert.False(Match.IsFor(Match.SeatView(1), PlayerAt(2)));
            Assert.False(Match.IsFor(Match.SeatView(1), Admin));
            Assert.False(Match.IsFor(Match.SeatView(1), Spectator));
            Assert.True(Match.IsFor(Match.Admins, Admin));
            Assert.False(Match.IsFor(Match.Admins, Spectator));
            Assert.True(Match.IsFor(Match.Watchers, Spectator));
            Assert.False(Match.IsFor(Match.Watchers, PlayerAt(0)));
            Assert.True(Match.IsFor("player3", PlayerAt(3)));
            Assert.False(Match.IsFor("player3", PlayerAt(2)));
        }

        [Theory]
        [InlineData(1, false)]
        [InlineData(2, true)]
        [InlineData(3, false)]
        public void PlayersFollowTheirSeatsViewAndTheOthersTheFullGame(int seed, bool freeTrade)
        {
            using Match match = NewMatch(seed, freeTrade);
            List<Follower> followers = FollowAll(match);
            AssertInStep(match, followers);
            Random random = new Random(seed);
            for (int i = 0; i < 1500 && !match.Record.State.HasEnded; i++)
            {
                match.Handle(Admin, new PlayMove(RandomMove(match, random)));
                AssertInStep(match, followers);
            }
            foreach (Follower follower in followers) AssertNoHiddenOutcome(follower);
        }

        [Fact]
        public void UndoAndRedoKeepEverySeatsView()
        {
            using Match match = NewMatch(4);
            List<Follower> followers = FollowAll(match);
            Random random = new Random(4);
            for (int i = 0; i < 1500 && !match.Record.State.HasEnded; i++)
            {
                double r = random.NextDouble();
                if (r < 0.1) match.Handle(Admin, new UndoCommand());
                else if (r < 0.15) match.Handle(Admin, new RedoCommand());
                else match.Handle(Admin, new PlayMove(RandomMove(match, random)));
                AssertInStep(match, followers);

                // Undo goes back exactly: the trackers equal those of a replay of the remaining history
                RecordBeliefs replayed = RecordBeliefs.Of(match.Record);
                foreach (int observer in new[] { BeliefTracker.Public, 0, 1, 2, 3 })
                    Assert.Equal(replayed.For(observer), match.Beliefs.For(observer));
            }

            // A long undo and redo comes back to the same views
            Dictionary<int, GameState> views = Enumerable.Range(0, 4).ToDictionary(seat => seat, seat => match.Beliefs.View(match.Record.State, seat));
            for (int i = 0; i < 40; i++) match.Handle(Admin, new UndoCommand());
            AssertInStep(match, followers);
            for (int i = 0; i < 40; i++) match.Handle(Admin, new RedoCommand());
            AssertInStep(match, followers);
            foreach ((int seat, GameState view) in views) Assert.Equal(view, match.Beliefs.View(match.Record.State, seat));
            foreach (Follower follower in followers) AssertNoHiddenOutcome(follower);
        }

        [Fact]
        public void EditsLoadsAndNewGamesReachPlayersAsViews()
        {
            using Match match = NewMatch(5);
            List<Follower> followers = FollowAll(match);
            Random random = new Random(5);
            for (int i = 0; i < 400; i++) match.Handle(Admin, new PlayMove(RandomMove(match, random)));

            // An edit: the players get the views before and after it, not the full positions
            GameState before = new GameState(match.Record.State), after = new GameState(match.Record.State);
            after.ResourceBank.Remove(ResourceCardType.Ore, 1);
            after.Players[1].ResourceCards.Add(ResourceCardType.Ore, 1);
            match.Handle(Admin, new ApplyEdit(new BoardEdit("gave Red an ore card", 1, before, after)));
            AssertInStep(match, followers);
            foreach (Follower follower in followers.Where(f => Match.SeesView(f.Session)))
            {
                BoardEdit edit = follower.Game.Record.History[^1].Edit!;
                Assert.Equal(match.Beliefs.View(match.Record.State, follower.Session.Seat!.Value), edit.After);
            }
            match.Handle(Admin, new UndoCommand());
            AssertInStep(match, followers);
            match.Handle(Admin, new RedoCommand());
            AssertInStep(match, followers);

            // Loading the game again from its moves
            GameState start = RecordBeliefs.Of(match.Record).Start;
            List<Action> played = match.Record.ActionsSinceLastEdit();
            GameState loadFrom = new GameState(match.Record.State);
            match.Handle(Admin, new LoadGame(loadFrom, played, []));
            AssertInStep(match, followers);

            match.Handle(Admin, new NewGame(AllHuman, false));
            AssertInStep(match, followers);
            for (int i = 0; i < 200; i++) match.Handle(Admin, new PlayMove(RandomMove(match, random)));
            AssertInStep(match, followers);
            foreach (Follower follower in followers) AssertNoHiddenOutcome(follower);
        }

        [Fact]
        public void ALateJoinerGetsItsSeatsView()
        {
            using Match match = NewMatch(6);
            Random random = new Random(6);
            for (int i = 0; i < 800 && !match.Record.State.HasEnded; i++) match.Handle(Admin, new PlayMove(RandomMove(match, random)));
            for (int i = 0; i < 5; i++) match.Handle(Admin, new UndoCommand());

            List<Follower> followers = FollowAll(match);
            AssertInStep(match, followers);
            foreach (Follower follower in followers.Where(f => Match.SeesView(f.Session)))
            {
                SnapshotEvent snapshot = Assert.IsType<SnapshotEvent>(follower.Received[0]);
                Assert.Equal(match.Beliefs.View(match.Record.State, follower.Session.Seat!.Value), snapshot.State);
                Assert.NotNull(snapshot.Start);
            }
            foreach (Follower follower in followers) AssertNoHiddenOutcome(follower);
        }
    }
}
