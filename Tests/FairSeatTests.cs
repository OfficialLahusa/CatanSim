using Agents;
using Agents.MCTS;
using Common;
using Common.Actions;
using Simulation;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The fair seats (Fair MCTS Easy/Medium/Hard): agents that play under hidden information (DeterminizedMCTSAgent) with the
    /// presets' thinking times, chosen separately from the default seats. The match keeps their beliefs as it keeps its own,
    /// through undo, redo, loading and seat changes.
    /// </summary>
    public class FairSeatTests
    {
        private static readonly SessionInfo Admin = new("admin", "Admin", Role.Admin, null);

        [Fact]
        public void FairSeatsHaveTheirNamesTimesAndWorlds()
        {
            Assert.Equal("Fair MCTS Easy", Seats.Name(SeatKind.PimcEasy));
            Assert.Equal("Fair MCTS Medium", Seats.Name(SeatKind.PimcMedium));
            Assert.Equal("Fair MCTS Hard", Seats.Name(SeatKind.PimcHard));
            foreach ((SeatKind fair, SeatKind plain) in new[] { (SeatKind.PimcEasy, SeatKind.MctsEasy), (SeatKind.PimcMedium, SeatKind.MctsMedium), (SeatKind.PimcHard, SeatKind.MctsHard) })
            {
                Assert.True(Seats.NeedsValueNet(fair));
                Assert.Equal(Seats.PresetSeconds(plain), Seats.PresetSeconds(fair));
            }
            Assert.Equal(1, Seats.Determinizations(SeatKind.PimcEasy));
            Assert.Equal(1, Seats.Determinizations(SeatKind.PimcMedium));
            Assert.Equal(4, Seats.Determinizations(SeatKind.PimcHard));

            // The default seats stay as they were
            Assert.Equal(SeatKind.MctsHard, ServerConsole.ParseKind("hard"));
            Assert.Equal(SeatKind.PimcHard, ServerConsole.ParseKind("fair-hard"));
            Assert.Equal(SeatKind.PimcEasy, ServerConsole.ParseKind("PimcEasy"));
        }

        [SkippableFact]
        public void AFairSeatsAgentPlaysUnderHiddenInformation()
        {
            TestUtil.SkipWithoutModel();
            GameState state = TestUtil.NewGame();
            DeterminizedMCTSAgent hard = Assert.IsType<DeterminizedMCTSAgent>(Seats.CreateAgent(SeatKind.PimcHard, 2, state, AgentSettings.Default, true));
            Assert.Equal(2, hard.PlayerIndex);
            Assert.Equal(4, hard.Determinizations);
            Assert.Equal(2, hard.Search.TimeLimitSeconds);

            DeterminizedMCTSAgent easy = Assert.IsType<DeterminizedMCTSAgent>(Seats.CreateAgent(SeatKind.PimcEasy, 1, state, new AgentSettings(null, 500, OpponentModel.MaxN), true));
            Assert.Equal(1, easy.Determinizations);
            Assert.Equal(500, easy.Search.IterationLimit);

            // Without the net they fall back to the builder, like the other seats that need it
            Assert.IsType<BuilderAgent>(Seats.CreateAgent(SeatKind.PimcMedium, 0, state, AgentSettings.Default, false));
        }

        [SkippableFact]
        public void TheMatchGivesFairSeatsItsBeliefsThroughUndoLoadAndSeatChanges()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(41);
            using Match match = new Match(TestUtil.NewGame(), [SeatKind.PimcMedium, SeatKind.Human, SeatKind.PimcEasy, SeatKind.Human], valueNetAvailable: true);
            Random random = new Random(41);

            void AssertBeliefs()
            {
                for (int seat = 0; seat < 4; seat++)
                {
                    if (match.AgentOf(seat) is DeterminizedMCTSAgent fair)
                        Assert.Equal(match.Beliefs.For(seat), fair.Beliefs);
                }
            }

            for (int i = 0; i < 600 && !match.Record.State.HasEnded; i++)
            {
                double r = random.NextDouble();
                if (r < 0.08) match.Handle(Admin, new UndoCommand());
                else if (r < 0.12) match.Handle(Admin, new RedoCommand());
                else
                {
                    GameState state = match.Record.State;
                    List<Action> legal = LegalActionProvider.GetActionsForState(state, TestUtil.ActingPlayer(state));
                    match.Handle(Admin, new PlayMove(legal[random.Next(legal.Count)]));
                }
                AssertBeliefs();
            }

            // A seat that turns fair in the middle of the game knows only what the seat could know
            match.Handle(Admin, new SetSeat(1, SeatKind.PimcHard));
            Assert.IsType<DeterminizedMCTSAgent>(match.AgentOf(1));
            AssertBeliefs();

            // Loading a game: the beliefs follow its moves from its start
            match.Handle(Admin, new LoadGame(new GameState(match.Record.State), match.Record.ActionsSinceLastEdit(), []));
            AssertBeliefs();
        }
    }
}
