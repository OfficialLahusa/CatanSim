using Agents;
using Agents.MCTS;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The PIMC agent (DeterminizedMCTSAgent) searches only what its seat may know: the same view gives the same
    /// determinizations, whatever the hidden truth, and its moves are legal in the true game
    /// </summary>
    public class DeterminizedAgentTests
    {
        private static DeterminizedMCTSAgent NewAgent(sbyte seat, GameState start, int seed = 3) =>
            new DeterminizedMCTSAgent(new MCTSAgent(seat, start, iterationLimit: 256), start, determinizations: 4, seed: seed);

        private static bool IsSteal(GameState state, Action action) =>
            action is RobberAction { TargetPlayerIndex: sbyte victim } && victim != action.PlayerIndex
            && CardTypeExtensions.ResourceTypes.Count(t => state.Players[victim].ResourceCards.Get(t) > 0) >= 2;

        [SkippableFact]
        public void WithoutHiddenInformationItSearchesTheTrueState()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(21);
            GameState start = TestUtil.NewGame();
            DeterminizedMCTSAgent agent = NewAgent(1, start);
            IReadOnlyList<GameState> searched = agent.Determinize(start);
            Assert.Single(searched);
            Assert.Equal(start, searched[0]);
        }

        [SkippableFact]
        public void ItSearchesOnlyWhatItsSeatSees()
        {
            TestUtil.SkipWithoutModel();

            // A game up to a steal, played on with two different stolen cards: an agent outside the steal must search the
            // same determinizations in both
            for (int seed = 1; seed < 300; seed++)
            {
                TestUtil.Seed(seed);
                GameState start = TestUtil.NewGame();
                GameState state = new GameState(start);
                List<Action> played = [];
                Agent[] agents = TestUtil.RandomAgents();
                Action? steal = null;
                while (!state.HasEnded && played.Count < 4000)
                {
                    Action action = agents[TestUtil.ActingPlayer(state)].Act(state);
                    if (IsSteal(state, action)) { steal = action; break; }
                    action.Apply(state);
                    played.Add(action);
                }
                if (steal == null) continue;

                RobberAction robber = (RobberAction)steal;
                sbyte outsider = (sbyte)Enumerable.Range(0, 4).First(p => p != robber.PlayerIndex && p != robber.TargetPlayerIndex);
                for (int outcomeSeed = 1; outcomeSeed < 100; outcomeSeed++)
                {
                    (GameState truthA, DeterminizedMCTSAgent agentA, RobberAction stealA) = PlayOn(start, played, steal, outsider, outcomeSeed);
                    (GameState truthB, DeterminizedMCTSAgent agentB, RobberAction stealB) = PlayOn(start, played, steal, outsider, outcomeSeed + 500);
                    if (stealA.History!.StolenCard == stealB.History!.StolenCard) continue;

                    Assert.Equal(agentA.Beliefs, agentB.Beliefs);
                    IReadOnlyList<GameState> searchedA = agentA.Determinize(truthA), searchedB = agentB.Determinize(truthB);
                    Assert.Equal(searchedA.Count, searchedB.Count);
                    Assert.True(searchedA.Count > 1);
                    for (int i = 0; i < searchedA.Count; i++) Assert.Equal(searchedA[i], searchedB[i]);
                    return;
                }
            }
            Assert.Fail("No random game had a steal with two possible outcomes");
        }

        private static (GameState truth, DeterminizedMCTSAgent agent, RobberAction steal) PlayOn(GameState start, List<Action> played, Action steal, sbyte seat, int outcomeSeed)
        {
            GameState truth = new GameState(start);
            DeterminizedMCTSAgent agent = NewAgent(seat, start);
            foreach (Action action in played)
            {
                Action copy = action.Clone();
                copy.Apply(truth);
                agent.HandleAction(copy);
            }
            TestUtil.Seed(outcomeSeed);
            RobberAction played2 = (RobberAction)steal.Clone();
            played2.Apply(truth);
            agent.HandleAction(played2);
            return (truth, agent, played2);
        }

        [SkippableFact]
        public void ItsMovesAreLegalInTheTrueGame()
        {
            TestUtil.SkipWithoutModel();
            TestUtil.Seed(22);
            GameState state = TestUtil.NewGame();
            DeterminizedMCTSAgent pimc = NewAgent(0, state);
            Agent[] agents = TestUtil.RandomAgents();
            int decisions = 0;
            while (!state.HasEnded && decisions < 60)
            {
                sbyte acting = TestUtil.ActingPlayer(state);
                Action action = acting == 0 ? pimc.Act(new GameState(state)) : agents[acting].Act(state);
                if (acting == 0)
                {
                    decisions++;
                    Assert.True(action.IsValidFor(state), $"{action} is not legal in the true game");
                }
                action.Apply(state);
                pimc.HandleAction(action);
            }
        }
    }
}
