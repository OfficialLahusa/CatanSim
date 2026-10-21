using Common;
using Common.Actions;
using System.Reflection;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// Generic properties of all action types: coverage, Apply/Revert round trips, provider validity,
    /// and the copying/matching helpers used by the MCTS tree
    /// </summary>
    public class ActionTests
    {
        [Fact]
        public void EveryActionTypeOccursInRandomGames()
        {
            TestUtil.Seed(2);
            HashSet<Type> seen = [];

            // Guards the reflection-based type list the other tests rely on
            Assert.Equal(19, TestUtil.ActionTypes.Count);

            for (int game = 0; game < 200 && seen.Count < TestUtil.ActionTypes.Count; game++)
            {
                TestUtil.PlayRandomGame(TestUtil.NewGame(), beforeApply: (_, action) => seen.Add(action.GetType()));
            }

            Assert.Empty(TestUtil.ActionTypes.Except(seen).Select(type => type.Name));
        }

        [Fact]
        public void ApplyThenRevertRestoresTheStateForEveryActionType()
        {
            TestUtil.Seed(3);
            Dictionary<Type, int> checkedPerType = [];

            for (int game = 0; game < 120 && checkedPerType.Count < TestUtil.ActionTypes.Count; game++)
            {
                TestUtil.PlayRandomGame(TestUtil.NewGame(), beforeApply: (state, action) =>
                {
                    // Revert the action on a copy; a clone keeps the original free of history
                    GameState before = new GameState(state);
                    GameState work = new GameState(state);
                    Action clone = action.Clone();

                    clone.Apply(work);
                    clone.Revert(work);

                    Assert.True(work.Equals(before), $"Apply + Revert of {action} did not restore the state");
                    checkedPerType[action.GetType()] = checkedPerType.GetValueOrDefault(action.GetType()) + 1;
                });
            }

            Assert.Empty(TestUtil.ActionTypes.Except(checkedPerType.Keys).Select(type => type.Name));
        }

        [Fact]
        public void RevertingAWholeGameRestoresTheInitialState()
        {
            TestUtil.Seed(4);

            for (int game = 0; game < 10; game++)
            {
                GameState state = TestUtil.NewGame();
                GameState initial = new GameState(state);
                List<Action> played = TestUtil.PlayRandomGame(state);
                GameState final = new GameState(state);

                for (int i = played.Count - 1; i >= 0; i--)
                    played[i].Revert(state);
                Assert.True(state.Equals(initial), $"game {game}: reverting all {played.Count} actions did not restore the initial state");

                // Replaying the stored outcomes reaches the same final state
                foreach (Action action in played)
                    action.Apply(state);
                Assert.True(state.Equals(final), $"game {game}: replaying all actions did not reach the final state");
            }
        }

        [Fact]
        public void ProvidedActionsAreValidAndTheActingPlayerAlwaysHasOne()
        {
            TestUtil.Seed(5);
            Random sample = new Random(5);
            int checkedStates = 0;

            for (int game = 0; game < 15; game++)
            {
                TestUtil.PlayRandomGame(TestUtil.NewGame(), beforeApply: (state, _) =>
                {
                    if (sample.NextDouble() > 0.1) return;
                    checkedStates++;

                    for (sbyte playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
                    {
                        List<Action> actions = TestUtil.AllProvidedActions(state, playerIdx);

                        foreach (Action action in actions)
                            Assert.True(action.IsValidFor(state), $"provided action {action} is not valid");

                        if (playerIdx == TestUtil.ActingPlayer(state))
                            Assert.NotEmpty(actions);
                        else if (!state.CanPlayerAct(playerIdx))
                            Assert.Empty(actions);
                    }
                });
            }

            Assert.True(checkedStates > 500);
        }

        [Fact]
        public void CloneKeepsTheOutcomeAndLeavesTheOriginalUntouched()
        {
            TestUtil.Seed(6);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.MustRoll && RollAction.GetActionsForState(s, (sbyte)s.Turn.PlayerIndex).Count > 0));

            RollAction roll = (RollAction)RollAction.GetActionsForState(state, (sbyte)state.Turn.PlayerIndex)[0];
            RollAction variant = (RollAction)roll.GetOutcomeVariants(state, (sbyte)state.Turn.PlayerIndex).First(v => ((RollAction)v.Action).History!.RollResult.Total == 8).Action;
            var originalHistory = variant.History;

            RollAction clone = (RollAction)variant.Clone();
            clone.Apply(new GameState(state));

            Assert.Same(originalHistory, variant.History);
            Assert.NotSame(variant.History, clone.History);
            Assert.Equal(8, clone.History!.RollResult.Total);
        }

        [Fact]
        public void IsSameDecisionComparesTypePlayerAndParametersButNotHistory()
        {
            Assert.True(new RoadAction(1, 10).IsSameDecision(new RoadAction(1, 10)));
            Assert.False(new RoadAction(1, 10).IsSameDecision(new RoadAction(1, 11)));
            Assert.False(new RoadAction(1, 10).IsSameDecision(new RoadAction(2, 10)));
            Assert.False(new RoadAction(1, 10).IsSameDecision(new SecondInitialRoadAction(1, 10)));

            Assert.True(new TwoToOneTradeAction(0, ResourceCardType.Ore, ResourceCardType.Wool).IsSameDecision(new TwoToOneTradeAction(0, ResourceCardType.Ore, ResourceCardType.Wool)));
            Assert.False(new TwoToOneTradeAction(0, ResourceCardType.Ore, ResourceCardType.Wool).IsSameDecision(new TwoToOneTradeAction(0, ResourceCardType.Ore, ResourceCardType.Grain)));
            Assert.True(new RobberAction(0, 12, null).IsSameDecision(new RobberAction(0, 12, null)));
            Assert.False(new RobberAction(0, 12, null).IsSameDecision(new RobberAction(0, 12, 2)));
            Assert.True(new YearOfPlentyAction(0, ResourceCardType.Brick, null).IsSameDecision(new YearOfPlentyAction(0, ResourceCardType.Brick, null)));
            Assert.False(new YearOfPlentyAction(0, ResourceCardType.Brick, null).IsSameDecision(new YearOfPlentyAction(0, ResourceCardType.Brick, ResourceCardType.Ore)));

            // Card sets are compared by content
            CardSet<ResourceCardType> discardA = new();
            discardA.Add(ResourceCardType.Lumber, 2);
            CardSet<ResourceCardType> discardB = new(discardA);
            CardSet<ResourceCardType> discardC = new();
            discardC.Add(ResourceCardType.Ore, 2);
            Assert.True(new DiscardAction(3, discardA).IsSameDecision(new DiscardAction(3, discardB)));
            Assert.False(new DiscardAction(3, discardA).IsSameDecision(new DiscardAction(3, discardC)));

            // History (e.g. a resolved roll) is not part of the decision
            TestUtil.Seed(7);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.MustRoll && RollAction.GetActionsForState(s, (sbyte)s.Turn.PlayerIndex).Count > 0));
            sbyte roller = (sbyte)state.Turn.PlayerIndex;
            RollAction rolled = new RollAction(roller);
            rolled.Apply(new GameState(state));
            Assert.True(rolled.IsSameDecision(new RollAction(roller)));
        }

        [Fact]
        public void DistinctProvidedActionsAreNeverTheSameDecision()
        {
            // The MCTS tree matches played actions to its children by decision, so the legal actions of a state must be
            // pairwise distinguishable (otherwise a played action could be matched to the wrong child)
            TestUtil.Seed(9);
            Random sample = new Random(9);
            int comparedSets = 0;

            for (int game = 0; game < 10; game++)
            {
                TestUtil.PlayRandomGame(TestUtil.NewGame(), beforeApply: (state, _) =>
                {
                    if (sample.NextDouble() > 0.2) return;

                    List<Action> actions = TestUtil.AllProvidedActions(state, TestUtil.ActingPlayer(state));
                    for (int i = 0; i < actions.Count; i++)
                        for (int j = i + 1; j < actions.Count; j++)
                            Assert.False(actions[i].IsSameDecision(actions[j]), $"{actions[i]} and {actions[j]} are considered the same decision");

                    comparedSets++;
                });
            }

            Assert.True(comparedSets > 100);
        }

        [Fact]
        public void HasSameOutcomeMatchesEquivalentOutcomes()
        {
            TestUtil.Seed(8);
            GameState state = TestUtil.NewGame();
            Assert.True(TestUtil.PlayUntil(state, s => s.Turn.MustRoll && RollAction.GetActionsForState(s, (sbyte)s.Turn.PlayerIndex).Count > 0));
            sbyte roller = (sbyte)state.Turn.PlayerIndex;

            RollAction Rolled(byte first, byte second)
            {
                RollAction action = new RollAction(roller);
                SetHistory(action, new RollAction.RollActionHistory(new RollResult { First = first, Second = second }, state.Turn.LastRoll));
                return action;
            }

            Assert.True(Rolled(1, 6).HasSameOutcome(Rolled(6, 1)));
            Assert.True(Rolled(3, 4).HasSameOutcome(Rolled(5, 2)));
            Assert.False(Rolled(1, 6).HasSameOutcome(Rolled(2, 6)));
            Assert.False(Rolled(1, 6).HasSameOutcome(new RollAction(roller)));
        }

        internal static void SetHistory(Action action, object? history)
        {
            action.GetType().GetProperty("History")!.SetValue(action, history);
        }
    }
}
