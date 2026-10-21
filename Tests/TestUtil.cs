using Agents;
using Agents.Inference;
using Common;
using Common.Actions;
using System.Reflection;
using Action = Common.Actions.Action;

// The simulator draws from one global random source (Utils.Random); running tests one after another keeps seeded runs reproducible
[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace Tests
{
    /// <summary>
    /// Shared helpers for building and playing games in tests
    /// </summary>
    public static class TestUtil
    {
        public const int PlayerCount = 4;

        /// <summary>
        /// All concrete action types of the simulator
        /// </summary>
        public static readonly IReadOnlyList<Type> ActionTypes = typeof(Action).Assembly.GetTypes()
            .Where(type => type.IsSubclassOf(typeof(Action)) && !type.IsAbstract)
            .OrderBy(type => type.Name)
            .ToList();

        /// <summary>
        /// Seeds the simulator's global random source (map generation, dice, card draws, random agents)
        /// </summary>
        public static void Seed(int seed)
        {
            Utils.Random = new ThreadSafeRandom(seed);
        }

        public static GameState NewGame()
        {
            GameState state = new GameState(MapGenerator.GenerateRandomClassic(), PlayerCount);
            state.Board.Adjacency.PrecomputeAll();
            return state;
        }

        public static Agent[] RandomAgents()
        {
            return Enumerable.Range(0, PlayerCount).Select(idx => (Agent)new RandomAgent((sbyte)idx)).ToArray();
        }

        /// <summary>
        /// The player that acts next (the lowest-index player able to act, as in the Client)
        /// </summary>
        public static sbyte ActingPlayer(GameState state)
        {
            for (sbyte playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                if (state.CanPlayerAct(playerIdx)) return playerIdx;
            }

            throw new InvalidOperationException("No player can act.");
        }

        /// <summary>
        /// Whether the turn player is free to build, trade or end the turn
        /// </summary>
        public static bool IsFreeTurn(GameState state)
        {
            return state.Turn.TypeOfRound == TurnState.RoundType.Normal
                && !state.Turn.MustRoll && !state.Turn.MustDiscard && !state.Turn.MustMoveRobber;
        }

        /// <summary>
        /// Plays random actions until the game ends or maxActions were played. The callbacks receive the state
        /// before and after each action. Returns the played actions in order.
        /// </summary>
        public static List<Action> PlayRandomGame(GameState state,
            System.Action<GameState, Action>? beforeApply = null,
            System.Action<GameState, Action>? afterApply = null,
            int maxActions = int.MaxValue)
        {
            Agent[] agents = RandomAgents();
            List<Action> played = [];

            while (!state.HasEnded && played.Count < maxActions)
            {
                Action action = agents[ActingPlayer(state)].Act(state);
                beforeApply?.Invoke(state, action);
                action.Apply(state);
                played.Add(action);
                afterApply?.Invoke(state, action);
            }

            return played;
        }

        /// <summary>
        /// Plays random actions until the condition holds. Returns false if the game ended first.
        /// </summary>
        public static bool PlayUntil(GameState state, Func<GameState, bool> condition, List<Action>? played = null)
        {
            Agent[] agents = RandomAgents();

            while (!condition(state))
            {
                if (state.HasEnded) return false;

                Action action = agents[ActingPlayer(state)].Act(state);
                action.Apply(state);
                played?.Add(action);
            }

            return true;
        }

        /// <summary>
        /// A fresh board with no buildings, switched to a normal turn of the given player
        /// </summary>
        public static GameState EmptyNormalTurn(sbyte playerIdx, bool mustRoll = false)
        {
            GameState state = NewGame();
            state.Turn.TypeOfRound = TurnState.RoundType.Normal;
            state.Turn.PlayerIndex = playerIdx;
            state.Turn.MustRoll = mustRoll;
            return state;
        }

        /// <summary>
        /// Moves cards from the resource bank to a player's hand
        /// </summary>
        public static void GiveCards(GameState state, int playerIdx, ResourceCardType type, uint amount)
        {
            state.ResourceBank.Remove(type, amount);
            state.Players[playerIdx].ResourceCards.Add(type, amount);
        }

        /// <summary>
        /// All actions the static GetActionsForState providers offer for the given player
        /// </summary>
        public static List<Action> AllProvidedActions(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            foreach (Type type in ActionTypes)
            {
                MethodInfo? provider = type.GetMethod("GetActionsForState", BindingFlags.Public | BindingFlags.Static, [typeof(GameState), typeof(sbyte)]);
                if (provider == null) continue;

                actions.AddRange((List<Action>)provider.Invoke(null, [state, playerIdx])!);
            }

            return actions;
        }

        public static string FixturePath(params string[] parts)
        {
            return Path.Combine([AppContext.BaseDirectory, "Fixtures", .. parts]);
        }

        /// <summary>
        /// The Fixtures folder in the source tree (for regenerating fixtures)
        /// </summary>
        public static string SourceFixtureDirectory()
        {
            DirectoryInfo? dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "Tests.csproj")))
                dir = dir.Parent;

            return Path.Combine(dir?.FullName ?? throw new DirectoryNotFoundException("Tests.csproj not found above the test output."), "Fixtures");
        }

        public static string ModelPath => StateValueNet.DefaultModelPath;

        /// <summary>
        /// Skips the test when the deployed value net is not available (it is not tracked in git)
        /// </summary>
        public static void SkipWithoutModel()
        {
            Skip.IfNot(File.Exists(ModelPath), $"Value net {StateValueNet.DefaultModelFile} not found next to the tests.");
        }

        public static int CountBuildings(GameState state, int playerIdx, Intersection.BuildingType type)
        {
            return state.Board.Intersections.Count(intersection => intersection.Owner == playerIdx && intersection.Building == type);
        }

        public static int CountRoads(GameState state, int playerIdx)
        {
            return state.Board.Edges.Count(edge => edge.Owner == playerIdx && edge.Building == Edge.BuildingType.Road);
        }
    }
}
