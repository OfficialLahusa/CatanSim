using Agents;
using Agents.MCTS;
using Common;
using System.Diagnostics;
using Action = Common.Actions.Action;

namespace SelfPlayCollector
{
    /// <summary>
    /// A position recorded after a full search: the state vector, the player deciding there, the search's mean result of
    /// every player at the root, and the visits of the root moves by their index in MCTSTree.GetValidActions (the order of
    /// the root's children), only those with visits; Moves are the root's moves in that order
    /// </summary>
    internal sealed record RecordedPosition(int ActionIndex, int Round, int Actor, Half[] Features, double[] RootValues, uint RootVisits,
        long Iterations, (int move, uint visits)[] Visits, Action[] Moves);

    /// <summary>
    /// A played game: its ID and seed, the final state with all actions (a save file replays them from the seed's initial
    /// state), the recorded positions, and whether it ended (games stopped at the action limit have no winner)
    /// </summary>
    internal sealed record GameRecord(long Game, ulong Seed, GameState FinalState, List<Action> Actions, List<RecordedPosition> Positions,
        int SearchedMoves, double Seconds)
    {
        public bool Finished => FinalState.HasEnded;

        // Players can only win on their own turn, so the turn player of an ended game is the winner
        public int Winner => Finished ? FinalState.Turn.PlayerIndex : -1;
    }

    /// <summary>
    /// One self-play game: every decision with more than one option is searched with MCTS (max^n) on one tree for all
    /// seats, re-rooted after every action. With FullSearchProbability a decision gets a full search and its position is
    /// recorded, otherwise a fast one (playout cap randomization, Wu 2019). The first TemperatureMoves searched decisions
    /// sample their move from the visits.
    /// </summary>
    internal static class SelfPlayGame
    {
        public const int PlayerCount = 4;

        /// <summary>
        /// The game's initial state: map and development card deck come from the seed, as in DatasetCollector.PlayMatch
        /// </summary>
        public static GameState InitialState(ulong seed, bool centerDesert)
        {
            Utils.Random.SeedCurrentThread(Seeds.ToInt32(seed));
            return new GameState(MapGenerator.GenerateRandomClassic(centerDesert), PlayerCount);
        }

        /// <summary>
        /// Plays the game, or returns null if it was cancelled
        /// </summary>
        public static async Task<GameRecord?> PlayAsync(long game, ulong seed, SelfPlaySettings settings, MCTSSearch search, IStateVectorizer vectorizer,
            System.Action<long> countIterations, CancellationToken cancellation)
        {
            var stopwatch = Stopwatch.StartNew();
            GameState state = InitialState(seed, settings.CenterDesert);
            Random random = new(Seeds.ToInt32(Seeds.Derive(seed, 1)));
            List<Action> actions = [];
            List<RecordedPosition> positions = [];
            MCTSTree? tree = null;
            int searchedMoves = 0;
            float[] features = new float[vectorizer.FeatureDim];

            while (!state.HasEnded && actions.Count < settings.MaxActions)
            {
                if (cancellation.IsCancellationRequested) return null;

                List<Action> valid = MCTSTree.GetValidActions(state);
                Action action;
                if (valid.Count == 1)
                {
                    action = valid[0];
                }
                else
                {
                    if (tree == null || !tree.RootState.Equals(state))
                        tree = new MCTSTree(new GameState(state), (sbyte)state.Turn.PlayerIndex, settings.Exploration, model: OpponentModel.MaxN);

                    bool full = random.NextDouble() < settings.FullSearchProbability;
                    int iterations = full ? settings.FullIterations : settings.FastIterations;
                    // Root noise for the full searches only, whose visits become policy targets (as in KataGo)
                    search.RootNoiseWeight = full ? settings.RootNoise : 0;
                    search.RootNoiseConcentration = settings.RootNoiseConcentration;
                    search.NoiseRandom = random;
                    await search.RunAsync(tree, double.PositiveInfinity, iterations).ConfigureAwait(false);
                    countIterations(search.LastSearch!.Iterations);

                    if (full)
                        positions.Add(Record(state, actions.Count, tree, search.LastSearch.Iterations, vectorizer, features,
                            settings.PruneTargets ? search.LastRootPolicy : null));

                    action = searchedMoves < settings.TemperatureMoves ? tree.SampleMove(settings.Temperature, random) : tree.GetMostPromisingMove();
                    searchedMoves++;
                }

                action.Apply(state);
                actions.Add(action);
                tree = Advance(tree, action);
            }

            return new GameRecord(game, seed, state, actions, positions, searchedMoves, stopwatch.Elapsed.TotalSeconds);
        }

        /// <param name="cleanPriors">The root's priors before the root noise, to record the visits without those the noise
        /// caused (MCTSTree.PrunedRootVisits); null records the visits as they are</param>
        private static RecordedPosition Record(GameState state, int actionIndex, MCTSTree tree, long iterations, IStateVectorizer vectorizer, float[] features,
            float[]? cleanPriors)
        {
            vectorizer.VectorizeInto(state, features);
            Half[] row = new Half[features.Length];
            for (int i = 0; i < features.Length; i++)
                row[i] = (Half)features[i];

            IReadOnlyList<MCTSNode> children = tree.RootNode.Children;
            uint[] counts = cleanPriors != null ? tree.PrunedRootVisits(cleanPriors) : children.Select(child => child.VisitCount).ToArray();
            var visits = new List<(int, uint)>();
            for (int i = 0; i < children.Count; i++)
                if (counts[i] > 0) visits.Add((i, counts[i]));

            return new RecordedPosition(actionIndex, state.Turn.RoundCounter, tree.RootNode.DeciderIndex, row, tree.RootValues(), tree.RootNode.VisitCount,
                iterations, visits.ToArray(), children.Select(child => child.LastAction!).ToArray());
        }

        /// <summary>
        /// The tree after the action was applied to the game, as MCTSAgent.HandleAction keeps it; null to start over
        /// </summary>
        private static MCTSTree? Advance(MCTSTree? tree, Action action)
        {
            if (tree == null) return null;

            MCTSNode? nextRoot = tree.FindChildForAction(action);
            GameState nextState = new(tree.RootState);
            Action replay = action.Clone();
            if (nextRoot == null || !replay.IsValidFor(nextState)) return null;

            // The real action keeps the outcome that happened (e.g. the dice pair behind a roll total)
            replay.Apply(nextState);
            tree.ReRoot(nextRoot, nextState);
            return tree;
        }
    }
}
