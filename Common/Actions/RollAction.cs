using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;
using static Common.Actions.BuyDevelopmentCardAction;

namespace Common.Actions
{
    public class RollAction : Action, IReplayAction, IActionProvider, IOutputRandomnessAction
    {
        public record RollActionHistory
        {
            public RollResult PrevRollResult { get; init; }
            public bool TriggeredRobber { get; init; }
            [YamlIgnore]
            public uint[,]? AwardedYields { get; set; } = null;

            // YamlDotNet cannot serialize multidimensional arrays, so expose a
            // jagged-array view which is YAML-serializable and converts to/from the
            // internal uint[,] representation.
            [YamlMember(Alias = "AwardedYields")]
            public uint[][]? AwardedYieldsSerialized
            {
                get
                {
                    if (AwardedYields == null) return null;

                    int rows = AwardedYields.GetLength(0);
                    int cols = AwardedYields.GetLength(1);
                    uint[][] result = new uint[rows][];
                    for (int i = 0; i < rows; i++)  
                    {
                        result[i] = new uint[cols];
                        for (int j = 0; j < cols; j++) result[i][j] = AwardedYields[i, j];
                    }

                    return result;
                }
                set
                {
                    if (value == null)
                    {
                        AwardedYields = null;
                        return;
                    }

                    int rows = value.Length;
                    int cols = rows > 0 ? value[0].Length : 0;
                    uint[,] arr = new uint[rows, cols];

                    for (int i = 0; i < rows; i++)
                    {
                        if (value[i].Length != cols) throw new InvalidOperationException("Inconsistent row lengths in AwardedYieldsSerialized");
                        for (int j = 0; j < cols; j++) arr[i, j] = value[i][j];
                    }

                    AwardedYields = arr;
                }
            }
            public uint RobbedYields { get; init; } = 0;
            public uint CappedYields { get; init; } = 0;

            public RollActionHistory(RollResult PrevRollResult, bool TriggeredRobber, uint[,]? AwardedYields = null, uint RobbedYields = 0, uint CappedYields = 0)
            {
                this.PrevRollResult = PrevRollResult;
                this.TriggeredRobber = TriggeredRobber;
                this.AwardedYields = AwardedYields;
                this.RobbedYields = RobbedYields;
                this.CappedYields = CappedYields;
            }

            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private RollActionHistory() : this(new(), false) { }
        }

        public RollActionHistory? History { get; private set; }

        public RollResult RollResult { get; init; }

        // Randomize roll when null is passed
        // The option to specify the roll result is primarily used for action tree exploration
        // TODO: Potentially rewrite by forcing randomization and moving the result to history
        public RollAction(sbyte playerIdx, RollResult? rollResult)
            : base(playerIdx)
        {
            RollResult = rollResult ?? RollResult.GetRandom();
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private RollAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            bool robberTriggered = RollResult.Total == 7;

            if (robberTriggered)
            {
                // Require discard if card limit is exceeded
                for (int player = 0; player < state.Players.Length; player++)
                {
                    bool mustDiscard = state.Players[player].ResourceCards.Count() > state.Settings.RobberCardLimit;
                    state.Turn.AwaitedPlayerDiscards[player] = mustDiscard;
                }

                // Require robber move
                state.Turn.MustMoveRobber = true;

                History = new RollActionHistory(state.Turn.LastRoll, robberTriggered);
            }
            else
            {
                (uint[,] yieldSummary, uint robbedYields, uint cappedYields) = AwardYields(state, RollResult.Total);

                History = new RollActionHistory(state.Turn.LastRoll, robberTriggered, yieldSummary, robbedYields, cappedYields);
            }

            // Update turn state
            state.Turn.LastRoll = RollResult;
            state.Turn.MustRoll = false;
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Cancel required discards and robber movement
            if (History!.TriggeredRobber)
            {
                Array.Fill(state.Turn.AwaitedPlayerDiscards, false);

                state.Turn.MustMoveRobber = false;
            }
            // Return yields
            else
            {
                for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
                {
                    for (int resourceTypeIdx = 0; resourceTypeIdx < CardSet<ResourceCardType>.Values.Count; resourceTypeIdx++)
                    {
                        uint awardedAmount = History.AwardedYields![playerIdx, resourceTypeIdx];
                        if (awardedAmount == 0) continue;

                        ResourceCardType resourceType = CardSet<ResourceCardType>.Values[resourceTypeIdx];

                        state.ResourceBank.Add(resourceType, awardedAmount);
                        state.Players[playerIdx].ResourceCards.Remove(resourceType, awardedAmount);
                    }
                }
            }

            // Update turn state
            state.Turn.LastRoll = History.PrevRollResult;
            state.Turn.MustRoll = true;
        }

        private (uint[,] yieldSummary, uint robbedYields, uint cappedYields) AwardYields(GameState state, int number)
        {
            uint[,] yieldSummary = new uint[state.Players.Length, CardSet<ResourceCardType>.Values.Count];
            uint robbedYields = 0;

            // Calculate tile yields
            foreach (Tile tile in state.Board.Map.Where(x => x.HasYield() && x.Number == number))
            {
                foreach (Intersection intersection in state.Board.Adjacency.GetIntersections(tile))
                {
                    uint yieldCount = intersection.Building switch
                    {
                        Intersection.BuildingType.City => 2,
                        Intersection.BuildingType.Settlement => 1,
                        _ => 0
                    };

                    if (yieldCount > 0)
                    {
                        if (tile != state.Board.Robber)
                        {
                            yieldSummary[intersection.Owner, CardSet<ResourceCardType>.ToInt(tile.Type.ToCardType())] += yieldCount;
                        }
                        else
                        {
                            robbedYields += yieldCount;
                        }
                    }
                }
            }

            // Award yields according to limited bank stock
            uint cappedYields = 0;
            for (int resourceTypeIdx = 0; resourceTypeIdx < CardSet<ResourceCardType>.Values.Count; resourceTypeIdx++)
            {
                ResourceCardType resourceType = CardSet<ResourceCardType>.Values[resourceTypeIdx];
                uint bankStock = state.ResourceBank.Get(resourceType);

                uint totalAwardedAmount = 0;
                uint affectedPlayers = 0;

                for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
                {
                    uint awardedAmount = yieldSummary[playerIdx, resourceTypeIdx];
                    totalAwardedAmount += awardedAmount;
                    affectedPlayers += awardedAmount > 0 ? 1u : 0u;
                }

                // Do not award yields, if bank stock is insufficient and more than one player is affected
                bool insufficientStock = bankStock < totalAwardedAmount;
                if (insufficientStock && affectedPlayers > 1)
                {
                    cappedYields += totalAwardedAmount;

                    // Remove capped yields from summary
                    for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
                    {
                        yieldSummary[playerIdx, resourceTypeIdx] = 0;
                    }

                    continue;
                }

                // Transfer cards from bank to player
                for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
                {
                    uint awardedAmount = yieldSummary[playerIdx, resourceTypeIdx];
                    if (awardedAmount > bankStock) awardedAmount = bankStock;

                    // Remove capped yields from summary
                    yieldSummary[playerIdx, resourceTypeIdx] = awardedAmount;

                    state.ResourceBank.Remove(resourceType, awardedAmount);
                    state.Players[playerIdx].ResourceCards.Add(resourceType, awardedAmount);
                }
            }

            return (yieldSummary, robbedYields, cappedYields);
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && turn.MustRoll
                && !turn.MustMoveRobber;
        }

        public bool HasHistory()
        {
            return History != null;
        }

        public void ClearHistory()
        {
            History = null;
        }


        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            RollAction randomRollAction = new RollAction(playerIdx, null);
            if(randomRollAction.IsValidFor(state)) actions.Add(randomRollAction);

            return actions;
        }

        public List<Action> GetOutcomeVariants(GameState state, sbyte playerIdx)
        {
            if (HasHistory())
                throw new InvalidOperationException("The output randomness of this action was already resolved.");

            List<Action> variants = new List<Action>();

            if (!IsTurnValid(state.Turn, playerIdx)) return variants;

            // Output randomness: What is the dice roll result?
            // This action is an exception from the other output randomness actions, since the initial parameters of the RollResult have to be disregarded.
            for (byte first = 1; first <= 6; first++)
            {
                for (byte second = 1; second <= 6; second++)
                {
                    RollResult roll = new() { First = first, Second = second };
                    RollAction action = new((sbyte)state.Turn.PlayerIndex, roll);

                    if (action.IsValidFor(state))
                    {
                        variants.Add(action);
                    }
                }
            }

            return variants;
        }

        public override string ToString()
        {
            return base.ToString() + $", {RollResult.First}+{RollResult.Second}={RollResult.Total}";
        }
    }
}
