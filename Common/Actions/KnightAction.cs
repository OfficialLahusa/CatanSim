using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class KnightAction : Action, IReplayAction, IActionProvider
    {
        public record KnightActionHistory(int PrevLargestArmyHolder)
        {
            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private KnightActionHistory() : this(-1) { }
        }

        public KnightActionHistory? History { get; private set; }

        public KnightAction(sbyte playerIdx)
            : base(playerIdx)
        { }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private KnightAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            int currentLargestArmyHolder = -1;

            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                if (state.Players[playerIdx].VictoryPoints.LargestArmyPoints > 0)
                {
                    currentLargestArmyHolder = playerIdx;
                    break;
                }
            }

            History = new KnightActionHistory(currentLargestArmyHolder);

            // Remove card
            state.Players[PlayerIndex].DevelopmentCards.Remove(DevelopmentCardType.Knight, 1);

            // Update turn state
            state.Turn.MustMoveRobber = true;
            state.Turn.HasPlayedDevelopmentCard = true;

            // Increment largest army and recalculate VPs
            ++state.Players[PlayerIndex].PlayedKnights;
            state.CalculateLargestArmy(PlayerIndex);

            // Check for match completion
            state.CheckForCompletion();
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Return card
            state.Players[PlayerIndex].DevelopmentCards.Add(DevelopmentCardType.Knight, 1);

            // Update turn state
            state.Turn.MustMoveRobber = false;
            state.Turn.HasPlayedDevelopmentCard = false;

            // Decrement largest army
            --state.Players[PlayerIndex].PlayedKnights;

            // Un-complete match
            state.Turn.TypeOfRound = TurnState.RoundType.Normal;

            // Reassign VPs to previous owner (or nobody if previously -1)
            for (int playerIdx = 0; playerIdx < state.Players.Length; playerIdx++)
            {
                state.Players[playerIdx].VictoryPoints.LargestArmyPoints = (byte)(playerIdx == History!.PrevLargestArmyHolder ? 2 : 0);
            }
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && turn.PendingTrade == null
                && !turn.MustDiscard
                && !turn.MustMoveRobber
                && !turn.HasPlayedDevelopmentCard;
        }

        public bool IsBoardValid(GameState state)
        {
            bool hasCard = state.Players[PlayerIndex].DevelopmentCards.Contains(DevelopmentCardType.Knight);

            // Check for dev card age
            bool cardAgeSufficient = state.Players[PlayerIndex].HasOldDevelopmentCard(DevelopmentCardType.Knight);

            return hasCard && cardAgeSufficient;
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
            KnightAction action = new(playerIdx);

            return action.IsValidFor(state) ? [action] : [];
        }
    }
}
