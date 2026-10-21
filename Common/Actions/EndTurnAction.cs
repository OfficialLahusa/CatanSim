using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class EndTurnAction : Action, IReplayAction, IActionProvider
    {
        public record EndTurnActionHistory(uint PrevFreeRoads, CardSet<DevelopmentCardType> PrevNewDevelopmentCards, bool PrevHasPlayedDevelopmentCard)
        {
            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private EndTurnActionHistory() : this(0, null!, false) { }
        }

        public EndTurnActionHistory? History { get; private set; }

        public EndTurnAction(sbyte playerIdx)
            : base(playerIdx)
        { }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private EndTurnAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            History = new EndTurnActionHistory(
                state.Players[PlayerIndex].BuildingStock.FreeRoads,
                // Copy, so that the history never shares a live card set with a player state
                new CardSet<DevelopmentCardType>(state.Players[PlayerIndex].NewDevelopmentCards),
                state.Turn.HasPlayedDevelopmentCard
            );

            // Return remaining free stock from road building dev card to main stock
            BuildingStock stock = state.Players[PlayerIndex].BuildingStock;
            if(stock.FreeRoads > 0)
            {
                stock.RemainingRoads += stock.FreeRoads;
                stock.FreeRoads = 0;
            }

            // Allow newly acquired dev cards to be played
            state.Players[PlayerIndex].NewDevelopmentCards = new();

            // Update turn state
            state.Turn.PlayerIndex = ++state.Turn.PlayerIndex % state.Players.Length;
            if(state.Turn.PlayerIndex == 0) state.Turn.RoundCounter++;
            state.Turn.MustRoll = true;
            // TODO: Remove?
            state.Turn.MustMoveRobber = false;
            Array.Fill(state.Turn.AwaitedPlayerDiscards, false);
            state.Turn.HasPlayedDevelopmentCard = false;

            // Players might have had enough VPs to win before their own turn
            state.CheckForCompletion();
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Return previous free roadbuilding stock
            BuildingStock stock = state.Players[PlayerIndex].BuildingStock;
            if (History!.PrevFreeRoads > 0)
            {
                stock.RemainingRoads -= History.PrevFreeRoads;
                stock.FreeRoads = History.PrevFreeRoads;
            }

            // Block previous new dev cards
            state.Players[PlayerIndex].NewDevelopmentCards = History.PrevNewDevelopmentCards;

            // Update turn state
            if (state.Turn.PlayerIndex == 0) state.Turn.RoundCounter--;
            state.Turn.PlayerIndex--;
            if (state.Turn.PlayerIndex < 0) state.Turn.PlayerIndex = state.Players.Length - 1;
            state.Turn.MustRoll = false;
            state.Turn.HasPlayedDevelopmentCard = History.PrevHasPlayedDevelopmentCard;

            // Un-complete match
            state.Turn.TypeOfRound = TurnState.RoundType.Normal;
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && !turn.MustRoll
                && !turn.MustDiscard
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
            EndTurnAction action = new EndTurnAction(playerIdx);

            return action.IsValidFor(state) ? [action] : [];
        }
    }
}
