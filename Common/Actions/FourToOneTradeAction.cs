using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class FourToOneTradeAction : Action, IActionProvider
    {
        public ResourceCardType InputType { get; init; }
        public ResourceCardType OutputType { get; init; }

        public FourToOneTradeAction(sbyte playerIdx, ResourceCardType inputType, ResourceCardType outputType)
            : base(playerIdx)
        {
            InputType = inputType;
            OutputType = outputType;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private FourToOneTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Road Building roads that were not placed yet are given up
            ForfeitFreeRoads(state);

            // Remove input from hand
            state.Players[PlayerIndex].ResourceCards.Remove(InputType, 4);

            // Remove output from bank
            state.ResourceBank.Remove(OutputType, 1);

            // Add input to bank
            state.ResourceBank.Add(InputType, 4);

            // Add output to hand
            state.Players[PlayerIndex].ResourceCards.Add(OutputType, 1);
        }

        public override void Revert(GameState state)
        {
            // Return input to hand
            state.Players[PlayerIndex].ResourceCards.Add(InputType, 4);

            // Return output to bank
            state.ResourceBank.Add(OutputType, 1);

            // Remove input from bank
            state.ResourceBank.Remove(InputType, 4);

            // Remove output from hand
            state.Players[PlayerIndex].ResourceCards.Remove(OutputType, 1);

            ReturnForfeitedFreeRoads(state);
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            return turn.PlayerIndex == playerIdx
                && turn.TypeOfRound == TurnState.RoundType.Normal
                && !turn.MustRoll
                && !turn.MustDiscard
                && !turn.MustMoveRobber;
        }

        public bool IsBoardValid(GameState state)
        {
            bool cardTypesDiffer = InputType != OutputType;
            bool playerHasInput = state.Players[PlayerIndex].ResourceCards.Contains(InputType, 4);
            bool bankHasOutput = state.ResourceBank.Contains(OutputType, 1);

            return cardTypesDiffer && playerHasInput && bankHasOutput;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            if (!IsTurnValid(state.Turn, playerIdx)) return actions;

            foreach (var inputType in CardSet<ResourceCardType>.Values)
            {
                if (inputType == ResourceCardType.Unknown) continue;

                foreach (var outputType in CardSet<ResourceCardType>.Values)
                {
                    if (outputType == ResourceCardType.Unknown) continue;

                    FourToOneTradeAction action = new(playerIdx, inputType, outputType);

                    if (action.IsBoardValid(state))
                    {
                        actions.Add(action);
                    }
                }
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", {InputType.GetAbbreviation()}{InputType.GetAbbreviation()}{InputType.GetAbbreviation()}{InputType.GetAbbreviation()} -> {OutputType.GetAbbreviation()}";
        }
    }
}
