using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class TwoToOneTradeAction : Action, IActionProvider
    {
        public ResourceCardType InputType { get; init; }
        public ResourceCardType OutputType { get; init; }

        public TwoToOneTradeAction(sbyte playerIdx, ResourceCardType inputType, ResourceCardType outputType)
            : base(playerIdx)
        {
            InputType = inputType;
            OutputType = outputType;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private TwoToOneTradeAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Road Building roads that were not placed yet are given up
            ForfeitFreeRoads(state);

            // Remove input from hand
            state.Players[PlayerIndex].ResourceCards.Remove(InputType, 2);

            // Remove output from bank
            state.ResourceBank.Remove(OutputType, 1);

            // Add input to bank
            state.ResourceBank.Add(InputType, 2);

            // Add output to hand
            state.Players[PlayerIndex].ResourceCards.Add(OutputType, 1);
        }

        public override void Revert(GameState state)
        {
            // Return input to hand
            state.Players[PlayerIndex].ResourceCards.Add(InputType, 2);

            // Return output to bank
            state.ResourceBank.Add(OutputType, 1);

            // Remove input from bank
            state.ResourceBank.Remove(InputType, 2);

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
            bool playerHasInput = state.Players[PlayerIndex].ResourceCards.Contains(InputType, 2);
            bool bankHasOutput = state.ResourceBank.Contains(OutputType, 1);

            // Check port privilege
            bool hasPort = state.Players[PlayerIndex].PortPrivileges.HasFlag(RequiredPort(InputType));

            return cardTypesDiffer && playerHasInput && bankHasOutput && hasPort;
        }

        private static PortPrivileges RequiredPort(ResourceCardType inputType) => inputType switch
        {
            ResourceCardType.Lumber => PortPrivileges.LumberTwoToOne,
            ResourceCardType.Brick  => PortPrivileges.BrickTwoToOne,
            ResourceCardType.Wool   => PortPrivileges.WoolTwoToOne,
            ResourceCardType.Grain  => PortPrivileges.GrainTwoToOne,
            ResourceCardType.Ore    => PortPrivileges.OreTwoToOne,
            _ => throw new InvalidOperationException(),
        };

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            if (!IsTurnValid(state.Turn, playerIdx)) return actions;

            // The trades IsBoardValid accepts, without creating the rejected ones
            CardSet<ResourceCardType> hand = state.Players[playerIdx].ResourceCards;
            foreach (ResourceCardType inputType in CardTypeExtensions.ResourceTypes)
            {
                if (!hand.Contains(inputType, 2) || !state.Players[playerIdx].PortPrivileges.HasFlag(RequiredPort(inputType))) continue;

                foreach (ResourceCardType outputType in CardTypeExtensions.ResourceTypes)
                {
                    if (outputType != inputType && state.ResourceBank.Contains(outputType, 1))
                        actions.Add(new TwoToOneTradeAction(playerIdx, inputType, outputType));
                }
            }

            return actions;
        }

        public override string ToString()
        {
            return base.ToString() + $", {InputType.GetAbbreviation()}{InputType.GetAbbreviation()} -> {OutputType.GetAbbreviation()}";
        }
    }
}
