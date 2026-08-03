using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;

namespace Common.Actions
{
    public class DiscardAction : Action, IActionProvider
    {
        // Cards that are discarded by the player
        public CardSet<ResourceCardType> SelectedCards;

        public DiscardAction(sbyte playerIdx, CardSet<ResourceCardType> selectedCards)
            : base(playerIdx)
        {
            SelectedCards = selectedCards;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private DiscardAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            // Remove selected cards from hand
            CardSet<ResourceCardType> cardSet = state.Players[PlayerIndex].ResourceCards;
            cardSet.Remove(SelectedCards);

            // Return discarded cards to bank
            state.ResourceBank.Add(SelectedCards);

            // Mark discard as completed
            state.Turn.AwaitedPlayerDiscards[PlayerIndex] = false;
        }

        public override void Revert(GameState state)
        {
            // Return selected cards to hand
            CardSet<ResourceCardType> cardSet = state.Players[PlayerIndex].ResourceCards;
            cardSet.Add(SelectedCards);

            // Remove discarded cards from bank
            state.ResourceBank.Remove(SelectedCards);

            // Mark discard as awaited
            state.Turn.AwaitedPlayerDiscards[PlayerIndex] = true;
        }

        public override bool IsValidFor(GameState state)
        {
            return IsTurnValid(state.Turn, PlayerIndex) && IsBoardValid(state);
        }

        public static bool IsTurnValid(TurnState turn, int playerIdx)
        {
            // Note: Player index does NOT need to match turn player index
            return turn.TypeOfRound == TurnState.RoundType.Normal
                && !turn.MustRoll
                && turn.MustDiscard
                && turn.AwaitedPlayerDiscards[playerIdx];
        }

        public bool IsBoardValid(GameState state)
        {
            CardSet<ResourceCardType> playerCards = state.Players[PlayerIndex].ResourceCards;

            int excessCards = (int)playerCards.Count() - state.Settings.RobberCardLimit;

            // Terminate early if discarding isn't required
            if (excessCards <= 0) return false;

            bool validAmount = SelectedCards.Count() == playerCards.Count() / 2;
            bool validSubset = playerCards.Contains(SelectedCards);

            return validAmount && validSubset;
        }

        public static List<Action> GetActionsForState(GameState state, sbyte playerIdx)
        {
            List<Action> actions = [];

            if(!IsTurnValid(state.Turn, playerIdx)) return actions;

            CardSet<ResourceCardType> playerCards = state.Players[playerIdx].ResourceCards;
            int excessCards = (int)playerCards.Count() - state.Settings.RobberCardLimit;

            // Skip player, if no discard is needed
            if (excessCards <= 0) return actions;

            // Generate all discardable subsets
            // TODO: Optimize this, since the order of the cards doesn't matter, and we can just generate combinations instead of permutations
            ReadOnlySpan<ResourceCardType> heldResources = CardSet<ResourceCardType>.Values
                .SelectMany(
                    resourceType => Enumerable.Repeat(resourceType, (int)playerCards.Get(resourceType))
                )
                .ToArray();

            List<ResourceCardType[]> subsets = Utils.GetSubsets(heldResources, heldResources.Length / 2);

            // Create actions for each subset
            foreach (ResourceCardType[] subset in subsets)
            {
                // Add to new CardSet
                CardSet<ResourceCardType> cardSubset = new();
                foreach (var card in subset)
                {
                    cardSubset.Add(card, 1);
                }

                // Validate action
                DiscardAction action = new DiscardAction(playerIdx, cardSubset);
                if (action.IsBoardValid(state))
                {
                    actions.Add(action);
                }
            }

            return actions;
        }

        public static DiscardAction GetRandomDiscard(GameState state, sbyte playerIdx)
        {
            // Get list of types of held resource cards
            List<ResourceCardType> heldResources = CardSet<ResourceCardType>.Values
                .SelectMany(
                    resourceType => Enumerable.Repeat(resourceType, (int)state.Players[playerIdx].ResourceCards.Get(resourceType))
                )
                .ToList();

            // Randomize card order
            Utils.Shuffle(heldResources);

            // Remove the cards that are kept
            heldResources.RemoveRange(0, heldResources.Count - heldResources.Count / 2);

            // Add to new CardSet
            CardSet<ResourceCardType> discardedSet = new();
            foreach (var card in heldResources)
            {
                discardedSet.Add(card, 1);
            }

            return new DiscardAction(playerIdx, discardedSet);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder(base.ToString());
            sb.Append(", ");


            foreach(ResourceCardType resourceType in CardSet<ResourceCardType>.Values)
            {
                for (int i = 0; i < SelectedCards.Get(resourceType); i++)
                {
                    sb.Append(resourceType.GetAbbreviation());
                }
            }

            return sb.ToString();
        }
    }
}
