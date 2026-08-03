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

            int requiredDiscards = (int)(playerCards.Count() / 2);

            // Generate all discardable combinations
            int[] handCounts = CardSet<ResourceCardType>.Values
                .Select(resourceType => (int)playerCards.Get(resourceType))
                .ToArray();
            List<int[]> possibleDiscards = GetDiscardCombinations(handCounts, requiredDiscards).ToList();

            // Create actions for each combination
            foreach (int[] discard in possibleDiscards)
            {
                // Add to new CardSet
                CardSet<ResourceCardType> cardSubset = new();
                for (int i = 0; i < CardSet<ResourceCardType>.Values.Count; i++)
                {
                    ResourceCardType card = CardSet<ResourceCardType>.Values[i];
                    for (int j = 0; j < discard[i]; j++)
                    {
                        cardSubset.Add(card, 1);
                    }
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


        private static IEnumerable<int[]> GetDiscardCombinations(int[] handCounts, int requiredDiscards)
        {
            int n = handCounts.Length;

            if (requiredDiscards < 0 || requiredDiscards > handCounts.Sum())
                throw new InvalidOperationException("Invalid number of required discards.");

            // possibleDiscardsRemaining[i] = max cards that could possibly be discarded from types [i .. n-1] combined
            // => Used for recursion tree pruning, to avoid exploring impossible branches
            var possibleDiscardsRemaining = new int[n + 1];
            for (int i = n - 1; i >= 0; i--)
                possibleDiscardsRemaining[i] = possibleDiscardsRemaining[i + 1] + handCounts[i];

            // Shared buffer for current discard combination being built up in recursion
            var current = new int[n];

            foreach (var combo in Recurse(0, requiredDiscards, handCounts, possibleDiscardsRemaining, current))
                // Return a shallow copy of the current combination, since the current array is reused in recursion
                yield return (int[])combo.Clone();
        }

        private static IEnumerable<int[]> Recurse(int typeIndex, int remaining, int[] handCounts, int[] possibleDiscardsRemaining, int[] current)
        {
            // Terminate if we have reached the last card type
            if (typeIndex == handCounts.Length)
            {
                // If we have discarded enough cards, yield the current combination
                if (remaining == 0)
                    yield return current;
                // otherwise terminate this branch
                yield break;
            }

            // Interval of how many cards of the current type can be discarded
            int lower = Math.Max(0, remaining - possibleDiscardsRemaining[typeIndex + 1]);
            int upper = Math.Min(handCounts[typeIndex], remaining);

            // Branch for each possible amount of cards of the current type to discard
            for (int amountOfCurrentTypeToDiscard = lower; amountOfCurrentTypeToDiscard <= upper; amountOfCurrentTypeToDiscard++)
            {
                current[typeIndex] = amountOfCurrentTypeToDiscard;
                foreach (var combo in Recurse(typeIndex + 1, remaining - amountOfCurrentTypeToDiscard, handCounts, possibleDiscardsRemaining, current))
                    yield return combo;
            }

            // Backtrack if current branch is done to avoid affecting other branches
            current[typeIndex] = 0; 
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
