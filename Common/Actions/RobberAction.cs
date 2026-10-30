using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Common.Actions.RollAction;

namespace Common.Actions
{
    public class RobberAction : Action, IReplayAction, IActionProvider, IOutputRandomnessAction
    {
        // The previous robber position is stored as a tile index rather than a Tile reference, so that
        // Revert keeps pointing at the tile inside the board's map after copying or deserialization
        public record RobberActionHistory(int? PrevRobberTileIndex, ResourceCardType? StolenCard = null)
        {
            /// <summary>
            /// Parameterless constructor for deserialization
            /// </summary>
            private RobberActionHistory() : this(null, null) { }
        }

        public RobberActionHistory? History { get; private set; }

        public int TargetTileIndex { get; init; }
        public sbyte? TargetPlayerIndex { get; init; }

        public RobberAction(sbyte playerIdx, int targetTileIdx, sbyte? targetPlayerIdx)
            : base(playerIdx)
        {
            TargetTileIndex = targetTileIdx;
            TargetPlayerIndex = targetPlayerIdx;
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private RobberAction()
            : base(-1)
        { }

        public override void Apply(GameState state)
        {
            base.Apply(state);

            Tile tile = state.Board.Map.Values.ElementAt(TargetTileIndex);
            int? prevRobberTileIndex = state.Board.GetRobberTileIndex();
            state.Board.Robber = tile;

            // Draw from adjacent player
            if (TargetPlayerIndex.HasValue)
            {
                if (state.Players[TargetPlayerIndex.Value].ResourceCards.Count() > 0)
                {
                    ResourceCardType drawnCard;

                    // Draw random card
                    if (!HasHistory())
                    {
                        drawnCard = state.Players[TargetPlayerIndex.Value].ResourceCards.Draw(true);

                        // Store outcome
                        History = new RobberActionHistory(prevRobberTileIndex, drawnCard);
                    }
                    // Replay stored outcome
                    else
                    {
                        drawnCard = History!.StolenCard!.Value;
                        if (drawnCard == ResourceCardType.Unknown) StealUnseen(state.Players[TargetPlayerIndex.Value].ResourceCards);
                        else state.Players[TargetPlayerIndex.Value].ResourceCards.Remove(drawnCard, 1);
                    }

                    state.Players[PlayerIndex].ResourceCards.Add(drawnCard, 1);
                }
            }

            if (History == null)
            {
                History = new RobberActionHistory(prevRobberTileIndex);
            }

            // Update turn state
            state.Turn.MustMoveRobber = false;
        }

        public override Action RedactedFor(int observer)
        {
            // Only the thief and the victim see the stolen card
            RobberAction copy = (RobberAction)Clone();
            if (History?.StolenCard != null && observer != PlayerIndex && observer != TargetPlayerIndex)
                copy.History = History with { StolenCard = ResourceCardType.Unknown };
            return copy;
        }

        /// <summary>
        /// A redacted steal replayed on a view (see RedactedFor): as the observer sees it (see BeliefTracker), every type the
        /// victim holds may have lost the card, so one card of each becomes unknown, less the stolen one
        /// </summary>
        private static void StealUnseen(CardSet<ResourceCardType> victim)
        {
            uint target = victim.Count() - 1;
            foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes)
                if (victim.Get(type) > 0) victim.Remove(type, 1);
            uint count = victim.Count();
            if (count > target) victim.Remove(ResourceCardType.Unknown, count - target);
            else victim.Add(ResourceCardType.Unknown, target - count);
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Return robber to previous tile
            state.Board.Robber = History!.PrevRobberTileIndex.HasValue
                ? state.Board.Map.Values[History.PrevRobberTileIndex.Value]
                : null;

            // Return card to adjacent player
            if (TargetPlayerIndex.HasValue && History.StolenCard.HasValue)
            {
                state.Players[TargetPlayerIndex.Value].ResourceCards.Add(History.StolenCard.Value, 1);
                state.Players[PlayerIndex].ResourceCards.Remove(History.StolenCard.Value, 1);
            }

            // Update turn state
            state.Turn.MustMoveRobber = true;
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
                // Robber actions can happen before rolling when a knight is used before the roll
                && !turn.MustDiscard
                && turn.MustMoveRobber;
        }

        public bool IsBoardValid(GameState state)
        {
            Tile tile = state.Board.Map.Values.ElementAt(TargetTileIndex);

            bool isLandTile = tile.IsLandTile();
            bool alreadyHasRobber = state.Board.HasRobber(tile);

            if (!isLandTile || alreadyHasRobber) return false;

            // Target player specification
            bool hasValidTarget;

            if(TargetPlayerIndex.HasValue)
            {
                bool isTargetPlayerAdjacent = false;

                foreach (Intersection intersection in state.Board.Adjacency.GetIntersections(tile))
                {
                    if (state.Board.GetBuildingOwner(intersection.Index) == TargetPlayerIndex && state.Board.GetBuilding(intersection.Index) != Intersection.BuildingType.None)
                    {
                        isTargetPlayerAdjacent = true;
                        break;
                    }
                }

                hasValidTarget = TargetPlayerIndex.Value != PlayerIndex && isTargetPlayerAdjacent;
            }
            else
            {
                bool hasOtherAdjacentPlayer = false;

                foreach (Intersection intersection in state.Board.Adjacency.GetIntersections(tile))
                {
                    if (state.Board.GetBuildingOwner(intersection.Index) != PlayerIndex && state.Board.GetBuilding(intersection.Index) != Intersection.BuildingType.None)
                    {
                        hasOtherAdjacentPlayer = true;
                        break;
                    }
                }

                hasValidTarget = !hasOtherAdjacentPlayer;
            }

            // If stored history exists, ensure it is valid
            bool historyValid = !HasHistory()
                || !History!.StolenCard.HasValue && (!TargetPlayerIndex.HasValue || TargetPlayerIndex.Value == PlayerIndex || state.Players[TargetPlayerIndex.Value].ResourceCards.Count() == 0)
                // Contains, so a redacted steal (an Unknown card) replays on a view (see CardSet)
                || History.StolenCard.HasValue && TargetPlayerIndex.HasValue && TargetPlayerIndex != PlayerIndex && state.Players[TargetPlayerIndex.Value].ResourceCards.Contains(History.StolenCard.Value);

            return hasValidTarget && historyValid;
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

            if(!IsTurnValid(state.Turn, playerIdx)) return actions;

            // The actions IsBoardValid accepts for every tile and target, without creating the rejected ones
            for(int tileIdx = 0; tileIdx < state.Board.Map.Length; tileIdx++)
            {
                Tile tile = state.Board.Map.Values[tileIdx];
                if (!tile.IsLandTile() || state.Board.HasRobber(tile)) continue;

                // Bit per other owner of a building on the tile (a building without owner counts as someone else)
                UInt128 corners = state.Board.Adjacency.GetCornerMask(tile) & state.Board.BuildingMask;
                int otherOwners = 0;
                for (int ownerIdx = 0; ownerIdx < state.Players.Length && corners != UInt128.Zero; ownerIdx++)
                {
                    UInt128 owned = corners & state.Board.GetBuildingMask(ownerIdx);
                    if (owned != UInt128.Zero && ownerIdx != playerIdx) otherOwners |= 1 << ownerIdx;
                    corners &= ~owned;
                }
                if (corners != UInt128.Zero) otherOwners |= 1 << 31;

                // Without victims the robber steals from nobody, otherwise from one of them
                if (otherOwners == 0)
                    actions.Add(new RobberAction(playerIdx, tileIdx, null));

                for(sbyte targetPlayerIdx = 0; targetPlayerIdx < state.Players.Length; targetPlayerIdx++)
                {
                    if ((otherOwners & (1 << targetPlayerIdx)) != 0)
                        actions.Add(new RobberAction(playerIdx, tileIdx, targetPlayerIdx));
                }
            }

            return actions;
        }

        public bool HasSameOutcome(Action other)
        {
            return other is RobberAction otherRobber && HasHistory() && otherRobber.HasHistory()
                && History!.StolenCard == otherRobber.History!.StolenCard;
        }

        public List<OutcomeVariant> GetOutcomeVariants(GameState state, sbyte playerIdx)
        {
            if (HasHistory())
                throw new InvalidOperationException("The output randomness of this action was already resolved.");

            List<OutcomeVariant> variants = new();

            if (!IsTurnValid(state.Turn, playerIdx)) return variants;

            // Output randomness: Which resource card was drawn from the target player, proportional to their hand
            // It is possible to not draw a card at all, hence the nullability
            ResourceCardType?[] stolenCardOptions = [null, .. CardTypeExtensions.ResourceTypes];
            CardSet<ResourceCardType>? targetHand = TargetPlayerIndex.HasValue ? state.Players[TargetPlayerIndex.Value].ResourceCards : null;
            uint targetHandSize = targetHand?.Count() ?? 0;

            foreach (ResourceCardType? stolenCard in stolenCardOptions)
            {
                RobberAction actionCopy = new RobberAction(PlayerIndex, TargetTileIndex, TargetPlayerIndex);
                RobberActionHistory outcome = new RobberActionHistory(state.Board.GetRobberTileIndex(), stolenCard);
                actionCopy.History = outcome;

                if (actionCopy.IsBoardValid(state))
                {
                    // IsBoardValid only accepts "no card" when there is nothing to steal, and a card only if the target holds it
                    double probability = stolenCard.HasValue ? (double)targetHand!.Get(stolenCard.Value) / targetHandSize : 1.0;
                    variants.Add(new OutcomeVariant(actionCopy, probability));
                }
            }

            return variants;
        }

        public override string ToString()
        {
            return base.ToString() + $", T{TargetTileIndex}" + (TargetPlayerIndex.HasValue ? $", D{TargetPlayerIndex}" : "");
        }
    }
}
