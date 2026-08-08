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
        public record RobberActionHistory(Tile? PrevRobber, ResourceCardType? StolenCard = null)
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
            Tile? prevRobber = state.Board.Robber;
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
                        History = new RobberActionHistory(prevRobber, drawnCard);
                    }
                    // Replay stored outcome
                    else
                    {
                        drawnCard = History!.StolenCard!.Value;
                        state.Players[TargetPlayerIndex.Value].ResourceCards.Remove(drawnCard, 1);
                    }

                    state.Players[PlayerIndex].ResourceCards.Add(drawnCard, 1);
                }
            }

            if (History == null)
            {
                History = new RobberActionHistory(prevRobber);
            }

            // Update turn state
            state.Turn.MustMoveRobber = false;
        }

        public override void Revert(GameState state)
        {
            // Ensure action was applied before
            if (!HasHistory()) throw new InvalidOperationException();

            // Return robber to previous tile
            state.Board.Robber = History!.PrevRobber;

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
                // Robber actions can happen before rolling when a knight is used before the roll
                && !turn.MustDiscard
                && turn.MustMoveRobber;
        }

        public bool IsBoardValid(GameState state)
        {
            Tile tile = state.Board.Map.Values.ElementAt(TargetTileIndex);

            bool isLandTile = tile.IsLandTile();
            bool alreadyHasRobber = state.Board.Robber == tile;

            if (!isLandTile || alreadyHasRobber) return false;

            // Target player specification
            bool hasValidTarget;

            if(TargetPlayerIndex.HasValue)
            {
                bool isTargetPlayerAdjacent = false;

                foreach (Intersection intersection in state.Board.Adjacency.GetIntersections(tile))
                {
                    if (intersection.Owner == TargetPlayerIndex && intersection.Building != Intersection.BuildingType.None)
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
                    if (intersection.Owner != PlayerIndex && intersection.Building != Intersection.BuildingType.None)
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
                || History.StolenCard.HasValue && TargetPlayerIndex.HasValue && TargetPlayerIndex != PlayerIndex && state.Players[TargetPlayerIndex.Value].ResourceCards.Get(History.StolenCard.Value) > 0;

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

            for(int tileIdx = 0; tileIdx < state.Board.Map.Length; tileIdx++)
            {
                for(sbyte targetPlayerIdx = -1; targetPlayerIdx < state.Players.Length; targetPlayerIdx++)
                {
                    RobberAction action = new(playerIdx, tileIdx, targetPlayerIdx < 0 ? null : targetPlayerIdx);

                    if(action.IsBoardValid(state))
                    {
                        actions.Add(action);
                    }
                }
            }

            return actions;
        }

        public List<Action> GetOutcomeVariants(GameState state, sbyte playerIdx)
        {
            if (HasHistory())
                throw new InvalidOperationException("The output randomness of this action was already resolved.");

            List<Action> variants = new List<Action>();

            if (!IsTurnValid(state.Turn, playerIdx)) return variants;

            // Output randomness: Which resource card was drawn from the target player
            // It is possible to not draw a card at all, hence the nullability
            ResourceCardType?[] stolenCardOptions = [null, .. CardSet<ResourceCardType>.Values];

            foreach (ResourceCardType? stolenCard in stolenCardOptions)
            {
                RobberAction actionCopy = new RobberAction(PlayerIndex, TargetTileIndex, TargetPlayerIndex);
                RobberActionHistory outcome = new RobberActionHistory(state.Board.Robber, stolenCard);
                actionCopy.History = outcome;

                if (actionCopy.IsBoardValid(state))
                {
                    variants.Add(actionCopy);
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
