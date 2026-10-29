using Common;
using Common.Actions;
using Agents.Inference;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Action = Common.Actions.Action;

namespace Agents
{
    public class GreedyAgent : Agent
    {
        protected static StateValueNet _stateValueNet;
        protected static StateVectorizer _stateVectorizer;

        public GreedyAgent(sbyte playerIndex)
            : base(playerIndex)
        {
            
        }

        static GreedyAgent()
        {
            _stateValueNet = new StateValueNet(StateValueNet.DefaultModelFile);
            _stateVectorizer = StateVectorizer.For(_stateValueNet);
        }

        public override Action Act(GameState state)
        {
            // Trades between players: answers to offers and, on turn, the pick of a partner
            if (TradeDecisions.Decide(state, PlayerIndex) is Action trade) return trade;

            List<Action> firstInitialSettlementActions = FirstInitialSettlementAction.GetActionsForState(state, PlayerIndex);
            if (firstInitialSettlementActions.Count > 0)
            {
                return GreedySelectAction(firstInitialSettlementActions, state);
            }

            List<Action> secondInitialSettlementActions = SecondInitialSettlementAction.GetActionsForState(state, PlayerIndex);
            if (secondInitialSettlementActions.Count > 0)
            {
                return GreedySelectAction(secondInitialSettlementActions, state);
            }

            if (DiscardAction.IsTurnValid(state.Turn, PlayerIndex))
            {
                return GreedySelectAction(DiscardAction.GetActionsForState(state, PlayerIndex), state);
            }

            List<Action> actions = [
                .. EndTurnAction.GetActionsForState(state, PlayerIndex),

                .. RollAction.GetActionsForState(state, PlayerIndex),
                .. RobberAction.GetActionsForState(state, PlayerIndex),

                .. FirstInitialRoadAction.GetActionsForState(state, PlayerIndex),
                .. SecondInitialRoadAction.GetActionsForState(state, PlayerIndex),

                .. RoadAction.GetActionsForState(state, PlayerIndex),
                .. SettlementAction.GetActionsForState(state, PlayerIndex),
                .. CityAction.GetActionsForState(state, PlayerIndex),
                .. BuyDevelopmentCardAction.GetActionsForState(state, PlayerIndex),

                .. KnightAction.GetActionsForState(state, PlayerIndex),
                .. MonopolyAction.GetActionsForState(state, PlayerIndex),
                .. RoadBuildingAction.GetActionsForState(state, PlayerIndex),
                .. YearOfPlentyAction.GetActionsForState(state, PlayerIndex),

                .. FourToOneTradeAction.GetActionsForState(state, PlayerIndex),
                .. ThreeToOneTradeAction.GetActionsForState(state, PlayerIndex),
                .. TwoToOneTradeAction.GetActionsForState(state, PlayerIndex)
            ];

            if (actions.Count == 0) throw new InvalidOperationException();

            return GreedySelectAction(actions, state);
        }

        protected Action GreedySelectAction(List<Action> actions, GameState state)
        {
            return actions
                .OrderByDescending(a => PredictValue(a, state))
                .First();
        }

        protected float PredictValue(Action action, GameState currentState)
        {
            // If action is an output randomness action, average the value over all outcomes
            if (action is IOutputRandomnessAction randomAction)
            {
                List<OutcomeVariant> outcomeVariants = randomAction.GetOutcomeVariants(currentState, PlayerIndex);

                // Expected score over the outcome distribution
                float expectedScore = 0;

                foreach ((Action actionVariant, double probability) in outcomeVariants)
                {
                    GameState variantStateCopy = new GameState(currentState);
                    actionVariant.Apply(variantStateCopy);
                    expectedScore += (float)probability * ScoreState(variantStateCopy);
                }

                return expectedScore;
            }
            
            // Otherwise evaluate the result of the individual action directly
            GameState stateCopy = new GameState(currentState);
            action.Apply(stateCopy);
            return ScoreState(stateCopy);
        }

        protected float ScoreState(GameState state)
        {
            // Ended matches have a known result (the turn player is the winner), no need to ask the net
            if (state.HasEnded)
                return state.Turn.PlayerIndex == PlayerIndex ? 1 : 0;

            return ValuationToScore(StateValueFunc(state));
        }

        protected float ValuationToScore(float[] valuation)
        {
            return valuation[PlayerIndex] / valuation.Sum();
        }

        public static float[] StateValueFunc(GameState state)
        {
            var inputTensor = _stateVectorizer.Vectorize(state);
            return _stateValueNet.Run(inputTensor);
        }
    }
}
