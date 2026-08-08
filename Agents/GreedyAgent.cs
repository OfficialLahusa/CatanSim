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
            _stateValueNet = new StateValueNet("state_value_net-0mw2c1bh.onnx");
            _stateVectorizer = new StateVectorizer();
        }

        public override Action Act(GameState state, uint playedActionsCount)
        {
            List<Action> firstInitialSettlementActions = FirstInitialSettlementAction.GetActionsForState(state, PlayerIndex);
            if (firstInitialSettlementActions.Count > 0)
            {
                return GreedySelectAction(firstInitialSettlementActions, state, playedActionsCount);
            }

            List<Action> secondInitialSettlementActions = SecondInitialSettlementAction.GetActionsForState(state, PlayerIndex);
            if (secondInitialSettlementActions.Count > 0)
            {
                return GreedySelectAction(secondInitialSettlementActions, state, playedActionsCount);
            }

            if (DiscardAction.IsTurnValid(state.Turn, PlayerIndex))
            {
                return GreedySelectAction(DiscardAction.GetActionsForState(state, PlayerIndex), state, playedActionsCount);
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

            return GreedySelectAction(actions, state, playedActionsCount);
        }

        protected Action GreedySelectAction(List<Action> actions, GameState state, uint playedActionsCount)
        {
            return actions
                .OrderByDescending(a => PredictValue(a, state, playedActionsCount))
                .First();
        }

        protected float PredictValue(Action action, GameState currentState, uint playedActionsCount)
        {
            // If action is an output randomness action, average the value over all outcomes
            if (action is IOutputRandomnessAction randomAction)
            {
                List<Action> outcomeVariants = randomAction.GetOutcomeVariants(currentState, PlayerIndex);

                float scoreSum = 0;

                foreach (Action actionVariant in outcomeVariants)
                {
                    GameState variantStateCopy = new GameState(currentState);
                    actionVariant.Apply(variantStateCopy);
                    scoreSum += ValuationToScore(StateValueFunc(variantStateCopy, playedActionsCount + 1));
                }

                return scoreSum / outcomeVariants.Count;
            }
            
            // Otherwise evaluate the result of the individual action directly
            GameState stateCopy = new GameState(currentState);
            action.Apply(stateCopy);
            return ValuationToScore(StateValueFunc(stateCopy, playedActionsCount + 1));
        }

        protected float ValuationToScore(float[] valuation)
        {
            return valuation[PlayerIndex] / valuation.Sum();
        }

        public static float[] StateValueFunc(GameState state, uint playedActionsCount)
        {
            var inputTensor = _stateVectorizer.Vectorize(state, playedActionsCount);
            return _stateValueNet.Run(inputTensor);
        }
    }
}
