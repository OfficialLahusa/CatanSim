using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Agents
{
    /// <summary>
    /// Fast rule-based agent, meant for playouts: builds whenever it can (best settlement spot, best city, roads only toward
    /// a new settlement spot), plays development cards, buys them with what is left, trades only to complete its next build
    /// (cheapest port first) and otherwise ends the turn. The robber goes where it hurts the opponents most.
    /// With the exploration rate it plays a random action instead (as RandomAgent), which keeps playouts varied.
    /// </summary>
    public class BuilderAgent : Agent
    {
        private static readonly ResourceCardType[] Resources =
            [ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain, ResourceCardType.Ore];

        // Costs indexed like Resources
        private static readonly uint[] SettlementCost = [1, 1, 1, 1, 0];
        private static readonly uint[] CityCost = [0, 0, 0, 2, 3];
        private static readonly uint[] RoadCost = [1, 1, 0, 0, 0];
        private static readonly uint[] DevelopmentCardCost = [0, 0, 1, 1, 1];

        private readonly double _explorationRate;
        private readonly RandomAgent _randomAgent;

        public BuilderAgent(sbyte playerIndex, double explorationRate = 0)
            : base(playerIndex)
        {
            _explorationRate = explorationRate;
            _randomAgent = new RandomAgent(playerIndex);
        }

        public override Action Act(GameState state)
        {
            // Trades between players: answers to offers and, on turn, the pick of a partner
            if (TradeDecisions.Decide(state, PlayerIndex) is Action trade) return trade;

            sbyte p = PlayerIndex;

            if (DiscardAction.IsTurnValid(state.Turn, p))
                return DiscardAction.GetRandomDiscard(state, p);

            if (_explorationRate > 0 && Utils.Random.NextDouble() < _explorationRate)
                return _randomAgent.Act(state);

            // Setup placements are only asked for outside of normal rounds
            if (state.Turn.TypeOfRound != TurnState.RoundType.Normal)
                return SetupAction(state, p) ?? _randomAgent.Act(state);

            return RobberMove(state, p)
                ?? FreeRoad(state, p)
                ?? BeforeRoll(state, p)
                ?? Build(state, p)
                ?? PlayDevelopmentCard(state, p)
                ?? BuyDevelopmentCard(state, p)
                ?? Trade(state, p)
                ?? First(EndTurnAction.GetActionsForState(state, p))
                ?? _randomAgent.Act(state);
        }

        // -- Turn phases (each returns null if it doesn't apply) --

        private static Action? SetupAction(GameState state, sbyte p)
        {
            Action? settlement = BestByIntersection(FirstInitialSettlementAction.GetActionsForState(state, p), state)
                ?? BestByIntersection(SecondInitialSettlementAction.GetActionsForState(state, p), state);
            if (settlement != null) return settlement;

            List<Action> roads = FirstInitialRoadAction.GetActionsForState(state, p);
            if (roads.Count == 0) roads = SecondInitialRoadAction.GetActionsForState(state, p);
            return roads.Count > 0 ? RoadTowardBuildingSite(roads, state) : null;
        }

        /// <summary>
        /// The robber goes on the tile that blocks the most opponent production and the least own production,
        /// stealing from the victim with the most cards
        /// </summary>
        private static Action? RobberMove(GameState state, sbyte p)
        {
            List<Action> moves = RobberAction.GetActionsForState(state, p);
            if (moves.Count == 0) return null;

            Action best = moves[0];
            float bestScore = float.NegativeInfinity;
            foreach (Action move in moves)
            {
                RobberAction robber = (RobberAction)move;
                Tile tile = state.Board.Map.Values[robber.TargetTileIndex];
                float score = 0;

                foreach (int intersectionIdx in BitMask.Of(state.Board.Adjacency.GetCornerMask(tile) & state.Board.BuildingMask))
                {
                    float blocked = tile.YieldPoints * (state.Board.GetBuilding(intersectionIdx) == Intersection.BuildingType.City ? 2 : 1);
                    score += state.Board.GetBuildingOwner(intersectionIdx) == p ? -2 * blocked : blocked;
                }

                if (robber.TargetPlayerIndex is sbyte victim)
                    score += 0.1f * state.Players[victim].ResourceCards.Count();

                if (score > bestScore)
                {
                    best = move;
                    bestScore = score;
                }
            }

            return best;
        }

        private static Action? FreeRoad(GameState state, sbyte p)
        {
            if (!RoadAction.IsPlacingFreeRoads(state, p)) return null;

            List<Action> roads = RoadAction.GetActionsForState(state, p);
            return roads.Count > 0 ? RoadTowardBuildingSite(roads, state) : null;
        }

        private static Action? BeforeRoll(GameState state, sbyte p)
        {
            if (!state.Turn.MustRoll) return null;

            // Chase the robber off the own production before rolling
            if (RobberBlocksOwnProduction(state, p))
            {
                Action? knight = First(KnightAction.GetActionsForState(state, p));
                if (knight != null) return knight;
            }

            return First(RollAction.GetActionsForState(state, p));
        }

        private static Action? Build(GameState state, sbyte p)
        {
            Action? settlement = BestByIntersection(SettlementAction.GetActionsForState(state, p), state);
            if (settlement != null) return settlement;

            Action? city = BestByIntersection(CityAction.GetActionsForState(state, p), state);
            if (city != null) return city;

            // Roads only while no settlement spot is reachable, otherwise the cards are saved for the settlement
            if (NeedsRoad(state, p))
            {
                List<Action> roads = RoadAction.GetActionsForState(state, p);
                if (roads.Count > 0) return RoadTowardBuildingSite(roads, state);
            }

            return null;
        }

        private static Action? PlayDevelopmentCard(GameState state, sbyte p)
        {
            Action? knight = First(KnightAction.GetActionsForState(state, p));
            if (knight != null) return knight;

            if (NeedsRoad(state, p))
            {
                Action? roadBuilding = First(RoadBuildingAction.GetActionsForState(state, p));
                if (roadBuilding != null) return roadBuilding;
            }

            // Year of Plenty: the two resources that bring the next build closest
            List<Action> plenty = YearOfPlentyAction.GetActionsForState(state, p);
            if (plenty.Count > 0)
            {
                uint[] cost = NextBuildCost(state, p);
                return plenty.MinBy(action =>
                {
                    YearOfPlentyAction choice = (YearOfPlentyAction)action;
                    uint[] hand = Hand(state, p);
                    hand[ResourceIndex(choice.FirstChoice)]++;
                    if (choice.SecondChoice is ResourceCardType second) hand[ResourceIndex(second)]++;
                    return Missing(hand, cost);
                });
            }

            // Monopoly: the resource the opponents hold most of
            List<Action> monopoly = MonopolyAction.GetActionsForState(state, p);
            if (monopoly.Count > 0)
            {
                return monopoly.MaxBy(action =>
                {
                    ResourceCardType type = ((MonopolyAction)action).ChosenType;
                    uint held = 0;
                    for (int other = 0; other < state.Players.Length; other++)
                        if (other != p) held += state.Players[other].ResourceCards.Get(type);
                    return held;
                });
            }

            return null;
        }

        private static Action? BuyDevelopmentCard(GameState state, sbyte p)
        {
            return First(BuyDevelopmentCardAction.GetActionsForState(state, p));
        }

        /// <summary>
        /// Trades a surplus resource for one missing for the next build, at the cheapest available rate
        /// </summary>
        private static Action? Trade(GameState state, sbyte p)
        {
            uint[] hand = Hand(state, p);
            uint[] cost = NextBuildCost(state, p);
            if (Missing(hand, cost) == 0) return null;

            foreach ((List<Action> trades, uint rate) in new[] {
                (TwoToOneTradeAction.GetActionsForState(state, p), 2u),
                (ThreeToOneTradeAction.GetActionsForState(state, p), 3u),
                (FourToOneTradeAction.GetActionsForState(state, p), 4u) })
            {
                foreach (Action trade in trades)
                {
                    (ResourceCardType input, ResourceCardType output) = TradeTypes(trade);
                    int inputIdx = ResourceIndex(input), outputIdx = ResourceIndex(output);

                    // Only trades that add a missing card and give away cards the build doesn't need
                    if (hand[outputIdx] < cost[outputIdx] && hand[inputIdx] >= cost[inputIdx] + rate)
                        return trade;
                }
            }

            return null;
        }

        // -- Helpers --

        private static Action? First(List<Action> actions) => actions.Count > 0 ? actions[0] : null;

        private static Action? BestByIntersection(List<Action> actions, GameState state)
        {
            Action? best = null;
            float bestValue = float.NegativeInfinity;
            foreach (Action action in actions)
            {
                int intersectionIdx = action switch
                {
                    FirstInitialSettlementAction a => a.IntersectionIndex,
                    SecondInitialSettlementAction a => a.IntersectionIndex,
                    SettlementAction a => a.IntersectionIndex,
                    CityAction a => a.IntersectionIndex,
                    _ => throw new ArgumentException($"{action} is not placed on an intersection.")
                };

                float value = SimpleAgent.IntersectionValueFunc(state.Board.Intersections[intersectionIdx], state);
                if (value > bestValue)
                {
                    best = action;
                    bestValue = value;
                }
            }

            return best;
        }

        /// <summary>
        /// A random road whose far end is a free building site, or a random road if there is none
        /// </summary>
        private static Action RoadTowardBuildingSite(List<Action> roads, GameState state)
        {
            UInt128 freeSites = state.Board.GetFreeBuildingSites();
            int start = Utils.Random.Next(roads.Count);
            for (int i = 0; i < roads.Count; i++)
            {
                Action road = roads[(start + i) % roads.Count];
                int edgeIdx = road switch
                {
                    RoadAction a => a.EdgeIndex,
                    FirstInitialRoadAction a => a.EdgeIndex,
                    SecondInitialRoadAction a => a.EdgeIndex,
                    _ => throw new ArgumentException($"{road} is not a road.")
                };

                if ((state.Board.Adjacency.GetIntersectionMask(edgeIdx) & freeSites) != UInt128.Zero)
                    return road;
            }

            return roads[start];
        }

        private static bool NeedsRoad(GameState state, sbyte p)
        {
            return state.Board.GetSettlementSites(p) == UInt128.Zero && state.Players[p].BuildingStock.RemainingSettlements > 0;
        }

        private static bool RobberBlocksOwnProduction(GameState state, sbyte p)
        {
            int? robberTileIdx = state.Board.GetRobberTileIndex();
            if (robberTileIdx == null) return false;

            return (state.Board.Adjacency.GetCornerMask(state.Board.Map.Values[robberTileIdx.Value]) & state.Board.GetBuildingMask(p)) != UInt128.Zero;
        }

        /// <summary>
        /// Cost of the build the agent saves for: a settlement if a spot is reachable, else a city, else a road toward a spot, else a development card
        /// </summary>
        private static uint[] NextBuildCost(GameState state, sbyte p)
        {
            BuildingStock stock = state.Players[p].BuildingStock;
            if (stock.RemainingSettlements > 0 && state.Board.GetSettlementSites(p) != UInt128.Zero) return SettlementCost;
            if (stock.RemainingCities > 0 && OwnsSettlement(state, p)) return CityCost;
            if (NeedsRoad(state, p) && stock.RemainingRoads > 0) return RoadCost;
            return DevelopmentCardCost;
        }

        private static bool OwnsSettlement(GameState state, sbyte p)
        {
            foreach (int intersectionIdx in BitMask.Of(state.Board.GetBuildingMask(p)))
                if (state.Board.GetBuilding(intersectionIdx) == Intersection.BuildingType.Settlement) return true;

            return false;
        }

        private static uint[] Hand(GameState state, sbyte p)
        {
            uint[] hand = new uint[Resources.Length];
            for (int i = 0; i < Resources.Length; i++)
                hand[i] = state.Players[p].ResourceCards.Get(Resources[i]);
            return hand;
        }

        private static uint Missing(uint[] hand, uint[] cost)
        {
            uint missing = 0;
            for (int i = 0; i < cost.Length; i++)
                if (hand[i] < cost[i]) missing += cost[i] - hand[i];
            return missing;
        }

        private static int ResourceIndex(ResourceCardType type) => Array.IndexOf(Resources, type);

        private static (ResourceCardType input, ResourceCardType output) TradeTypes(Action trade) => trade switch
        {
            TwoToOneTradeAction t => (t.InputType, t.OutputType),
            ThreeToOneTradeAction t => (t.InputType, t.OutputType),
            FourToOneTradeAction t => (t.InputType, t.OutputType),
            _ => throw new ArgumentException($"{trade} is not a trade.")
        };
    }
}
