using Agents;
using Common;
using Common.Actions;
using Action = Common.Actions.Action;

namespace Tests
{
    /// <summary>
    /// The playout agents: legal moves through whole games, and the rules BuilderAgent follows
    /// </summary>
    public class AgentTests
    {
        private const int GamesPerAgent = 20;
        private const int RoundLimit = 1500;

        public static IEnumerable<object[]> PlayoutAgents()
        {
            yield return ["RandomAgent"];
            yield return ["SimpleAgent"];
            yield return ["BuilderAgent"];
            yield return ["BuilderAgent with exploration"];
        }

        private static Agent Create(string name, sbyte playerIdx) => name switch
        {
            "RandomAgent" => new RandomAgent(playerIdx),
            "SimpleAgent" => new SimpleAgent(playerIdx),
            "BuilderAgent" => new BuilderAgent(playerIdx),
            "BuilderAgent with exploration" => new BuilderAgent(playerIdx, explorationRate: 0.2),
            _ => throw new ArgumentException(name)
        };

        [Theory]
        [MemberData(nameof(PlayoutAgents))]
        public void AgentsOnlyPlayLegalActionsAndFinishTheirGames(string agentName)
        {
            TestUtil.Seed(180);

            for (int game = 0; game < GamesPerAgent; game++)
            {
                GameState state = TestUtil.NewGame();
                Agent[] agents = Enumerable.Range(0, 4).Select(i => Create(agentName, (sbyte)i)).ToArray();

                while (!state.HasEnded)
                {
                    sbyte actingPlayer = TestUtil.ActingPlayer(state);
                    Action action = agents[actingPlayer].Act(state);

                    Assert.True(action.PlayerIndex == actingPlayer, $"{agentName}, game {game}: {action} acts for another player");
                    Assert.True(action.IsValidFor(state), $"{agentName}, game {game}: {action} is not valid");
                    action.Apply(state);

                    Assert.True(state.Turn.RoundCounter < RoundLimit, $"{agentName}, game {game}: no winner after {RoundLimit} rounds");
                }
            }
        }

        [Fact]
        public void BuilderAgentBuildsTheBestReachableSettlement()
        {
            TestUtil.Seed(181);
            GameState state = TestUtil.EmptyNormalTurn(0);
            (int[] chain, _) = RoadChain(state, 0, length: 2);
            foreach (ResourceCardType type in new[] { ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain })
                TestUtil.GiveCards(state, 0, type, 1);

            Action action = new BuilderAgent(0).Act(state);

            SettlementAction settlement = Assert.IsType<SettlementAction>(action);
            float bestValue = SettlementAction.GetActionsForState(state, 0)
                .Max(a => SimpleAgent.IntersectionValueFunc(state.Board.Intersections[((SettlementAction)a).IntersectionIndex], state));
            Assert.Equal(bestValue, SimpleAgent.IntersectionValueFunc(state.Board.Intersections[settlement.IntersectionIndex], state));
            Assert.Contains(settlement.IntersectionIndex, chain);
        }

        [Fact]
        public void BuilderAgentTradesOnlyTowardItsNextBuild()
        {
            TestUtil.Seed(182);
            GameState state = TestUtil.EmptyNormalTurn(0);
            RoadChain(state, 0, length: 2);

            // Four ore and a reachable settlement spot: ore isn't needed, so trade it for a missing settlement resource
            TestUtil.GiveCards(state, 0, ResourceCardType.Ore, 4);
            Action trade = new BuilderAgent(0).Act(state);
            FourToOneTradeAction fourToOne = Assert.IsType<FourToOneTradeAction>(trade);
            Assert.Equal(ResourceCardType.Ore, fourToOne.InputType);
            Assert.Contains(fourToOne.OutputType, new[] { ResourceCardType.Lumber, ResourceCardType.Brick, ResourceCardType.Wool, ResourceCardType.Grain });

            // Three ore can't be traded without a port, so the turn ends
            GameState fewer = TestUtil.EmptyNormalTurn(0);
            RoadChain(fewer, 0, length: 2);
            TestUtil.GiveCards(fewer, 0, ResourceCardType.Ore, 3);
            Assert.IsType<EndTurnAction>(new BuilderAgent(0).Act(fewer));
        }

        [Fact]
        public void BuilderAgentPutsTheRobberOnAnOpponentAndOffItsOwnProduction()
        {
            TestUtil.Seed(183);
            GameState state = TestUtil.EmptyNormalTurn(0);
            state.Turn.MustMoveRobber = true;

            // One own and one opponent settlement without a shared tile
            Intersection own = state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).Count(t => t.HasYield()) == 3);
            HashSet<Tile> ownTiles = [.. state.Board.Adjacency.GetTiles(own)];
            Intersection opponent = state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).Count(t => t.HasYield()) == 3
                && !state.Board.Adjacency.GetTiles(i).Any(ownTiles.Contains));
            state.Board.SetBuilding(own.Index, Intersection.BuildingType.Settlement, 0);
            state.Board.SetBuilding(opponent.Index, Intersection.BuildingType.Settlement, 1);

            RobberAction robber = Assert.IsType<RobberAction>(new BuilderAgent(0).Act(state));
            Tile target = state.Board.Map.Values[robber.TargetTileIndex];

            Assert.Contains(target, state.Board.Adjacency.GetTiles(opponent));
            Assert.DoesNotContain(target, ownTiles);
        }

        /// <summary>
        /// Roads of the player in a row starting from a free intersection, on a board without buildings
        /// </summary>
        private static (int[] intersections, int[] edges) RoadChain(GameState state, sbyte playerIdx, int length)
        {
            int current = state.Board.Intersections.First(i => state.Board.Adjacency.GetTiles(i).All(t => t.IsLandTile())).Index;
            List<int> intersections = [current];
            List<int> edges = [];

            for (int step = 0; step < length; step++)
            {
                int edge = state.Board.Adjacency.GetEdgeIndices(current).ToArray().First(e => !edges.Contains(e));
                (int top, int bottom) = state.Board.Adjacency.GetIntersectionIndices(edge);
                current = top == current ? bottom : top;

                state.Board.SetRoad(edge, Edge.BuildingType.Road, playerIdx);
                edges.Add(edge);
                intersections.Add(current);
            }

            return ([.. intersections], [.. edges]);
        }
    }
}
