using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;
using static Common.Direction;

namespace Common
{
    public class GameState
    {
        public GameSettings Settings { get; set; }
        public TurnState Turn { get; set; }
        public Board Board { get; set; }
        public CardSet<ResourceCardType> ResourceBank { get; set; }
        public CardSet<DevelopmentCardType> DevelopmentBank { get; set; }
        public PlayerState[] Players { get; set; }
        [YamlIgnore]
        public bool HasEnded => Turn.TypeOfRound == TurnState.RoundType.MatchEnded;

        public GameState(Board board, uint playerCount)
        {
            Settings = new GameSettings();
            Turn = new TurnState(playerCount);
            Board = board;
            (ResourceBank, DevelopmentBank) = CreateBank();
            Players = new PlayerState[playerCount];

            for (int i = 0; i < playerCount; i++)
            {
                Players[i] = new PlayerState();
            }
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public GameState(GameState copy)
        {
            Settings = new(copy.Settings);
            Turn = new(copy.Turn);
            Board = new(copy.Board);
            ResourceBank = new(copy.ResourceBank);
            DevelopmentBank = new(copy.DevelopmentBank);
            Players = new PlayerState[copy.Players.Length];

            for (int i = 0; i < Players.Length; i++)
            {
                Players[i] = new(copy.Players[i]);
            }
        }

        /// <summary>
        /// Parameterless constructor for deserialization
        /// </summary>
        private GameState()
        {

        }

        public void CalculateLargestArmy(int causingPlayerIdx)
        {
            uint playerKnights = Players[causingPlayerIdx].PlayedKnights;

            // Abort if minimum army size wasn't reached
            if (playerKnights < 3) return;

            // Minimum reached => Abort if player isn't ahead in army size
            for (int playerIdx = 0; playerIdx < Players.Length; playerIdx++)
            {
                if (playerIdx != causingPlayerIdx && Players[playerIdx].PlayedKnights >= playerKnights)
                {
                    return;
                }
            }
            
            // Army is largest => Reallocate points to player
            for (int playerIdx = 0; playerIdx < Players.Length; playerIdx++)
            {
                Players[playerIdx].VictoryPoints.LargestArmyPoints = (byte)(playerIdx == causingPlayerIdx ? 2 : 0);
            }
        }

        public void CalculateLongestRoad(int causingPlayerIdx, bool checkForBreak = false)
        {
            Stopwatch stopwatch = Stopwatch.StartNew();
            uint longestRoadFast = CalculateLongestRoadFast(causingPlayerIdx, checkForBreak);
            double secondsFast = stopwatch.Elapsed.TotalSeconds;
            Console.WriteLine($"Fast longest road calculation took {secondsFast}s");

            stopwatch.Restart();
            uint longestRoadSlow = CalculateLongestRoadSlow(causingPlayerIdx, checkForBreak);
            double secondsSlow = stopwatch.Elapsed.TotalSeconds;
            Console.WriteLine($"Slow longest road calculation took {secondsSlow}s");

            Console.WriteLine($"Speedup: {secondsSlow/secondsFast:P2}");

            AwardLongestRoadVPs(causingPlayerIdx, checkForBreak, longestRoadSlow);

            if (longestRoadFast != longestRoadSlow)
                throw new InvalidOperationException($"The methods produced different lengths for player {causingPlayerIdx}. Fast: {longestRoadFast}, Slow: {longestRoadSlow}");
        }

        private uint CalculateLongestRoadFast(int causingPlayerIdx, bool checkForBreak = false)
        {
            // Board indices of the roads owned by the player
            // Max 15 since we assume default building stock
            Span<byte> globalRoadIndices = stackalloc byte[15];

            // Bitmask of the player's roads that are currently considered for the longest road
            // Indices are for the span above and therefore <15
            ushort playerRoadMask = 0;

            // Number of roads owned by the player
            byte roadCount = 0;

            // Gather all roads owned by the player
            for (int edgeIdx = 0; edgeIdx < Board.Edges.Count; edgeIdx++)
            {
                if (Board.Edges[edgeIdx].Owner == causingPlayerIdx
                    && Board.Edges[edgeIdx].Building == Edge.BuildingType.Road)
                {
                    globalRoadIndices[roadCount] = (byte)edgeIdx;
                    playerRoadMask |= (ushort)(1 << roadCount);
                    roadCount++;
                }
            }

            uint longestRoadLength = 0;

            // Check possible paths starting from each owned road
            for (int startingRoadIdx = 0; startingRoadIdx < roadCount; startingRoadIdx++)
            {
                // Get global idx of starting road
                byte edgeIdx = globalRoadIndices[startingRoadIdx];

                // Get adjacent intersection indices
                (Intersection top, Intersection bottom) = Board.Adjacency.GetIntersections(Board.Edges[edgeIdx]);
                byte topIdx = top.Index;
                byte bottomIdx = bottom.Index;

                uint topLen    = GetMaxPathDFS(causingPlayerIdx, startingRoadIdx, topIdx,    playerRoadMask, roadCount, ref globalRoadIndices);
                uint bottomLen = GetMaxPathDFS(causingPlayerIdx, startingRoadIdx, bottomIdx, playerRoadMask, roadCount, ref globalRoadIndices);

                uint lenStartingHere = Math.Max(topLen, bottomLen);

                if (lenStartingHere > longestRoadLength)
                {
                    longestRoadLength = lenStartingHere;

                    // Stop checking other options if maximum length was reached once
                    if (longestRoadLength == roadCount)
                        break;
                }
            }

            // Update longest road length of the handled player
            Players[causingPlayerIdx].LongestRoadLength = longestRoadLength;

            return longestRoadLength;
        }

        private uint GetMaxPathDFS(int playerIdx, int playerRoadIdx, int currentIntersectionIdx, ushort remainingMask, byte roadCount, ref Span<byte> globalRoadIndices)
        {
            // Remove current road from the mask of roads left to consider
            remainingMask &= (ushort)~(1 << playerRoadIdx);

            // Terminate if the intersection towards which we are expanding is blocked by another player
            if (Board.Intersections[currentIntersectionIdx].Building != Intersection.BuildingType.None
                && Board.Intersections[currentIntersectionIdx].Owner != playerIdx)
                return 1;

            uint maxDepth = 0;

            foreach (Edge roadAdjToIntersection in Board.Adjacency.GetEdges(Board.Intersections[currentIntersectionIdx]))
            {
                // Don't go back along the road we came from
                if (roadAdjToIntersection.Index == globalRoadIndices[playerRoadIdx])
                    continue;

                // Only proceed if the road was built and owned by the player
                if (roadAdjToIntersection.Owner != playerIdx || roadAdjToIntersection.Building == Edge.BuildingType.None)
                    continue;

                // Find ouf if the road is in our bitmask of valid remaining player roads
                int maskIdxOfEdge = -1;

                for (int maskBitIdx = 0; maskBitIdx < roadCount; maskBitIdx++)
                {
                    // Check if this bit of the mask represents the edge we want to branch off along
                    if (globalRoadIndices[maskBitIdx] == roadAdjToIntersection.Index)
                    {
                        // Check if the bit is marked as remaining
                        if ((remainingMask & (ushort)(1 << maskBitIdx)) != 0)
                        {
                            maskIdxOfEdge = maskBitIdx;
                        }

                        // No need to continue the search, the road was found to be either remaining or already used
                        break;
                    }
                }

                if (maskIdxOfEdge == -1)
                    continue;

                // If the player-owned road is a valid branch, continue DFS
                // Get the next intersection on the other end of the road
                (Intersection top, Intersection bottom) = Board.Adjacency.GetIntersections(roadAdjToIntersection);
                int nextIntersectionIdx = top.Index != currentIntersectionIdx ? top.Index : bottom.Index;

                uint branchDepth = GetMaxPathDFS(playerIdx, maskIdxOfEdge, nextIntersectionIdx, remainingMask, roadCount, ref globalRoadIndices);

                // Update max depth if we found a longer path
                if (branchDepth > maxDepth)
                    maxDepth = branchDepth;
            }

            return 1 + maxDepth;
        }

        private uint CalculateLongestRoadSlow(int causingPlayerIdx, bool checkForBreak = false)
        {
            Dictionary<Edge, int> roadIndexLookup = Board.Edges
                .Select((edge, idx) => (edge, idx))
                .Where(x => x.edge.Owner == causingPlayerIdx && x.edge.Building != Edge.BuildingType.None)
                .ToDictionary(x => x.edge, x => x.idx);

            ImmutableHashSet<Edge> playerRoads = roadIndexLookup.Keys.ToImmutableHashSet();
            HashSet<Edge> longestPlayerRoad = [];

            // Recursively calculate longest road from each possible starting road
            foreach (Edge startingRoad in playerRoads)
            {
                // Create one candidate each for the top and bottom search direction along the starting road
                (Intersection top, Intersection bottom) = Board.Adjacency.GetIntersections(startingRoad);

                foreach (Intersection startingDirection in new Intersection[] {top, bottom})
                {
                    HashSet<Edge> candidate = CalculateLongestRoadRec(causingPlayerIdx, startingRoad, startingDirection, playerRoads.Remove(startingRoad), [], Board);
                    if (candidate.Count > longestPlayerRoad.Count)
                    {
                        longestPlayerRoad = candidate;

                        // Skip remaining branches, if the candidate length is guaranteed to be maximal
                        // => No longer road achievable, only permutations
                        if (longestPlayerRoad.Count == playerRoads.Count) break;
                    }
                }
            }

            uint longestRoadLength = (uint)longestPlayerRoad.Count;
            Players[causingPlayerIdx].LongestRoadLength = longestRoadLength;

            return longestRoadLength;
        }

        private static HashSet<Edge> CalculateLongestRoadRec(int playerIdx, Edge current, Intersection intersectionToFollow, ImmutableHashSet<Edge> remaining, ImmutableHashSet<Edge> contained, Board board)
        {
            HashSet<Edge> longestPlayerRoad = [.. contained, current];

            // Terminate if all player roads are contained
            if (remaining.IsEmpty) return longestPlayerRoad;

            // Find possible branches
            bool intersectionBlocked = intersectionToFollow.Owner != playerIdx && intersectionToFollow.Building != Intersection.BuildingType.None;

            // Don't branch if intersection is blocked
            if (intersectionBlocked)
                return longestPlayerRoad;

            // Otherwise, branch along remaining roads located at intersection
            var roadsOnIntersection = board.Adjacency.GetEdges(intersectionToFollow).Where(edge => edge.Owner == playerIdx && edge != current);
            var remainingTopRoads = roadsOnIntersection.Intersect(remaining);

            // Recursively evaluate branches
            foreach (Edge branch in remainingTopRoads)
            {
                // Get next intersection along the branching edge
                (Intersection top, Intersection bottom) = board.Adjacency.GetIntersections(branch);
                Intersection nextIntersection = top != intersectionToFollow ? top : bottom;

                HashSet<Edge> candidate = CalculateLongestRoadRec(playerIdx, branch, nextIntersection, remaining.Remove(branch), [.. contained, current], board);
                if (candidate.Count > longestPlayerRoad.Count)
                {
                    longestPlayerRoad = candidate;

                    // Skip remaining branches, if the candidate length is guaranteed to be maximal
                    // => No longer road achievable, only permutations
                    if (longestPlayerRoad.Count == contained.Count + remaining.Count) break;
                }
            }

            return longestPlayerRoad;
        }

        private void AwardLongestRoadVPs(int causingPlayerIdx, bool checkForBreak, uint roadLength)
        {
            // Award VPs
            // Cause: Road broken by settlement placement
            if (checkForBreak)
            {
                // Check if player was the leader
                if (Players[causingPlayerIdx].VictoryPoints.LongestRoadPoints == 0) return;

                // Check for minimum length
                uint globalLongestRoad = Players.Max(state => state.LongestRoadLength);
                if (globalLongestRoad < 5) return;

                bool isLeading = roadLength == globalLongestRoad;
                bool isTied = Players.Count(state => state.LongestRoadLength == globalLongestRoad) > 1;

                // Keep VPs if still higher or tied
                if (isLeading) return;

                // Set VPs aside if tied and behind the tie
                if (!isLeading && isTied)
                {
                    Players[causingPlayerIdx].VictoryPoints.LongestRoadPoints = 0;
                    return;
                }

                // Give VPs to new leader if behind and untied
                for (int playerIdx = 0; playerIdx < Players.Length; playerIdx++)
                {
                    bool isNewLeader = Players[playerIdx].LongestRoadLength == globalLongestRoad;
                    Players[playerIdx].VictoryPoints.LongestRoadPoints = (byte)(isNewLeader ? 2 : 0);
                }
            }
            // Cause: New road placed
            else
            {
                // Check for minimum length
                if (roadLength < 5) return;

                // Check if another player has at least an equally long road
                if (Players.Any(player => player != Players[causingPlayerIdx] && player.LongestRoadLength >= roadLength)) return;

                // Move VPs to player
                for (int playerIdx = 0; playerIdx < Players.Length; playerIdx++)
                {
                    Players[playerIdx].VictoryPoints.LongestRoadPoints = (byte)(playerIdx == causingPlayerIdx ? 2 : 0);
                }
            }
        }

        public void CheckForCompletion()
        {
            // Players can only win on their own turn
            if (Players[Turn.PlayerIndex].VictoryPoints.Total >= Settings.VictoryPoints)
            {
                Turn.TypeOfRound = TurnState.RoundType.MatchEnded;
            }
        }

        public bool CanPlayerAct(int playerIdx)
        {
            // TODO: Eventually account for trade offers from other players to target player
            return !Turn.MustDiscard && Turn.PlayerIndex == playerIdx || Turn.MustDiscard && Turn.AwaitedPlayerDiscards[playerIdx];
        }

        public static (CardSet<ResourceCardType> resources, CardSet<DevelopmentCardType> development) CreateBank()
        {
            CardSet<ResourceCardType> resources = new();
            CardSet<DevelopmentCardType> development = new();

            resources.Add(ResourceCardType.Lumber, 19);
            resources.Add(ResourceCardType.Brick, 19);
            resources.Add(ResourceCardType.Wool, 19);
            resources.Add(ResourceCardType.Grain, 19);
            resources.Add(ResourceCardType.Ore, 19);

            development.Add(DevelopmentCardType.Knight, 14);
            development.Add(DevelopmentCardType.RoadBuilding, 2);
            development.Add(DevelopmentCardType.YearOfPlenty, 2);
            development.Add(DevelopmentCardType.Monopoly, 2);
            development.Add(DevelopmentCardType.VictoryPoint, 5);

            return (resources, development);
        }

        public void ResetCards()
        {
            (ResourceBank, DevelopmentBank) = CreateBank();

            foreach(PlayerState player in Players)
            {
                player.ResourceCards = new();
                player.DevelopmentCards = new();
            }
        }

        public void Reset()
        {
            Turn = new TurnState((uint)Players.Length);
            (ResourceBank, DevelopmentBank) = CreateBank();
            Players = new PlayerState[Players.Length];

            for (int i = 0; i < Players.Length; i++)
            {
                Players[i] = new PlayerState();
            }
        }

        public override bool Equals(object? obj)
        {
            return obj is GameState other
                && Settings.Equals(other.Settings)
                && Turn.Equals(other.Turn)
                && Board.Equals(other.Board)
                && ResourceBank.Equals(other.ResourceBank)
                && DevelopmentBank.Equals(other.DevelopmentBank)
                && Players.SequenceEqual(other.Players);
        }

        public override int GetHashCode()
        {
            HashCode hash = new HashCode();

            foreach (PlayerState player in Players)
            {
                hash.Add(player);
            }

            hash.Add(Settings);
            hash.Add(Turn);
            hash.Add(Board);
            hash.Add(ResourceBank);
            hash.Add(DevelopmentBank);

            return hash.ToHashCode();
        }

        public int GetVerboseHashCode()
        {
            HashCode hash = new HashCode();

            foreach (PlayerState player in Players)
            {
                hash.Add(player);
                Console.WriteLine("Player: " + player.GetHashCode());
            }

            hash.Add(Settings);
            Console.WriteLine("Settings: " + Settings.GetHashCode());
            hash.Add(Turn);
            Console.WriteLine("Turn: " + Turn.GetHashCode());
            hash.Add(Board);
            Console.WriteLine("Board: " + Board.GetHashCode());
            hash.Add(ResourceBank);
            Console.WriteLine("ResourceBank: " + ResourceBank.GetHashCode());
            hash.Add(DevelopmentBank);
            Console.WriteLine("DevelopmentBank: " + DevelopmentBank.GetHashCode());

            Console.WriteLine("Total: " + hash.ToHashCode());

            Console.WriteLine("===============================");

            return hash.ToHashCode();
        }
    }
}
