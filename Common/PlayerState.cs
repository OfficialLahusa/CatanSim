using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class PlayerState
    {
        public VictoryPointTracker VictoryPoints;
        public uint PlayedKnights;
        public uint LongestRoadLength;

        public CardSet<ResourceCardType> ResourceCards;
        public CardSet<DevelopmentCardType> DevelopmentCards;
        public CardSet<DevelopmentCardType> NewDevelopmentCards;

        public BuildingStock BuildingStock;
        public PortPrivileges PortPrivileges;

        public PlayerState()
        {
            VictoryPoints = new VictoryPointTracker();
            PlayedKnights = 0;
            LongestRoadLength = 0;

            ResourceCards = new();
            DevelopmentCards = new();
            NewDevelopmentCards = new();

            BuildingStock = new BuildingStock();
            PortPrivileges = PortPrivileges.None;
        }

        /// <summary>
        /// Deep copy constructor
        /// </summary>
        /// <param name="copy">Instance to copy</param>
        public PlayerState(PlayerState copy)
        {
            VictoryPoints = new(copy.VictoryPoints);
            PlayedKnights = copy.PlayedKnights;
            LongestRoadLength = copy.LongestRoadLength;

            ResourceCards = new(copy.ResourceCards);
            DevelopmentCards = new(copy.DevelopmentCards);
            NewDevelopmentCards = new(copy.NewDevelopmentCards);

            BuildingStock = new(copy.BuildingStock);
            PortPrivileges = copy.PortPrivileges;
        }

        public bool CanAffordRoad()
        {
            bool hasFreeRoad = BuildingStock.FreeRoads > 0;
            bool canBuyNormally = BuildingStock.RemainingRoads > 0 && ResourceCards.Contains(ResourceCardType.Lumber) && ResourceCards.Contains(ResourceCardType.Brick);
            return hasFreeRoad || canBuyNormally;
        }

        public bool CanAffordSettlement()
        {
            return BuildingStock.RemainingSettlements > 0 
                && ResourceCards.Contains(ResourceCardType.Lumber) && ResourceCards.Contains(ResourceCardType.Brick) 
                && ResourceCards.Contains(ResourceCardType.Wool) && ResourceCards.Contains(ResourceCardType.Grain);
        }

        public bool CanAffordCity()
        {
            return BuildingStock.RemainingCities > 0 && ResourceCards.Contains(ResourceCardType.Grain, 2) && ResourceCards.Contains(ResourceCardType.Ore, 3);
        }

        /// <summary>
        /// Whether the player holds a card of the type that wasn't bought this turn (NewDevelopmentCards are among the
        /// DevelopmentCards). On a view, where another player's cards are Unknown (see BeliefTracker), any old card may be one.
        /// </summary>
        public bool HasOldDevelopmentCard(DevelopmentCardType type)
        {
            return DevelopmentCards.Get(type) + DevelopmentCards.Get(DevelopmentCardType.Unknown)
                > NewDevelopmentCards.Get(type) + NewDevelopmentCards.Get(DevelopmentCardType.Unknown);
        }

        public bool CanAffordDevelopmentCard()
        {
            return ResourceCards.Contains(ResourceCardType.Wool) && ResourceCards.Contains(ResourceCardType.Grain) && ResourceCards.Contains(ResourceCardType.Ore);
        }

        public override bool Equals(object? obj)
        {
            return obj is PlayerState state 
                && VictoryPoints.Equals(state.VictoryPoints) 
                && PlayedKnights == state.PlayedKnights 
                && LongestRoadLength == state.LongestRoadLength 
                && ResourceCards.Equals(state.ResourceCards) 
                && DevelopmentCards.Equals(state.DevelopmentCards) 
                && NewDevelopmentCards.Equals(state.NewDevelopmentCards)
                && BuildingStock.Equals(state.BuildingStock)
                && PortPrivileges == state.PortPrivileges;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(VictoryPoints, PlayedKnights, LongestRoadLength, ResourceCards, DevelopmentCards, NewDevelopmentCards, BuildingStock, PortPrivileges);
        }

        public int GetVerboseHashCode()
        {
            Console.WriteLine($"VictoryPoints: {VictoryPoints.GetHashCode()}");
            Console.WriteLine($"PlayedKnights: {PlayedKnights.GetHashCode()}");
            Console.WriteLine($"LongestRoadLength: {LongestRoadLength.GetHashCode()}");
            Console.WriteLine($"ResourceCards: {ResourceCards.GetHashCode()}");
            Console.WriteLine($"DevelopmentCards: {DevelopmentCards.GetHashCode()}");
            Console.WriteLine($"NewDevelopmentCards: {NewDevelopmentCards.GetHashCode()}");
            Console.WriteLine($"BuildingStock: {BuildingStock.GetHashCode()}");
            Console.WriteLine($"PortPrivileges: {PortPrivileges.GetHashCode()}");

            Console.WriteLine("Total HashCode: " + GetHashCode());

            return GetHashCode();
        }
    }


}
