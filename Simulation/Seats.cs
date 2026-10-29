using Agents;
using Agents.Inference;
using Agents.MCTS;
using Common;

namespace Simulation
{
    /// <summary>
    /// Who plays a seat
    /// </summary>
    public enum SeatKind { Human, MctsEasy, MctsMedium, MctsHard, Greedy, Builder, Simple, Random }

    /// <summary>
    /// What a connection may do: an admin controls the match (seats, undo, edits, playback), a player and an agent (bridge)
    /// move for their seat, a spectator only watches
    /// </summary>
    public enum Role { Admin, Player, Agent, Spectator }

    /// <summary>
    /// A connection to a match: its role and, for players and agents, its seat
    /// </summary>
    public sealed record SessionInfo(string Id, string Name, Role Role, int? Seat, string Address = "");

    /// <summary>
    /// How agents move: paused or not, whether human seats are played by stand-in agents (watching all seats play), and the
    /// pause between moves so the game can be followed
    /// </summary>
    public sealed record PlaybackState(bool Paused, bool HumanStandIns, float PaceSeconds)
    {
        public static readonly PlaybackState Default = new(false, false, 0.6f);
    }

    /// <summary>
    /// Search settings of the MCTS seats: the presets' thinking times, or one time or iteration budget for all, and the
    /// opponent model
    /// </summary>
    public sealed record AgentSettings(double? SecondsForAll, long IterationsForAll, OpponentModel OpponentModel)
    {
        public static readonly AgentSettings Default = new(null, 0, OpponentModel.MaxN);
    }

    public static class Seats
    {
        public static string Name(SeatKind kind) => kind switch
        {
            SeatKind.Human => "Human",
            SeatKind.MctsEasy => "MCTS · Easy",
            SeatKind.MctsMedium => "MCTS · Medium",
            SeatKind.MctsHard => "MCTS · Hard",
            SeatKind.Greedy => "Greedy",
            SeatKind.Builder => "Builder",
            SeatKind.Simple => "Simple",
            SeatKind.Random => "Random",
            _ => kind.ToString()
        };

        /// <summary>
        /// Whether the seat's agent evaluates states with the value net (a human seat's stand-in does)
        /// </summary>
        public static bool NeedsValueNet(SeatKind kind) => kind is SeatKind.MctsEasy or SeatKind.MctsMedium or SeatKind.MctsHard or SeatKind.Greedy or SeatKind.Human;

        public static double PresetSeconds(SeatKind kind) => kind switch
        {
            SeatKind.MctsEasy => 0.1,
            SeatKind.MctsHard => 2,
            _ => 0.5
        };

        /// <summary>
        /// The agent of a seat; a human seat gets a stand-in that plays like the medium preset (the builder without the net)
        /// </summary>
        public static Agent CreateAgent(SeatKind kind, sbyte seat, GameState state, AgentSettings settings, bool valueNetAvailable)
        {
            if (kind == SeatKind.Human) kind = valueNetAvailable ? SeatKind.MctsMedium : SeatKind.Builder;
            if (NeedsValueNet(kind) && !valueNetAvailable) kind = SeatKind.Builder;

            return kind switch
            {
                SeatKind.Random => new RandomAgent(seat),
                SeatKind.Simple => new SimpleAgent(seat),
                SeatKind.Builder => new BuilderAgent(seat),
                SeatKind.Greedy => new GreedyAgent(seat),
                _ => new MCTSAgent(seat, state, timeLimitSeconds: settings.SecondsForAll ?? PresetSeconds(kind), opponentModel: settings.OpponentModel,
                    iterationLimit: settings.IterationsForAll)
            };
        }

        public static bool ValueNetAvailable => StateValueNet.IsDefaultModelAvailable;
    }
}
