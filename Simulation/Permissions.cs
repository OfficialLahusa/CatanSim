using Common;

namespace Simulation
{
    /// <summary>
    /// What each role may do: admins everything; players and agent bridges move for their own seat while it is played from
    /// outside (a human seat); spectators nothing
    /// </summary>
    public static class Permissions
    {
        /// <summary>
        /// Why the session may not send the command, or null if it may
        /// </summary>
        public static string? Check(SessionInfo sender, Command command, GameState state, SeatKind[] seats)
        {
            if (sender.Role == Role.Admin) return null;

            if (command is PlayMove move)
            {
                if (sender.Role == Role.Spectator) return "Spectators can't move.";
                if (sender.Seat != move.Action.PlayerIndex) return $"You play seat {sender.Seat + 1}, not seat {move.Action.PlayerIndex + 1}.";
                if (seats[move.Action.PlayerIndex] != SeatKind.Human) return "An agent plays this seat.";
                return null;
            }

            return $"Only admins can {Describe(command)}.";
        }

        private static string Describe(Command command) => command switch
        {
            UndoCommand => "undo",
            RedoCommand => "redo",
            ApplyEdit => "edit the position",
            NewGame => "start a new game",
            LoadGame => "load a game",
            SetSeat => "change the seats",
            SetPlayback or StepCommand => "control the playback",
            ConfigureAgents => "configure the agents",
            KickSession => "close connections",
            _ => "do that"
        };
    }
}
