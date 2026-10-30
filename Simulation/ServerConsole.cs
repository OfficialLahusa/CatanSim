using System.Text;

namespace Simulation
{
    /// <summary>
    /// The admin console of a dedicated server: the match's state, the connections, tokens and match control, one text
    /// command at a time
    /// </summary>
    public sealed class ServerConsole
    {
        private static readonly SessionInfo ConsoleSession = new("console", "Server console", Role.Admin, null);

        private readonly Match _match;
        private readonly string? _path;

        public ServerConfig Config { get; }
        public SimulationServer? Server { get; set; }

        /// <param name="path">Where token changes are saved, or null to keep them in memory</param>
        public ServerConsole(Match match, ServerConfig config, string? path)
        {
            _match = match;
            Config = config;
            _path = path;
        }

        public const string Help = """
            Commands:
              status                       seats, turn, playback and connections
              sessions                     the connections with their IDs
              kick ID                      closes a connection
              tokens                       the tokens with their IDs (the secrets are never stored)
              token admin [LABEL]          a new admin token
              token player SEAT [LABEL]    a new token for the player of a seat (1-4)
              token agent SEAT [LABEL]     a new token for an agent bridge playing a seat
              token spectator [LABEL]      a new spectator token
              revoke ID                    removes a token and closes its connections
              spectators on|off            whether anyone may watch without a token
              seat SEAT KIND               who plays a seat (human, easy, medium, hard, open-hand-easy, open-hand-medium,
                                           open-hand-hard, greedy, builder, simple, random)
              newgame KIND KIND KIND KIND  a new game on a new map with these seats (add "desert" for a central desert,
                                           "notrade" for no trading between players)
              pause / resume               stops or restarts the agents
              quit                         stops the server
            Tokens are shown once and stored as hashes.
            """;

        public string Execute(string line)
        {
            string[] words = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0) return "";

            try
            {
                switch (words[0].ToLowerInvariant())
                {
                    case "help": return Help;
                    case "status": return Status();
                    case "sessions": return Sessions();
                    case "kick" when words.Length == 2:
                        _match.Handle(ConsoleSession, new KickSession(words[1]));
                        return $"Closing {words[1]}.";
                    case "token": return AddToken(words);
                    case "tokens": return TokenList();
                    case "revoke" when words.Length == 2:
                        if (Server != null) return Server.RevokeToken(words[1]) ? $"Revoked {words[1]}." : $"There is no token {words[1]}.";
                        if (Config.RemoveToken(words[1]) == null) return $"There is no token {words[1]}.";
                        if (_path != null) Config.Save(_path);
                        return $"Revoked {words[1]}.";
                    case "spectators" when words.Length == 2 && words[1].ToLowerInvariant() is "on" or "off":
                        {
                            bool allow = words[1].Equals("on", StringComparison.OrdinalIgnoreCase);
                            if (Server != null) Server.SetAllowAnonymousSpectators(allow);
                            else
                            {
                                Config.AllowAnonymousSpectators = allow;
                                if (_path != null) Config.Save(_path);
                            }
                            return allow ? "Anyone may watch without a token." : "Watching needs a token.";
                        }
                    case "seat" when words.Length == 3:
                        {
                            int seat = ParseSeat(words[1]);
                            if (ParseKind(words[2]) is not SeatKind kind) return $"Unknown seat kind \"{words[2]}\".";
                            _match.Handle(ConsoleSession, new SetSeat(seat, kind));
                            return $"Seat {seat + 1}: {Seats.Name(kind)}.";
                        }
                    case "newgame" when words.Length >= 5:
                        {
                            SeatKind[] kinds = new SeatKind[4];
                            for (int i = 0; i < 4; i++)
                            {
                                if (ParseKind(words[i + 1]) is not SeatKind kind) return $"Unknown seat kind \"{words[i + 1]}\".";
                                kinds[i] = kind;
                            }
                            _match.Handle(ConsoleSession, new NewGame(kinds, words.Skip(5).Any(w => w.Equals("desert", StringComparison.OrdinalIgnoreCase)),
                                FreeTrade: !words.Skip(5).Any(w => w.Equals("notrade", StringComparison.OrdinalIgnoreCase))));
                            return "New game started.";
                        }
                    case "pause":
                        _match.Handle(ConsoleSession, new SetPlayback(_match.Playback with { Paused = true }));
                        return "Agents paused.";
                    case "resume":
                        _match.Handle(ConsoleSession, new SetPlayback(_match.Playback with { Paused = false }));
                        return "Agents resumed.";
                }
            }
            catch (FormatException e)
            {
                return e.Message;
            }
            return $"Unknown command \"{line.Trim()}\". Type help for the commands.";
        }

        private string Status()
        {
            StringBuilder text = new StringBuilder();
            SeatKind[] seats = _match.Seats;
            for (int seat = 0; seat < seats.Length; seat++)
                text.AppendLine($"Seat {seat + 1}: {Seats.Name(seats[seat])}");
            Common.GameState state = _match.Record.State;
            int acting = _match.ActingPlayer;
            text.AppendLine(state.HasEnded ? $"Game over, won by seat {state.Turn.PlayerIndex + 1}" : $"Round {state.Turn.RoundCounter}, seat {acting + 1} to act");
            text.AppendLine($"Agents {(_match.Playback.Paused ? "paused" : "playing")}, {_match.Playback.PaceSeconds:0.0} s between moves");
            text.Append(Server != null ? $"Listening on port {Server.Port}, {Server.Sessions.Count} connected" : "Not listening");
            return text.ToString();
        }

        private string Sessions()
        {
            IReadOnlyList<SessionInfo> sessions = Server?.Sessions ?? [];
            if (sessions.Count == 0) return "No connections.";
            return string.Join(Environment.NewLine, sessions.Select(s =>
                $"{s.Id}  {s.Name}  {s.Role.ToString().ToLower()}{(s.Seat is int seat ? $" of seat {seat + 1}" : "")}  {s.Address}"));
        }

        private string AddToken(string[] words)
        {
            if (words.Length < 2) return "Which role? admin, player SEAT, agent SEAT or spectator.";
            Role role;
            int? seat = null;
            int labelStart = 2;
            switch (words[1].ToLowerInvariant())
            {
                case "admin": role = Role.Admin; break;
                case "spectator": role = Role.Spectator; break;
                case "player":
                case "agent":
                    role = words[1].Equals("player", StringComparison.OrdinalIgnoreCase) ? Role.Player : Role.Agent;
                    if (words.Length < 3) return $"A {words[1].ToLower()} token needs a seat (1-4).";
                    seat = ParseSeat(words[2]);
                    labelStart = 3;
                    break;
                default:
                    return $"Unknown role \"{words[1]}\": admin, player SEAT, agent SEAT or spectator.";
            }

            string label = string.Join(' ', words.Skip(labelStart));
            string token;
            TokenInfo info;
            if (Server != null)
            {
                (info, token) = Server.CreateToken(role, seat, label);
            }
            else
            {
                (TokenEntry entry, token) = Config.AddToken(role, seat, label);
                info = entry.Info;
                if (_path != null) Config.Save(_path);
            }
            return $"New {role.ToString().ToLower()} token{(seat is int s ? $" for seat {s + 1}" : "")} {info.Id} (shown once): {token}";
        }

        private string TokenList()
        {
            List<TokenInfo> tokens = Config.TokenInfos();
            if (tokens.Count == 0) return "No tokens.";
            return string.Join(Environment.NewLine, tokens.Select(t =>
                $"{t.Id}  {t.Role.ToString().ToLower()}{(t.Seat is int seat ? $" of seat {seat + 1}" : "")}  {(t.Label.Length > 0 ? t.Label + "  " : "")}created {t.Created:yyyy-MM-dd HH:mm} UTC"));
        }

        private static int ParseSeat(string word)
        {
            if (int.TryParse(word, out int seat) && seat is >= 1 and <= 4) return seat - 1;
            throw new FormatException($"\"{word}\" is not a seat (1-4).");
        }

        /// <summary>
        /// A seat kind by its name or its preset: easy, medium, hard for the MCTS presets that see what their seat sees (also
        /// fair-easy and so on), open-hand-easy and so on for those that see every hand
        /// </summary>
        public static SeatKind? ParseKind(string word) => word.ToLowerInvariant() switch
        {
            "easy" or "fair-easy" => SeatKind.PimcEasy,
            "medium" or "fair-medium" => SeatKind.PimcMedium,
            "hard" or "fair-hard" => SeatKind.PimcHard,
            "open-hand-easy" => SeatKind.MctsEasy,
            "open-hand-medium" => SeatKind.MctsMedium,
            "open-hand-hard" => SeatKind.MctsHard,
            _ => Enum.TryParse(word, true, out SeatKind kind) ? kind : null
        };
    }
}
