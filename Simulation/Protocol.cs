using Agents.MCTS;
using Common;
using Common.Serialization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Action = Common.Actions.Action;

namespace Simulation
{
    /// <summary>
    /// The first message of a connection: who connects and as what. A reconnecting client sends its session ID to keep it.
    /// </summary>
    public sealed record Hello(string Name, string? Token, Role Role, int? Seat, string? SessionId);

    /// <summary>
    /// The wire format: one JSON object per WebSocket text message, with a "type" field. Game states and moves are embedded as
    /// the save files' YAML, which keeps the moves' resolved randomness (dice, stolen cards), so clients replay them exactly.
    /// </summary>
    public static class Protocol
    {
        public const string Path = "/match";

        #region Game payloads
        private static string Yaml(GameState state) => SaveFileSerializer.Serialize(state);
        private static GameState State(JsonNode? node)
        {
            GameState state = SaveFileSerializer.Deserialize<GameState>((string)node!);

            // As when loading a save file: the adjacency is serialized without its link to the board
            state.Board.Adjacency.Attach(state.Board);
            return state;
        }

        private static string Yaml(Action action) => SaveFileSerializer.Serialize(new List<Action> { action });
        private static Action Action(JsonNode? node) => SaveFileSerializer.Deserialize<List<Action>>((string)node!)[0];

        private static string Yaml(List<Action> actions) => SaveFileSerializer.Serialize(actions);
        private static List<Action> Actions(JsonNode? node) => SaveFileSerializer.Deserialize<List<Action>>((string)node!) ?? [];

        private static JsonObject Entry(HistoryEntry entry) => entry.Action is Action action
            ? new JsonObject { ["action"] = Yaml(action) }
            : new JsonObject { ["edit"] = Edit(entry.Edit!) };

        private static HistoryEntry Entry(JsonNode? node) => node!["action"] is JsonNode action
            ? HistoryEntry.Of(Action(action))
            : HistoryEntry.Of(Edit(node["edit"]));

        private static JsonObject Edit(BoardEdit edit) => new()
        {
            ["description"] = edit.Description,
            ["player"] = edit.Player,
            ["before"] = Yaml(edit.Before),
            ["after"] = Yaml(edit.After),
        };

        private static BoardEdit Edit(JsonNode? node) => new((string)node!["description"]!, (int)node["player"]!, State(node["before"]), State(node["after"]));

        private static JsonArray Entries(IEnumerable<HistoryEntry> entries) => new(entries.Select(e => (JsonNode)Entry(e)).ToArray());
        private static List<HistoryEntry> Entries(JsonNode? node) => node!.AsArray().Select(Entry).ToList();

        private static JsonArray SeatKinds(SeatKind[] seats) => new(seats.Select(s => (JsonNode)s.ToString()).ToArray());
        private static SeatKind[] SeatKinds(JsonNode? node) => node!.AsArray().Select(n => Enum.Parse<SeatKind>((string)n!)).ToArray();

        private static JsonObject Playback(PlaybackState p) => new() { ["paused"] = p.Paused, ["standIns"] = p.HumanStandIns, ["pace"] = p.PaceSeconds };
        private static PlaybackState Playback(JsonNode? node) => new((bool)node!["paused"]!, (bool)node["standIns"]!, (float)node["pace"]!);

        private static JsonObject Agents(AgentSettings a) => new()
        {
            ["seconds"] = a.SecondsForAll,
            ["iterations"] = a.IterationsForAll,
            ["opponentModel"] = a.OpponentModel.ToString()
        };

        private static AgentSettings Agents(JsonNode? node) => new((double?)node!["seconds"], (long)node["iterations"]!, Enum.Parse<OpponentModel>((string)node["opponentModel"]!));

        private static JsonObject Session(SessionInfo s) => new()
        {
            ["id"] = s.Id,
            ["name"] = s.Name,
            ["role"] = s.Role.ToString(),
            ["seat"] = s.Seat,
            ["address"] = s.Address
        };

        private static SessionInfo Session(JsonNode? node) => new((string)node!["id"]!, (string)node["name"]!, Enum.Parse<Role>((string)node["role"]!), (int?)node["seat"], (string?)node["address"] ?? "");
        #endregion

        #region Commands
        public static string Serialize(Command command)
        {
            JsonObject json = command switch
            {
                PlayMove m => new JsonObject { ["action"] = Yaml(m.Action) },
                UndoCommand => [],
                RedoCommand => [],
                ApplyEdit e => new JsonObject { ["edit"] = Edit(e.Edit) },
                NewGame n => new JsonObject { ["seats"] = SeatKinds(n.Seats), ["centerDesert"] = n.CenterDesert, ["clearBoard"] = n.ClearBoard },
                LoadGame l => new JsonObject { ["state"] = Yaml(l.State), ["played"] = Yaml(l.Played), ["redo"] = Yaml(l.Redo) },
                SetSeat s => new JsonObject { ["seat"] = s.Seat, ["kind"] = s.Kind.ToString() },
                SetPlayback p => new JsonObject { ["playback"] = Playback(p.Playback) },
                StepCommand => [],
                ConfigureAgents c => new JsonObject { ["settings"] = Agents(c.Settings) },
                KickSession k => new JsonObject { ["session"] = k.SessionId },
                _ => throw new ArgumentException($"Unknown command {command.GetType().Name}")
            };
            json["type"] = command.GetType().Name;
            return json.ToJsonString();
        }

        public static Command DeserializeCommand(string text)
        {
            JsonNode json = JsonNode.Parse(text) ?? throw new JsonException("Empty message");
            return (string?)json["type"] switch
            {
                nameof(PlayMove) => new PlayMove(Action(json["action"])),
                nameof(UndoCommand) => new UndoCommand(),
                nameof(RedoCommand) => new RedoCommand(),
                nameof(ApplyEdit) => new ApplyEdit(Edit(json["edit"])),
                nameof(NewGame) => new NewGame(SeatKinds(json["seats"]), (bool)json["centerDesert"]!, (bool?)json["clearBoard"] ?? false),
                nameof(LoadGame) => new LoadGame(State(json["state"]), Actions(json["played"]), Actions(json["redo"])),
                nameof(SetSeat) => new SetSeat((int)json["seat"]!, Enum.Parse<SeatKind>((string)json["kind"]!)),
                nameof(SetPlayback) => new SetPlayback(Playback(json["playback"])),
                nameof(StepCommand) => new StepCommand(),
                nameof(ConfigureAgents) => new ConfigureAgents(Agents(json["settings"])),
                nameof(KickSession) => new KickSession((string)json["session"]!),
                string type => throw new JsonException($"Unknown command type \"{type}\""),
                null => throw new JsonException("The message has no type")
            };
        }
        #endregion

        #region Events
        public static string Serialize(MatchEvent e)
        {
            JsonObject json = e switch
            {
                WelcomeEvent w => new JsonObject { ["session"] = Session(w.Session) },
                SnapshotEvent s => new JsonObject
                {
                    ["state"] = Yaml(s.State),
                    ["history"] = Entries(s.History),
                    ["redo"] = Entries(s.Redo),
                    ["seats"] = SeatKinds(s.Seats),
                    ["playback"] = Playback(s.Playback),
                    ["agents"] = Agents(s.Agents),
                },
                AppliedEvent a => new JsonObject { ["entry"] = Entry(a.Entry), ["redo"] = a.IsRedo },
                UndoneEvent => [],
                SeatsEvent s => new JsonObject { ["seats"] = SeatKinds(s.Seats) },
                PlaybackEvent p => new JsonObject { ["playback"] = Playback(p.Playback) },
                AgentsEvent a => new JsonObject { ["settings"] = Agents(a.Settings) },
                ThinkingEvent t => new JsonObject { ["seat"] = t.Seat, ["thinking"] = t.Thinking },
                SessionsEvent s => new JsonObject { ["sessions"] = new JsonArray(s.Sessions.Select(x => (JsonNode)Session(x)).ToArray()) },
                ErrorEvent err => new JsonObject { ["message"] = err.Message },
                ClosedEvent c => new JsonObject { ["reason"] = c.Reason },
                _ => throw new ArgumentException($"Unknown event {e.GetType().Name}")
            };
            json["type"] = e.GetType().Name;
            return json.ToJsonString();
        }

        public static MatchEvent DeserializeEvent(string text)
        {
            JsonNode json = JsonNode.Parse(text) ?? throw new JsonException("Empty message");
            return (string?)json["type"] switch
            {
                nameof(WelcomeEvent) => new WelcomeEvent(Session(json["session"])),
                nameof(SnapshotEvent) => new SnapshotEvent(State(json["state"]), Entries(json["history"]), Entries(json["redo"]), SeatKinds(json["seats"]),
                    Playback(json["playback"]), Agents(json["agents"])),
                nameof(AppliedEvent) => new AppliedEvent(Entry(json["entry"]), (bool)json["redo"]!),
                nameof(UndoneEvent) => new UndoneEvent(),
                nameof(SeatsEvent) => new SeatsEvent(SeatKinds(json["seats"])),
                nameof(PlaybackEvent) => new PlaybackEvent(Playback(json["playback"])),
                nameof(AgentsEvent) => new AgentsEvent(Agents(json["settings"])),
                nameof(ThinkingEvent) => new ThinkingEvent((int)json["seat"]!, (bool)json["thinking"]!),
                nameof(SessionsEvent) => new SessionsEvent(json["sessions"]!.AsArray().Select(Session).ToList()),
                nameof(ErrorEvent) => new ErrorEvent((string)json["message"]!),
                nameof(ClosedEvent) => new ClosedEvent((string)json["reason"]!),
                string type => throw new JsonException($"Unknown event type \"{type}\""),
                null => throw new JsonException("The message has no type")
            };
        }
        #endregion

        #region Hello
        public static string Serialize(Hello hello) => new JsonObject
        {
            ["type"] = nameof(Hello),
            ["name"] = hello.Name,
            ["token"] = hello.Token,
            ["role"] = hello.Role.ToString(),
            ["seat"] = hello.Seat,
            ["sessionId"] = hello.SessionId,
        }.ToJsonString();

        public static Hello DeserializeHello(string text)
        {
            JsonNode json = JsonNode.Parse(text) ?? throw new JsonException("Empty message");
            if ((string?)json["type"] != nameof(Hello)) throw new JsonException("The first message must be a hello");
            return new Hello((string?)json["name"] ?? "Guest", (string?)json["token"], Enum.Parse<Role>((string)json["role"]!), (int?)json["seat"], (string?)json["sessionId"]);
        }
        #endregion
    }
}
