using Agents;
using Client.UI;
using Common;
using Common.Actions;
using SFML.Graphics;
using Action = Common.Actions.Action;

using Simulation;

namespace Client
{
    /// <summary>
    /// Screenshots (F12, or after a delay from the command line) and preset scenes to start in, for looking at the UI in a
    /// given situation without playing up to it
    /// </summary>
    public partial class GameScreen
    {
        private string? _screenshotPath;

        /// <summary>
        /// Saves the next frame as a PNG (by default into the working directory with the time in the name)
        /// </summary>
        public void SaveScreenshot(string? path = null)
        {
            _screenshotPath = path ?? $"screenshot_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}.png";
        }

        public bool ScreenshotPending => _screenshotPath != null;

        /// <summary>
        /// Writes the pending screenshot from the frame drawn so far (before it is displayed)
        /// </summary>
        private void CaptureScreenshot()
        {
            if (_screenshotPath == null) return;

            // glReadPixels resolves the multisampled back buffer (copying it into a texture would take one sample per pixel)
            uint width = _window.Size.X, height = _window.Size.Y;
            byte[] pixels = new byte[width * height * 4];
            glReadPixels(0, 0, (int)width, (int)height, 0x1908, 0x1401, pixels);

            // OpenGL rows run bottom to top
            byte[] flipped = new byte[pixels.Length];
            int stride = (int)width * 4;
            for (int row = 0; row < height; row++)
                Array.Copy(pixels, row * stride, flipped, (height - 1 - row) * stride, stride);
            using Image image = new Image(width, height, flipped);
            image.SaveToFile(_screenshotPath);
            Console.WriteLine($"Screenshot saved to {Path.GetFullPath(_screenshotPath)}");
            _screenshotPath = null;
        }

        [System.Runtime.InteropServices.DllImport("opengl32.dll")]
        private static extern void glReadPixels(int x, int y, int width, int height, uint format, uint type, byte[] pixels);

        public static readonly string[] Scenes = ["setup", "main", "road", "city", "roll", "discard", "robber", "victim", "trade", "devcards", "plenty",
            "monopoly", "newgame", "save", "developer", "curtain", "offer", "tradeanswer", "tradepick", "spectate", "edit", "end", "hand", "rolls", "banktrade"];

        /// <summary>
        /// Plays fast agent moves up to a situation and sets the mode for it. Seat 4 (Orange) is the human.
        /// </summary>
        public void RunScene(string scene, int seed)
        {
            // The scene plays its own moves; the match's agents wait meanwhile and take over at the end
            Utils.Random.SeedCurrentThread(seed);
            SeatKind bot = _valueNetAvailable ? SeatKind.MctsMedium : SeatKind.Builder;
            Send(new SetPlayback(Playback with { Paused = true, HumanStandIns = false }));
            Send(new Simulation.NewGame([bot, bot, bot, SeatKind.Human], false));

            Agent[] fast = Enumerable.Range(0, PLAYER_COUNT).Select(i => (Agent)new BuilderAgent((sbyte)i)).ToArray();
            void PlayUntil(Func<bool> done, int maxMoves = 5000)
            {
                for (int i = 0; i < maxMoves && !done() && !State.HasEnded; i++)
                    Send(new PlayMove(fast[ActingPlayer].Act(State)));
            }
            bool HumanMainPhase(int minRound) => ActingPlayer == 3 && State.Turn.RoundCounter >= minRound && !State.Turn.MustRoll
                && !State.Turn.MustDiscard && !State.Turn.MustMoveRobber && State.Turn.TypeOfRound == TurnState.RoundType.Normal;

            switch (scene)
            {
                case "setup":
                    PlayUntil(() => ActingPlayer == 3);
                    break;

                case "main":
                case "road":
                case "city":
                case "trade":
                case "devcards":
                case "plenty":
                case "monopoly":
                case "newgame":
                case "save":
                case "developer":
                case "curtain":
                case "offer":
                case "tradepick":
                case "edit":
                case "hand":
                case "rolls":
                case "banktrade":
                    PlayUntil(() => HumanMainPhase(4));
                    if (scene == "road")
                    {
                        GiveCards(3, ResourceCardType.Lumber, 2);
                        GiveCards(3, ResourceCardType.Brick, 2);
                        _buildMode = BuildMode.Road;
                    }
                    else if (scene == "city")
                    {
                        GiveCards(3, ResourceCardType.Grain, 2);
                        GiveCards(3, ResourceCardType.Ore, 3);
                        _buildMode = BuildMode.City;
                    }
                    else if (scene == "trade")
                    {
                        GiveCards(3, ResourceCardType.Lumber, 4);
                        _dialog = Dialog.Trade;
                        _tradeGive = ResourceCardType.Lumber;
                        _tradeGet = ResourceCardType.Ore;
                    }
                    else if (scene == "devcards")
                    {
                        GameState after = new GameState(State);
                        after.Players[3].DevelopmentCards.Add(DevelopmentCardType.Knight, 2);
                        after.Players[3].DevelopmentCards.Add(DevelopmentCardType.Monopoly, 1);
                        after.Players[3].DevelopmentCards.Add(DevelopmentCardType.YearOfPlenty, 1);
                        after.Players[3].NewDevelopmentCards.Add(DevelopmentCardType.YearOfPlenty, 1);
                        after.DevelopmentBank.Remove(DevelopmentCardType.Knight, 2);
                        after.DevelopmentBank.Remove(DevelopmentCardType.Monopoly, 1);
                        after.DevelopmentBank.Remove(DevelopmentCardType.YearOfPlenty, 1);
                        SendEdit(new BoardEdit("gave Orange development cards", 3, State, after));
                        _dialog = Dialog.DevelopmentCard;
                    }
                    else if (scene == "banktrade")
                    {
                        GiveCards(3, ResourceCardType.Lumber, 4);
                    }
                    else if (scene == "rolls")
                    {
                        _rollChartPinned = true;
                    }
                    else if (scene == "hand")
                    {
                        // A big hand with playable development cards, one bought this turn and a victory point
                        foreach (ResourceCardType type in CardTypeExtensions.ResourceTypes) GiveCards(3, type, 5);
                        GameState after = new GameState(State);
                        (DevelopmentCardType type, uint amount)[] cards = [(DevelopmentCardType.Knight, 3), (DevelopmentCardType.Monopoly, 1),
                            (DevelopmentCardType.YearOfPlenty, 2), (DevelopmentCardType.VictoryPoint, 2)];
                        foreach ((DevelopmentCardType type, uint wanted) in cards)
                        {
                            uint amount = Math.Min(wanted, after.DevelopmentBank.Get(type));
                            after.Players[3].DevelopmentCards.Add(type, amount);
                            after.DevelopmentBank.Remove(type, amount);
                        }
                        after.Players[3].NewDevelopmentCards.Add(DevelopmentCardType.YearOfPlenty, 1);
                        SendEdit(new BoardEdit("gave Orange development cards", 3, State, after));
                    }
                    else if (scene is "plenty" or "monopoly")
                    {
                        // A card bought in an earlier turn, so it can be played now
                        DevelopmentCardType card = scene == "plenty" ? DevelopmentCardType.YearOfPlenty : DevelopmentCardType.Monopoly;
                        GameState after = new GameState(State);
                        after.Players[3].DevelopmentCards.Add(card, 1);
                        after.DevelopmentBank.Remove(card, 1);
                        SendEdit(new BoardEdit($"gave Orange a {card.GetName().ToLower()} card", 3, State, after));
                        _dialog = scene == "plenty" ? Dialog.YearOfPlenty : Dialog.Monopoly;
                    }
                    else if (scene == "newgame")
                    {
                        Array.Copy(Seats, _newGameSeats, PLAYER_COUNT);
                        _dialog = Dialog.NewGame;
                    }
                    else if (scene == "save") _openSaveModal = true;
                    else if (scene == "developer") _showDeveloperWindow = true;
                    else if (scene == "curtain")
                    {
                        // Hot seat: Blue is a second human at this screen, and the screen waits to be passed to Orange
                        Send(new Simulation.SetSeat(0, SeatKind.Human));
                        _curtainFor = 3;
                    }
                    else if (scene == "offer")
                    {
                        GiveCards(3, ResourceCardType.Ore, 3);
                        OpenOfferDialog(3);
                        _offerGive[4] = 2;
                        _offerGet[0] = 1;
                        _offerGet[1] = 1;
                    }
                    else if (scene == "tradepick")
                    {
                        // A generous offer to everyone; the agents answer once the scene hands the match back to them
                        GiveCards(3, ResourceCardType.Ore, 3);
                        GiveCards(3, ResourceCardType.Grain, 2);
                        CardSet<ResourceCardType> give = new(), get = new();
                        give.Add(ResourceCardType.Ore, 2);
                        give.Add(ResourceCardType.Grain, 1);
                        get.Add(ResourceCardType.Lumber, 1);
                        Send(new PlayMove(new OfferTradeAction(3, give, get, OfferTradeAction.Everyone(State, 3))));
                    }
                    else if (scene == "edit")
                    {
                        SetMode(InteractionMode.Edit);
                        _editPlayer = 1;
                    }
                    break;

                case "roll":
                    PlayUntil(() => ActingPlayer == 3 && State.Turn.MustRoll && State.Turn.RoundCounter >= 4
                        && Legal(3).Any(a => a is RollAction));
                    // The roll's flying cards and lit tiles stay: they are what the scene is for
                    ClearSceneEffects();
                    RollOrEndTurn();
                    ApplyPlaybackForMode(playing: false);
                    return;

                case "discard":
                    for (int game = 0; game < 50 && !(ActingPlayer == 3 && Legal(3).Any(a => a is DiscardAction)); game++)
                    {
                        if (game > 0) NewGame();
                        PlayUntil(() => ActingPlayer == 3 && Legal(3).Any(a => a is DiscardAction));
                    }
                    break;

                case "robber":
                    for (int game = 0; game < 50 && !(ActingPlayer == 3 && Legal(3).Any(a => a is RobberAction)); game++)
                    {
                        if (game > 0) NewGame();
                        PlayUntil(() => ActingPlayer == 3 && Legal(3).Any(a => a is RobberAction));
                    }
                    break;

                case "victim":
                    {
                        // A robber move with a tile where two or more players could be robbed
                        int? Contested() => ActingPlayer == 3 ? Legal(3).Where(a => a is RobberAction).GroupBy(a => TargetOf(a))
                            .Where(g => g.Count() > 1).Select(g => g.Key?.Index).FirstOrDefault() : null;
                        for (int game = 0; game < 50 && Contested() == null; game++)
                        {
                            if (game > 0) NewGame();
                            PlayUntil(() => Contested() != null, 20000);
                        }
                        if (Contested() is int tile)
                        {
                            _victimTile = tile;
                            _dialog = Dialog.Victim;
                        }
                        break;
                    }

                case "spectate":
                    PlayUntil(() => State.Turn.RoundCounter >= 5 && !State.Turn.MustRoll && State.Turn.TypeOfRound == TurnState.RoundType.Normal);
                    // Hard agents think for 2 s, so the live search shows up in the heat map
                    if (_valueNetAvailable)
                        for (int seat = 0; seat < 3; seat++) Send(new Simulation.SetSeat(seat, SeatKind.MctsHard));
                    SetMode(InteractionMode.Spectate);
                    ClearSceneEffects();
                    Send(new SetPlayback(new PlaybackState(false, true, 0.3f)));
                    return;

                case "end":
                    PlayUntil(() => State.HasEnded, 20000);
                    break;

                case "tradeanswer":
                    {
                        // Blue offers Orange a trade in Blue's main phase; Orange answers
                        PlayUntil(() => ActingPlayer == 0 && State.Turn.RoundCounter >= 4 && !State.Turn.MustRoll && !State.Turn.MustDiscard
                            && !State.Turn.MustMoveRobber && State.Turn.TypeOfRound == TurnState.RoundType.Normal);
                        GiveCards(0, ResourceCardType.Wool, 2);
                        GiveCards(3, ResourceCardType.Brick, 2);
                        CardSet<ResourceCardType> give = new(), get = new();
                        give.Add(ResourceCardType.Wool, 2);
                        get.Add(ResourceCardType.Brick, 1);
                        Send(new PlayMove(new OfferTradeAction(0, give, get, (byte)(1 << 3 | 1 << 2))));
                        break;
                    }
            }

            ClearSceneEffects();

            // A trade with the bank right as the scene starts, so its cards are in flight
            if (scene == "banktrade") PlayFirst<FourToOneTradeAction>(3);

            // The agents play their seats again (none in Edit mode)
            ApplyPlaybackForMode(playing: false);
        }

        /// <summary>
        /// The moves a scene played through all ran before the first frame: their flying cards and placement rings are stale
        /// </summary>
        private void ClearSceneEffects()
        {
            _particles.Clear();
            _overlay.ClearPlacements();
        }

        /// <summary>
        /// Moves cards from the bank into a hand, for scenes
        /// </summary>
        private void GiveCards(int player, ResourceCardType type, uint amount)
        {
            amount = Math.Min(amount, State.ResourceBank.Get(type));
            GameState after = new GameState(State);
            after.ResourceBank.Remove(type, amount);
            after.Players[player].ResourceCards.Add(type, amount);
            SendEdit(new BoardEdit($"gave {Theme.PlayerName(player)} {amount} {type.GetName().ToLower()}", player, State, after));
        }
    }
}
