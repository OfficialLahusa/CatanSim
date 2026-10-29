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

        public static readonly string[] Scenes = ["setup", "main", "road", "city", "roll", "discard", "robber", "trade", "devcards", "spectate", "edit", "end"];

        /// <summary>
        /// Plays fast agent moves up to a situation and sets the mode for it. Seat 4 (Orange) is the human.
        /// </summary>
        public void RunScene(string scene, int seed)
        {
            // The scene plays its own moves; the match's agents wait meanwhile and take over at the end
            Utils.Random.SeedCurrentThread(seed);
            SeatKind bot = _stateValueNet != null ? SeatKind.MctsMedium : SeatKind.Builder;
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
                case "edit":
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
                    else if (scene == "edit")
                    {
                        SetMode(InteractionMode.Edit);
                        _editPlayer = 1;
                    }
                    break;

                case "roll":
                    PlayUntil(() => ActingPlayer == 3 && State.Turn.MustRoll && State.Turn.RoundCounter >= 4
                        && Legal(3).Any(a => a is RollAction));
                    RollOrEndTurn();
                    break;

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

                case "spectate":
                    PlayUntil(() => State.Turn.RoundCounter >= 5 && !State.Turn.MustRoll && State.Turn.TypeOfRound == TurnState.RoundType.Normal);
                    // Hard agents think for 2 s, so the live search shows up in the heat map
                    if (_stateValueNet != null)
                        for (int seat = 0; seat < 3; seat++) Send(new Simulation.SetSeat(seat, SeatKind.MctsHard));
                    SetMode(InteractionMode.Spectate);
                    Send(new SetPlayback(new PlaybackState(false, true, 0.3f)));
                    return;
                    break;

                case "end":
                    PlayUntil(() => State.HasEnded, 20000);
                    break;
            }

            // The agents play their seats again (none in Edit mode)
            ApplyPlaybackForMode(playing: false);
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
