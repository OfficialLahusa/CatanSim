using Agents;
using Common;
using Common.Actions;
using SFML.Graphics;
using Action = Common.Actions.Action;

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
            using Texture texture = new Texture(_window.Size.X, _window.Size.Y);
            texture.Update(_window);
            using Image image = texture.CopyToImage();
            image.SaveToFile(_screenshotPath);
            Console.WriteLine($"Screenshot saved to {Path.GetFullPath(_screenshotPath)}");
            _screenshotPath = null;
        }

        public static readonly string[] Scenes = ["setup", "main", "road", "roll", "discard", "trade", "devcards", "spectate", "edit", "end"];

        /// <summary>
        /// Plays fast agent moves up to a situation and sets the mode for it. Seat 4 (Orange) is the human.
        /// </summary>
        public void RunScene(string scene, int seed)
        {
            Utils.Random.SeedCurrentThread(seed);
            for (int seat = 0; seat < PLAYER_COUNT; seat++)
                _seats[seat] = seat == 3 ? SeatKind.Human : _stateValueNet != null ? SeatKind.MctsMedium : SeatKind.Builder;
            for (sbyte seat = 0; seat < PLAYER_COUNT; seat++)
                _agents[seat] = CreateAgent(_seats[seat], seat);
            NewGame();

            Agent[] fast = Enumerable.Range(0, PLAYER_COUNT).Select(i => (Agent)new BuilderAgent((sbyte)i)).ToArray();
            void PlayUntil(Func<bool> done, int maxMoves = 5000)
            {
                for (int i = 0; i < maxMoves && !done() && !_state.HasEnded; i++)
                    ApplyMove(fast[ActingPlayer].Act(_state), false);
            }
            bool HumanMainPhase(int minRound) => ActingPlayer == 3 && _state.Turn.RoundCounter >= minRound && !_state.Turn.MustRoll
                && !_state.Turn.MustDiscard && !_state.Turn.MustMoveRobber && _state.Turn.TypeOfRound == TurnState.RoundType.Normal;

            switch (scene)
            {
                case "setup":
                    PlayUntil(() => ActingPlayer == 3);
                    break;

                case "main":
                case "road":
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
                    else if (scene == "trade")
                    {
                        GiveCards(3, ResourceCardType.Lumber, 4);
                        _dialog = Dialog.Trade;
                        _tradeGive = ResourceCardType.Lumber;
                        _tradeGet = ResourceCardType.Ore;
                    }
                    else if (scene == "devcards")
                    {
                        _state.Players[3].DevelopmentCards.Add(DevelopmentCardType.Knight, 2);
                        _state.Players[3].DevelopmentCards.Add(DevelopmentCardType.Monopoly, 1);
                        _state.Players[3].NewDevelopmentCards.Add(DevelopmentCardType.YearOfPlenty, 1);
                        _state.DevelopmentBank.Remove(DevelopmentCardType.Knight, 2);
                        _state.DevelopmentBank.Remove(DevelopmentCardType.Monopoly, 1);
                        _state.DevelopmentBank.Remove(DevelopmentCardType.YearOfPlenty, 1);
                        AfterPositionChanged();
                        _dialog = Dialog.DevelopmentCard;
                    }
                    else if (scene == "edit")
                    {
                        SetMode(InteractionMode.Edit);
                        _editPlayer = 1;
                    }
                    break;

                case "roll":
                    PlayUntil(() => ActingPlayer == 3 && _state.Turn.MustRoll && _state.Turn.RoundCounter >= 4
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

                case "spectate":
                    PlayUntil(() => _state.Turn.RoundCounter >= 5 && !_state.Turn.MustRoll && _state.Turn.TypeOfRound == TurnState.RoundType.Normal);
                    // Hard agents think for 2 s, so the live search shows up in the heat map
                    for (sbyte seat = 0; seat < PLAYER_COUNT; seat++)
                        if (_stateValueNet != null) _agents[seat] = CreateAgent(seat == 3 ? SeatKind.Human : _seats[seat] = SeatKind.MctsHard, seat);
                    SetMode(InteractionMode.Spectate);
                    _spectatePlaying = true;
                    _botPaceSeconds = 0.3f;
                    break;

                case "end":
                    PlayUntil(() => _state.HasEnded, 20000);
                    break;
            }
            _sinceLastMove.Restart();
        }

        /// <summary>
        /// Moves cards from the bank into a hand, for scenes
        /// </summary>
        private void GiveCards(int player, ResourceCardType type, uint amount)
        {
            amount = Math.Min(amount, _state.ResourceBank.Get(type));
            _state.ResourceBank.Remove(type, amount);
            _state.Players[player].ResourceCards.Add(type, amount);
            AfterPositionChanged();
        }
    }
}
