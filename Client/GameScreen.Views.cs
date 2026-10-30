using Client.UI;
using Common;
using ImGuiNET;
using Simulation;
using System.Numerics;

namespace Client
{
    /// <summary>
    /// Whose knowledge the screen shows (see BeliefTracker): in Play mode the human's, so the other hands show their size or,
    /// if the table shows them (GameSettings.ShowTrackedCards), the cards confirmed by what happened in public; in Spectate
    /// mode every hand, none, or what any seat knows
    /// </summary>
    public partial class GameScreen
    {
        // Spectate mode's view: null shows every hand, BeliefTracker.Public none, a seat what that seat knows
        private int? _spectateObserver;

        // Whether a new game started from here shows the confirmed cards (GameSettings.ShowTrackedCards)
        private bool _showTrackedCards;

        // The view of the shown observer, kept until the position or the observer changes
        private GameState? _shownView;
        private int _shownVersion = -1;
        private int? _shownViewObserver;

        /// <summary>
        /// Whose knowledge the panels show: null everything, BeliefTracker.Public what everyone sees, a seat what it knows
        /// </summary>
        private int? ShownObserver => _mode switch
        {
            InteractionMode.Play => Viewer is int viewer ? Math.Max(viewer, BeliefTracker.Public) : null,
            InteractionMode.Spectate => _spectateObserver,
            _ => null
        };

        /// <summary>
        /// The position as the shown observer knows it: the full one, or its view from the trackers. A player following its
        /// seat's view over the network has only that view.
        /// </summary>
        private GameState Shown
        {
            get
            {
                if (ShownObserver is not int observer || _sim.Game.Beliefs is not RecordBeliefs beliefs || State.HasEnded) return State;
                if (_shownView == null || _shownVersion != _stateVersion || _shownViewObserver != observer)
                {
                    _shownView = beliefs.View(State, observer);
                    _shownVersion = _stateVersion;
                    _shownViewObserver = observer;
                }
                return _shownView;
            }
        }

        /// <summary>
        /// What the shown observer knows of a seat's unknown cards, if its tracker is at hand
        /// </summary>
        private HandBelief? ShownBelief(int seat) => ShownObserver is int observer && _sim.Game.Beliefs is RecordBeliefs beliefs ? beliefs.For(observer).Hand(seat) : null;

        /// <summary>
        /// How a seat's hand is shown: every card, the confirmed cards and the unknown ones, or only its size
        /// </summary>
        private (bool full, bool tracked) HandShown(int seat)
        {
            int? observer = ShownObserver;
            bool full = observer == null || observer == seat;
            bool tracked = !full && observer >= 0 && (_mode == InteractionMode.Spectate || State.Settings.ShowTrackedCards);
            return (full, tracked);
        }

        /// <summary>
        /// Sets Spectate mode's view by name (all, hidden, or a seat 1-4), for --view
        /// </summary>
        public void SetSpectateView(string view) => _spectateObserver = view switch
        {
            "hidden" => BeliefTracker.Public,
            _ when int.TryParse(view, out int seat) && seat is >= 1 and <= PLAYER_COUNT => seat - 1,
            _ => null
        };

        private static string ObserverName(int? observer) => observer switch
        {
            null => "All hands",
            BeliefTracker.Public => "Hands hidden",
            int seat => $"{Theme.PlayerName(seat)}'s view"
        };

        /// <summary>
        /// The view picker of Spectate mode in the top bar: every hand, none, or what a seat knows; returns its width
        /// </summary>
        private float DrawViewPicker(ImDrawListPtr dl, Vector2 min)
        {
            string label = ObserverName(_spectateObserver);
            float width = 40 + Theme.TextSize(Theme.Regular15, label).X;
            if (Theme.HitArea("viewpicker", min, new Vector2(width, 32), out bool hovered)) ImGui.OpenPopup("view-picker");
            dl.RoundRectFilled(min, min + new Vector2(width, 32), Theme.U32(hovered ? Theme.PanelRaised : Theme.Panel), 16);
            dl.RoundRect(min, min + new Vector2(width, 32), Theme.U32(Theme.Border), 16);
            Theme.DrawText(dl, Theme.Regular15, min + new Vector2(14, 6), Theme.TextSoft, label);
            Vector2 arrow = min + new Vector2(width - 18, 14);
            dl.AddTriangleFilled(arrow + new Vector2(-5, -2), arrow + new Vector2(5, -2), arrow + new Vector2(0, 4), Theme.U32(Theme.TextMuted));

            if (ImGui.BeginPopup("view-picker"))
            {
                foreach (int? observer in new int?[] { null, BeliefTracker.Public, 0, 1, 2, 3 })
                {
                    if (ImGui.MenuItem(ObserverName(observer), "", _spectateObserver == observer)) _spectateObserver = observer;
                }
                ImGui.EndPopup();
            }
            return width;
        }

        /// <summary>
        /// A tooltip over a hand with unknown cards: how many, and how many of each type they can be at most
        /// </summary>
        private static string UnknownCardsText(HandBelief belief)
        {
            IEnumerable<string> types = CardTypeExtensions.ResourceTypes.Where(belief.CanBe)
                .Select(type => $"{belief.Room(type)} {type.GetName().ToLower()}");
            string cards = belief.Unknown == 1 ? "1 unknown card" : $"{belief.Unknown} unknown cards";
            return $"{cards}, at most {string.Join(", ", types)}";
        }
    }
}
