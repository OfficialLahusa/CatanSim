using ImGuiNET;
using SFML.Graphics;

namespace Client.Logging
{
    /// <summary>
    /// A log line made of colored pieces, e.g. a player name in its color followed by what happened. A line with hidden
    /// information (a stolen card, a drawn development card) names it only for the players who know it and shows the public
    /// version to everyone else.
    /// </summary>
    public class SentenceEntry : ILogEntry
    {
        private readonly (string text, Color color)[] _segments;
        private readonly (string text, Color color)[]? _publicSegments;
        private readonly int[] _knownTo;

        public SentenceEntry((string text, Color color)[] segments, (string text, Color color)[]? publicSegments = null, params int[] knownTo)
        {
            _segments = segments;
            _publicSegments = publicSegments;
            _knownTo = knownTo;
        }

        private (string text, Color color)[] Visible =>
            _publicSegments == null || EventLog.Viewer is not int viewer || _knownTo.Contains(viewer) ? _segments : _publicSegments;

        public void Draw()
        {
            (string text, Color color)[] segments = Visible;
            for (int i = 0; i < segments.Length; i++)
            {
                if (i > 0) ImGui.SameLine(0, 0);
                ImGui.PushStyleColor(ImGuiCol.Text, ColorPalette.ColorToVec4(segments[i].color));
                ImGui.TextUnformatted(segments[i].text);
                ImGui.PopStyleColor();
            }
        }

        public string GetText() => string.Concat(Visible.Select(s => s.text));
    }
}
