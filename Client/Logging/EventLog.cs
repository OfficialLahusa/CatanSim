using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Common;
using ImGuiNET;
using SFML.Graphics;

namespace Client.Logging
{
    public class EventLog
    {
        // Closed sections
        private List<List<ILogEntry>> _sections;

        // Open section
        private List<ILogEntry> _lines;

        // Frames left in which the log scrolls to its end
        private int _scrollToBottom = 0;

        /// <summary>
        /// The player whose hidden information the log may show, or null for all of it
        /// </summary>
        public static int? Viewer { get; set; }

        public EventLog()
        {
            _sections = new List<List<ILogEntry>>();
            _lines = new List<ILogEntry>();
        }

        public void Draw()
        {
            Draw(new Vector2(250, 350), true);
        }

        public void Draw(Vector2 size, bool border)
        {
            ImGui.BeginChild("ScrollLog", size, border, ImGuiWindowFlags.NavFlattened);

            // Draw closed sections
            foreach(List<ILogEntry> section in _sections)
            {
                foreach (ILogEntry line in section)
                {
                    line.Draw();
                }
            }

            // Draw open section
            foreach (ILogEntry line in _lines)
            {
                line.Draw();
            }

            // A scroll target far past the end, which ImGui clamps to the end of the content at the next frame's begin. For
            // two frames, since the end the first frame clamps to was measured before the newest lines were laid out.
            if (_scrollToBottom > 0)
            {
                ImGui.SetScrollY(1e9f);

                _scrollToBottom--;
            }

            ImGui.EndChild();
        }

        public void WriteLine(params ILogEntry[] entries)
        {
            if (entries.Length == 1)
            {
                _lines.Add(entries[0]);
            }
            else if (entries.Length > 1)
            {
                _lines.Add(new ListEntry(entries));
            }

            _scrollToBottom = 2;
        }

        public void PushSection()
        {
            _sections.Add(_lines);
            _lines = new();
        }

        /// <summary>
        /// Removes the section of the last logged action (on undo). Lines written since then belong to no action and stay.
        /// </summary>
        public void PopSection()
        {
            if (_sections.Count > 0)
            {
                _sections.RemoveAt(_sections.Count - 1);
            }
        }

        public void Clear()
        {
            _sections.Clear();
            _lines.Clear();
        }
    }
}
