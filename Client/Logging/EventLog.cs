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

        private bool _scrollToBottom = false;

        public EventLog()
        {
            _sections = new List<List<ILogEntry>>();
            _lines = new List<ILogEntry>();
        }

        public void Draw()
        {
            ImGui.BeginChild("ScrollLog", new Vector2(250, 350), true, ImGuiWindowFlags.NavFlattened);

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

            if (_scrollToBottom)
            {
                ImGui.SetScrollHereY();

                _scrollToBottom = false;
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

            _scrollToBottom = true;
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
