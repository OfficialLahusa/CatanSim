using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ImGuiNET;
using System;
using System.Collections.Generic;
using System.IO;
using Num = System.Numerics;
using SFML.Graphics;

namespace Client
{
    /// <summary>
    /// Source: https://gist.github.com/prime31/91d1582624eb2635395417393018016e
    /// (Adapted for current use case)
    /// </summary>
    public class FilePicker
    {
        static readonly Dictionary<object, FilePicker> _filePickers = new Dictionary<object, FilePicker>();

        // Key this picker is registered under, so that it can unregister itself when cancelled
        private object? _owner;

        public bool SaveMode;
        public string RootFolder;
        public string CurrentFolder;
        public string SelectedFile;
        public string OutputFileName;
        public List<string> AllowedExtensions;
        public bool OnlyAllowFolders;

        // Why the current folder could not be listed (no access, removed meanwhile, drive not ready)
        private string? _listError;

        public static FilePicker GetFolderPicker(object o, string startingPath, bool saveMode = false)
            => GetFilePicker(o, startingPath, saveMode, null, true);

        public static FilePicker GetFilePicker(object o, string startingPath, bool saveMode = false, string? searchFilter = null, bool onlyAllowFolders = false, string? defaultOutputFileName = null)
        {
            if (File.Exists(startingPath))
            {
                startingPath = new FileInfo(startingPath).DirectoryName!;
            }
            else if (string.IsNullOrEmpty(startingPath) || !Directory.Exists(startingPath))
            {
                startingPath = Environment.CurrentDirectory;
                if (string.IsNullOrEmpty(startingPath))
                    startingPath = AppContext.BaseDirectory;
            }

            if (!_filePickers.TryGetValue(o, out FilePicker fp))
            {
                fp = new FilePicker();
                fp._owner = o;
                fp.RootFolder = startingPath;
                fp.CurrentFolder = startingPath;
                fp.OnlyAllowFolders = onlyAllowFolders;
                fp.SaveMode = saveMode;
                fp.OutputFileName = defaultOutputFileName ?? string.Empty;

                if (searchFilter != null)
                {
                    if (fp.AllowedExtensions != null)
                        fp.AllowedExtensions.Clear();
                    else
                        fp.AllowedExtensions = new List<string>();

                    fp.AllowedExtensions.AddRange(searchFilter.Split(new char[] { '|' }, StringSplitOptions.RemoveEmptyEntries));
                }

                _filePickers.Add(o, fp);
            }

            return fp;
        }

        public static void RemoveFilePicker(object o) => _filePickers.Remove(o);

        public bool Draw()
        {
            ImGui.AlignTextToFramePadding();
            ImGui.Text("Current Folder: " + CurrentFolder.Split(Path.DirectorySeparatorChar).Last()); //Path.GetFileName(RootFolder) + CurrentFolder.Replace(RootFolder, ""));
            if (_listError != null) ImGui.TextColored(ColorPalette.ColorToVec4(new Color(255, 120, 120)), _listError);
            bool result = false;

            if (ImGui.BeginChildFrame(1, new Num.Vector2(400, 400)))
            {
                var di = new DirectoryInfo(CurrentFolder);
                if (di.Exists)
                {
                    if (di.Parent != null)// && CurrentFolder != RootFolder)
                    {
                        ImGui.PushStyleColor(ImGuiCol.Text, ColorPalette.ColorToVec4(Color.Yellow));
                        if (ImGui.Selectable("../", false, ImGuiSelectableFlags.DontClosePopups))
                            CurrentFolder = di.Parent.FullName;

                        ImGui.PopStyleColor();
                    }

                    var fileSystemEntries = GetFileSystemEntries(di.FullName);
                    foreach (var fse in fileSystemEntries)
                    {
                        if (Directory.Exists(fse))
                        {
                            var name = Path.GetFileName(fse);
                            ImGui.PushStyleColor(ImGuiCol.Text, ColorPalette.ColorToVec4(Color.Yellow));
                            if (ImGui.Selectable(name + "/", false, ImGuiSelectableFlags.DontClosePopups))
                                CurrentFolder = fse;
                            ImGui.PopStyleColor();
                        }
                        else
                        {
                            var name = Path.GetFileName(fse);
                            bool isSelected = SelectedFile == fse;
                            if (ImGui.Selectable(name, isSelected, ImGuiSelectableFlags.DontClosePopups))
                            {
                                SelectedFile = fse;
                                OutputFileName = Path.GetFileName(fse);
                            }
                                

                            // Only a double-click on this file entry confirms the dialog
                            // (a double-click elsewhere in the list, e.g. on a folder, must not)
                            if (ImGui.IsItemHovered() && ImGui.IsMouseDoubleClicked(0))
                            {
                                SelectedFile = fse;
                                OutputFileName = Path.GetFileName(fse);
                                result = true;
                                ImGui.CloseCurrentPopup();
                            }
                        }
                    }
                }
            }
            ImGui.EndChildFrame();

            if (SaveMode)
            {
                ImGui.PushItemWidth(325);
                ImGui.InputText("Filename", ref OutputFileName, 255);
                ImGui.PopItemWidth();
            }

            if (ImGui.Button("Cancel"))
            {
                result = false;
                ImGui.CloseCurrentPopup();

                // Unregister, otherwise the next dialog with the same owner reuses this picker's mode and selection
                if (_owner != null)
                    _filePickers.Remove(_owner);
            }

            if (OnlyAllowFolders)
            {
                ImGui.SameLine();
                if (ImGui.Button(SaveMode ? "Save" : "Load"))
                {
                    result = true;
                    SelectedFile = CurrentFolder;
                    ImGui.CloseCurrentPopup();
                }
            }
            else if (SelectedFile != null || (OutputFileName != null && OutputFileName.Length > 0))
            {
                ImGui.SameLine();
                if (ImGui.Button(SaveMode ? "Save" : "Load"))
                {
                    result = true;
                    ImGui.CloseCurrentPopup();
                }
            }

            return result;
        }

        public string GetSaveFilePath()
        {
            if (!SaveMode)
                throw new InvalidOperationException("FilePicker is not in SaveMode.");

            if (OnlyAllowFolders)
                return CurrentFolder;

            return Path.Combine(CurrentFolder, OutputFileName);
        }

        bool TryGetFileInfo(string fileName, out FileInfo realFile)
        {
            try
            {
                realFile = new FileInfo(fileName);
                return true;
            }
            catch
            {
                realFile = null;
                return false;
            }
        }

        List<string> GetFileSystemEntries(string fullName)
        {
            var files = new List<string>();
            var dirs = new List<string>();

            string[] entries;
            try
            {
                entries = Directory.GetFileSystemEntries(fullName, "");
                _listError = null;
            }
            catch (Exception e) when (e is UnauthorizedAccessException or IOException)
            {
                _listError = $"This folder can't be listed: {e.Message}";
                return files;
            }

            foreach (var fse in entries)
            {
                if (Directory.Exists(fse))
                {
                    dirs.Add(fse);
                }
                else if (!OnlyAllowFolders)
                {
                    if (AllowedExtensions != null)
                    {
                        var ext = Path.GetExtension(fse);
                        if (AllowedExtensions.Contains(ext))
                            files.Add(fse);
                    }
                    else
                    {
                        files.Add(fse);
                    }
                }
            }

            var ret = new List<string>(dirs);
            ret.AddRange(files);

            return ret;
        }

    }
}