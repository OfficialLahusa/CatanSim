using System;

namespace Common.Serialization
{
    /// <summary>
    /// Thrown when a save file's format version does not match SaveFile.CurrentVersion
    /// </summary>
    public class UnsupportedSaveFileVersionException : Exception
    {
        public int Version { get; }

        public UnsupportedSaveFileVersionException(int version)
            : base($"Unsupported save file version {version} (this build reads version {SaveFile.CurrentVersion}).")
        {
            Version = version;
        }
    }
}
