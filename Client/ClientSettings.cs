using System.Text.Json;

namespace Client
{
    /// <summary>
    /// The user's settings from the main menu, kept in %AppData%\Hexfarer\settings.json. A missing or unreadable file gives the
    /// defaults, so the client always starts.
    /// </summary>
    public sealed class ClientSettings
    {
        public bool Sounds { get; set; } = true;

        // Scales the volume of every sound effect, 0 to 1
        public float Volume { get; set; } = 1f;

        // Cards flying between the board, the hands and the bank
        public bool Particles { get; set; } = true;

        // Pause between two agent moves in a local game, in seconds, so they can be followed
        public float BotPace { get; set; } = 0.6f;

        private static string FilePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Hexfarer", "settings.json");

        private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

        public static ClientSettings Load()
        {
            try
            {
                if (File.Exists(FilePath) && JsonSerializer.Deserialize<ClientSettings>(File.ReadAllText(FilePath), Options) is ClientSettings settings)
                {
                    settings.Volume = Math.Clamp(settings.Volume, 0, 1);
                    settings.BotPace = Math.Clamp(settings.BotPace, 0, 3);
                    return settings;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not read the settings, using the defaults: {e.Message}");
            }
            return new ClientSettings();
        }

        /// <summary>
        /// Writes the settings; why that failed, or null
        /// </summary>
        public string? Save()
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
                File.WriteAllText(FilePath, JsonSerializer.Serialize(this, Options));
                return null;
            }
            catch (Exception e)
            {
                Console.WriteLine($"Could not save the settings: {e.Message}");
                return e.Message;
            }
        }
    }
}
