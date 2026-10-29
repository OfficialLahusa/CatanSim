using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Simulation
{
    /// <summary>
    /// A token that admits a connection with a role (and for players and agents a seat). Only the token's hash is stored;
    /// the ID names the token in lists and for revoking it.
    /// </summary>
    public sealed record TokenEntry(string Hash, Role Role, int? Seat)
    {
        public string Id { get; init; } = "";
        public string Label { get; init; } = "";
        public DateTimeOffset Created { get; init; } = DateTimeOffset.UtcNow;

        public TokenInfo Info => new(Id, Role, Seat, Label, Created);
    }

    /// <summary>
    /// What admins see of a token: everything but the secret
    /// </summary>
    public sealed record TokenInfo(string Id, Role Role, int? Seat, string Label, DateTimeOffset Created);

    /// <summary>
    /// What admins see of the server they are connected to
    /// </summary>
    public sealed record ServerInfo(int Port, string BindAddress, bool Secure, string? CertificateFingerprint, bool AllowAnonymousSpectators, bool AllowRemoteAdmins);

    /// <summary>
    /// How a server is reached and who may connect
    /// </summary>
    public sealed record ServerConfig
    {
        /// <summary>
        /// 0 picks a free port
        /// </summary>
        public int Port { get; init; } = 17017;

        /// <summary>
        /// The address to listen on: 127.0.0.1 for this machine only, 0.0.0.0 for the network
        /// </summary>
        public string BindAddress { get; init; } = "127.0.0.1";

        public List<TokenEntry> Tokens { get; init; } = [];

        /// <summary>
        /// Whether anyone may watch without a token (admins can switch it while the server runs)
        /// </summary>
        public bool AllowAnonymousSpectators { get; set; } = true;

        /// <summary>
        /// Whether admin tokens are accepted from other computers. Off by default: without TLS the token travels in plain text,
        /// so remote admins should only be allowed together with a certificate.
        /// </summary>
        public bool AllowRemoteAdmins { get; init; }

        /// <summary>
        /// A PFX file with the server's certificate for wss:// (created as a self-signed one if missing), or null for ws://
        /// </summary>
        public string? CertificatePath { get; init; }
        public string? CertificatePassword { get; init; }

        /// <summary>
        /// A certificate given in code (a client hosting with TLS, tests); takes precedence over CertificatePath
        /// </summary>
        [JsonIgnore]
        public X509Certificate2? Certificate { get; init; }

        /// <summary>
        /// Failed logins from one address before it is locked out, and for how long
        /// </summary>
        public int MaxFailedLogins { get; init; } = 5;
        public int LockoutSeconds { get; init; } = 300;

        public int MaxConnections { get; init; } = 64;

        /// <summary>
        /// The largest message a client may send (a loaded game with its history is far below this)
        /// </summary>
        public int MaxMessageBytes { get; init; } = 8 * 1024 * 1024;

        public static string HashToken(string token) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(token))).ToLowerInvariant();

        /// <summary>
        /// A random token with 256 bits, URL-safe. A fast hash is enough for tokens this long: there is nothing to guess.
        /// </summary>
        public static string NewToken() => Base64Url(RandomNumberGenerator.GetBytes(32));

        private static string NewId() => Base64Url(RandomNumberGenerator.GetBytes(6));

        private static string Base64Url(byte[] bytes) => Convert.ToBase64String(bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=');

        /// <summary>
        /// The token entry a token matches, compared in constant time
        /// </summary>
        public TokenEntry? Find(string? token)
        {
            if (string.IsNullOrEmpty(token) || token.Length > 256) return null;
            byte[] hash = Encoding.ASCII.GetBytes(HashToken(token));
            lock (Tokens)
            {
                TokenEntry? found = null;
                // Every entry is compared, so the time doesn't tell which one matched
                foreach (TokenEntry entry in Tokens)
                    if (CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(entry.Hash), hash)) found ??= entry;
                return found;
            }
        }

        /// <summary>
        /// Adds a new token for a role; the token itself is returned once and only its hash kept
        /// </summary>
        public (TokenEntry entry, string token) AddToken(Role role, int? seat, string? label)
        {
            if (!Enum.IsDefined(role)) throw new ArgumentException($"Unknown role {role}");
            if (role is Role.Player or Role.Agent && seat is not (>= 0 and < 4)) throw new ArgumentException($"A {role.ToString().ToLower()} token needs a seat (1-4).");
            if (role is Role.Admin or Role.Spectator) seat = null;

            string token = NewToken();
            TokenEntry entry = new TokenEntry(HashToken(token), role, seat) { Id = NewId(), Label = CleanLabel(label) };
            lock (Tokens) Tokens.Add(entry);
            return (entry, token);
        }

        /// <summary>
        /// Removes a token; the hash of the removed one, or null if there was none with this ID
        /// </summary>
        public string? RemoveToken(string id)
        {
            lock (Tokens)
            {
                int index = Tokens.FindIndex(t => t.Id == id);
                if (index < 0) return null;
                string hash = Tokens[index].Hash;
                Tokens.RemoveAt(index);
                return hash;
            }
        }

        public List<TokenInfo> TokenInfos()
        {
            lock (Tokens) return Tokens.Select(t => t.Info).ToList();
        }

        /// <summary>
        /// A name or label from outside: no control characters, at most 40 characters
        /// </summary>
        public static string CleanLabel(string? text, int maxLength = 40)
        {
            if (text == null) return "";
            // No control, formatting (zero-width, bidirectional overrides) or line separator characters
            string clean = new string(text.Where(c => char.GetUnicodeCategory(c) is not (System.Globalization.UnicodeCategory.Control
                or System.Globalization.UnicodeCategory.Format or System.Globalization.UnicodeCategory.LineSeparator
                or System.Globalization.UnicodeCategory.ParagraphSeparator)).ToArray()).Trim();
            return clean.Length > maxLength ? clean[..maxLength].TrimEnd() : clean;
        }

        private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };

        /// <summary>
        /// Loads a configuration; tokens from older files get IDs, which are saved right away so they stay the same
        /// </summary>
        public static ServerConfig Load(string path)
        {
            ServerConfig config = JsonSerializer.Deserialize<ServerConfig>(File.ReadAllText(path), JsonOptions) ?? new ServerConfig();
            if (config.Tokens.Any(t => string.IsNullOrEmpty(t.Id)))
            {
                for (int i = 0; i < config.Tokens.Count; i++)
                    if (string.IsNullOrEmpty(config.Tokens[i].Id)) config.Tokens[i] = config.Tokens[i] with { Id = NewId() };
                config.Save(path);
            }
            return config;
        }

        /// <summary>
        /// Saves the configuration, readable only by its owner where the file system has permissions
        /// </summary>
        public void Save(string path)
        {
            string json;
            lock (Tokens) json = JsonSerializer.Serialize(this, JsonOptions);
            WritePrivate(path, Encoding.UTF8.GetBytes(json));
        }

        /// <summary>
        /// Writes a file that only its owner may read (on Unix; Windows keeps the profile's permissions)
        /// </summary>
        internal static void WritePrivate(string path, byte[] content)
        {
            string temp = path + ".tmp";
            File.Delete(temp);
            if (OperatingSystem.IsWindows())
            {
                File.WriteAllBytes(temp, content);
            }
            else
            {
                using FileStream stream = new FileStream(temp, new FileStreamOptions
                {
                    Mode = FileMode.Create, Access = FileAccess.Write, UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite
                });
                stream.Write(content);
            }
            File.Move(temp, path, true);
        }
    }
}
