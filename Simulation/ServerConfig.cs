using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Simulation
{
    /// <summary>
    /// A token that admits a connection with a role (and for players and agents a seat). Only the token's hash is stored.
    /// </summary>
    public sealed record TokenEntry(string Hash, Role Role, int? Seat);

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
        /// Whether anyone may watch without a token
        /// </summary>
        public bool AllowAnonymousSpectators { get; init; } = true;

        public static string HashToken(string token) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(token))).ToLowerInvariant();

        /// <summary>
        /// A random token, e.g. for the players of a hosted game
        /// </summary>
        public static string NewToken() => Convert.ToBase64String(RandomNumberGenerator.GetBytes(12)).Replace('+', 'x').Replace('/', 'y').TrimEnd('=');

        /// <summary>
        /// The token entry a token matches, compared in constant time
        /// </summary>
        public TokenEntry? Find(string? token)
        {
            if (string.IsNullOrEmpty(token)) return null;
            byte[] hash = Encoding.ASCII.GetBytes(HashToken(token));
            return Tokens.FirstOrDefault(t => CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(t.Hash), hash));
        }

        private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true, Converters = { new JsonStringEnumConverter() } };

        public static ServerConfig Load(string path) => JsonSerializer.Deserialize<ServerConfig>(File.ReadAllText(path), JsonOptions) ?? new ServerConfig();

        public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));
    }
}
