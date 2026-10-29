using System.Net;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Simulation
{
    /// <summary>
    /// Certificates for wss://: a self-signed one the server creates on first start, and the fingerprint clients pin instead
    /// of trusting a certificate authority
    /// </summary>
    public static class ServerCertificates
    {
        /// <summary>
        /// A self-signed server certificate for this machine (localhost, its name and loopback addresses), valid for ten years
        /// </summary>
        public static X509Certificate2 CreateSelfSigned()
        {
            // Through PFX, since TLS on Windows can't use the ephemeral key of a freshly created certificate
            return X509CertificateLoader.LoadPkcs12(NewPfx(null), null);
        }

        /// <summary>
        /// The certificate in a PFX file, created there first if the file doesn't exist
        /// </summary>
        public static X509Certificate2 LoadOrCreate(string path, string? password = null)
        {
            if (!File.Exists(path)) ServerConfig.WritePrivate(path, NewPfx(password));
            return X509CertificateLoader.LoadPkcs12FromFile(path, password);
        }

        private static byte[] NewPfx(string? password)
        {
            using ECDsa key = ECDsa.Create(ECCurve.NamedCurves.nistP256);
            CertificateRequest request = new CertificateRequest("CN=CatanSim server", key, HashAlgorithmName.SHA256);
            SubjectAlternativeNameBuilder names = new SubjectAlternativeNameBuilder();
            names.AddDnsName("localhost");
            names.AddDnsName(Environment.MachineName);
            names.AddIpAddress(IPAddress.Loopback);
            names.AddIpAddress(IPAddress.IPv6Loopback);
            request.CertificateExtensions.Add(names.Build());
            request.CertificateExtensions.Add(new X509BasicConstraintsExtension(false, false, 0, true));
            request.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.DigitalSignature, true));
            request.CertificateExtensions.Add(new X509EnhancedKeyUsageExtension([new Oid("1.3.6.1.5.5.7.3.1")], false));

            using X509Certificate2 created = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddYears(10));
            return created.Export(X509ContentType.Pfx, password);
        }

        /// <summary>
        /// The SHA-256 fingerprint of a certificate as lowercase hex, the form clients pin
        /// </summary>
        public static string Fingerprint(X509Certificate certificate) => Convert.ToHexString(SHA256.HashData(certificate.GetRawCertData())).ToLowerInvariant();

        /// <summary>
        /// A fingerprint as typed or pasted (with colons, spaces, any case) in the form Fingerprint returns
        /// </summary>
        public static string NormalizeFingerprint(string fingerprint) => new string(fingerprint.Where(Uri.IsHexDigit).ToArray()).ToLowerInvariant();
    }
}
