using Common;
using ImGuiNET;
using System.Numerics;

namespace Client.UI
{
    /// <summary>
    /// Yields of a roll as small cards flying in an arc from the producing tile to the receiving player's hand or seat panel,
    /// with a fading streak behind them and a flash on arrival. Positions are in HUD units. Start and destination are asked for
    /// every frame: the board may move and the panels may not be laid out yet when a card starts (a roll before the first frame).
    /// </summary>
    public class Particles
    {
        private sealed class Particle
        {
            public ResourceCardType Card;
            public Func<Vector2> From = () => Vector2.Zero;
            public Func<Vector2> To = () => Vector2.Zero;
            public float Delay, Age;
        }

        private sealed class Flash
        {
            public Vector2 At;
            public SFML.Graphics.Color Color;
            public float Age;
        }

        private const float FlightSeconds = 1.1f, FlashSeconds = 0.35f, Streak = 0.12f;

        private readonly List<Particle> _particles = [];
        private readonly List<Flash> _flashes = [];

        public bool IsEmpty => _particles.Count == 0 && _flashes.Count == 0;

        /// <summary>
        /// A card flying from a point to a destination, starting after a delay
        /// </summary>
        public void Add(ResourceCardType card, Func<Vector2> from, Func<Vector2> to, float delay)
        {
            _particles.Add(new Particle { Card = card, From = from, To = to, Delay = delay });
        }

        public void Clear()
        {
            _particles.Clear();
            _flashes.Clear();
        }

        public void Update(float deltaSeconds)
        {
            foreach (Particle p in _particles) p.Age += deltaSeconds;
            foreach (Flash f in _flashes) f.Age += deltaSeconds;

            foreach (Particle p in _particles.Where(p => p.Age - p.Delay >= FlightSeconds).ToList())
            {
                _flashes.Add(new Flash { At = p.To(), Color = ColorPalette.GetCardColor(p.Card) });
                _particles.Remove(p);
            }
            _flashes.RemoveAll(f => f.Age >= FlashSeconds);
        }

        public void Draw(ImDrawListPtr dl)
        {
            foreach (Particle p in _particles)
            {
                float u = (p.Age - p.Delay) / FlightSeconds;
                if (u < 0) continue;
                Vector2 from = p.From(), to = p.To();

                float eased = 0.5f - 0.5f * MathF.Cos(MathF.PI * u);
                float opacity = u < 0.88f ? 1 : (1 - u) / 0.12f;
                SFML.Graphics.Color color = ColorPalette.GetCardColor(p.Card);

                // Streak: the last part of the arc, fading out behind the card
                const int segments = 12;
                for (int i = 0; i < segments; i++)
                {
                    float a = MathF.Max(0, eased - Streak * (i + 1) / segments), b = MathF.Max(0, eased - Streak * i / segments);
                    float fade = 1 - (float)i / segments;
                    dl.AddLine(Bezier(from, to, a), Bezier(from, to, b), Theme.U32(Theme.WithAlpha(color, 0.55f * opacity * fade)), 7 * fade + 1);
                }

                // The card grows on the way up and shrinks into the destination
                float scale = u < 0.45f ? 0.5f + 0.75f * MathF.Sin(MathF.PI * u / 0.9f) : 1.25f - 0.65f * (u - 0.45f) / 0.55f;
                Vector2 size = new Vector2(24, 32) * scale;
                Theme.ResourceCard(dl, p.Card, Bezier(from, to, eased) - size / 2, size, true, opacity);
            }

            foreach (Flash f in _flashes)
            {
                float v = f.Age / FlashSeconds;
                dl.AddCircle(f.At, 8 + 26 * v, Theme.U32(Theme.WithAlpha(f.Color, 0.9f * (1 - v))), 32, 4);
            }
        }

        /// <summary>
        /// A point on the arc above the straight line, higher for longer flights
        /// </summary>
        private static Vector2 Bezier(Vector2 from, Vector2 to, float t)
        {
            Vector2 control = (from + to) / 2 + new Vector2(0, -0.35f * Vector2.Distance(from, to));
            return (1 - t) * (1 - t) * from + 2 * (1 - t) * t * control + t * t * to;
        }
    }
}
