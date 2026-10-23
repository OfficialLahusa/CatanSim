using Agents;
using Common;
using System.Globalization;

namespace DatasetCollector
{
    public enum AgentKind : byte
    {
        Random = 0,
        Simple = 1,
        Builder = 2,
    }

    /// <summary>
    /// The population of agents for matches and playouts: every seat draws its agent kind by weight, and SimpleAgents and
    /// BuilderAgents draw an exploration rate (share of random actions) between the minimum and maximum.
    /// Pure random win rates undervalue plans RandomAgents never carry out (cities, development cards, ports).
    /// </summary>
    public sealed record AgentMix(double RandomWeight, double SimpleWeight, double BuilderWeight, double MinExploration, double MaxExploration)
    {
        public static readonly AgentMix Default = new(0.30, 0.35, 0.35, 0.10, 0.20);

        private static readonly char[] KindLetters = ['R', 'S', 'B'];

        public void Validate()
        {
            double[] weights = [RandomWeight, SimpleWeight, BuilderWeight];
            if (weights.Any(weight => weight < 0 || !double.IsFinite(weight)) || weights.Sum() <= 0)
                throw new ArgumentException("Agent weights must be non-negative and not all zero.");
            if (MinExploration < 0 || MaxExploration > 1 || MinExploration > MaxExploration)
                throw new ArgumentException("Exploration rates must satisfy 0 <= min <= max <= 1.");
        }

        public AgentKind DrawKind()
        {
            double draw = Utils.Random.NextDouble() * (RandomWeight + SimpleWeight + BuilderWeight);
            if (draw < RandomWeight) return AgentKind.Random;
            if (draw < RandomWeight + SimpleWeight || BuilderWeight == 0) return AgentKind.Simple;
            return AgentKind.Builder;
        }

        /// <summary>
        /// Draws the agent kind of every seat into kinds and creates the agents
        /// </summary>
        public Agent[] Draw(Span<AgentKind> kinds)
        {
            Agent[] agents = new Agent[kinds.Length];
            for (sbyte seat = 0; seat < kinds.Length; seat++)
            {
                kinds[seat] = DrawKind();
                agents[seat] = Create(kinds[seat], seat);
            }
            return agents;
        }

        public Agent Create(AgentKind kind, sbyte seat) => kind switch
        {
            AgentKind.Random => new RandomAgent(seat),
            AgentKind.Simple => new SimpleAgent(seat, DrawExploration()),
            AgentKind.Builder => new BuilderAgent(seat, DrawExploration()),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private double DrawExploration()
        {
            return MinExploration + Utils.Random.NextDouble() * (MaxExploration - MinExploration);
        }

        /// <summary>
        /// One letter per seat, e.g. "RSBB"
        /// </summary>
        public static string Letters(ReadOnlySpan<AgentKind> kinds)
        {
            Span<char> letters = stackalloc char[kinds.Length];
            for (int i = 0; i < kinds.Length; i++) letters[i] = KindLetters[(int)kinds[i]];
            return new string(letters);
        }

        /// <summary>
        /// Parses weights like "random=0.3,simple=0.35,builder=0.35" (kinds left out get weight 0) and exploration like "0.1-0.2" or "0.15"
        /// </summary>
        public static AgentMix Parse(string weights, string exploration)
        {
            double random = 0, simple = 0, builder = 0;
            foreach (string part in weights.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                string[] pair = part.Split('=', 2, StringSplitOptions.TrimEntries);
                if (pair.Length != 2) throw new ArgumentException($"Expected kind=weight, got \"{part}\".");
                double weight = double.Parse(pair[1], CultureInfo.InvariantCulture);
                switch (pair[0].ToLowerInvariant())
                {
                    case "random": random = weight; break;
                    case "simple": simple = weight; break;
                    case "builder": builder = weight; break;
                    default: throw new ArgumentException($"Unknown agent kind \"{pair[0]}\" (random, simple, builder).");
                }
            }

            string[] range = exploration.Split('-', 2, StringSplitOptions.TrimEntries);
            double min = double.Parse(range[0], CultureInfo.InvariantCulture);
            double max = range.Length == 2 ? double.Parse(range[1], CultureInfo.InvariantCulture) : min;

            AgentMix mix = new(random, simple, builder, min, max);
            mix.Validate();
            return mix;
        }

        public override string ToString()
        {
            return string.Create(CultureInfo.InvariantCulture,
                $"random={RandomWeight},simple={SimpleWeight},builder={BuilderWeight}, exploration {MinExploration}-{MaxExploration}");
        }
    }
}
