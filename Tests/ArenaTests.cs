using Agents.Inference;
using Agents.MCTS;
using Arena;

namespace Tests
{
    /// <summary>
    /// The Arena duel tool: agent specs, the command line, the pair statistics with the SPRT, and a short mirror duel of two
    /// tiny untrained nets (only the plumbing, not playing strength)
    /// </summary>
    public class ArenaTests
    {
        private static string PolicyModelPath => TestUtil.FixturePath("ValueNet", "policy_tiny.onnx");
        private static string PlainModelPath => TestUtil.FixturePath("ValueNet", "placements_tiny.onnx");

        [Fact]
        public void AnAgentSpecWithoutOptionsHasTheDefaults()
        {
            AgentSpec spec = AgentSpec.Parse("I2=nets/i2.onnx");

            Assert.Equal("I2", spec.Name);
            Assert.Equal("nets/i2.onnx", spec.ModelPath);
            Assert.Equal(OpponentModel.MaxN, spec.OpponentModel);
            Assert.Equal(1.414, spec.Exploration);
            Assert.Equal(MCTSAgent.DefaultLeavesPerBatch, spec.LeavesPerBatch);
            Assert.Equal(0.1, spec.FirstPlayReduction);
            Assert.Equal(0, spec.PlacementWeight);
            Assert.False(spec.UniformPriors);
            Assert.Equal("I2 i2.onnx MaxN", spec.ToString());
        }

        [Fact]
        public void AnAgentSpecReadsItsOptions()
        {
            AgentSpec spec = AgentSpec.Parse("c1=a.onnx=Paranoid=c1.0=b64=f0.25=p0.5=u");

            Assert.Equal(OpponentModel.Paranoid, spec.OpponentModel);
            Assert.Equal(1.0, spec.Exploration);
            Assert.Equal(64, spec.LeavesPerBatch);
            Assert.Equal(0.25, spec.FirstPlayReduction);
            Assert.Equal(0.5, spec.PlacementWeight);
            Assert.True(spec.UniformPriors);
            Assert.Equal("c1 a.onnx Paranoid c1 b64 f0.25 p0.5 u", spec.ToString());
        }

        [Fact]
        public void AnAgentSpecReadsItsHiddenInformationSearch()
        {
            Assert.Equal(4, AgentSpec.Parse("P=a.onnx=d4").Determinizations);
            AgentSpec spec = AgentSpec.Parse("S=a.onnx=is");
            Assert.True(spec.InformationSets);
            Assert.Equal("S a.onnx MaxN is", spec.ToString());
            Assert.False(spec.FullKnowledge);
            AgentSpec full = AgentSpec.Parse("F=a.onnx=isfull");
            Assert.True(full.InformationSets && full.FullKnowledge);
            Assert.Equal("F a.onnx MaxN isfull", full.ToString());
            AgentSpec ablation = AgentSpec.Parse("A=a.onnx=isfull=ob=norm");
            Assert.True(ablation.BranchHiddenOutcomes);
            Assert.True(ablation.NormalizeValues);
            Assert.Equal("A a.onnx MaxN isfull norm ob", ablation.ToString());
            Assert.False(AgentSpec.Parse("M=a.onnx=nonorm").NormalizeValues);
            Assert.Null(AgentSpec.Parse("M=a.onnx").NormalizeValues);
            AgentSpec fresh = AgentSpec.Parse("N=a.onnx=noreuse");
            Assert.False(fresh.ReuseTree);
            Assert.Equal("N a.onnx MaxN noreuse", fresh.ToString());
            Assert.True(AgentSpec.Parse("L=a.onnx=leaf").EvaluateNewLeaves);
        }

        [Theory]
        [InlineData("a.onnx")]
        [InlineData("=a.onnx")]
        [InlineData("A=")]
        [InlineData("A=a.onnx=x3")]
        public void AnInvalidAgentSpecIsRejected(string spec)
        {
            Assert.ThrowsAny<Exception>(() => AgentSpec.Parse(spec));
        }

        [Fact]
        public void TheCommandLineSetsTheDuel()
        {
            MirrorSettings settings = CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx=c1.0", "--iterations", "800",
                "--parallel", "8", "--max-games", "1000", "--name", "loop_g009", "--sprt", "-10", "20", "--max-minutes", "90", "--provider", "cuda"]).Settings!;

            Assert.Equal("C", settings.Challenger.Name);
            Assert.Equal(1.0, settings.Baseline.Exploration);
            Assert.Equal(800, settings.Iterations);
            Assert.Equal(8, settings.Parallel);
            Assert.Equal(1000, settings.MaxGames);
            Assert.Equal(CommandLine.SeedOf("loop_g009"), settings.SeedBase);
            Assert.Equal(-10, settings.Elo0);
            Assert.Equal(20, settings.Elo1);
            Assert.Equal(TimeSpan.FromMinutes(90), settings.MaxDuration);
            Assert.Equal(InferenceProvider.Cuda, settings.Provider);
        }

        [Fact]
        public void ATimeBudgetReplacesTheIterations()
        {
            MirrorSettings settings = CommandLine.Parse(["mirror", "--challenger", "C=c.onnx", "--baseline", "B=b.onnx", "--seconds", "0.5"]).Settings!;

            Assert.Equal(0, settings.Iterations);
            Assert.Equal(0.5, settings.Seconds);
            Assert.Equal(30_000, settings.SeedBase);
        }

        [Fact]
        public void TheSeedOfANameIsStableAndSeparatesNames()
        {
            Assert.Equal(CommandLine.SeedOf("loop_g003"), CommandLine.SeedOf("loop_g003"));
            Assert.NotEqual(CommandLine.SeedOf("loop_g003"), CommandLine.SeedOf("loop_g006"));
            Assert.InRange(CommandLine.SeedOf("loop_g003"), 100_000, int.MaxValue - 10_000);
        }

        [Theory]
        [InlineData("mirror", "--challenger", "A=a.onnx")]
        [InlineData("mirror", "--challenger", "A=a.onnx", "--baseline", "A=b.onnx")]
        [InlineData("mirror", "--challenger", "A=a.onnx", "--baseline", "B=b.onnx", "--sprt", "40", "0")]
        [InlineData("duel", "--challenger", "A=a.onnx", "--baseline", "B=b.onnx")]
        [InlineData("mirror", "--challenger", "A=a.onnx", "--baseline", "B=b.onnx", "--parallel")]
        public void AnInvalidCommandLineIsRejected(params string[] args)
        {
            Assert.Throws<ArgumentException>(() => CommandLine.Parse(args));
        }

        [Fact]
        public void ClearPairResultsAcceptH1()
        {
            double[] scores = [.. Enumerable.Repeat(1.0, 42), .. Enumerable.Repeat(0.0, 7), .. Enumerable.Repeat(0.5, 55)];

            PairStatistics s = PairStatistics.Of(scores, 0, 40);

            Assert.Equal((104, 42, 7, 55), (s.Pairs, s.ChallengerTwice, s.BaselineTwice, s.Split));
            Assert.Equal(9.211, s.Llr, 3);
            Assert.Equal(121.7, s.Elo, 1);
            Assert.True(s.EloLow > 0 && s.EloHigh > s.Elo);
            Assert.Equal(-2.944, s.LowerBound, 3);
            Assert.Equal(2.944, s.UpperBound, 3);
            Assert.Equal(1, s.Verdict);
        }

        [Fact]
        public void EvenPairResultsLeaveTheTestOpen()
        {
            double[] scores = [.. Enumerable.Repeat(1.0, 10), .. Enumerable.Repeat(0.0, 12), .. Enumerable.Repeat(0.5, 30)];

            PairStatistics s = PairStatistics.Of(scores, 0, 40);

            Assert.Equal(-1.328, s.Llr, 3);
            Assert.Equal(-13.4, s.Elo, 1);
            Assert.True(s.EloLow < 0 && s.EloHigh > 0);
            Assert.Equal(0, s.Verdict);
        }

        [Fact]
        public void IdenticalFirstPairsDecideNothing()
        {
            Assert.Equal(0, PairStatistics.Of([0.5, 0.5], 0, 40).Verdict);
            PairStatistics tenSplits = PairStatistics.Of([.. Enumerable.Repeat(0.5, 10)], 0, 40);
            Assert.InRange(tenSplits.Llr, tenSplits.LowerBound, 0);
            Assert.Equal(0, tenSplits.Verdict);
        }

        [Fact]
        public void FewerThanTwoPairsGiveNoEstimate()
        {
            PairStatistics s = PairStatistics.Of([1.0], 0, 40);

            Assert.Equal(1, s.Pairs);
            Assert.Equal(0, s.Llr);
            Assert.Equal(double.NegativeInfinity, s.EloLow);
            Assert.Equal(0, s.Verdict);
        }

        [Fact]
        public void AShortMirrorDuelPlaysAllPairs()
        {
            var settings = new MirrorSettings
            {
                Challenger = AgentSpec.Parse($"P={PolicyModelPath}=b8"),
                Baseline = AgentSpec.Parse($"V={PlainModelPath}=b8"),
                Iterations = 16,
                Parallel = 4,
                MaxGames = 4,
                SeedBase = 123,
                Provider = InferenceProvider.Cpu,
            };
            var log = new StringWriter();

            PairStatistics result = new MirrorDuel(settings, log).Run();

            string text = log.ToString();
            Assert.Equal(2, result.Pairs);
            Assert.Equal(4, text.Split('\n').Count(line => line.Contains(" game ")));
            Assert.Contains("seats C B C B", text);
            Assert.Contains("seats B C B C", text);
            Assert.Contains("all games played: 4 games", text);
            Assert.Matches(@"[+-]?\d+ Elo \[[+-]?\d+, [+-]?\d+\]", text.TrimEnd().Split('\n')[^1]);
        }
    }
}
