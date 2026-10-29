namespace Arena
{
    /// <summary>
    /// Statistics of a mirror duel over its completed map pairs, each scored for the challenger: 1 if it won both games,
    /// 0.5 if each side won one, 0 if it lost both (a game without a winner counts as lost for both). The Elo estimate and
    /// its 95% interval come from the mean pair score; the sequential probability ratio test (Wald) between H0 "the gain is
    /// Elo0" and H1 "the gain is Elo1" uses the normal approximation of Fishtest on the pair scores. It decides only from
    /// MinPairs on and with a floor under the variance: the first pairs often all score the same (a few splits), and their
    /// variance of 0 would otherwise accept a hypothesis at once.
    /// </summary>
    internal sealed record PairStatistics(int Pairs, int ChallengerTwice, int BaselineTwice, double Elo, double EloLow, double EloHigh,
        double Llr, double LowerBound, double UpperBound)
    {
        public const int MinPairs = 30;
        public const double MinVariance = 0.01;

        public int Split => Pairs - ChallengerTwice - BaselineTwice;

        /// <summary>
        /// 1 if H1 is accepted, -1 if H0 is accepted, 0 while the test is open (always before MinPairs)
        /// </summary>
        public int Verdict => Pairs < MinPairs ? 0 : Llr >= UpperBound ? 1 : Llr <= LowerBound ? -1 : 0;

        public static PairStatistics Of(IReadOnlyList<double> scores, double elo0, double elo1, double alpha = 0.05, double beta = 0.05)
        {
            int n = scores.Count;
            int twice = scores.Count(s => s == 1), lost = scores.Count(s => s == 0);
            double lower = Math.Log(beta / (1 - alpha)), upper = Math.Log((1 - beta) / alpha);
            if (n < 2)
                return new PairStatistics(n, twice, lost, 0, double.NegativeInfinity, double.PositiveInfinity, 0, lower, upper);

            double mean = scores.Average();
            double variance = scores.Sum(s => (s - mean) * (s - mean)) / (n - 1);
            double half = 1.96 * Math.Sqrt(variance / n);
            double s0 = ExpectedScore(elo0), s1 = ExpectedScore(elo1);
            double llr = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * Math.Max(variance, MinVariance));
            return new PairStatistics(n, twice, lost, EloOf(mean), EloOf(mean - half), EloOf(mean + half), llr, lower, upper);
        }

        public static double ExpectedScore(double elo) => 1 / (1 + Math.Pow(10, -elo / 400));

        public static double EloOf(double score)
        {
            score = Math.Clamp(score, 1e-4, 1 - 1e-4);
            return -400 * Math.Log10(1 / score - 1);
        }
    }
}
