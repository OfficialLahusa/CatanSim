using Agents.MCTS;
using System.Globalization;

namespace Arena
{
    /// <summary>
    /// An MCTS agent of a duel, given as "NAME=MODEL.onnx[=OPTION...]". Options: an opponent model (MaxN, Paranoid, Random;
    /// MaxN if none), cX exploration constant, bN leaves per batch, fX first play reduction, pX placement weight, u PUCT
    /// with uniform priors instead of the policy head's, dN hidden information: only its seat's view, searched as N
    /// determinizations (PIMC, see DeterminizedMCTSAgent), is hidden information searched as one tree over the seat's
    /// information sets (see InformationSetMCTSAgent), isfull the same search on the true position (an ablation); without
    /// these the agent sees every hand. Ablations of the search: norm or nonorm selects with or without values scaled to
    /// the range seen in the tree (MCTS normalizes, ISMCTS not), ob lets ISMCTS branch on outcomes its seat can't see,
    /// noreuse starts MCTS on a new tree every decision (ISMCTS always does), noleaf lets MCTS evaluate one of a new leaf's
    /// moves instead of the leaf itself (the old search; leaf is the default).
    /// </summary>
    internal sealed record AgentSpec(string Name, string ModelPath, OpponentModel OpponentModel, double Exploration, int LeavesPerBatch,
        double FirstPlayReduction, double PlacementWeight, bool UniformPriors, int Determinizations = 0, bool InformationSets = false,
        bool FullKnowledge = false, bool? NormalizeValues = null, bool BranchHiddenOutcomes = false, bool ReuseTree = true,
        bool EvaluateNewLeaves = true)
    {
        public static AgentSpec Parse(string spec)
        {
            string[] parts = spec.Split('=');
            if (parts.Length < 2 || parts[0].Length == 0 || parts[1].Length == 0)
                throw new ArgumentException($"Agent \"{spec}\" is not NAME=MODEL.onnx[=OPTION...].");

            var result = new AgentSpec(parts[0], parts[1], OpponentModel.MaxN, 1.414, MCTSAgent.DefaultLeavesPerBatch, 0.1, 0, false);
            foreach (string option in parts.Skip(2))
            {
                double Number() => double.Parse(option[1..], CultureInfo.InvariantCulture);
                result = option switch
                {
                    "u" => result with { UniformPriors = true },
                    "is" => result with { InformationSets = true },
                    "isfull" => result with { InformationSets = true, FullKnowledge = true },
                    "norm" => result with { NormalizeValues = true },
                    "nonorm" => result with { NormalizeValues = false },
                    "ob" => result with { BranchHiddenOutcomes = true },
                    "noreuse" => result with { ReuseTree = false },
                    "leaf" => result with { EvaluateNewLeaves = true },
                    "noleaf" => result with { EvaluateNewLeaves = false },
                    _ when Enum.TryParse(option, out OpponentModel model) => result with { OpponentModel = model },
                    ['c', ..] => result with { Exploration = Number() },
                    ['b', ..] => result with { LeavesPerBatch = int.Parse(option[1..], CultureInfo.InvariantCulture) },
                    ['f', ..] => result with { FirstPlayReduction = Number() },
                    ['p', ..] => result with { PlacementWeight = Number() },
                    ['d', ..] => result with { Determinizations = int.Parse(option[1..], CultureInfo.InvariantCulture) },
                    _ => throw new ArgumentException($"Unknown option \"{option}\" in agent \"{spec}\"."),
                };
            }
            return result;
        }

        /// <summary>
        /// The name, the model's file name and the options that differ from the defaults
        /// </summary>
        public override string ToString()
        {
            List<string> options = [OpponentModel.ToString()];
            if (Exploration != 1.414) options.Add($"c{Exploration.ToString(CultureInfo.InvariantCulture)}");
            if (LeavesPerBatch != MCTSAgent.DefaultLeavesPerBatch) options.Add($"b{LeavesPerBatch}");
            if (FirstPlayReduction != 0.1) options.Add($"f{FirstPlayReduction.ToString(CultureInfo.InvariantCulture)}");
            if (PlacementWeight != 0) options.Add($"p{PlacementWeight.ToString(CultureInfo.InvariantCulture)}");
            if (UniformPriors) options.Add("u");
            if (Determinizations > 0) options.Add($"d{Determinizations}");
            if (InformationSets) options.Add(FullKnowledge ? "isfull" : "is");
            if (NormalizeValues is bool normalize) options.Add(normalize ? "norm" : "nonorm");
            if (BranchHiddenOutcomes) options.Add("ob");
            if (!ReuseTree) options.Add("noreuse");
            if (!EvaluateNewLeaves) options.Add("noleaf");
            return $"{Name} {Path.GetFileName(ModelPath)} {string.Join(" ", options)}";
        }
    }
}
