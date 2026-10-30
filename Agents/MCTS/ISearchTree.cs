using Common;
using Action = Common.Actions.Action;

namespace Agents.MCTS
{
    /// <summary>
    /// A leaf the search selected: the tree's node (opaque to the search) and the state to evaluate there
    /// </summary>
    public readonly record struct SearchLeaf(object Node, GameState State);

    /// <summary>
    /// An agent whose search can be shown while it thinks: the root moves of its current tree and the position they belong to
    /// (empty without a tree, or for a chance node)
    /// </summary>
    public interface IShowsSearch
    {
        (GameState? state, IReadOnlyList<MCTSAgent.RootMove> moves) GetRootMoves();
    }

    /// <summary>
    /// What MCTSSearch needs of a tree: a leaf per iteration (selected, expanded and marked as pending), the priors of its
    /// moves, and the backpropagation of its utilities. MCTSTree searches the game itself, ISMCTSTree the information sets of
    /// one seat, a world drawn per iteration.
    /// </summary>
    public interface ISearchTree
    {
        /// <summary>
        /// A position of the root, for its priors: its moves are the root's in the order of MCTSTree.GetValidActions
        /// </summary>
        GameState RootState { get; }

        sbyte OwnerIndex { get; }

        /// <summary>
        /// Whether selection uses priors (PUCT), set by the search
        /// </summary>
        bool UsesPolicy { get; set; }

        bool RootHasPolicyPriors { get; }

        /// <summary>
        /// The root's priors, in the order of MCTSTree.GetValidActions(RootState)
        /// </summary>
        void SetRootPriors(float[] priors);

        /// <summary>
        /// Selects a leaf, expanding on the way, and marks its path as pending; thread-safe
        /// </summary>
        SearchLeaf SelectLeaf();

        /// <summary>
        /// Whether a leaf has the priors of its moves already (it can be selected again before)
        /// </summary>
        bool HasPolicyPriors(SearchLeaf leaf);

        /// <summary>
        /// Gives a leaf the priors of the moves (MCTSTree.GetValidActions of its state, in that order)
        /// </summary>
        void ApplyPriors(SearchLeaf leaf, float[] priors, List<Action> moves);

        /// <summary>
        /// Adds a leaf's utilities (one per player) to its path, resolving the pending visits
        /// </summary>
        void Backpropagate(SearchLeaf leaf, ReadOnlySpan<double> utilities);
    }
}
