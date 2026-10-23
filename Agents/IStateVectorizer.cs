using Common;

namespace Agents
{
    /// <summary>
    /// Turns a game state into the input vector of a value net, so agents can be given a net together with the encoding
    /// it was trained on
    /// </summary>
    public interface IStateVectorizer
    {
        int FeatureDim { get; }

        /// <summary>
        /// Writes the feature vector of the state into the destination, which holds exactly FeatureDim values
        /// </summary>
        void VectorizeInto(GameState state, Span<float> destination);
    }
}
