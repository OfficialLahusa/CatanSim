using SFML.Audio;

namespace Client
{
    public static class Sounds
    {
        public static readonly SoundBuffer DiceRolling;
        public static readonly SoundBuffer Place;

        static Sounds()
        {
            DiceRolling = new SoundBuffer(ClientUtils.Resource("dice_rolling.wav"));
            Place = new SoundBuffer(ClientUtils.Resource("place.wav"));
        }
    }
}
