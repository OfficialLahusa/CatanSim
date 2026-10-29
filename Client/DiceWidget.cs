using Common;
using SFML.Audio;
using SFML.Graphics;
using SFML.System;

namespace Client
{
    public class DiceWidget : Drawable
    {
        private RectangleShape _firstDie;
        private RectangleShape _secondDie;
        // Size of a die in pixels
        public const int Size = 120;
        private const int _size = Size;
        private static readonly Color _inactiveColor = new Color(255, 255, 255, 85);
        private bool _active;

        public RollResult RollResult { get; set; }

        public bool Active { 
            get { return _active; }
            set
            {
                _active = value;
                _firstDie.FillColor = value ? Color.White : _inactiveColor;
                _secondDie.FillColor = _firstDie.FillColor;
            }
        }

        public DiceWidget(RenderWindow window)
        {
            _firstDie = new RectangleShape(new SFML.System.Vector2f(_size, _size));
            _firstDie.Texture = TextureAtlas.Texture;

            _secondDie = new RectangleShape(new SFML.System.Vector2f(_size, _size));
            _secondDie.Texture = TextureAtlas.Texture;

            Active = false;

            RollResult = RollResult.GetRandom();
            UpdateSprites();
            UpdatePosition(window);

            window.Resized += Window_Resized;
        }

        private void Window_Resized(object? sender, SFML.Window.SizeEventArgs e)
        {
            if(sender != null)
            {
                UpdatePosition((RenderWindow)sender);
            }
        }

        public int Roll()
        {
            RollResult = RollResult.GetRandom();
            UpdateSprites();

            return RollResult.Total;
        }

        private void UpdatePosition(RenderWindow window)
        {
            // Position at the bottom right corner of view coordinate system (center = 0, 0)
            _firstDie.Position = new Vector2f(window.Size.X / 2 - 2 * _size - 40, window.Size.Y / 2 - _size - 20);
            _secondDie.Position = new Vector2f(window.Size.X / 2 - _size - 20, window.Size.Y / 2 - _size - 20);
        }

        public void UpdateSprites()
        {
            _firstDie.TextureRect = TextureAtlas.GetTextureRect(TextureAtlas.Sprite.DiceOne + RollResult.First - 1);
            _secondDie.TextureRect = TextureAtlas.GetTextureRect(TextureAtlas.Sprite.DiceOne + RollResult.Second - 1);
        }

        public void Draw(RenderTarget target, RenderStates states)
        {
            target.Draw(_firstDie, states);
            target.Draw(_secondDie, states);
        }

        public bool Contains(float x, float y)
        {
            return _firstDie.GetGlobalBounds().Contains(x, y) || _secondDie.GetGlobalBounds().Contains(x, y);
        }
    }
}
