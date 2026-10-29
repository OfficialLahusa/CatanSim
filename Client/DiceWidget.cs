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
        // Size of a die in UI units (GuiImpl.UiScale pixels each)
        public const int Size = 120;

        // Between the two dice, in UI units
        public const int Gap = 20;

        // The HUD column the dice are centered in (UI units)
        private readonly float _columnWidth;
        private static float _size => Size * GuiImpl.UiScale;
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

        public DiceWidget(RenderWindow window, float columnWidth)
        {
            _columnWidth = columnWidth;
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
            // At the bottom of the right HUD column, centered in it (the view's origin is the window's center)
            float margin = 20 * GuiImpl.UiScale, gap = Gap * GuiImpl.UiScale;
            float columnCenter = window.Size.X / 2 - margin - _columnWidth * GuiImpl.UiScale / 2;
            float left = MathF.Round(columnCenter - _size - gap / 2);
            _firstDie.Position = new Vector2f(left, window.Size.Y / 2 - _size - margin);
            _secondDie.Position = new Vector2f(left + _size + gap, window.Size.Y / 2 - _size - margin);
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
