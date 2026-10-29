using Common;
using ImGuiNET;
using SFML.Graphics;
using System.Numerics;
using Color = SFML.Graphics.Color;

namespace Client.UI
{
    /// <summary>
    /// Look of the HUD: colors, fonts and drawing helpers on ImGui draw lists (panels, atlas icons, cards in the client's card
    /// style), so every panel is drawn the same way
    /// </summary>
    public static class Theme
    {
        // Panel colors
        public static readonly Color Background = new Color(8, 25, 76);
        public static readonly Color Panel = new Color(15, 33, 86, 250);
        public static readonly Color PanelRaised = new Color(26, 44, 99);
        public static readonly Color PanelActive = new Color(42, 59, 114);
        public static readonly Color Border = new Color(44, 69, 136);
        public static readonly Color BorderLight = new Color(68, 96, 159);
        public static readonly Color Text = new Color(255, 255, 255);
        public static readonly Color TextSoft = new Color(201, 214, 245);
        public static readonly Color TextMuted = new Color(159, 176, 216);
        public static readonly Color TextFaint = new Color(111, 128, 168);
        public static readonly Color Accent = new Color(42, 91, 215);
        public static readonly Color AccentBorder = new Color(127, 162, 255);
        public static readonly Color Highlight = new Color(255, 226, 122);
        public static readonly Color Shadow = new Color(0, 0, 0, 90);

        // Fonts, from small to large; the regular one is ImGui's default
        public static ImFontPtr Regular13, Regular15, Regular17, SemiBold15, SemiBold17, Bold13, Bold15, Bold17, Bold20, Bold22, ExtraBold36;

        public const float Rounding = 14f;

        private static bool _fontsLoaded;

        /// <summary>
        /// Loads the Open Sans fonts into ImGui's atlas and sets the style; once, before the first frame
        /// </summary>
        public static void Init(float uiScale = 1f)
        {
            if (_fontsLoaded) return;
            _fontsLoaded = true;
            GuiImpl.UiScale = uiScale;

            // Rasterized at the pixel size they end up at, laid out at their size in UI units
            ImGuiIOPtr io = ImGui.GetIO();
            ImFontPtr Load(string file, float size)
            {
                ImFontPtr font = io.Fonts.AddFontFromFileTTF(ClientUtils.Resource("Open_Sans", "static", file), size * uiScale);
                font.Scale = 1f / uiScale;
                return font;
            }
            Regular15 = Load("OpenSans-Regular.ttf", 15);
            Regular13 = Load("OpenSans-Regular.ttf", 13);
            Regular17 = Load("OpenSans-Regular.ttf", 17);
            SemiBold15 = Load("OpenSans-SemiBold.ttf", 15);
            SemiBold17 = Load("OpenSans-SemiBold.ttf", 17);
            Bold13 = Load("OpenSans-Bold.ttf", 13);
            Bold15 = Load("OpenSans-Bold.ttf", 15);
            Bold17 = Load("OpenSans-Bold.ttf", 17);
            Bold20 = Load("OpenSans-Bold.ttf", 20);
            Bold22 = Load("OpenSans-Bold.ttf", 22);
            ExtraBold36 = Load("OpenSans-ExtraBold.ttf", 36);
            unsafe { io.NativePtr->FontDefault = Regular15.NativePtr; }
            GuiImpl.UpdateFontTexture();

            // Small atlas icons need mipmaps, or they shimmer
            TextureAtlas.Texture.Smooth = true;
            TextureAtlas.Texture.GenerateMipmap();

            ImGuiStylePtr style = ImGui.GetStyle();
            style.WindowRounding = 10;
            style.ChildRounding = 8;
            style.FrameRounding = 7;
            style.PopupRounding = 8;
            style.GrabRounding = 7;
            style.ScrollbarRounding = 8;
            style.TabRounding = 7;
            style.WindowPadding = new Vector2(12, 10);
            style.FramePadding = new Vector2(8, 4);
            style.ItemSpacing = new Vector2(8, 6);
            style.WindowBorderSize = 1;

            var colors = style.Colors;
            colors[(int)ImGuiCol.WindowBg] = Vec4(Panel);
            colors[(int)ImGuiCol.PopupBg] = Vec4(new Color(15, 33, 86, 250));
            colors[(int)ImGuiCol.ChildBg] = Vec4(new Color(11, 22, 64, 160));
            colors[(int)ImGuiCol.Border] = Vec4(Border);
            colors[(int)ImGuiCol.TitleBg] = Vec4(new Color(15, 33, 86));
            colors[(int)ImGuiCol.TitleBgActive] = Vec4(PanelActive);
            colors[(int)ImGuiCol.FrameBg] = Vec4(new Color(11, 22, 64));
            colors[(int)ImGuiCol.FrameBgHovered] = Vec4(PanelRaised);
            colors[(int)ImGuiCol.FrameBgActive] = Vec4(PanelActive);
            colors[(int)ImGuiCol.Button] = Vec4(PanelRaised);
            colors[(int)ImGuiCol.ButtonHovered] = Vec4(PanelActive);
            colors[(int)ImGuiCol.ButtonActive] = Vec4(Accent);
            colors[(int)ImGuiCol.Header] = Vec4(PanelRaised);
            colors[(int)ImGuiCol.HeaderHovered] = Vec4(PanelActive);
            colors[(int)ImGuiCol.HeaderActive] = Vec4(Accent);
            colors[(int)ImGuiCol.CheckMark] = Vec4(PlayerColor(3));
            colors[(int)ImGuiCol.SliderGrab] = Vec4(PlayerColor(3));
            colors[(int)ImGuiCol.SliderGrabActive] = Vec4(Highlight);
            colors[(int)ImGuiCol.Separator] = Vec4(Border);
            colors[(int)ImGuiCol.Text] = Vec4(TextSoft);
            colors[(int)ImGuiCol.TextDisabled] = Vec4(TextFaint);
            colors[(int)ImGuiCol.ScrollbarBg] = Vec4(new Color(0, 0, 0, 0));
            colors[(int)ImGuiCol.ScrollbarGrab] = Vec4(Border);
            colors[(int)ImGuiCol.ModalWindowDimBg] = Vec4(new Color(0, 0, 0, 110));
        }

        #region Colors
        public static Vector4 Vec4(Color c) => new Vector4(c.R / 255f, c.G / 255f, c.B / 255f, c.A / 255f);

        public static uint U32(Color c) => (uint)(c.A << 24 | c.B << 16 | c.G << 8 | c.R);

        public static Color WithAlpha(Color c, float alpha) => new Color(c.R, c.G, c.B, (byte)Math.Clamp(alpha * 255, 0, 255));

        public static Color Lerp(Color a, Color b, float t)
        {
            t = Math.Clamp(t, 0, 1);
            return new Color((byte)(a.R + (b.R - a.R) * t), (byte)(a.G + (b.G - a.G) * t), (byte)(a.B + (b.B - a.B) * t), (byte)(a.A + (b.A - a.A) * t));
        }

        public static Color PlayerColor(int player) => ColorPalette.GetPlayerColor(player);

        /// <summary>
        /// A lighter shade of a player color for text on dark panels (pure blue and red are hard to read there)
        /// </summary>
        public static Color PlayerTextColor(int player) => player switch
        {
            0 => new Color(111, 140, 255),
            1 => new Color(255, 96, 96),
            2 => new Color(62, 240, 62),
            3 => new Color(254, 168, 1),
            _ => Text
        };

        /// <summary>
        /// Readable text on top of a player color
        /// </summary>
        public static Color TextOnPlayer(int player) => player is 0 or 1 ? Text : new Color(40, 30, 0);

        public static string PlayerName(int player) => player switch
        {
            0 => "Blue",
            1 => "Red",
            2 => "Green",
            3 => "Orange",
            _ => $"Player {player}"
        };

        /// <summary>
        /// Heat map colors from dark plum (rare) over magenta and pink to white (most chosen), for t on [0, 1]: no red, orange,
        /// green or blue hues, so heat never looks like a player's pieces
        /// </summary>
        public static Color Heat(float t)
        {
            ReadOnlySpan<(float at, byte r, byte g, byte b)> stops = [(0f, 70, 18, 70), (0.25f, 140, 30, 130), (0.5f, 214, 62, 196), (0.75f, 255, 150, 232), (1f, 255, 245, 253)];
            t = Math.Clamp(t, 0, 1);
            for (int i = 1; i < stops.Length; i++)
            {
                if (t <= stops[i].at)
                {
                    float f = (t - stops[i - 1].at) / (stops[i].at - stops[i - 1].at);
                    return new Color((byte)(stops[i - 1].r + (stops[i].r - stops[i - 1].r) * f), (byte)(stops[i - 1].g + (stops[i].g - stops[i - 1].g) * f),
                        (byte)(stops[i - 1].b + (stops[i].b - stops[i - 1].b) * f));
                }
            }
            return new Color(255, 245, 253);
        }
        #endregion

        #region Drawing
        /// <summary>
        /// Like ImDrawList.AddRectFilled, but with round corners: ImGui 1.78 draws every corner with only 3 segments, which looks
        /// faceted at our sizes and UI scales, so the corners get as many segments as their radius on screen needs
        /// </summary>
        public static void RoundRectFilled(this ImDrawListPtr dl, Vector2 min, Vector2 max, uint color, float rounding = 0, ImDrawCornerFlags corners = ImDrawCornerFlags.All)
        {
            (min, max) = SnapToPixels(min, max);
            if (rounding <= 0 || corners == ImDrawCornerFlags.None)
            {
                dl.AddRectFilled(min, max, color);
                return;
            }
            PathRoundRect(dl, min, max, rounding, corners);
            dl.PathFillConvex(color);
        }

        /// <summary>
        /// Like ImDrawList.AddRect with smooth corners (see RoundRectFilled)
        /// </summary>
        public static void RoundRect(this ImDrawListPtr dl, Vector2 min, Vector2 max, uint color, float rounding = 0, ImDrawCornerFlags corners = ImDrawCornerFlags.All, float thickness = 1f)
        {
            // The outline covers whole pixels on screen: snapped to the pixel grid and as many pixels thick as the scaled thickness
            // rounds to, so at a fractional UI scale a 1 unit border doesn't fall between two pixels and turn faint on some sides
            (min, max) = SnapToPixels(min, max);
            float pixels = MathF.Max(1, MathF.Round(thickness * GuiImpl.UiScale));
            thickness = pixels / GuiImpl.UiScale;
            Vector2 half = new Vector2(thickness / 2);
            if (rounding <= 0 || corners == ImDrawCornerFlags.None)
            {
                dl.AddRect(min + half, max - half, color, 0, corners, thickness);
                return;
            }
            PathRoundRect(dl, min + half, max - half, rounding, corners);
            dl.PathStroke(color, true, thickness);
        }

        /// <summary>
        /// A rect in HUD units moved to the nearest whole pixels of the scaled screen
        /// </summary>
        private static (Vector2 min, Vector2 max) SnapToPixels(Vector2 min, Vector2 max)
        {
            float scale = GuiImpl.UiScale;
            return (Vector2.Round(min * scale) / scale, Vector2.Round(max * scale) / scale);
        }

        /// <summary>
        /// The path of ImDrawList.PathRect (the same clamping of the radius), with corner arcs of enough segments
        /// </summary>
        private static void PathRoundRect(ImDrawListPtr dl, Vector2 a, Vector2 b, float rounding, ImDrawCornerFlags corners)
        {
            bool top = (corners & ImDrawCornerFlags.Top) == ImDrawCornerFlags.Top, bottom = (corners & ImDrawCornerFlags.Bot) == ImDrawCornerFlags.Bot;
            bool left = (corners & ImDrawCornerFlags.Left) == ImDrawCornerFlags.Left, right = (corners & ImDrawCornerFlags.Right) == ImDrawCornerFlags.Right;
            rounding = MathF.Min(rounding, MathF.Abs(b.X - a.X) * (top || bottom ? 0.5f : 1f) - 1f);
            rounding = MathF.Min(rounding, MathF.Abs(b.Y - a.Y) * (left || right ? 0.5f : 1f) - 1f);
            if (rounding <= 0)
            {
                dl.PathLineTo(a);
                dl.PathLineTo(new Vector2(b.X, a.Y));
                dl.PathLineTo(b);
                dl.PathLineTo(new Vector2(a.X, b.Y));
                return;
            }
            float tl = (corners & ImDrawCornerFlags.TopLeft) != 0 ? rounding : 0, tr = (corners & ImDrawCornerFlags.TopRight) != 0 ? rounding : 0;
            float br = (corners & ImDrawCornerFlags.BotRight) != 0 ? rounding : 0, bl = (corners & ImDrawCornerFlags.BotLeft) != 0 ? rounding : 0;
            Corner(dl, new Vector2(a.X + tl, a.Y + tl), tl, MathF.PI, 1.5f * MathF.PI);
            Corner(dl, new Vector2(b.X - tr, a.Y + tr), tr, 1.5f * MathF.PI, 2 * MathF.PI);
            Corner(dl, new Vector2(b.X - br, b.Y - br), br, 0, 0.5f * MathF.PI);
            Corner(dl, new Vector2(a.X + bl, b.Y - bl), bl, 0.5f * MathF.PI, MathF.PI);
        }

        /// <summary>
        /// A quarter arc whose segments deviate at most a quarter pixel on screen from the circle
        /// </summary>
        private static void Corner(ImDrawListPtr dl, Vector2 center, float radius, float from, float to)
        {
            if (radius <= 0)
            {
                dl.PathLineTo(center);
                return;
            }
            float pixels = radius * GuiImpl.UiScale;
            float step = MathF.Acos(MathF.Max(0, 1 - 0.25f / pixels)) * 2;
            int segments = Math.Clamp((int)MathF.Ceiling((to - from) / step), 2, 32);
            dl.PathArcTo(center, radius, from, to, segments);
        }

        /// <summary>
        /// A rounded panel with a soft shadow and a border
        /// </summary>
        public static void DrawPanel(ImDrawListPtr dl, Vector2 min, Vector2 max, Color? border = null, float rounding = Rounding, Color? fill = null, float borderWidth = 1f)
        {
            dl.RoundRectFilled(min + new Vector2(0, 4), max + new Vector2(0, 6), U32(Shadow), rounding);
            dl.RoundRectFilled(min, max, U32(fill ?? Panel), rounding);
            dl.RoundRect(min, max, U32(border ?? Border), rounding, ImDrawCornerFlags.All, borderWidth);
        }

        /// <summary>
        /// A glow around a rounded rect fading out over 10 units: one-pixel rings side by side, so it is a smooth gradient
        /// (outlines spaced apart showed as stripes)
        /// </summary>
        public static void Glow(ImDrawListPtr dl, Vector2 min, Vector2 max, Color color, float rounding = Rounding, float strength = 1f)
        {
            float pixel = 1 / GuiImpl.UiScale;
            int rings = Math.Max(1, (int)MathF.Round(10 * GuiImpl.UiScale));
            for (int i = 0; i < rings; i++)
            {
                float fade = 1 - (i + 0.5f) / rings;
                Vector2 grow = new Vector2((i + 1) * pixel);
                dl.RoundRect(min - grow, max + grow, U32(WithAlpha(color, 0.3f * strength * fade * fade)), rounding + (i + 1) * pixel, ImDrawCornerFlags.All, pixel);
            }
        }

        public static void DrawText(ImDrawListPtr dl, ImFontPtr font, Vector2 pos, Color color, string text)
        {
            dl.AddText(font, font.FontSize * font.Scale, pos, U32(color), text);
        }

        public static Vector2 TextSize(ImFontPtr font, string text)
        {
            ImGui.PushFont(font);
            Vector2 size = ImGui.CalcTextSize(text);
            ImGui.PopFont();
            return size;
        }

        /// <summary>
        /// Text centered on a point (horizontally, and vertically on the font's line height)
        /// </summary>
        public static void TextCentered(ImDrawListPtr dl, ImFontPtr font, Vector2 center, Color color, string text)
        {
            Vector2 size = TextSize(font, text);
            DrawText(dl, font, center - size / 2, color, text);
        }

        public static void TextRight(ImDrawListPtr dl, ImFontPtr font, Vector2 rightTop, Color color, string text)
        {
            Vector2 size = TextSize(font, text);
            DrawText(dl, font, rightTop - new Vector2(size.X, 0), color, text);
        }

        public static IntPtr AtlasId => GuiImpl.TextureId(TextureAtlas.Texture);

        /// <summary>
        /// An atlas sprite tinted with a color, like the board draws its icons
        /// </summary>
        public static void Icon(ImDrawListPtr dl, TextureAtlas.Sprite sprite, Vector2 min, Vector2 max, Color tint)
        {
            IntRect rect = sprite.GetTextureRect();
            Vector2 size = new Vector2(TextureAtlas.Texture.Size.X, TextureAtlas.Texture.Size.Y);

            // The atlas is mipmapped and some icons fill their cell to the edge, so at small sizes the coarse levels blend the
            // neighbouring cell into the border (a thin line along the quad's edge). The UVs are pulled in by about one texel of
            // the level in use and the quad shrinks by the same amount, so the icon keeps its size and place.
            float pixels = MathF.Max(1, (max.X - min.X) * GuiImpl.UiScale);
            float inset = Math.Clamp(rect.Width / pixels, 0, rect.Width / 16f);
            Vector2 shrink = (max - min) * (inset / rect.Width);
            Vector2 uv0 = new Vector2(rect.Left + inset, rect.Top + inset) / size;
            Vector2 uv1 = new Vector2(rect.Left + rect.Width - inset, rect.Top + rect.Height - inset) / size;
            dl.AddImage(AtlasId, min + shrink, max - shrink, uv0, uv1, U32(tint));
        }

        public static void IconCentered(ImDrawListPtr dl, TextureAtlas.Sprite sprite, Vector2 center, float size, Color tint)
        {
            Icon(dl, sprite, center - new Vector2(size / 2), center + new Vector2(size / 2), tint);
        }

        /// <summary>
        /// A resource card in the client's card style: border and icon in the darker icon color, fill in the tile color
        /// </summary>
        public static void ResourceCard(ImDrawListPtr dl, ResourceCardType type, Vector2 min, Vector2 size, bool withIcon = true, float alpha = 1f)
        {
            Card(dl, ColorPalette.GetCardColor(type), ColorPalette.GetCardIconColor(type), TextureAtlas.GetSprite(type), min, size, withIcon, alpha);
        }

        /// <summary>
        /// A development card; muted ones (not playable) are faded toward the panel but stay opaque, so the icon keeps its contrast
        /// </summary>
        public static void DevelopmentCard(ImDrawListPtr dl, DevelopmentCardType type, Vector2 min, Vector2 size, bool withIcon = true, float alpha = 1f, bool muted = false)
        {
            Color fill = ColorPalette.GetCardColor(type), icon = ColorPalette.GetCardIconColor(type);
            if (muted)
            {
                fill = Lerp(fill, Panel, 0.45f);
                icon = Lerp(icon, Panel, 0.2f);
            }
            Card(dl, fill, icon, TextureAtlas.GetSprite(type), min, size, withIcon, alpha);
        }

        public static void Card(ImDrawListPtr dl, Color fill, Color icon, TextureAtlas.Sprite sprite, Vector2 min, Vector2 size, bool withIcon, float alpha)
        {
            // The border is 6 of 80 units wide, as in Card
            float border = MathF.Max(1.5f, size.X * 6f / 80f);
            dl.RoundRectFilled(min, min + size, U32(WithAlpha(icon, alpha)), MathF.Min(2f, border));
            dl.RoundRectFilled(min + new Vector2(border), min + size - new Vector2(border), U32(WithAlpha(fill, alpha)));
            if (withIcon)
            {
                float iconSize = size.X * 0.75f;
                IconCentered(dl, sprite, min + size / 2, iconSize, WithAlpha(icon, alpha));
            }
        }

        /// <summary>
        /// The back of a card (hidden hand cards)
        /// </summary>
        public static void CardBack(ImDrawListPtr dl, Vector2 min, Vector2 size, float alpha = 1f)
        {
            float border = MathF.Max(1.5f, size.X * 6f / 80f);
            dl.RoundRectFilled(min, min + size, U32(WithAlpha(new Color(201, 214, 245), alpha)), 2f);
            dl.RoundRectFilled(min + new Vector2(border), min + size - new Vector2(border), U32(WithAlpha(new Color(62, 98, 173), alpha)));
        }

        /// <summary>
        /// A small card per resource of a cost, side by side and centered on a point
        /// </summary>
        public static void CostRow(ImDrawListPtr dl, IEnumerable<ResourceCardType> cost, Vector2 center, float cardWidth = 15, float alpha = 1f)
        {
            ResourceCardType[] cards = cost.ToArray();
            float gap = 2;
            float width = cards.Length * cardWidth + (cards.Length - 1) * gap;
            Vector2 size = new Vector2(cardWidth, cardWidth * 4 / 3);
            Vector2 pos = new Vector2(center.X - width / 2, center.Y - size.Y / 2);
            foreach (ResourceCardType card in cards)
            {
                ResourceCard(dl, card, pos, size, true, alpha);
                pos.X += cardWidth + gap;
            }
        }

        /// <summary>
        /// A pill with centered text
        /// </summary>
        public static void Pill(ImDrawListPtr dl, Vector2 min, Vector2 max, Color fill, ImFontPtr font, Color textColor, string text, Color? border = null)
        {
            float r = (max.Y - min.Y) / 2;
            dl.RoundRectFilled(min, max, U32(fill), r);
            if (border.HasValue) dl.RoundRect(min, max, U32(border.Value), r, ImDrawCornerFlags.All, 2f);
            TextCentered(dl, font, (min + max) / 2, textColor, text);
        }
        #endregion

        #region Windows
        public const ImGuiWindowFlags HudFlags = ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.NoBackground | ImGuiWindowFlags.NoMove
            | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoFocusOnAppearing | ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNav;

        private const float HudBleed = 4f;

        /// <summary>
        /// Opens a borderless, transparent window covering a screen rect, for a panel drawn on its draw list; close with ImGui.End
        /// </summary>
        public static ImDrawListPtr BeginHud(string id, Vector2 min, Vector2 size)
        {
            // The window reaches a few units past the rect, so outlines and their anti-aliased edges on the rect's border aren't
            // clipped (a panel drawn at the window's edge got a thinner top border); the cursor still starts at the rect
            Vector2 bleed = new Vector2(HudBleed);
            ImGui.SetNextWindowPos(min - bleed);
            ImGui.SetNextWindowSize(size + 2 * bleed);
            ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, Vector2.Zero);
            ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, 0f);
            ImGui.Begin(id, HudFlags);
            ImGui.PopStyleVar(2);
            ImGui.SetCursorScreenPos(min);
            return ImGui.GetWindowDrawList();
        }

        /// <summary>
        /// An invisible button over a screen rect; whether it was clicked, and whether the mouse is over it
        /// </summary>
        public static bool HitArea(string id, Vector2 min, Vector2 size, out bool hovered)
        {
            ImGui.SetCursorScreenPos(min);
            bool clicked = ImGui.InvisibleButton(id, size);
            hovered = ImGui.IsItemHovered();
            return clicked;
        }
        #endregion
    }
}
