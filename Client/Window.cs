using SFML.Graphics;
using SFML.System;
using SFML.Window;
using System.Globalization;
using System.Runtime.InteropServices;
using VideoMode = SFML.Window.VideoMode;

namespace Client
{
    public class Window
    {
        private RenderWindow window;
        private Clock deltaClock;

        private Screen screen;

        /// <summary>
        /// Options: --scene NAME [--seed N] starts in a preset situation (see GameScreen.Scenes), --size WIDTHxHEIGHT sets
        /// the window size instead of maximizing it, --screenshot FILE [--after SECONDS] saves a screenshot after a while and quits
        /// </summary>
        public static void Main(String[] args)
        {
            Window window = new Window(args);
        }

        public Window(string[] args)
        {
            string? Option(string name) => Array.IndexOf(args, name) is int i && i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
            string? scene = Option("--scene");
            string? screenshot = Option("--screenshot");
            double after = double.Parse(Option("--after") ?? "3", CultureInfo.InvariantCulture);
            int seed = int.Parse(Option("--seed") ?? "7", CultureInfo.InvariantCulture);
            string? size = Option("--size");

            VideoMode mode = VideoMode.DesktopMode;
            if (size != null)
            {
                string[] parts = size.Split('x');
                mode = new VideoMode(uint.Parse(parts[0]), uint.Parse(parts[1]));
            }

            window = new RenderWindow(mode, "CatanSim", Styles.Default, new ContextSettings() { AntialiasingLevel = 8 });

            // Maximize window
            if (size == null) ShowWindow(window.SystemHandle, 3);

            // Init ImGui
            GuiImpl.Init(window);
            UI.Theme.Init();

            // Init Screen
            GameScreen gameScreen = new GameScreen(window);
            screen = gameScreen;
            if (scene != null) gameScreen.RunScene(scene, seed);

            window.Closed += Window_Closed;

            deltaClock = new Clock();
            Clock runClock = new Clock();
            bool screenshotRequested = false;

            while (window.IsOpen)
            {
                Time deltaTime = deltaClock.Restart();

                // Input handling
                if (window.HasFocus())
                {
                    screen.HandleInput(deltaTime);
                }

                // Update
                screen.Update(deltaTime);

                // Rendering
                screen.Draw(deltaTime);

                // A scripted screenshot quits once it is written
                if (screenshot != null)
                {
                    if (!screenshotRequested && runClock.ElapsedTime.AsSeconds() >= after)
                    {
                        gameScreen.SaveScreenshot(screenshot);
                        screenshotRequested = true;
                    }
                    else if (screenshotRequested && !gameScreen.ScreenshotPending)
                    {
                        window.Close();
                    }
                }
            }

            // Agents may still search in the background
            Environment.Exit(0);
        }

        private void Window_Closed(object? sender, EventArgs e)
        {
            window.Close();
        }


        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    }
}
