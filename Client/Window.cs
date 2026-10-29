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
        /// the window size instead of maximizing it, --screenshot FILE [--after SECONDS] saves a screenshot after a while and quits,
        /// --host PORT (or localhost:PORT) hosts the game for others, --join ADDRESS [--name N] [--role player|spectator|admin|agent] [--seat 1-4]
        /// [--token T, or HEXFARER_TOKEN] [--fingerprint F] joins a game on a server (F: the certificate fingerprint for wss), --network opens the network window, --admin [TAB] the admin panel
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

            window = new RenderWindow(mode, "Hexfarer", Styles.Default, new ContextSettings() { AntialiasingLevel = 8 });

            // Maximize window
            if (size == null) ShowWindow(window.SystemHandle, 3);

            // Init ImGui
            GuiImpl.Init(window);
            // The HUD is laid out for a window about 1100 pixels high and scaled to the screen in quarter steps
            float uiScale = Math.Clamp(MathF.Round(window.Size.Y / 1100f * 4) / 4, 0.75f, 2f);
            UI.Theme.Init(uiScale);

            // Init Screen
            GameScreen gameScreen = new GameScreen(window);
            screen = gameScreen;
            if (scene != null) gameScreen.RunScene(scene, seed);
            if (Option("--host") is string host) gameScreen.HostFromCommandLine(host);
            if (Option("--join") is string join)
                // The token may also come from HEXFARER_TOKEN (or the older CATANSIM_TOKEN), which keeps it out of process lists and
                // shell history
                gameScreen.JoinFromCommandLine(join, Option("--name"), Option("--role"), Option("--seat") is string seatText ? int.Parse(seatText) : null,
                    Option("--token") ?? Environment.GetEnvironmentVariable("HEXFARER_TOKEN") ?? Environment.GetEnvironmentVariable("CATANSIM_TOKEN"),
                    Option("--fingerprint"));
            if (args.Contains("--network")) gameScreen.ShowNetworkWindow();
            if (args.Contains("--admin")) gameScreen.ShowAdminWindow(Option("--admin") is string tab && !tab.StartsWith("--") ? tab : null);
            string? selfTest = Option("--selftest");
            if (selfTest != null) gameScreen.StartSelfTest(int.Parse(selfTest), seed);

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

                if (gameScreen.SelfTestFinished) window.Close();

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
