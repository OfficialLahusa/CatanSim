using System.Text.Json;

namespace DatasetCollector
{
    /// <summary>
    /// Collects a run in parts: the matches are split into settings.Parts consecutive ranges, each collected into its own
    /// subdirectory of the run (part_00, part_01, ...). Match indices continue across parts, so the parts together equal a
    /// single run with the same seed, and every part stays small enough to list and to process on its own.
    ///
    /// Running the same command again resumes: finished parts (metadata.json with status finished) are skipped, an unfinished
    /// one is deleted and collected again. That loses nothing, since the same seeds reproduce the same samples, and happens
    /// only if its metadata shows the same settings; a part collected with other settings stops the run instead.
    /// </summary>
    public static class PartialRuns
    {
        public static CollectionResult Collect(CollectorSettings settings, TextWriter log)
        {
            settings.Validate();
            if (settings.Parts == 1)
                return new DatasetCollector(settings, log).Collect();

            string runDirectory = Path.Combine(settings.OutputDirectory, settings.DatasetName);
            if (settings.Name == null)
                log.WriteLine($"Run directory {runDirectory} (pass --name {Path.GetFileName(runDirectory)} to resume it later)");

            CollectionResult total = new(0, 0, 0, 0, TimeSpan.Zero, false);
            int firstMatch = settings.FirstMatch;
            for (int part = 0; part < settings.Parts; part++)
            {
                int matches = settings.Matches / settings.Parts + (part < settings.Matches % settings.Parts ? 1 : 0);
                CollectorSettings partSettings = settings with
                {
                    OutputDirectory = runDirectory,
                    Name = $"part_{part:D2}",
                    FirstMatch = firstMatch,
                    Matches = matches,
                    Parts = 1,
                };
                firstMatch += matches;

                string partDirectory = Path.Combine(runDirectory, partSettings.Name);
                if (Directory.Exists(partDirectory))
                {
                    string? status = ReadStatus(partDirectory, partSettings);
                    if (status == "finished")
                    {
                        log.WriteLine($"\nPart {part + 1}/{settings.Parts} ({partSettings.Name}) is finished, skipping it.");
                        continue;
                    }

                    log.WriteLine($"\nPart {part + 1}/{settings.Parts} ({partSettings.Name}) is unfinished ({status ?? "no metadata"}), collecting it again.");
                    Directory.Delete(partDirectory, recursive: true);
                }

                log.WriteLine($"\nPart {part + 1}/{settings.Parts}: matches {partSettings.FirstMatch} to {partSettings.FirstMatch + matches - 1}");
                CollectionResult result = new DatasetCollector(partSettings, log).Collect();
                total = new CollectionResult(total.Samples + result.Samples, total.SkippedSamples + result.SkippedSamples, total.Playouts + result.Playouts,
                    total.Errors + result.Errors, total.Elapsed + result.Elapsed, result.Cancelled);
                if (result.Cancelled) break;
            }

            return total;
        }

        /// <summary>
        /// The status in an existing part's metadata.json (null without one), after checking that the part was collected
        /// with the same settings
        /// </summary>
        private static string? ReadStatus(string partDirectory, CollectorSettings expected)
        {
            string path = Path.Combine(partDirectory, "metadata.json");
            if (!File.Exists(path)) return null;

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;
            JsonElement settings = root.GetProperty("settings");
            JsonElement agents = settings.GetProperty("agents");

            bool same = root.GetProperty("seed").GetString() == expected.Seed.ToString(System.Globalization.CultureInfo.InvariantCulture)
                && settings.TryGetProperty("first_match", out JsonElement first) && first.GetInt32() == expected.FirstMatch
                && settings.GetProperty("matches").GetInt32() == expected.Matches
                && settings.GetProperty("samples_per_match").GetInt32() == expected.SamplesPerMatch
                && settings.GetProperty("playouts_per_sample").GetInt32() == expected.PlayoutsPerSample
                && settings.GetProperty("sibling_groups_per_match").GetInt32() == expected.SiblingGroupsPerMatch
                && settings.GetProperty("siblings_per_group").GetInt32() == expected.SiblingsPerGroup
                && settings.GetProperty("center_desert").GetBoolean() == expected.CenterDesert
                && agents.GetProperty("random").GetDouble() == expected.Agents.RandomWeight
                && agents.GetProperty("simple").GetDouble() == expected.Agents.SimpleWeight
                && agents.GetProperty("builder").GetDouble() == expected.Agents.BuilderWeight
                && agents.GetProperty("min_exploration").GetDouble() == expected.Agents.MinExploration
                && agents.GetProperty("max_exploration").GetDouble() == expected.Agents.MaxExploration;

            if (!same)
                throw new InvalidOperationException($"\"{partDirectory}\" was collected with other settings (see its metadata.json). "
                    + "Resume a run with the same command (seed, matches, parts and the other settings), or choose another --name.");

            return root.GetProperty("status").GetString();
        }
    }
}
