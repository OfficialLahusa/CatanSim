"""The self-play loop's evaluation command: run without a shell, so the paths filled into it can't add arguments or commands.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))

from selfplay_loop import clearly_worse, command_arguments  # noqa: E402

TEMPLATE = ('"D:/Some Dir/Arena.exe" mirror --challenger "NEW={new}" --baseline "OLD={old}" --name {name} '
            '--iterations 5000 --sprt 0 40')


class CommandArgumentsTest(unittest.TestCase):
    def test_quoted_parts_stay_one_argument(self):
        args = command_arguments(TEMPLATE, new="C:/nets/new.onnx", old="C:/nets/old.onnx", name="loop_g003")
        self.assertEqual(args[:3], ["D:/Some Dir/Arena.exe", "mirror", "--challenger"])
        self.assertEqual(args[3], "NEW=C:/nets/new.onnx")
        self.assertEqual(args[5], "OLD=C:/nets/old.onnx")
        self.assertEqual(args[6:8], ["--name", "loop_g003"])

    def test_filled_in_paths_cannot_add_arguments_or_commands(self):
        args = command_arguments(TEMPLATE, new="C:/a b/x;calc & y.onnx", old="C:/o\" --evil.onnx", name="n")
        self.assertEqual(args[3], "NEW=C:/a b/x;calc & y.onnx")
        self.assertEqual(args[5], "OLD=C:/o\" --evil.onnx")
        self.assertEqual(len(args), len(command_arguments(TEMPLATE, new="a", old="b", name="n")))


class ClearlyWorseTest(unittest.TestCase):
    def test_only_an_interval_entirely_below_zero_counts(self):
        self.assertTrue(clearly_worse("... -147 Elo [-273, -50] ..."))
        self.assertFalse(clearly_worse("... -14 Elo [-56, +28] ..."))
        self.assertFalse(clearly_worse("no output"))


if __name__ == "__main__":
    unittest.main()
