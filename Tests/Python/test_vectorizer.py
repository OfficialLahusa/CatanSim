"""Python side of the vectorizer tests.

The fixtures in Tests/Fixtures/Vectorizer are written by the C# test VectorizerTests.RegenerateParityFixtures:
each state is stored as a save file (.yaml) together with the C# feature vector (.csv). The C# test checks
StateVectorizer against the .csv, these tests check state_to_vector.py against the same .csv, so both
vectorizers agree feature by feature.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import copy
import glob
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from state_to_vector import (  # noqa: E402
    FEATURE_DIM,
    load_state_yaml,
    load_state_yaml_fast,
    state_to_feature_vector,
    yaml_to_numpy,
)

FIXTURES = os.path.join(REPO, "Tests", "Fixtures", "Vectorizer")


def fixture_files():
    files = sorted(glob.glob(os.path.join(FIXTURES, "*.yaml")))
    if not files:
        raise unittest.SkipTest(f"no vectorizer fixtures in {FIXTURES}")
    return files


def csharp_vector(yaml_path):
    with open(yaml_path[: -len(".yaml")] + ".csv") as stream:
        return np.array([float(value) for value in stream.read().split(",")], dtype=np.float32)


class VectorizerParityTest(unittest.TestCase):
    def test_feature_dim_matches_the_csharp_vectors(self):
        for path in fixture_files():
            self.assertEqual(FEATURE_DIM, csharp_vector(path).shape[0], path)

    def test_python_vectors_match_the_csharp_vectors(self):
        for path in fixture_files():
            expected = csharp_vector(path)
            actual = yaml_to_numpy(path).astype(np.float32)

            # The Python pipeline stores float16, so compare after the same rounding
            expected16 = expected.astype(np.float16).astype(np.float32)
            mismatches = np.flatnonzero(expected16 != actual)
            self.assertEqual(
                0, mismatches.size,
                f"{os.path.basename(path)}: features {mismatches[:10].tolist()} differ "
                f"(C# {expected16[mismatches[:10]].tolist()}, Python {actual[mismatches[:10]].tolist()})",
            )

    def test_equivalent_rolls_give_identical_vectors(self):
        state = load_state_yaml(os.path.join(FIXTURES, "midgame.yaml"))["GameState"]

        a = copy.deepcopy(state)
        b = copy.deepcopy(state)
        a["Turn"]["LastRoll"] = {"First": 1, "Second": 6}
        b["Turn"]["LastRoll"] = {"First": 6, "Second": 1}
        self.assertEqual(state_to_feature_vector(a), state_to_feature_vector(b))

        b["Turn"]["LastRoll"] = {"First": 2, "Second": 6}
        self.assertNotEqual(state_to_feature_vector(a), state_to_feature_vector(b))


    def test_the_last_roll_counts_only_once_someone_rolled(self):
        def with_roll(state, total):
            changed = copy.deepcopy(state)
            changed["Turn"]["LastRoll"] = {"First": total - 1, "Second": 1}
            return state_to_feature_vector(changed)

        # During setup and in the first normal turn before its roll, the last roll is a random placeholder
        setup = load_state_yaml(os.path.join(FIXTURES, "setup.yaml"))["GameState"]
        self.assertEqual(with_roll(setup, 5), with_roll(setup, 9))

        first_turn = copy.deepcopy(setup)
        first_turn["Turn"].update({"TypeOfRound": "Normal", "RoundCounter": 2, "PlayerIndex": 0, "MustRoll": True})
        self.assertEqual(with_roll(first_turn, 5), with_roll(first_turn, 9))

        first_turn["Turn"]["MustRoll"] = False
        self.assertNotEqual(with_roll(first_turn, 5), with_roll(first_turn, 9))


class SaveFileLoadingTest(unittest.TestCase):
    def test_fast_loader_matches_a_full_parse_of_the_current_format(self):
        for path in fixture_files():
            with open(path) as stream:
                self.assertTrue(stream.readline().startswith("Version: 2"), f"{path} is not a current save file")

            fast_state, played_count, undo_count = load_state_yaml_fast(path)
            full = load_state_yaml(path)

            self.assertEqual(len(full.get("PlayedActions") or []), played_count, path)
            self.assertEqual(len(full.get("UndoHistory") or []), undo_count, path)
            self.assertEqual(state_to_feature_vector(full["GameState"]), state_to_feature_vector(fast_state), path)


class BaselineLayoutTest(unittest.TestCase):
    """baselines.py reads victory points from hardcoded player block offsets"""

    def setUp(self):
        try:
            sys.path.insert(0, os.path.join(REPO, "Agents", "Training"))
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            import baselines
        except ImportError as error:
            self.skipTest(f"baselines.py needs torch: {error}")
        self.baselines = baselines

    def test_player_blocks_start_with_the_victory_points(self):
        self.assertEqual(FEATURE_DIM, self.baselines.EXPECTED_VECTOR_DIM)

        for path in fixture_files():
            vector = yaml_to_numpy(path).astype(np.float32)
            players = load_state_yaml(path)["GameState"]["Players"]

            for start, player in zip(self.baselines.PLAYER_BLOCK_STARTS, players):
                expected = sum(float(points) for points in player["VictoryPoints"].values())
                actual = vector[start: start + self.baselines.VP_FIELDS_PER_PLAYER].sum()
                self.assertEqual(expected, actual, f"{os.path.basename(path)}: block at {start}")


if __name__ == "__main__":
    unittest.main()
