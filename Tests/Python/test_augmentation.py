"""
Tests for the player rotation in augmentation.py: rotating a game state itself (players, owners, the player to act,
awaited discards) and vectorizing it must give exactly the permuted vector of the original state.
"""

import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Agents", "Training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "DatasetProcessing"))

from augmentation import is_rotatable, rotate_features, rotate_player_index, rotate_players  # noqa: E402
from state_to_vector import load_state_yaml, state_to_feature_vector  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "Fixtures", "Vectorizer")


def rotate_state(state, k):
    """The same game with player p relabeled as p + k"""
    rotated = copy.deepcopy(state)
    players = rotated["Players"]
    rotated["Players"] = players[-k:] + players[:-k] if k else players
    for element in rotated["Board"]["Intersections"] + rotated["Board"]["Edges"]:
        element["Owner"] = rotate_player_index(int(element["Owner"]), k)
    turn = rotated["Turn"]
    turn["PlayerIndex"] = rotate_player_index(int(turn["PlayerIndex"]), k)
    discards = list(turn["AwaitedPlayerDiscards"])
    turn["AwaitedPlayerDiscards"] = discards[-k:] + discards[:-k] if k else discards
    return rotated


def vector(state):
    return np.array(state_to_feature_vector(copy.deepcopy(state)), dtype=np.float64)


def fixture(name):
    return load_state_yaml(os.path.join(FIXTURES, f"{name}.yaml"))["GameState"]


class PlayerRotationTest(unittest.TestCase):
    # States after the setup phase. first_roll (the very first turn) is left out: the vectorizer encodes its last roll
    # as "nobody rolled yet" only for player 0, a rule of the encoding and not of the game.
    AFTER_SETUP = ("midgame", "robber", "discard", "ended")

    def test_rotating_the_game_state_gives_the_permuted_vector(self):
        for name in self.AFTER_SETUP:
            state = fixture(name)
            original = vector(state)
            self.assertTrue(is_rotatable(original), name)
            for k in range(1, 4):
                np.testing.assert_allclose(rotate_features(original, k), vector(rotate_state(state, k)), atol=1e-6,
                                           err_msg=f"{name}, rotated by {k}")

    def test_the_discard_state_moves_its_awaited_discards(self):
        state = fixture("discard")
        self.assertTrue(any(state["Turn"]["AwaitedPlayerDiscards"]))
        original = vector(state)
        self.assertFalse(np.array_equal(original, rotate_features(original, 1)))

    def test_four_rotations_are_the_identity_and_setup_states_are_never_rotated(self):
        original = vector(fixture("midgame"))
        np.testing.assert_array_equal(original, rotate_features(rotate_features(rotate_features(rotate_features(original, 1), 1), 1), 1))
        np.testing.assert_array_equal(original, rotate_features(original, 0))
        for name in ("initial", "setup"):
            self.assertFalse(is_rotatable(vector(fixture(name))), name)

    def test_labels_rotate_with_their_player(self):
        win_rates = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal([0.4, 0.1, 0.2, 0.3], rotate_players(win_rates, 1))
        placements = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(placements[3], rotate_players(placements, 1, axis=0)[0])
        self.assertEqual(0, rotate_player_index(3, 1))
        self.assertEqual(-1, rotate_player_index(-1, 2))


class AugmentedDatasetTest(unittest.TestCase):
    def test_features_and_labels_of_an_item_are_rotated_together(self):
        try:
            import tempfile
            from train import StateDataset
        except ImportError as error:
            self.skipTest(f"training dependencies missing: {error}")

        states = np.stack([vector(fixture("midgame")), vector(fixture("setup"))]).astype(np.float32)
        labels = np.array([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
        # Windows keeps memory-mapped files locked until the dataset is gone, so the cleanup may not delete them at once
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            np.save(os.path.join(directory, "states.npy"), states)
            np.save(os.path.join(directory, "labels.npy"), labels)
            np.save(os.path.join(directory, "actors.npy"), np.array([2, -1], dtype=np.int8))
            dataset = StateDataset(os.path.join(directory, "states.npy"), os.path.join(directory, "labels.npy"), directory,
                                   augment_rotation=True)

            seen = set()
            for _ in range(200):
                item = dataset[0]
                k = next(k for k in range(4) if np.allclose(item["x"].numpy(), rotate_features(states[0], k)))
                np.testing.assert_allclose(item["y"].numpy(), rotate_players(labels[0], k))
                self.assertEqual(rotate_player_index(2, k), int(item["actors"]))
                seen.add(k)
                # The setup state is never rotated
                np.testing.assert_array_equal(dataset[1]["x"].numpy(), states[1])
            self.assertEqual({0, 1, 2, 3}, seen)
            del dataset


if __name__ == "__main__":
    unittest.main()
