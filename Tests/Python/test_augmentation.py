"""
Tests for augmentation.py: transforming a game state itself and vectorizing it must give exactly the transformed vector
of the original state, for the player rotation (players, owners, the player to act, awaited discards) and the number
token pairs (tile numbers, the robber's tile, the dice of the last roll).
"""

import copy
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Agents", "Training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "DatasetProcessing"))

from augmentation import (NUMBER_PAIR_COUNT, _LAST_ROLL, _NUMBERS, is_rotatable, rotate_features,  # noqa: E402
                          rotate_player_index, rotate_players, swap_number_pairs)
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


def swap_numbers_in_state(state, mask):
    """The same game with the number tokens of the pairs in mask swapped (n and 14 - n), the last roll along"""
    swapped = copy.deepcopy(state)

    def is_swapped(n):
        low = min(n, 14 - n)
        return 2 <= low <= 6 and (mask >> (low - 2)) & 1

    board = swapped["Board"]
    for tile in board["Map"]["Values"] + [board["Robber"]]:
        if tile["Number"] not in (None, "") and is_swapped(int(tile["Number"])):
            tile["Number"] = 14 - int(tile["Number"])
    # Turning both dice over (d => 7 - d) turns the total t into 14 - t
    roll = swapped["Turn"].get("LastRoll") or {}
    if roll.get("First") and roll.get("Second") and is_swapped(int(roll["First"]) + int(roll["Second"])):
        roll["First"], roll["Second"] = 7 - int(roll["First"]), 7 - int(roll["Second"])
    return swapped


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


class NumberPairTest(unittest.TestCase):
    ALL = ("initial", "setup", "first_roll", "midgame", "robber", "discard", "ended")

    def test_swapping_numbers_in_the_game_state_gives_the_swapped_vector(self):
        for name in self.ALL:
            state = fixture(name)
            original = vector(state)
            for mask in range(2 ** NUMBER_PAIR_COUNT):
                np.testing.assert_allclose(swap_number_pairs(original, mask), vector(swap_numbers_in_state(state, mask)),
                                           atol=1e-6, err_msg=f"{name}, mask {mask}")

    def test_only_the_numbers_and_the_last_roll_change(self):
        original = vector(fixture("robber"))
        swapped = swap_number_pairs(original, 2 ** NUMBER_PAIR_COUNT - 1)
        changed = np.flatnonzero(swapped != original)
        self.assertTrue(set(changed) <= set(_NUMBERS) | {_LAST_ROLL})
        self.assertGreater(len(changed), 10)
        np.testing.assert_array_equal(original, swap_number_pairs(swapped, 2 ** NUMBER_PAIR_COUNT - 1))
        np.testing.assert_array_equal(original, swap_number_pairs(original, 0))

    def test_each_state_of_a_batch_gets_its_own_mask(self):
        states = np.stack([vector(fixture("midgame")), vector(fixture("robber"))]).astype(np.float16)
        swapped = swap_number_pairs(states, np.array([0, 31]))
        self.assertEqual(np.float16, swapped.dtype)
        np.testing.assert_array_equal(states[0], swapped[0])
        np.testing.assert_array_equal(swap_number_pairs(states[1], 31), swapped[1])


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

            # Number pairs: any of the 32 views, also for the setup state, the labels stay
            dataset_numbers = StateDataset(os.path.join(directory, "states.npy"), os.path.join(directory, "labels.npy"),
                                           directory, augment_numbers=True)
            masks = set()
            for _ in range(200):
                for row in (0, 1):
                    item = dataset_numbers[row]
                    mask = next(m for m in range(32) if np.array_equal(item["x"].numpy(), swap_number_pairs(states[row], m)))
                    np.testing.assert_array_equal(item["y"].numpy(), labels[row])
                    if row == 0:
                        masks.add(mask)
            self.assertGreater(len(masks), 8)
            del dataset, dataset_numbers


if __name__ == "__main__":
    unittest.main()
