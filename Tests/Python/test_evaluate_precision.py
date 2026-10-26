"""Tests for evaluate_precision.py: the metrics on small hand-made arrays (no model needed)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Agents", "Training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "DatasetProcessing"))

try:
    from evaluate_precision import (calibration, effective_seeds, kl_divergence, noise_floor, paired_difference,  # noqa: E402
                                    ranking_metrics, sibling_pairs, victory_points)
    from playout_results import effective_match_seed  # noqa: E402
    from state_to_vector import _PLAYER_SIZE, _PLAYER_START  # noqa: E402
    IMPORT_ERROR = None
except ImportError as error:  # numpy-only environments still run the other tests
    IMPORT_ERROR = error


@unittest.skipIf(IMPORT_ERROR is not None, f"dependencies missing: {IMPORT_ERROR}")
class EvaluatePrecisionTest(unittest.TestCase):
    def test_effective_seeds_match_the_scalar_version(self):
        seeds = np.array([5, 5 << 32, 0xFFFFFFFB, 0x80000000, 12345678901234567, (1 << 64) - 1], dtype=np.uint64)
        self.assertEqual([effective_match_seed(int(s)) for s in seeds], effective_seeds(seeds).tolist())

    def test_kl_is_zero_for_the_labels_and_positive_otherwise(self):
        labels = np.array([[0.5, 0.5, 0, 0], [0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_allclose(kl_divergence(labels, labels), 0, atol=1e-9)
        self.assertTrue((kl_divergence(labels, np.full((2, 4), 0.25))[0] > 0))
        # Players without a chance in the labels add nothing, wherever the prediction puts them
        np.testing.assert_allclose(kl_divergence(labels[:1], np.array([[0.5, 0.5, 0, 0]])), 0, atol=1e-9)

    def test_the_noise_floor_shrinks_with_more_playouts(self):
        labels = np.array([[0.4, 0.3, 0.2, 0.1], [1, 0, 0, 0]])
        np.testing.assert_allclose(noise_floor(labels, np.array([2000, 2000])), [3 / 4000, 0])
        self.assertLess(noise_floor(labels, np.array([4000, 0]))[0], noise_floor(labels, np.array([2000, 0]))[0])

    def test_ranking_counts_agreeing_pairs_and_the_best_move(self):
        # One group of three siblings of player 1 (rows 1-3; row 0 is the root and stays out)
        labels = np.zeros((4, 4))
        labels[1:, 1] = [0.30, 0.20, 0.10]
        groups = sibling_pairs(np.array([0, 0, 0, 0]), np.array([False, True, True, True]), np.array([1, 1, 1, 1]))
        self.assertEqual(1, len(groups))
        self.assertEqual([1, 2, 3], groups[0][0].tolist())

        perfect = ranking_metrics(labels, labels, groups)
        self.assertEqual(1.0, perfect["pair_accuracy_0"])
        self.assertEqual(1.0, perfect["best_move"])
        self.assertEqual(0.0, perfect["regret_pp"])

        reversed_ = labels.copy()
        reversed_[1:, 1] = [0.10, 0.20, 0.30]
        worst = ranking_metrics(labels, reversed_, groups)
        self.assertEqual(0.0, worst["pair_accuracy_0"])
        self.assertEqual(0.0, worst["best_move"])
        self.assertAlmostEqual(20.0, worst["regret_pp"])
        # All three pairs differ by at least 10pp, so all of them count for the 3pp threshold too
        self.assertEqual(3, worst["pairs_0.03"])

    def test_calibration_buckets_by_points_behind_the_leader(self):
        vp = np.array([[5, 3, 2, 0]])
        labels = np.array([[0.7, 0.2, 0.1, 0.0]])
        rows = {name: (count, predicted, actual) for name, count, predicted, actual in calibration(labels, labels, vp)}
        self.assertEqual((1, 0.7, 0.7), rows["0"])
        self.assertEqual((1, 0.2, 0.2), rows["2"])
        self.assertEqual((1, 0.1, 0.1), rows["3-4"])
        self.assertEqual((1, 0.0, 0.0), rows["5+"])

    def test_victory_points_sum_the_trackers_of_each_player(self):
        states = np.zeros((1, _PLAYER_START + 4 * _PLAYER_SIZE), dtype=np.float32)
        states[0, _PLAYER_START + 1 * _PLAYER_SIZE: _PLAYER_START + 1 * _PLAYER_SIZE + 5] = [2, 2, 1, 2, 0]
        self.assertEqual([0, 7, 0, 0], victory_points(states)[0].tolist())

    def test_paired_differences_are_clustered_by_match(self):
        reference = np.zeros(6)
        values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        mean, half = paired_difference(values, reference, np.array([1, 1, 1, 2, 2, 2]))
        self.assertAlmostEqual(0.5, mean)
        # Two clusters that disagree completely: a wide interval, much wider than for six independent samples
        _, independent = paired_difference(values, reference, np.arange(6))
        self.assertGreater(half, independent)


if __name__ == "__main__":
    unittest.main()
