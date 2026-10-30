"""Turning self-play runs into training arrays (DatasetProcessing/selfplay_to_dataset.py).

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from selfplay_to_dataset import convert  # noqa: E402

FEATURES = 5
GAME_HEADER = "game,seed,finished,winner,vp0,vp1,vp2,vp3,rounds,actions,positions,searched_moves,seconds,record_chunk,record_offset,record_length"
SAMPLE_HEADER = "game,position,action,round,actor,iterations,root_visits,q0,q1,q2,q3,chunk,row,visits"


def write_run(directory, games, samples, chunks):
    """A run directory the way SelfPlayCollector writes it (with a tiny feature layout)."""
    os.makedirs(directory)
    with open(os.path.join(directory, "metadata.json"), "w") as file:
        json.dump({"model": {"feature_dim": FEATURES}}, file)
    with open(os.path.join(directory, "games.csv"), "w", newline="") as file:
        file.write(GAME_HEADER + "\n" + "".join(line + "\n" for line in games))
    with open(os.path.join(directory, "samples.csv"), "w", newline="") as file:
        file.write(SAMPLE_HEADER + "\n" + "".join(line + "\n" for line in samples))
    for chunk, rows in enumerate(chunks):
        np.asarray(rows, dtype="<f2").tofile(os.path.join(directory, f"states_c{chunk:04d}.f16"))


class SelfPlayToDatasetTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.base)

    def test_positions_of_games_with_a_result_get_their_labels_and_rows(self):
        rows = np.arange(4 * FEATURES, dtype=np.float32).reshape(4, FEATURES)
        write_run(
            os.path.join(self.base, "run"),
            games=[
                # Game 7: player 2 wins at round 30 with 10 VP, 1 and 3 share 2nd/3rd place with 5 VP
                "7,1001,1,2,3,5,10,5,30,400,2,150,12.0,0,0,100",
                # Game 8 was stopped at the action limit: no result, left out
                "8,1002,0,-1,4,4,4,4,90,3000,1,900,99.0,0,100,100",
                "9,1003,1,0,10,2,2,2,25,300,1,100,9.0,0,200,100",
            ],
            samples=[
                "7,0,10,3,1,800,810,0.1,0.2,0.6,0.1,0,0,0:500 1:300",
                "7,1,90,20,2,800,950,0.0,0.0,1.0,0.0,1,0,0:950",
                "8,0,50,10,0,800,800,0.25,0.25,0.25,0.25,0,1,0:800",
                "9,0,40,8,3,800,800,0.4,0.2,0.2,0.2,0,2,2:800",
            ],
            chunks=[rows[:3], rows[3:]],
        )
        output = os.path.join(self.base, "prepared")
        self.assertEqual(3, convert([os.path.join(self.base, "run")], output, target_lambda=0.5, val=0.5))

        def load(name):
            return np.load(os.path.join(output, f"{name}.npy"))

        np.testing.assert_array_equal(load("states"), np.stack([rows[0], rows[3], rows[2]]).astype(np.float16))
        np.testing.assert_array_equal(load("winners"), [2, 2, 0])
        np.testing.assert_allclose(load("root_q")[0], [0.1, 0.2, 0.6, 0.1], rtol=1e-6)
        np.testing.assert_allclose(load("win_rates")[0], [0.05, 0.1, 0.8, 0.05], rtol=1e-6)
        np.testing.assert_allclose(load("win_rates")[2], [0.7, 0.1, 0.1, 0.1], rtol=1e-6)
        np.testing.assert_allclose(load("placements")[0], [[0, 0, 0, 1], [0, 0.5, 0.5, 0], [1, 0, 0, 0], [0, 0.5, 0.5, 0]])
        np.testing.assert_allclose(load("vp_shares")[0], np.array([3, 5, 10, 5]) / 23, rtol=1e-6)
        np.testing.assert_array_equal(load("rounds_left"), [27, 10, 17])
        np.testing.assert_array_equal(load("actors"), [1, 2, 3])
        np.testing.assert_array_equal(load("match_seeds"), [1001, 1001, 1003])

        # Both positions of a game land in the same split, and every position in exactly one
        train, val = load("idx_train"), load("idx_val")
        self.assertEqual([0, 1, 2], sorted(np.concatenate([train, val]).tolist()))
        self.assertTrue({0, 1} <= set(train.tolist()) or {0, 1} <= set(val.tolist()))
        self.assertEqual(0, len(load("idx_test")))

    def test_policy_files_become_visit_shares_per_move(self):
        run = os.path.join(self.base, "run")
        write_run(run, ["1,5,1,0,10,2,2,2,20,100,2,50,1.0,0,0,1", "2,6,0,-1,4,4,4,4,90,3000,1,900,99.0,0,1,1"],
                  ["1,0,5,2,0,800,800,1,0,0,0,0,0,0:600 2:200", "2,0,7,3,1,800,800,0.25,0.25,0.25,0.25,0,1,1:800",
                   "1,1,9,4,1,800,800,0,1,0,0,0,2,1:800"], [np.zeros((3, FEATURES))])
        # Three positions with 3, 2 and 2 moves of up to 2 tokens; the second belongs to a game without a result
        np.array([3, 2, 2], dtype="<i4").tofile(os.path.join(run, "policy_counts.i32"))
        tokens = np.array([[0, -1], [5, 20], [6, 21], [1, -1], [2, -1], [0, -1], [7, 30]], dtype="<i2")
        tokens.tofile(os.path.join(run, "policy_tokens.i16"))
        np.where(tokens >= 0, 1, 0).astype(np.uint8).tofile(os.path.join(run, "policy_weights.u8"))
        with open(os.path.join(run, "policy_layout.json"), "w") as file:
            json.dump({"max_tokens": [2], "vocabulary": [0, 40]}, file)

        output = os.path.join(self.base, "prepared")
        self.assertEqual(2, convert([run], output, val=0.0))

        def load(name):
            return np.load(os.path.join(output, f"{name}.npy"))

        np.testing.assert_array_equal(load("policy_offsets"), [0, 3, 5])
        np.testing.assert_array_equal(load("policy_tokens"), tokens[[0, 1, 2, 5, 6]])
        np.testing.assert_allclose(load("policy_targets"), [0.75, 0, 0.25, 0, 1])
        self.assertTrue(os.path.exists(os.path.join(output, "policy_layout.json")))

    def test_view_features_follow_the_positions(self):
        # views.f16 (SelfPlayCollector view-features) holds one row per line of samples.csv, of its own width
        run = os.path.join(self.base, "run")
        write_run(run, ["1,5,1,0,10,2,2,2,20,100,2,50,1.0,0,0,1", "2,6,0,-1,4,4,4,4,90,3000,1,900,99.0,0,1,1"],
                  ["1,0,5,2,0,800,800,1,0,0,0,0,0,0:800", "2,0,7,3,1,800,800,0.25,0.25,0.25,0.25,0,1,1:800",
                   "1,1,9,4,1,800,800,0,1,0,0,0,2,1:800"], [np.zeros((3, FEATURES))])
        views = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
        views.astype("<f2").tofile(os.path.join(run, "views.f16"))

        output = os.path.join(self.base, "prepared")
        self.assertEqual(2, convert([run], output, val=0.0))
        np.testing.assert_array_equal(np.load(os.path.join(output, "views.npy")), views[[0, 2]].astype(np.float16))

        # A run without them gives no views for the mix
        other = os.path.join(self.base, "other")
        write_run(other, ["3,7,1,0,10,2,2,2,20,100,1,50,1.0,0,0,1"], ["3,0,5,2,0,800,800,1,0,0,0,0,0,0:800"], [np.zeros((1, FEATURES))])
        mixed = os.path.join(self.base, "mixed")
        convert([run, other], mixed, val=0.0)
        self.assertFalse(os.path.exists(os.path.join(mixed, "views.npy")))

    def test_runs_of_different_feature_layouts_are_rejected(self):
        write_run(os.path.join(self.base, "a"), ["1,5,1,0,10,2,2,2,20,100,1,50,1.0,0,0,1"],
                  ["1,0,5,2,0,800,800,1,0,0,0,0,0,0:800"], [np.zeros((1, FEATURES))])
        other = os.path.join(self.base, "b")
        write_run(other, [], [], [])
        with open(os.path.join(other, "metadata.json"), "w") as file:
            json.dump({"model": {"feature_dim": FEATURES + 1}}, file)
        with self.assertRaises(ValueError):
            convert([os.path.join(self.base, "a"), other], os.path.join(self.base, "out"))


if __name__ == "__main__":
    unittest.main()
