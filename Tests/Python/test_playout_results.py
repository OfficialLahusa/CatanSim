"""Reading the dataset collector's raw playout results and deriving labels from them (DatasetProcessing/playout_results.py).

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from playout_results import PLAYOUT_DTYPE, build_label_arrays, derive_labels, effective_match_seed, load_samples, placements  # noqa: E402

HEADER = ("name,group,match,sample,position,match_length,round,terminal,playouts,finished,playout_offset,"
          "match_winner,match_vp,match_rounds,match_agents,end_winner,end_vp,sibling_group,source,actor,move,chance")


def playout(winner, vp, end_round, agents=(2, 2, 1, 0)):
    record = np.zeros(1, dtype=PLAYOUT_DTYPE)
    record["winner"], record["vp"], record["agents"], record["end_round"] = winner, vp, agents, end_round
    return record


def write_dataset(directory, worker, lines, records):
    """Writes one worker's files the way the collector does (header, one line per sample, 12-byte records)."""
    os.makedirs(os.path.join(directory, "samples"), exist_ok=True)
    os.makedirs(os.path.join(directory, "playouts"), exist_ok=True)
    with open(os.path.join(directory, "samples", f"{worker}.csv"), "w", newline="") as file:
        file.write(HEADER + "\n" + "".join(line + "\n" for line in lines))
    np.concatenate(records).tofile(os.path.join(directory, "playouts", f"{worker}.bin"))


class PlayoutResultsTest(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.directory = os.path.join(self.base, "run")
        self.records = [
            playout(1, [4, 10, 4, 7], 30),
            playout(1, [3, 11, 6, 6], 34),
            playout(2, [5, 8, 10, 9], 40),
            playout(-1, [6, 6, 6, 6], 50000),  # hit the round limit
        ]
        self.records += [
            playout(1, [4, 10, 3, 7], 31),  # sibling r0_s0_c0, same seeds as r0_s0
            playout(3, [5, 9, 4, 10], 36),
            playout(1, [3, 10, 6, 6], 35),
            playout(-1, [6, 6, 6, 6], 50000),
        ]
        write_dataset(self.directory, "w00", [
            "r0_s0,123,0,0,40,200,8,0,4,3,0,1,4/10/5/5,32,BBSR,,,r0_s0,root,1,,",
            "r0_s0_c0,123,0,0,41,200,8,0,4,3,4,1,4/10/5/5,32,BBSR,,,r0_s0,match,1,RoadAction,0",
            "r0_s0_c1,123,0,0,41,200,8,1,0,0,8,1,4/10/5/5,32,BBSR,1,6/10/2/3,r0_s0,random,1,SettlementAction,0",
            "r0_s1,123,0,1,200,200,32,1,0,0,8,1,4/10/5/5,32,BBSR,1,4/10/5/5,,,,,",
        ], self.records)

    def tearDown(self):
        shutil.rmtree(self.base)

    def test_samples_refer_to_their_playouts(self):
        samples = load_samples(self.directory)
        self.assertEqual(["r0_s0", "r0_s0_c0", "r0_s0_c1", "r0_s1"], sorted(samples))
        self.assertEqual(4, len(samples["r0_s0"].playouts))
        self.assertEqual(0, len(samples["r0_s1"].playouts))
        self.assertEqual(123, samples["r0_s0"].match_seed)
        self.assertTrue(samples["r0_s1"].terminal)
        self.assertEqual("BBSR", samples["r0_s0"].match_agents)

    def test_labels_come_from_the_finished_playouts(self):
        labels = derive_labels(load_samples(self.directory)["r0_s0"])

        np.testing.assert_allclose(labels.win_rates, [0, 2 / 3, 1 / 3, 0])
        self.assertEqual(3, labels.finished)
        self.assertAlmostEqual(np.mean([30 - 8, 34 - 8, 40 - 8]), labels.rounds_left)
        np.testing.assert_allclose(labels.vp_shares, np.mean([[4 / 25, 10 / 25, 4 / 25, 7 / 25],
                                                              [3 / 26, 11 / 26, 6 / 26, 6 / 26],
                                                              [5 / 32, 8 / 32, 10 / 32, 9 / 32]], axis=0))
        np.testing.assert_allclose(labels.placements.sum(axis=0), 1)
        np.testing.assert_allclose(labels.placements.sum(axis=1), 1)

    def test_siblings_name_their_group_and_an_ended_sibling_its_own_result(self):
        samples = load_samples(self.directory)
        self.assertEqual({"r0_s0"}, {s.sibling_group for s in samples.values() if s.sibling_group})
        self.assertEqual(("root", "match", "random"), tuple(samples[n].source for n in ("r0_s0", "r0_s0_c0", "r0_s0_c1")))
        self.assertEqual("", samples["r0_s1"].sibling_group)
        self.assertEqual(-1, samples["r0_s1"].actor)

        # The winning settlement ends the match there, whatever the match's own result was
        labels = derive_labels(samples["r0_s0_c1"])
        np.testing.assert_allclose(labels.win_rates, [0, 1, 0, 0])
        np.testing.assert_allclose(labels.vp_shares, np.array([6, 10, 2, 3]) / 21)
        np.testing.assert_allclose(derive_labels(samples["r0_s0_c0"]).win_rates, [0, 2 / 3, 0, 1 / 3])

    def test_ended_matches_are_labeled_by_their_result(self):
        labels = derive_labels(load_samples(self.directory)["r0_s1"])

        np.testing.assert_allclose(labels.win_rates, [0, 1, 0, 0])
        self.assertEqual(0, labels.rounds_left)
        # Players 2 and 3 tie for 2nd and 3rd place, player 0 is last
        np.testing.assert_allclose(labels.placements, [[0, 0, 0, 1], [1, 0, 0, 0], [0, .5, .5, 0], [0, .5, .5, 0]])

    def test_the_winner_places_first_and_ties_split_their_places(self):
        # The winner is first even without the most VP, the others tie for 2nd to 4th
        result = placements(np.array([0]), np.array([[9, 9, 9, 9]]))[0]
        np.testing.assert_allclose(result, [[1, 0, 0, 0]] + [[0, 1 / 3, 1 / 3, 1 / 3]] * 3)

    def test_label_arrays_follow_the_index(self):
        index = [[0, 0, "run", "r0_s1"], [1, 0, "run", "r0_s0"], [2, 1, "old_run", "r5_s2"], [3, 0, "run", "r0_s0_c1"]]
        os.makedirs(os.path.join(self.base, "old_run"))  # an older dataset without raw results

        arrays = build_label_arrays(self.base, index)

        self.assertEqual([1, 3, 0, 1], arrays["finished_playouts"].tolist())
        self.assertEqual([123, 123, 0, 123], arrays["match_seeds"].tolist())
        self.assertEqual([-1, 0, -1, 0], arrays["sibling_groups"].tolist())
        self.assertEqual([-1, 1, -1, 1], arrays["actors"].tolist())
        self.assertTrue(np.isnan(arrays["rounds_left"][2]))
        np.testing.assert_allclose(arrays["placements"][0, 1], [1, 0, 0, 0])

    def test_match_seeds_that_give_the_same_dotnet_random_share_their_effective_seed(self):
        # Folded to 32 bits like Seeds.ToInt32, then the absolute value like Random(int)
        self.assertEqual(5, effective_match_seed(5))
        self.assertEqual(5, effective_match_seed(5 << 32))           # same low and high halves after folding
        self.assertEqual(5, effective_match_seed(0xFFFFFFFB))        # folds to -5
        self.assertEqual(2**31 - 1, effective_match_seed(0x80000000))  # int.MinValue
        self.assertNotEqual(effective_match_seed(5), effective_match_seed(6))


if __name__ == "__main__":
    unittest.main()
