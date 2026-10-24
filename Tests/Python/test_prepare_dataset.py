"""The one-step dataset pipeline (DatasetProcessing/prepare_dataset.py): finding datasets at any depth, stopping on
unmatched samples and labels, taking vectors from chunks (extending chunks of the base encoding) and splitting by group.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import glob
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from build_dataset_index import build_dataset_index, find_datasets  # noqa: E402
from prepare_dataset import prepare_dataset  # noqa: E402
from state_to_vector import BASE_FEATURE_DIM, FEATURE_DIM, yaml_to_numpy  # noqa: E402

FIXTURES = sorted(glob.glob(os.path.join(REPO, "Tests", "Fixtures", "Vectorizer", "*.yaml")))


def write_dataset(directory, matches, chunk_width=None):
    """A dataset of the fixture states: match m gets samples r<m>_s0.., labels with all win rates 1/4, optionally a vector chunk"""
    os.makedirs(os.path.join(directory, "input"))
    os.makedirs(os.path.join(directory, "output"))
    names, rows = [], []
    for match in range(matches):
        for sample in range(2):
            fixture = FIXTURES[(2 * match + sample) % len(FIXTURES)]
            name = f"r{match}_s{sample}"
            shutil.copy(fixture, os.path.join(directory, "input", name + ".yaml"))
            with open(os.path.join(directory, "output", name + ".txt"), "w") as file:
                file.write("0.25\n0.25\n0.25\n0.25\n")
            names.append(name)
            rows.append(yaml_to_numpy(fixture))

    if chunk_width is not None:
        os.makedirs(os.path.join(directory, "vectors"))
        np.stack(rows)[:, :chunk_width].astype("<f2").tofile(os.path.join(directory, "vectors", "w00_c0000.f16"))
        with open(os.path.join(directory, "vectors", "w00_c0000.txt"), "w") as file:
            file.write("".join(name + "\n" for name in names))
        with open(os.path.join(directory, "metadata.json"), "w") as file:
            json.dump({"feature_dim": chunk_width}, file)
    return names


class PrepareDatasetTest(unittest.TestCase):
    def setUp(self):
        if not FIXTURES:
            raise unittest.SkipTest("no vectorizer fixtures")
        self.root = tempfile.mkdtemp()
        self.base = os.path.join(self.root, "datasets")
        # A partial run with chunks of the base encoding (collected before the derived features) and an older dataset without chunks
        write_dataset(os.path.join(self.base, "v2", "part_00"), matches=4, chunk_width=BASE_FEATURE_DIM)
        write_dataset(os.path.join(self.base, "old"), matches=3)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_datasets_are_found_at_any_depth(self):
        self.assertEqual(["old", "v2/part_00"], find_datasets(self.base))

    def test_the_pipeline_vectorizes_and_splits_every_sample(self):
        output = os.path.join(self.root, "prepared")
        prepare_dataset(self.base, output, workers=1, test=0.25, val=0.25)

        states = np.load(os.path.join(output, "states.npy"))
        groups = np.load(os.path.join(output, "groups.npy"))
        self.assertEqual((14, FEATURE_DIM), states.shape)
        self.assertEqual(7, np.unique(groups).size)  # one group per match, matches of different datasets apart

        # Every row is the sample's vector, whether it came from an extended chunk or from the YAML file
        with open(os.path.join(output, "dataset_index.csv")) as file:
            rows = [line.strip().split(",") for line in file.readlines()[1:]]
        for entry_idx, _, subdir_name, entry_name in rows:
            expected = yaml_to_numpy(os.path.join(self.base, subdir_name, "input", entry_name + ".yaml"))
            np.testing.assert_array_equal(expected, states[int(entry_idx)])

        # The splits cover every sample once and keep each match in one split
        splits = [np.load(os.path.join(output, f"idx_{name}.npy")) for name in ("train", "val", "test")]
        self.assertEqual(list(range(14)), sorted(np.concatenate(splits).tolist()))
        for a in range(3):
            for b in range(a + 1, 3):
                self.assertEqual(0, np.intersect1d(groups[splits[a]], groups[splits[b]]).size)

        # Without raw playout results the extra labels are missing (NaN), not made up
        self.assertTrue(np.isnan(np.load(os.path.join(output, "placements.npy"))).all())

    def test_unmatched_samples_and_labels_stop_the_index(self):
        os.remove(os.path.join(self.base, "old", "output", "r1_s0.txt"))
        with self.assertRaises(ValueError):
            build_dataset_index(self.base)
        self.assertEqual(13, len(build_dataset_index(self.base, allow_missing=True)))


if __name__ == "__main__":
    unittest.main()
