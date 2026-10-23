"""Tests for DatasetProcessing/vector_chunks.py, the reader of the vector chunks the dataset collector writes.

Run from the repository root:  python -m unittest discover -s Tests/Python -v
"""

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DatasetProcessing"))

from state_to_vector import FEATURE_DIM  # noqa: E402
from vector_chunks import load_vector_chunks  # noqa: E402


def write_chunk(dataset_dir, name, rows, sample_names):
    vectors_dir = os.path.join(dataset_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    np.asarray(rows, dtype="<f2").tofile(os.path.join(vectors_dir, name + ".f16"))
    with open(os.path.join(vectors_dir, name + ".txt"), "w") as index_file:
        index_file.writelines(sample + "\n" for sample in sample_names)


def random_rows(count, seed):
    return np.random.default_rng(seed).random((count, FEATURE_DIM)).astype(np.float16)


class VectorChunkTests(unittest.TestCase):
    def test_rows_of_all_chunks_are_found_by_sample_name(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            first, second = random_rows(3, 1), random_rows(2, 2)
            write_chunk(dataset_dir, "w00_c0000", first, ["r0_s0", "r0_s1", "r0_s2"])
            write_chunk(dataset_dir, "w01_c0000", second, ["r1_s0", "r1_s1"])

            vectors = load_vector_chunks(dataset_dir)

            self.assertEqual(sorted(vectors), ["r0_s0", "r0_s1", "r0_s2", "r1_s0", "r1_s1"])
            np.testing.assert_array_equal(vectors["r0_s2"], first[2])
            np.testing.assert_array_equal(vectors["r1_s0"], second[0])
            self.assertEqual(vectors["r1_s1"].dtype, np.float16)

    def test_datasets_without_chunks_give_no_vectors(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            self.assertEqual(load_vector_chunks(dataset_dir), {})

    def test_names_without_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            write_chunk(dataset_dir, "w00_c0000", random_rows(1, 3), ["r0_s0", "r0_s1"])
            with self.assertRaises(ValueError):
                load_vector_chunks(dataset_dir)

    def test_partial_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            write_chunk(dataset_dir, "w00_c0000", random_rows(1, 4)[:, :-1], ["r0_s0"])
            with self.assertRaises(ValueError):
                load_vector_chunks(dataset_dir)

    def test_rows_without_names_from_an_interrupted_run_are_ignored(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            rows = random_rows(2, 5)
            write_chunk(dataset_dir, "w00_c0000", rows, ["r0_s0"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                vectors = load_vector_chunks(dataset_dir)

            self.assertEqual(list(vectors), ["r0_s0"])
            np.testing.assert_array_equal(vectors["r0_s0"], rows[0])
            self.assertEqual(len(caught), 1)

    def test_samples_in_two_chunks_are_rejected(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            write_chunk(dataset_dir, "w00_c0000", random_rows(1, 6), ["r0_s0"])
            write_chunk(dataset_dir, "w01_c0000", random_rows(1, 7), ["r0_s0"])
            with self.assertRaises(ValueError):
                load_vector_chunks(dataset_dir)


if __name__ == "__main__":
    unittest.main()
