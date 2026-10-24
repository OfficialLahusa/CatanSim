"""
Split an indexed dataset by group (all samples of a match land in the same split) into train, validation and test.

Writes idx_train.npy, idx_val.npy and idx_test.npy (rows of states.npy etc., which train.py reads) and the groups of
each split. Only groups.npy is needed; --copies also writes X_*/y_* copies of the states and win rates per split
(about as large as the dataset itself).

Usage: python split_dataset.py [--groups groups.npy] [--output-dir .] [--test 0.15] [--val 0.15] [--seed 42]
                               [--copies --states states.npy --labels win_rates.npy]
"""
import argparse
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def split_indices(groups_vector: np.ndarray, test_proportion: float, val_proportion: float, seed: int):
    # Group Split TrainVal/Test
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_proportion, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(np.arange(groups_vector.shape[0]), groups=groups_vector))

    # Group Split Train/Val
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_proportion*(1/(1-test_proportion)), random_state=seed)
    train_idx, val_idx = next(splitter.split(trainval_idx, groups=groups_vector[trainval_idx]))
    train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]

    # Verify no overlaps and group cohesion
    for a, b in [(train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)]:
        assert np.intersect1d(a, b).size == 0
        assert np.intersect1d(groups_vector[a], groups_vector[b]).size == 0
    assert train_idx.size + val_idx.size + test_idx.size == groups_vector.shape[0]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def split_dataset(group_vec_path: Path, output_dir: Path, test_proportion: float, val_proportion: float, seed: int,
                  state_mat_path: Path = None, win_rate_mat_path: Path = None):
    groups_vector = np.load(group_vec_path)
    print(f"Group label vector: shape {groups_vector.shape}, dtype {groups_vector.dtype}, {np.unique(groups_vector).size} groups")

    train_idx, val_idx, test_idx = split_indices(groups_vector, test_proportion, val_proportion, seed)
    print(f"Split sizes: train {train_idx.shape}, val {val_idx.shape}, test {test_idx.shape} (no overlap, groups kept together)")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.save(output_dir / f"idx_{name}.npy", indices)
        np.save(output_dir / f"groups_{name}.npy", groups_vector[indices])

    # Optional copies of the rows per split (memory mapped, so the dataset doesn't need to fit in RAM)
    if state_mat_path is not None and win_rate_mat_path is not None:
        state_matrix = np.load(state_mat_path, mmap_mode="r")
        win_rate_matrix = np.load(win_rate_mat_path, mmap_mode="r")
        for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            np.save(output_dir / f"X_{name}.npy", state_matrix[indices])
            np.save(output_dir / f"y_{name}.npy", win_rate_matrix[indices])

    print(f"Saved dataset splits to \"{output_dir}\".")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Split an indexed dataset by group into train, validation and test.")
    parser.add_argument("--groups", default="groups.npy")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--test", type=float, default=0.15, help="test proportion of all samples")
    parser.add_argument("--val", type=float, default=0.15, help="validation proportion of all samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copies", action="store_true", help="also write X_*/y_* copies (needs --states and --labels)")
    parser.add_argument("--states", default="states.npy")
    parser.add_argument("--labels", default="win_rates.npy")
    args = parser.parse_args(argv)

    if not 0 < args.test < 1 or not 0 < args.val < 1 or args.test + args.val >= 1:
        print("--test and --val must be between 0 and 1 and leave samples for training.", file=sys.stderr)
        return 1

    split_dataset(Path(args.groups), Path(args.output_dir), args.test, args.val, args.seed,
                  Path(args.states) if args.copies else None, Path(args.labels) if args.copies else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
