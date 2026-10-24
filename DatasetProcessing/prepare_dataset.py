"""
Prepare collected datasets for training in one step, without questions (e.g. on a server):

1. Index all datasets below the base path (build_dataset_index.py), stopping if samples and labels don't match
2. Vectorize: states.npy from the vector chunks (or the YAML samples), win_rates.npy, groups.npy
3. Labels from the raw playout results (playout_results.py): placements.npy, vp_shares.npy, rounds_left.npy,
   finished_playouts.npy, match_seeds.npy, sibling_groups.npy, actors.npy, chance_moves.npy
4. Split by group into idx_train.npy, idx_val.npy, idx_test.npy (split_dataset.py)

Everything is written to the output directory, together with dataset_index.csv. Train with e.g.
    python train.py --states <out>/states.npy --labels <out>/win_rates.npy --train-idx <out>/idx_train.npy
                    --val-idx <out>/idx_val.npy --aux-dir <out> --normalize-inputs

Usage: python prepare_dataset.py <base path> <output directory> [--workers 8] [--test 0.15] [--val 0.15] [--seed 42]
                                 [--allow-missing]
"""
import argparse
import os
import sys
from pathlib import Path

from build_dataset_index import build_dataset_index, write_index
from playout_results import build_label_arrays, save_label_arrays
from split_dataset import split_dataset
from vectorize_indexed_dataset import vectorize_dataset


def prepare_dataset(base_path: str, output_dir: str, workers: int = 8, test: float = 0.15, val: float = 0.15,
                    seed: int = 42, allow_missing: bool = False) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("\n== 1/4 Index")
    index = build_dataset_index(base_path, allow_missing)
    write_index(index, str(output / "dataset_index.csv"))

    print("\n== 2/4 Vectorize")
    vectorize_dataset(Path(base_path), index, workers, output_dir=output)

    print("\n== 3/4 Labels from the raw playout results")
    arrays = build_label_arrays(base_path, index)
    print(f"{int((arrays['finished_playouts'] > 0).sum())} of {len(index)} entries have raw playout results.")
    save_label_arrays(arrays, str(output))

    print("\n== 4/4 Split")
    split_dataset(output / "groups.npy", output, test, val, seed)

    print(f"\nDataset prepared in \"{os.path.abspath(output)}\".")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Index, vectorize, label and split collected datasets in one step.")
    parser.add_argument("base_path", help="directory with the datasets (at any depth, e.g. the parts of a partial run)")
    parser.add_argument("output_dir")
    parser.add_argument("--workers", type=int, default=8, help="processes for samples without vector chunks")
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="seed of the split")
    parser.add_argument("--allow-missing", action="store_true", help="skip samples without label and labels without sample")
    args = parser.parse_args(argv)

    try:
        prepare_dataset(args.base_path, args.output_dir, args.workers, args.test, args.val, args.seed, args.allow_missing)
    except (ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
