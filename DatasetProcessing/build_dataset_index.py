"""
Build the index of a dataset: one row per sample (entry_idx, group_idx, subdir_name, entry_name).

Datasets are the directories below the base path that hold an `input` directory with the YAML samples and an
`output` directory with the TXT win rates, at any depth (e.g. the parts of a partial run, datasets/v2/part_00).
subdir_name is the dataset's path relative to the base path. Samples of one match share a group: the seed the match was
played with when the collector wrote raw results (see playout_results.effective_match_seed; matches with the same one can
be identical, so they share the group as well), otherwise the run index within its dataset.

Every YAML sample needs its TXT label and vice versa; a mismatch stops the index (before the long vectorization),
unless --allow-missing skips the unmatched samples.

Usage: python build_dataset_index.py <base path> [--output index.csv] [--allow-missing]
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

from playout_results import effective_match_seed, load_match_seeds

SKIPPED_DIRECTORIES = {"input", "output", "vectors", "samples", "playouts"}


def find_datasets(base_path: str) -> List[str]:
    """Paths (relative to the base path, with forward slashes, sorted) of all directories holding input/ and output/"""
    datasets = []
    for directory, subdirectories, _ in os.walk(base_path):
        if "input" in subdirectories and "output" in subdirectories:
            relative = os.path.relpath(directory, base_path).replace(os.sep, "/")
            datasets.append("." if relative == "." else relative)
        # Never descend into the sample directories, they hold millions of files
        subdirectories[:] = sorted(d for d in subdirectories if d not in SKIPPED_DIRECTORIES)
    return sorted(datasets)


def _stems(directory: str, extension: str) -> set:
    with os.scandir(directory) as entries:
        return {entry.name[: -len(extension)] for entry in entries if entry.name.endswith(extension)}


def parse_sample_name(name: str) -> Tuple[int, int, int]:
    """Run index, state index and sibling index (-1 for none) of a sample name, e.g. r919_s4 or r919_s4_c2"""
    parts = name.split("_")
    return int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]) if len(parts) > 2 else -1


def build_dataset_index(base_path: str, allow_missing: bool = False) -> List[list]:
    """The index rows [entry_idx, group_idx, subdir_name, entry_name] of all datasets below the base path"""
    datasets = find_datasets(base_path)
    if not datasets:
        raise ValueError(f"No dataset (directory with input/ and output/) found below \"{os.path.abspath(base_path)}\".")

    index = []
    group_indices = {}
    problems = []

    for subdir_name in datasets:
        directory = os.path.join(base_path, subdir_name)
        yaml_names = _stems(os.path.join(directory, "input"), ".yaml")
        txt_names = _stems(os.path.join(directory, "output"), ".txt")
        print(f"Dataset \"{subdir_name}\": {len(yaml_names)} YAML samples, {len(txt_names)} TXT labels.")

        missing_labels = sorted(yaml_names - txt_names)
        missing_samples = sorted(txt_names - yaml_names)
        if missing_labels:
            problems.append(f"{subdir_name}: {len(missing_labels)} sample(s) without label, e.g. {missing_labels[:3]}")
        if missing_samples:
            problems.append(f"{subdir_name}: {len(missing_samples)} label(s) without sample, e.g. {missing_samples[:3]}")

        # Group: the effective match seed if the collector wrote raw results (also across runs), otherwise the run index
        # in the dataset; both are mapped to consecutive indices
        match_seeds = load_match_seeds(directory)
        names = sorted(yaml_names & txt_names, key=parse_sample_name)
        for name in names:
            run_idx = parse_sample_name(name)[0]
            group_key = ("seed", effective_match_seed(match_seeds[name])) if name in match_seeds else ("run", subdir_name, run_idx)
            group_idx = group_indices.setdefault(group_key, len(group_indices))
            index.append([len(index), group_idx, subdir_name, name])

    if problems:
        message = "Samples and labels don't match:\n  " + "\n  ".join(problems)
        if not allow_missing:
            raise ValueError(message + "\nFix the dataset or pass --allow-missing to skip the unmatched files.")
        print("Warning: " + message + "\nThe unmatched files are skipped.")

    print(f"Index: {len(index)} samples in {len(group_indices)} groups from {len(datasets)} dataset(s).")
    return index


def write_index(index: List[list], path: str) -> None:
    with open(path, "w") as output_file:
        output_file.write("entry_idx,group_idx,subdir_name,entry_name\n")
        for entry_idx, group_idx, subdir_name, entry_name in index:
            output_file.write(f"{entry_idx},{group_idx},{subdir_name},{entry_name}\n")
    print(f"Saved index as \"{os.path.realpath(path)}\".")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the index of the datasets below a base path.")
    parser.add_argument("base_path")
    parser.add_argument("--output", default=None, help="index CSV (default dataset_index_<time>.csv)")
    parser.add_argument("--allow-missing", action="store_true", help="skip samples without label and labels without sample")
    args = parser.parse_args(argv)

    try:
        index = build_dataset_index(args.base_path, args.allow_missing)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    write_index(index, args.output or "dataset_index_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
