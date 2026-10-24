"""
Extend a state matrix from the 1608-feature encoding to the current 1764-feature encoding.

The 1764 encoding appends the derived features (production, trade rates, affordability, race margins) to the 1608
base features, and they are computed from the base features (see derived_features in state_to_vector.py), so an
existing dataset can be extended without vectorizing its YAML files again. Labels, groups and split indices stay
valid, since rows keep their order.

Usage: python migrate_states_1608_to_1764.py <states_1608.npy> <states_1764.npy>
"""

import sys

import numpy as np

from state_to_vector import BASE_FEATURE_DIM, FEATURE_DIM, derived_features


def migrate_file(old_path: str, new_path: str, rows_per_block: int = 100_000) -> None:
    old = np.load(old_path, mmap_mode="r")
    assert old.shape[1] == BASE_FEATURE_DIM, f"expected {BASE_FEATURE_DIM} features, got {old.shape[1]}"
    new = np.lib.format.open_memmap(new_path, mode="w+", dtype=np.float16, shape=(old.shape[0], FEATURE_DIM))

    for start in range(0, old.shape[0], rows_per_block):
        end = min(start + rows_per_block, old.shape[0])
        block = np.asarray(old[start:end])
        new[start:end, :BASE_FEATURE_DIM] = block
        new[start:end, BASE_FEATURE_DIM:] = derived_features(block).astype(np.float16)
        print(f"\r{end}/{old.shape[0]} rows", end="", flush=True)

    new.flush()
    print(f"\nSaved {new.shape} to \"{new_path}\".")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    migrate_file(sys.argv[1], sys.argv[2])
