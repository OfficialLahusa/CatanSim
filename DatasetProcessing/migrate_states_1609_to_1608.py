"""
Migrate a state matrix from the 1609-feature encoding to the current 1608-feature encoding.

The 1608 encoding drops the round counter and encodes the last roll as 0 before anyone rolled (see state_to_vector.py).
Both can be computed from the old vectors, so an existing dataset can be migrated without vectorizing its YAML files
again. The labels, groups and split indices stay valid, since rows keep their order.

Usage: python migrate_states_1609_to_1608.py <states_1609.npy> <states_1608.npy>
"""

import sys

import numpy as np

from state_to_vector import FEATURE_DIM, FIRST_NORMAL_ROUND

OLD_FEATURE_DIM = 1609

# Old layout (columns of the 1609 encoding)
ROUND_COUNTER, LAST_ROLL = 2, 3
PLAYER_INDEX_0 = 4                                  # one-hot over 4 players
ROUND_NORMAL, ROUND_FIRST_INITIAL, ROUND_SECOND_INITIAL = 8, 9, 10
MUST_ROLL = 12


def migrate_rows(old: np.ndarray) -> np.ndarray:
    """Migrates a block of rows (n, 1609) to (n, 1608) as float16."""
    assert old.shape[1] == OLD_FEATURE_DIM, f"expected {OLD_FEATURE_DIM} features, got {old.shape[1]}"
    x = old.astype(np.float32)

    # The last roll is a random placeholder during setup and in the first normal turn before its roll
    setup = (x[:, ROUND_FIRST_INITIAL] == 1) | (x[:, ROUND_SECOND_INITIAL] == 1)
    first_turn_before_roll = (x[:, ROUND_NORMAL] == 1) & (x[:, ROUND_COUNTER] == FIRST_NORMAL_ROUND) \
        & (x[:, PLAYER_INDEX_0] == 1) & (x[:, MUST_ROLL] == 1)
    x[:, LAST_ROLL] = np.where(setup | first_turn_before_roll, 0.0, x[:, LAST_ROLL])

    migrated = np.delete(x, ROUND_COUNTER, axis=1)
    assert migrated.shape[1] == FEATURE_DIM
    return migrated.astype(np.float16)


def migrate_file(old_path: str, new_path: str, rows_per_block: int = 100_000) -> None:
    old = np.load(old_path, mmap_mode="r")
    new = np.lib.format.open_memmap(new_path, mode="w+", dtype=np.float16, shape=(old.shape[0], FEATURE_DIM))

    for start in range(0, old.shape[0], rows_per_block):
        end = min(start + rows_per_block, old.shape[0])
        new[start:end] = migrate_rows(np.asarray(old[start:end]))
        print(f"\r{end}/{old.shape[0]} rows", end="", flush=True)

    new.flush()
    print(f"\nSaved {new.shape} to \"{new_path}\".")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    migrate_file(sys.argv[1], sys.argv[2])
