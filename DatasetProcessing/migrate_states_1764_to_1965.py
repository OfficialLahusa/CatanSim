"""
Extend a state matrix from the 1764-feature encoding to the current 1965-feature encoding.

The 1965 encoding appends the ports (type per coastal intersection) and the endgame features to the 1764 features. The
endgame features are computed from the base features, the ports are not in the old vectors: they come from the maps
of the matches, regenerated from the match seeds by the collector. Every regenerated map is checked against the tiles
stored in the vectors of its samples. Labels, groups and split indices stay valid, since rows keep their order.

Usage:
  1. python migrate_states_1764_to_1965.py seeds <prepared dir> <seeds.txt>
     writes the distinct match seeds of the prepared dataset (match_seeds.npy)
  2. DatasetCollector --regenerate-maps <seeds.txt> <maps.txt> [the run's --agents, --exploration, --center-desert]
  3. python migrate_states_1764_to_1965.py migrate <prepared dir> <maps.txt> <states_1965.npy>
"""

import sys

import numpy as np

from state_to_vector import (BASE_FEATURE_DIM, FEATURE_DIM, HEX_TYPES, LEGACY_FEATURE_DIM, PORT_FEATURE_DIM, PORT_TYPES,
                             _TILE_COUNT, _TILE_SIZE, _TILE_START, endgame_features, slot_port_features)


def write_seeds(prepared: str, seeds_path: str) -> None:
    seeds = np.unique(np.load(f"{prepared}/match_seeds.npy"))
    np.savetxt(seeds_path, seeds, fmt="%d")
    print(f"Saved {len(seeds)} match seeds to \"{seeds_path}\".")


def load_maps(maps_path: str):
    """Seeds (m,), port type per slot (m, 9) and tile type and number per tile (m, tiles) of the regenerated maps"""
    seeds, ports, tile_types, tile_numbers = [], [], [], []
    with open(maps_path) as stream:
        for line in stream:
            seed, port_names, tiles = line.rstrip("\n").split(",")
            seeds.append(int(seed))
            ports.append([PORT_TYPES.index(name) for name in port_names.split()])
            types, numbers = zip(*(tile.split(":") for tile in tiles.split()))
            tile_types.append([HEX_TYPES.index(name) for name in types])
            tile_numbers.append([int(number) for number in numbers])
    order = np.argsort(np.array(seeds, dtype=np.uint64))
    return (np.array(seeds, dtype=np.uint64)[order], np.array(ports, dtype=np.int64)[order],
            np.array(tile_types, dtype=np.int64)[order], np.array(tile_numbers, dtype=np.int64)[order])


def stored_tiles(block: np.ndarray):
    """Tile types (index into HEX_TYPES) and numbers (0 without one) as stored in vectors"""
    tiles = np.asarray(block[:, _TILE_START:_TILE_START + _TILE_COUNT * _TILE_SIZE], dtype=np.float64).reshape(len(block), _TILE_COUNT, _TILE_SIZE)
    types = tiles[:, :, :len(HEX_TYPES)].argmax(axis=2)
    numbers = np.where(tiles[:, :, 9] > 0, np.rint(tiles[:, :, 8] * 10 + 2), 0).astype(np.int64)
    return types, numbers


def migrate(prepared: str, maps_path: str, new_path: str, rows_per_block: int = 100_000) -> None:
    old = np.load(f"{prepared}/states.npy", mmap_mode="r")
    assert old.shape[1] == LEGACY_FEATURE_DIM, f"expected {LEGACY_FEATURE_DIM} features, got {old.shape[1]}"
    sample_seeds = np.load(f"{prepared}/match_seeds.npy", mmap_mode="r")
    seeds, ports, tile_types, tile_numbers = load_maps(maps_path)

    new = np.lib.format.open_memmap(new_path, mode="w+", dtype=np.float16, shape=(old.shape[0], FEATURE_DIM))
    for start in range(0, old.shape[0], rows_per_block):
        end = min(start + rows_per_block, old.shape[0])
        block = np.asarray(old[start:end])
        block_seeds = np.asarray(sample_seeds[start:end], dtype=np.uint64)
        maps = np.searchsorted(seeds, block_seeds)
        if np.any(maps >= len(seeds)) or np.any(seeds[np.minimum(maps, len(seeds) - 1)] != block_seeds):
            raise ValueError(f"rows {start}-{end}: match seeds missing in \"{maps_path}\"")

        types, numbers = stored_tiles(block)
        wrong = np.flatnonzero(np.any(types != tile_types[maps], axis=1) | np.any(numbers != tile_numbers[maps], axis=1))
        if len(wrong):
            raise ValueError(f"{len(wrong)} rows of {start}-{end} don't have the tiles of their regenerated map, "
                             f"e.g. row {start + wrong[0]} (seed {block_seeds[wrong[0]]})")

        new[start:end, :LEGACY_FEATURE_DIM] = block
        new[start:end, LEGACY_FEATURE_DIM:LEGACY_FEATURE_DIM + PORT_FEATURE_DIM] = slot_port_features(ports[maps]).astype(np.float16)
        new[start:end, LEGACY_FEATURE_DIM + PORT_FEATURE_DIM:] = endgame_features(block[:, :BASE_FEATURE_DIM]).astype(np.float16)
        print(f"\r{end}/{old.shape[0]} rows", end="", flush=True)

    new.flush()
    print(f"\nSaved {new.shape} to \"{new_path}\", every map matched the tiles of its samples.")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "seeds":
        write_seeds(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5 and sys.argv[1] == "migrate":
        migrate(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(1)
