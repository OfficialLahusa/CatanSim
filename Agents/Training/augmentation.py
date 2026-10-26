"""
Data augmentation by exact symmetries of the rules.

Player rotation: relabeling the players cyclically (player p becomes player p + k) keeps the turn
order, so after the setup phase the rotated state is the same game situation and its labels rotate along. In the setup
phase it isn't (the snake order always starts with player 0), so setup states are never rotated. Only cyclic rotations
are equivalences; any other permutation of the players would change the turn order.

On the feature vector the rotation is a pure index permutation: the player to act and the awaited discards in the turn
state, the owner one-hots of intersections and edges, the player blocks and the derived and endgame features per player
move with their player; everything else stays (also the ports). Vectors of the legacy layout (the start of the current
one) are rotated by the start of the permutation.

Number token pairs: 2/12, 3/11, 4/10, 5/9 and 6/8 are rolled equally often, and a number token matters only through how
often it is rolled (the 7 has no token). Swapping the tokens of any pairs on the board (and the last roll along) gives
the same game situation with the same labels, so each of the 5 pairs can be swapped independently (32 views, the
red 6/8 stay red, the generator's token counts stay the same). In any phase of the game. On the feature vector only the
normalized numbers of the tiles and the robber's tile and the last roll change; the derived features depend on the
numbers only through their pips and stay.

Board symmetries: the classic board is symmetric under its 6 rotations around the center tile and their mirror images.
Rotating or mirroring everything on it (tiles, buildings, roads, ports, robber) gives the same game situation with the
same labels, in any phase. The ports must be features for this (rotated, they lie at other coastal edges than the
generator's slots), so it needs the current layout. On the feature vector it permutes the tile, intersection, edge and
port blocks and moves the robber's coordinates; the per-player features are sums over the board and stay.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DatasetProcessing"))
from state_to_vector import (BASE_FEATURE_DIM, CLASSIC_EDGE_TILES, CLASSIC_INTERSECTION_ALL_TILES,  # noqa: E402
                             CLASSIC_MAP_WIDTH, COASTAL_INTERSECTIONS, FEATURE_DIM, LEGACY_FEATURE_DIM, PLAYER_BLOCK_SIZE,
                             PLAYER_BLOCK_STARTS, PLAYER_COUNT, PORT_FEATURE_DIM, PORT_TYPES, ROUND_VOCAB,
                             _INTERSECTION_SIZE, _INTERSECTION_START, _TILE_COUNT, _TILE_SIZE, _TILE_START)

_TURN_PLAYER = 3                                    # one-hot of the player to act, after the settings (2) and the last roll
_TURN_ROUND = _TURN_PLAYER + PLAYER_COUNT           # TypeOfRound one-hot
_TURN_DISCARDS = _TURN_ROUND + ROUND_VOCAB.size + 1  # awaited discards per player, after MustRoll
_INTERSECTIONS, _INTERSECTION_OWNER = 54, 4         # owner one-hot over [-1, 0..3] at offsets 3..7, player p at 4 + p
_EDGE_START, _EDGES, _EDGE_SIZE, _EDGE_OWNER = _INTERSECTION_START + 54 * _INTERSECTION_SIZE, 72, 7, 3
# Features per player in each group of derived features, in their order (each group is stored player by player):
# production, blocked production, total production, trade rates, can build, missing, missing after trading, rounds, race
_DERIVED_GROUPS = (5, 5, 1, 5, 4, 4, 4, 4, 7)
# The endgame features per player (settlement spots, road spots, gain this turn, wins, knight lead), then one for all
_ENDGAME_PER_PLAYER = 5

_ROTATABLE_ROUNDS = [ROUND_VOCAB.index["Normal"], ROUND_VOCAB.index["MatchEnded"]]


def rotation_index(k: int) -> np.ndarray:
    """Source index per feature for the rotation by k players: rotated = features[..., rotation_index(k)]"""
    source = np.arange(FEATURE_DIM)
    for new in range(PLAYER_COUNT):
        old = (new - k) % PLAYER_COUNT

        def move(new_position: int, old_position: int):
            source[new_position] = old_position

        move(_TURN_PLAYER + new, _TURN_PLAYER + old)
        move(_TURN_DISCARDS + new, _TURN_DISCARDS + old)
        for i in range(_INTERSECTIONS):
            base = _INTERSECTION_START + i * _INTERSECTION_SIZE + _INTERSECTION_OWNER
            move(base + new, base + old)
        for e in range(_EDGES):
            base = _EDGE_START + e * _EDGE_SIZE + _EDGE_OWNER
            move(base + new, base + old)
        for j in range(PLAYER_BLOCK_SIZE):
            move(PLAYER_BLOCK_STARTS[new] + j, PLAYER_BLOCK_STARTS[old] + j)

        start = BASE_FEATURE_DIM
        for size in _DERIVED_GROUPS:
            for j in range(size):
                move(start + new * size + j, start + old * size + j)
            start += PLAYER_COUNT * size
        assert start == LEGACY_FEATURE_DIM

        start += PORT_FEATURE_DIM
        for j in range(_ENDGAME_PER_PLAYER):
            move(start + new * _ENDGAME_PER_PLAYER + j, start + old * _ENDGAME_PER_PLAYER + j)
        assert start + PLAYER_COUNT * _ENDGAME_PER_PLAYER + 1 == FEATURE_DIM
    return source


ROTATION_INDEX = np.stack([rotation_index(k) for k in range(PLAYER_COUNT)])


def is_rotatable(features: np.ndarray) -> np.ndarray:
    """Whether states may be rotated: after the setup phase (normal rounds and ended matches)"""
    rounds = np.asarray(features)[..., _TURN_ROUND:_TURN_ROUND + ROUND_VOCAB.size]
    return rounds[..., _ROTATABLE_ROUNDS].sum(axis=-1) > 0.5


def rotate_features(features: np.ndarray, k: int) -> np.ndarray:
    features = np.asarray(features)
    # The legacy layout is the start of the current one, and the permutation keeps the start within itself
    return features[..., ROTATION_INDEX[k % PLAYER_COUNT, :features.shape[-1]]]


def rotate_players(values: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Per-player labels (win rates, VP shares, the player axis of placements): the value of player p goes to p + k"""
    return np.roll(values, k, axis=axis)


def rotate_player_index(player, k: int):
    """A player index (-1 for none stays), or an array of them"""
    if np.ndim(player) == 0:
        return player if player < 0 else (player + k) % PLAYER_COUNT
    player = np.asarray(player)
    return np.where(player < 0, player, (player + k) % PLAYER_COUNT)


_LAST_ROLL = 2                                      # total of the last roll (0 before anyone rolled), after the settings
# Normalized numbers ((n - 2) / 10, the presence flag right after): the robber's tile, then the 49 tiles
_NUMBERS = np.array([28] + [_TILE_START + t * _TILE_SIZE + 8 for t in range(_TILE_COUNT)])
NUMBER_PAIR_COUNT = 5                               # pairs (2, 12) to (6, 8), bit p of a mask swaps the pair (2 + p, 12 - p)


def swap_number_pairs(features: np.ndarray, masks) -> np.ndarray:
    """Swaps the number token pairs chosen by masks (0 to 31, one per state or one for all) on the board and in the last roll"""
    features = np.array(features, copy=True)
    masks = np.asarray(masks, dtype=np.int64)[..., None]

    def swapped(numbers):
        # A number n is swapped with 14 - n if the bit of its pair is set (the 7 has no pair, absent numbers stay)
        low = np.minimum(numbers, 14 - numbers)
        return ((numbers >= 2) & (numbers <= 12) & (low <= 6) & ((masks >> np.clip(low - 2, 0, 4)) & 1).astype(bool))

    values = features[..., _NUMBERS].astype(np.float64)
    numbers = np.rint(values * 10 + 2).astype(np.int64)
    swap = swapped(numbers) & (features[..., _NUMBERS + 1] > 0.5)
    features[..., _NUMBERS] = np.where(swap, (12 - numbers) / 10, values)

    rolls = np.rint(features[..., [_LAST_ROLL]].astype(np.float64)).astype(np.int64)
    features[..., [_LAST_ROLL]] = np.where(swapped(rolls), 14 - rolls, rolls)
    return features


BOARD_SYMMETRY_COUNT = 12                           # 6 rotations, each also mirrored
_ROBBER_X, _ROBBER_Y = 18, 19                       # map coordinates of the robber's tile
_CENTER = (3, 3)


def _to_cube(index: int):
    x, y = index % CLASSIC_MAP_WIDTH, index // CLASSIC_MAP_WIDTH
    q = x - (y + (y % 2)) // 2                      # even-r offset to cube coordinates, as Coordinates.EvenRToCube
    return q, y, -q - y


def _from_cube(q: int, r: int) -> int:
    x = q + (r + (r & 1)) // 2
    return r * CLASSIC_MAP_WIDTH + x


def tile_symmetry(t: int) -> np.ndarray:
    """New map index of every tile within distance 3 of the center (land and the water ring) under symmetry t (0-11:
    t // 6 mirrors, then t % 6 rotates by 60 degrees); the non-playable tiles farther out stay"""
    cq, cr, cs = _to_cube(_CENTER[1] * CLASSIC_MAP_WIDTH + _CENTER[0])
    target = np.arange(_TILE_COUNT)
    for index in range(_TILE_COUNT):
        q, r, s = _to_cube(index)
        dq, dr, ds = q - cq, r - cr, s - cs
        if max(abs(dq), abs(dr), abs(ds)) > 3:
            continue
        if t // 6:
            dr, ds = ds, dr                         # a reflection swaps two cube axes
        for _ in range(t % 6):
            dq, dr, ds = -dr, -ds, -dq              # rotation by 60 degrees
        target[index] = _from_cube(cq + dq, cr + dr)
    return target


def _element_symmetry(tiles_of, tile_target):
    """New index of every intersection or edge (given by the tiles around it) under a tile mapping"""
    index_of = {frozenset(tiles): i for i, tiles in enumerate(tiles_of)}
    return np.array([index_of[frozenset(int(tile_target[t]) for t in tiles)] for tiles in tiles_of])


def board_symmetry_index(t: int):
    """Source index per feature for symmetry t (rotated = features[..., index]) and the new map index of every tile
    (for the robber's coordinates)"""
    tiles = tile_symmetry(t)
    intersections = _element_symmetry(CLASSIC_INTERSECTION_ALL_TILES, tiles)
    edges = _element_symmetry(CLASSIC_EDGE_TILES, tiles)
    coastal_of = {intersection: i for i, intersection in enumerate(COASTAL_INTERSECTIONS)}
    coastal = np.array([coastal_of[int(intersections[i])] for i in COASTAL_INTERSECTIONS])

    source = np.arange(FEATURE_DIM)

    def move_blocks(start, size, target):
        for old, new in enumerate(target):
            source[start + new * size:start + (new + 1) * size] = np.arange(start + old * size, start + (old + 1) * size)

    move_blocks(_TILE_START, _TILE_SIZE, tiles)
    move_blocks(_INTERSECTION_START, _INTERSECTION_SIZE, intersections)
    move_blocks(_EDGE_START, _EDGE_SIZE, edges)
    move_blocks(LEGACY_FEATURE_DIM, len(PORT_TYPES), coastal)
    assert sorted(source) == list(range(FEATURE_DIM))
    return source, tiles


_BOARD_SYMMETRIES = [board_symmetry_index(t) for t in range(BOARD_SYMMETRY_COUNT)]
BOARD_INDEX = np.stack([index for index, _ in _BOARD_SYMMETRIES])
BOARD_TILE_TARGET = np.stack([tiles for _, tiles in _BOARD_SYMMETRIES])


def transform_board(features: np.ndarray, t: int) -> np.ndarray:
    """The vectors (current layout) of the same situations on the board rotated and mirrored by symmetry t"""
    features = np.asarray(features)
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"board symmetries need the ports as features ({FEATURE_DIM} features), not {features.shape[-1]}")
    transformed = features[..., BOARD_INDEX[t]]
    robber = (np.rint(features[..., _ROBBER_Y].astype(np.float64)) * CLASSIC_MAP_WIDTH
              + np.rint(features[..., _ROBBER_X].astype(np.float64))).astype(np.int64)
    moved = BOARD_TILE_TARGET[t][robber]
    transformed[..., _ROBBER_X] = moved % CLASSIC_MAP_WIDTH
    transformed[..., _ROBBER_Y] = moved // CLASSIC_MAP_WIDTH
    return transformed
