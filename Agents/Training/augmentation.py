"""
Player rotation as data augmentation: relabeling the players cyclically (player p becomes player p + k) keeps the turn
order, so after the setup phase the rotated state is the same game situation and its labels rotate along. In the setup
phase it isn't (the snake order always starts with player 0), so setup states are never rotated. Only cyclic rotations
are equivalences; any other permutation of the players would change the turn order.

On the feature vector (1764 layout) the rotation is a pure index permutation: the player to act and the awaited
discards in the turn state, the owner one-hots of intersections and edges, the player blocks and the derived features
per player move with their player; everything else stays.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DatasetProcessing"))
from state_to_vector import (BASE_FEATURE_DIM, FEATURE_DIM, PLAYER_BLOCK_SIZE, PLAYER_BLOCK_STARTS,  # noqa: E402
                             PLAYER_COUNT, ROUND_VOCAB, _INTERSECTION_SIZE, _INTERSECTION_START)

_TURN_PLAYER = 3                                    # one-hot of the player to act, after the settings (2) and the last roll
_TURN_ROUND = _TURN_PLAYER + PLAYER_COUNT           # TypeOfRound one-hot
_TURN_DISCARDS = _TURN_ROUND + ROUND_VOCAB.size + 1  # awaited discards per player, after MustRoll
_INTERSECTIONS, _INTERSECTION_OWNER = 54, 4         # owner one-hot over [-1, 0..3] at offsets 3..7, player p at 4 + p
_EDGE_START, _EDGES, _EDGE_SIZE, _EDGE_OWNER = _INTERSECTION_START + 54 * _INTERSECTION_SIZE, 72, 7, 3
# Features per player in each group of derived features, in their order (each group is stored player by player):
# production, blocked production, total production, trade rates, can build, missing, missing after trading, rounds, race
_DERIVED_GROUPS = (5, 5, 1, 5, 4, 4, 4, 4, 7)

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
    assert start == FEATURE_DIM
    return source


ROTATION_INDEX = np.stack([rotation_index(k) for k in range(PLAYER_COUNT)])


def is_rotatable(features: np.ndarray) -> np.ndarray:
    """Whether states may be rotated: after the setup phase (normal rounds and ended matches)"""
    rounds = np.asarray(features)[..., _TURN_ROUND:_TURN_ROUND + ROUND_VOCAB.size]
    return rounds[..., _ROTATABLE_ROUNDS].sum(axis=-1) > 0.5


def rotate_features(features: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(features)[..., ROTATION_INDEX[k % PLAYER_COUNT]]


def rotate_players(values: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Per-player labels (win rates, VP shares, the player axis of placements): the value of player p goes to p + k"""
    return np.roll(values, k, axis=axis)


def rotate_player_index(player: int, k: int) -> int:
    return player if player < 0 else (player + k) % PLAYER_COUNT
