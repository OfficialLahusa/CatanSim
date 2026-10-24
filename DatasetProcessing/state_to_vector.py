"""
Convert GameState YAML snapshots into a flat fixed-length (currently 1764) numeric feature vector.

Design notes
------------
- Only the `GameState` sub-tree is encoded (Settings, Turn, Board,
  ResourceBank, DevelopmentBank, Players). `PlayedActions` / `UndoHistory`
  are an event log of *arbitrary length*, not a fixed-size state snapshot,
  so they are intentionally excluded from the vector. The action count is
  not encoded either: it is not observable at inference time in the same
  way (MCTS tree nodes) and leaks the sampled game's length.
- `Board.Adjacency` is dropped (as in the original script) since it is a
  constant function of the fixed board layout, not per-state information.
- `FacesDownwards` (Intersections) and `Direction` (Edges) are also
  board-layout constants, not per-state variables, and are dropped for the
  same reason -- they never change between snapshots of the same map.
- Every categorical field (hex resource type, building type, owner, round
  type...) is one-hot encoded via a small `Vocab` helper. Unseen values are
  logged as a warning and encoded as an all-zero vector rather than
  crashing outright -- if you see these warnings, extend the corresponding
  vocabulary list.
- Numbers that can be legitimately absent (hex `Number` on water/desert
  tiles) are encoded as (normalized_value, present_flag) pairs so "no
  number" isn't confused with "number == 2".
- The round counter is not encoded (no rule depends on it, and the net used
  it as a shortcut), and the last roll is 0 before anyone rolled (until then
  it is a random placeholder).
- Counts stay unscaled: a net trained on counts divided to about [0, 1] was
  clearly worse, most likely because the small count features get drowned
  out by the many binary board features.
- After the base features come derived ones (production, trade rates,
  affordability, race margins, see derived_features); with them and input
  normalization the value net's val KL fell from 0.087 to 0.032.
- Output ordering is 100% deterministic given a fixed board (Width/Height,
  hex count, intersection/edge count, player count), so tensors from
  different snapshots of the *same* map/player-count are directly
  comparable feature-for-feature. If you feed in boards of different size
  or player count, pad/bucket accordingly -- this script assumes a
  constant schema across your dataset (assert checks enforce this).
"""

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import yaml

# Size of the base features (the encoded state) and of the derived features after them
BASE_FEATURE_DIM = 1608
DERIVED_FEATURE_DIM = 156

# Total size of the final GameState feature vector
FEATURE_DIM = BASE_FEATURE_DIM + DERIVED_FEATURE_DIM

# The setup rounds count the round counter up to 2, so the first normal round is 2
FIRST_NORMAL_ROUND = 2

# --------------------------------------------------------------------------
# Vocabularies for categorical fields.
# If the simulator produces unknown categorical values that are not caught here
# (e.g. due to future updates), a warning is logged!
# --------------------------------------------------------------------------

RESOURCE_KEYS = ["Unknown", "Lumber", "Brick", "Wool", "Grain", "Ore"]
DEV_CARD_KEYS = [
    "Unknown",
    "Knight",
    "RoadBuilding",
    "YearOfPlenty",
    "Monopoly",
    "VictoryPoint",
]

HEX_TYPES = [
    "Water",
    "Lumber",
    "Brick",
    "Wool",
    "Grain",
    "Ore",
    "Desert",
    "NonPlayable",
]
INTERSECTION_BUILDING_TYPES = ["None", "Settlement", "City"]
EDGE_BUILDING_TYPES = ["None", "Road"]

# PortPrivileges is a [Flags] enum and encoded as a multi-hot-vector
PORT_PRIVILEGE_FLAGS = [
    "GenericThreeToOne",  # 1
    "LumberTwoToOne",     # 2
    "BrickTwoToOne",      # 4
    "WoolTwoToOne",       # 8
    "GrainTwoToOne",      # 16
    "OreTwoToOne",        # 32
]
TYPE_OF_ROUND_VALUES = [
    "Normal",
    "FirstInitial",
    "SecondInitial",
    "MatchEnded",
]


class Vocab:
    """One-hot encoder for a categorical field. Unseen values are logged as
    a warning and encoded as an all-zero vector."""

    def __init__(self, name: str, values: Sequence[str]):
        self.name = name
        self.values = list(values)
        self.index = {v: i for i, v in enumerate(self.values)}
        self.size = len(self.values)

    def one_hot(self, value: Any) -> List[float]:
        vec = [0.0] * self.size
        key = "None" if value is None else str(value)
        idx = self.index.get(key)
        if idx is None:
            print(
                f"[state_to_vector] WARNING: unseen value {key!r} for "
                f"field {self.name!r}; encoding as all-zero. Consider "
                f"adding it to the vocabulary."
            )
            return vec
        vec[idx] = 1.0
        return vec


HEX_VOCAB = Vocab("HexType", HEX_TYPES)
INTERSECTION_BUILDING_VOCAB = Vocab("IntersectionBuilding", INTERSECTION_BUILDING_TYPES)
EDGE_BUILDING_VOCAB = Vocab("EdgeBuilding", EDGE_BUILDING_TYPES)
ROUND_VOCAB = Vocab("TypeOfRound", TYPE_OF_ROUND_VALUES)

PORT_PRIVILEGE_INDEX = {flag: i for i, flag in enumerate(PORT_PRIVILEGE_FLAGS)}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def bool_feature(value: bool) -> List[float]:
    return [1.0 if value else 0.0]


def owner_one_hot(owner: int, num_players: int) -> List[float]:
    """Owner is -1 (unowned) or a player index in [0, num_players).
    Encoded as one-hot over [-1, 0, 1, ..., num_players-1]."""
    vec = [0.0] * (num_players + 1)
    idx = owner + 1  # shift -1 -> 0
    if 0 <= idx < len(vec):
        vec[idx] = 1.0
    else:
        print(f"[state_to_vector] WARNING: owner index {owner} out of range")
    return vec


def flatten_int_dict(d: Dict[str, int], keys: Sequence[str]) -> List[float]:
    """Flatten a {key: count} dict (ResourceCards, DevelopmentCards, ...)
    into a fixed-order numeric vector, defaulting missing keys to 0."""
    return [float(d.get(k, 0) or 0) for k in keys]


def port_privileges_multi_hot(value: Any) -> List[float]:
    """Encode a [Flags]-enum PortPrivileges value as a multi-hot vector
    over the individual flag bits (order: GenericThreeToOne, LumberTwoToOne,
    BrickTwoToOne, WoolTwoToOne, GrainTwoToOne, OreTwoToOne).

    Accepts either the YAML string form ("None", a single flag name, or a
    comma-separated combination like "LumberTwoToOne, OreTwoToOne") or a
    raw integer bitmask, in case the simulator ever serializes it that way.
    "None"/0/empty/missing all map to the all-zero vector.
    """
    vec = [0.0] * len(PORT_PRIVILEGE_FLAGS)

    if value is None:
        return vec

    if isinstance(value, int):
        for flag, idx in PORT_PRIVILEGE_INDEX.items():
            bit = 1 << idx
            if value & bit:
                vec[idx] = 1.0
        return vec

    text = str(value).strip()
    if text == "" or text == "None":
        return vec

    for token in text.split(","):
        token = token.strip()
        idx = PORT_PRIVILEGE_INDEX.get(token)
        if idx is None:
            print(
                f"[state_to_vector] WARNING: unseen PortPrivileges flag "
                f"{token!r}; ignoring it (not encoded). Consider adding it "
                f"to PORT_PRIVILEGE_FLAGS."
            )
            continue
        vec[idx] = 1.0
    return vec


def hex_number_feature(number: Any) -> List[float]:
    """Hex/robber 'Number' is 2-12 or None (water/desert/non-playable).
    Encoded as (normalized value in [0, 1], presence flag) so 'no number'
    isn't confused with an actual roll number."""
    if number is None:
        return [0.0, 0.0]
    normalized = (float(number) - 2.0) / 10.0  # 2..12 -> 0..1
    return [normalized, 1.0]


# --------------------------------------------------------------------------
# Section encoders
# --------------------------------------------------------------------------

def encode_settings(settings: Dict[str, Any]) -> List[float]:
    return [
        float(settings["RobberCardLimit"]),
        float(settings["VictoryPoints"]),
    ]


def encode_turn(turn: Dict[str, Any], num_players: int) -> List[float]:
    feats: List[float] = []

    # The round counter is not encoded by design: no rule depends on it, and a net trained with it used it as a
    # shortcut for game progress that doesn't carry over to stronger play (the board, hands and VP show progress).
    # Only the total of the last roll affects the game, so equivalent rolls get identical features.
    # Before anyone rolled, the last roll is a random placeholder and encoded as 0.
    if has_rolled(turn):
        last_roll = turn.get("LastRoll") or {}
        feats.append(float((last_roll.get("First") or 0) + (last_roll.get("Second") or 0)))
    else:
        feats.append(0.0)

    # PlayerIndex as one-hot (categorical, not ordinal).
    player_idx = turn["PlayerIndex"]
    one_hot = [0.0] * num_players
    if 0 <= player_idx < num_players:
        one_hot[player_idx] = 1.0
    feats.extend(one_hot)

    feats.extend(ROUND_VOCAB.one_hot(turn["TypeOfRound"]))
    feats.extend(bool_feature(turn["MustRoll"]))

    discards = turn.get("AwaitedPlayerDiscards") or []
    # Pad/truncate to num_players in case of mismatch.
    padded = list(discards) + [False] * max(0, num_players - len(discards))
    for d in padded[:num_players]:
        feats.extend(bool_feature(bool(d)))

    feats.extend(bool_feature(turn["MustMoveRobber"]))
    feats.extend(bool_feature(turn["HasPlayedDevelopmentCard"]))
    return feats


def has_rolled(turn: Dict[str, Any]) -> bool:
    """Whether the last roll is a real roll: not during the setup rounds and not in the first normal turn before its roll"""
    round_type = turn["TypeOfRound"]
    if round_type in ("FirstInitial", "SecondInitial"):
        return False
    if round_type == "Normal":
        return not (turn["RoundCounter"] == FIRST_NORMAL_ROUND and turn["PlayerIndex"] == 0 and turn["MustRoll"])
    return True


def encode_board(board: Dict[str, Any], num_players: int) -> List[float]:
    feats: List[float] = []

    # --- Robber ---
    robber = board["Robber"]
    # Robber X/Y are raw board coordinates; keep as plain numbers (small,
    # bounded, board-size-dependent) rather than one-hot over every cell.
    feats.append(float(robber["X"]))
    feats.append(float(robber["Y"]))
    feats.extend(HEX_VOCAB.one_hot(robber["Type"]))
    feats.extend(hex_number_feature(robber["Number"]))

    # --- Map hexes ---
    # Hex list order is fixed by the (constant) board layout, so X/Y are
    # implied by position in the list and are not re-encoded per hex.
    hex_values = board["Map"]["Values"]
    for hex_tile in hex_values:
        feats.extend(HEX_VOCAB.one_hot(hex_tile["Type"]))
        feats.extend(hex_number_feature(hex_tile["Number"]))

    # --- Intersections (settlements/cities) ---
    for inter in board["Intersections"]:
        feats.extend(INTERSECTION_BUILDING_VOCAB.one_hot(inter["Building"]))
        feats.extend(owner_one_hot(inter["Owner"], num_players))

    # --- Edges (roads) ---
    for edge in board["Edges"]:
        feats.extend(EDGE_BUILDING_VOCAB.one_hot(edge["Building"]))
        feats.extend(owner_one_hot(edge["Owner"], num_players))

    return feats


def encode_resource_bank(bank: Dict[str, Any]) -> List[float]:
    return flatten_int_dict(bank, RESOURCE_KEYS)


def encode_development_bank(bank: Dict[str, Any]) -> List[float]:
    return flatten_int_dict(bank, DEV_CARD_KEYS)


def encode_player(player: Dict[str, Any]) -> List[float]:
    feats: List[float] = []

    vp = player["VictoryPoints"]
    feats.extend(
        [
            float(vp["SettlementPoints"]),
            float(vp["CityPoints"]),
            float(vp["DevelopmentCardPoints"]),
            float(vp["LongestRoadPoints"]),
            float(vp["LargestArmyPoints"]),
        ]
    )

    feats.append(float(player["PlayedKnights"]))
    feats.append(float(player["LongestRoadLength"]))

    feats.extend(flatten_int_dict(player["ResourceCards"], RESOURCE_KEYS))
    feats.extend(flatten_int_dict(player["DevelopmentCards"], DEV_CARD_KEYS))
    feats.extend(flatten_int_dict(player["NewDevelopmentCards"], DEV_CARD_KEYS))

    stock = player["BuildingStock"]
    feats.extend(
        [
            float(stock["RemainingRoads"]),
            float(stock["RemainingSettlements"]),
            float(stock["RemainingCities"]),
            float(stock["FreeRoads"]),
        ]
    )

    feats.extend(port_privileges_multi_hot(player["PortPrivileges"]))
    return feats


def encode_players(players: List[Dict[str, Any]]) -> List[float]:
    feats: List[float] = []
    for player in players:
        feats.extend(encode_player(player))
    return feats


# --------------------------------------------------------------------------
# Top-level conversion
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Derived features: compound quantities of the board, hands and stocks that
# an MLP finds hard to compute from the one-hot encoding (e.g. production
# needs ownership x building x tile type x number through the geometry).
# Computed from the base vector, so stored datasets can be extended too, with
# integer arithmetic and float32 in the same order as Agents/StateVectorizer.cs
# (the parity fixtures compare both exactly).
# --------------------------------------------------------------------------

# Land tiles at the corners of every intersection of the classic board (map storage index y * 7 + x)
CLASSIC_INTERSECTION_TILES = [
    (9, 15), (9,), (9,), (10, 9), (10, 16, 9), (16, 15, 9), (10,), (11, 10), (11, 17, 10), (17, 16, 10), (11,), (11,),
    (18, 11), (18, 17, 11), (15, 22), (15,), (16, 23, 15), (23, 22, 15), (17, 24, 16), (24, 23, 16), (18, 25, 17),
    (25, 24, 17), (18,), (26, 18), (26, 25, 18), (22,), (22,), (23, 29, 22), (29, 22), (24, 30, 23), (30, 29, 23),
    (25, 31, 24), (31, 30, 24), (26, 32, 25), (32, 31, 25), (26,), (26,), (32, 26), (29,), (30, 37, 29), (37, 29),
    (31, 38, 30), (38, 37, 30), (32, 39, 31), (39, 38, 31), (32,), (39, 32), (37,), (38, 37), (37,), (39, 38), (38,),
    (39,), (39,),
]
CLASSIC_MAP_WIDTH = 7

# Layout of the base vector
_TILE_START, _TILE_SIZE, _TILE_COUNT = 30, 10, 49        # 8 type one-hot (HEX_TYPES), normalized number, present flag
_INTERSECTION_START, _INTERSECTION_SIZE = 520, 8         # building one-hot, owner one-hot over [-1, 0..3]
_DEVELOPMENT_BANK = slice(1463, 1468)                   # without Unknown

# The player blocks close the base features. Each holds the victory point trackers (VICTORY_POINT_FIELDS, their sum is
# the player's victory points), played knights, longest road, resource cards, development cards, new development cards,
# building stock and port privileges (see encode_player)
PLAYER_COUNT = 4
VICTORY_POINT_FIELDS = 5
PLAYER_BLOCK_SIZE = VICTORY_POINT_FIELDS + 2 + len(RESOURCE_KEYS) + 2 * len(DEV_CARD_KEYS) + 4 + len(PORT_PRIVILEGE_FLAGS)
PLAYER_BLOCK_STARTS = [BASE_FEATURE_DIM - (PLAYER_COUNT - p) * PLAYER_BLOCK_SIZE for p in range(PLAYER_COUNT)]
_PLAYER_START, _PLAYER_SIZE = PLAYER_BLOCK_STARTS[0], PLAYER_BLOCK_SIZE

# Costs in resource order (Lumber, Brick, Wool, Grain, Ore) of a road, a settlement, a city and a development card
BUILD_COSTS = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 2, 3], [0, 0, 1, 1, 1]], dtype=np.int64)

_INTERSECTION_TILE_MATRIX = np.zeros((len(CLASSIC_INTERSECTION_TILES), _TILE_COUNT), dtype=np.int64)
for _i, _tiles in enumerate(CLASSIC_INTERSECTION_TILES):
    _INTERSECTION_TILE_MATRIX[_i, list(_tiles)] = 1


def derived_features(base: np.ndarray) -> np.ndarray:
    """
    Derived features of base vectors (n, BASE_FEATURE_DIM), as float32 (n, DERIVED_FEATURE_DIM), all per player:
    - production per resource: expected cards per roll (pips / 36, cities count twice), without the robber's tile (20)
    - production blocked by the robber per resource (20), total production (4)
    - trade rate per resource: 2 with its port, 3 with a generic port, otherwise 4 (20)
    - per build (road, settlement, city, development card): can build now (cards after trading at the player's rates,
      stock, a settlement to upgrade, cards left in the bank), missing cards, missing cards after trading surplus,
      rounds until affordable at the current production (missing after trading / (4 rolls * production), at most 20) (4 x 16)
    - race: victory points, lead over the best other player, points still needed, longest road and knights against the
      best other player, hand size, over the robber card limit (28)
    """
    x = np.asarray(base, dtype=np.float32)
    n = x.shape[0]
    players = 4

    tiles = x[:, _TILE_START:_TILE_START + _TILE_COUNT * _TILE_SIZE].reshape(n, _TILE_COUNT, _TILE_SIZE)
    # Numbers are stored normalized (and maybe as float16), rounding gives the exact number back
    number = np.rint(tiles[:, :, 8].astype(np.float64) * 10 + 2).astype(np.int64)
    pips = np.where(tiles[:, :, 9] > 0, 6 - np.abs(7 - number), 0)
    tile_pips = np.rint(tiles[:, :, 1:6]).astype(np.int64) * pips[:, :, None]            # (n, tiles, resources)

    robber = np.rint(x[:, 19]).astype(np.int64) * CLASSIC_MAP_WIDTH + np.rint(x[:, 18]).astype(np.int64)
    on_robber = np.zeros((n, _TILE_COUNT), dtype=np.int64)
    on_robber[np.arange(n), robber] = 1

    inters = np.rint(x[:, _INTERSECTION_START:_INTERSECTION_START + 54 * _INTERSECTION_SIZE]).astype(np.int64).reshape(n, 54, _INTERSECTION_SIZE)
    owner = inters[:, :, 4:8]                                                               # (n, intersections, players)
    weight = (inters[:, :, 1] + 2 * inters[:, :, 2])[:, :, None] * owner
    settlements = (inters[:, :, 1:2] * owner).sum(axis=1)

    free_pips = np.matmul(weight.transpose(0, 2, 1), np.matmul(_INTERSECTION_TILE_MATRIX, tile_pips * (1 - on_robber)[:, :, None]))
    blocked_pips = np.matmul(weight.transpose(0, 2, 1), np.matmul(_INTERSECTION_TILE_MATRIX, tile_pips * on_robber[:, :, None]))
    production = free_pips.astype(np.float32) / np.float32(36)
    blocked = blocked_pips.astype(np.float32) / np.float32(36)
    total = free_pips.sum(axis=2).astype(np.float32) / np.float32(36)

    player = np.rint(x[:, _PLAYER_START:_PLAYER_START + players * _PLAYER_SIZE]).astype(np.int64).reshape(n, players, _PLAYER_SIZE)
    hand = player[:, :, 8:13]
    ports = player[:, :, 29:35]
    rate = np.where(ports[:, :, 1:6] > 0, 2, np.where(ports[:, :, 0:1] > 0, 3, 4))

    missing = np.maximum(BUILD_COSTS[None, None] - hand[:, :, None, :], 0).sum(axis=3)               # (n, players, builds)
    tradeable = (np.maximum(hand[:, :, None, :] - BUILD_COSTS[None, None], 0) // rate[:, :, None, :]).sum(axis=3)
    missing_traded = np.maximum(missing - tradeable, 0)
    development_left = np.rint(x[:, _DEVELOPMENT_BANK]).sum(axis=1) > 0
    stock = np.stack([
        player[:, :, 25] > 0,
        player[:, :, 26] > 0,
        (player[:, :, 27] > 0) & (settlements > 0),
        np.repeat(development_left[:, None], players, axis=1),
    ], axis=2)
    can_build = (missing_traded == 0) & stock
    rounds = np.minimum(missing_traded.astype(np.float32) / np.maximum(np.float32(4) * total, np.float32(0.05))[:, :, None], np.float32(20))

    def lead(values):
        best_other = np.stack([np.delete(values, p, axis=1).max(axis=1) for p in range(players)], axis=1)
        return values - best_other

    vp = player[:, :, 0:5].sum(axis=2)
    hand_size = hand.sum(axis=2)
    race = np.stack([
        vp, lead(vp), np.rint(x[:, 1:2]).astype(np.int64) - vp,
        lead(player[:, :, 6]), lead(player[:, :, 5]),
        hand_size, hand_size > np.rint(x[:, 0:1]),
    ], axis=2)

    return np.concatenate([
        production.reshape(n, -1), blocked.reshape(n, -1), total,
        rate.reshape(n, -1).astype(np.float32),
        can_build.reshape(n, -1).astype(np.float32), missing.reshape(n, -1).astype(np.float32),
        missing_traded.reshape(n, -1).astype(np.float32), rounds.reshape(n, -1),
        race.reshape(n, -1).astype(np.float32),
    ], axis=1)


def state_to_feature_vector(
    state: Dict[str, Any],
    num_players: int = 4,
) -> List[float]:
    """Convert a parsed `GameState` dict (as returned by
    `load_state_yaml_fast`, i.e. NOT the full root document) into a flat
    feature vector."""
 
    # Adjacency is a constant function of the fixed board layout, not
    # per-state info -- drop it if present.
    state["Board"].pop("Adjacency", None)
 
    assert len(state["Players"]) == num_players, (
        f"Expected {num_players} players, found {len(state['Players'])}. "
        "This script assumes a fixed player count across the dataset; "
        "pass the correct num_players if your games vary."
    )
 
    feats: List[float] = []
    feats.extend(encode_settings(state["Settings"]))
    feats.extend(encode_turn(state["Turn"], num_players))
    feats.extend(encode_board(state["Board"], num_players))
    feats.extend(encode_resource_bank(state["ResourceBank"]))
    feats.extend(encode_development_bank(state["DevelopmentBank"]))
    feats.extend(encode_players(state["Players"]))
    feats.extend(derived_features(np.array([feats]))[0].tolist())
    return feats



def load_state_yaml(path: str) -> Dict[str, Any]:
    with open(path) as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)

def load_state_yaml_fast(path: str):
    """Parses only the `GameState` subtree, skipping full parsing of
    `PlayedActions` / `UndoHistory`. Those logs can be an order of
    magnitude larger than GameState itself and are only ever used for
    their length here, so this avoids building throwaway Python objects
    for potentially thousands of logged actions per file.
 
    Relies on the root document having exactly three top-level keys in a
    fixed order -- `GameState`, `PlayedActions`, `UndoHistory` -- each
    starting at column 0, and PlayedActions/UndoHistory being YAML block
    sequences whose top-level items are lines starting with "- " at
    column 0 (both true for this simulator's output format). If a file
    doesn't match this shape, falls back to a full parse automatically.
 
    Returns (game_state_dict, played_actions_count, undo_history_count).
    """
    with open(path) as stream:
        lines = stream.readlines()
 
    played_start = None
    undo_start = None
    for i, line in enumerate(lines):
        if played_start is None and line.startswith("PlayedActions:"):
            played_start = i
        elif (
            played_start is not None
            and undo_start is None
            and line.startswith("UndoHistory:")
        ):
            undo_start = i
            break
 
    if played_start is None:
        # Doesn't match the expected shape -- fall back to a full parse.
        root = yaml.load("".join(lines), Loader=yaml.CSafeLoader)
        game_state = root["GameState"]
        played_count = len(root.get("PlayedActions") or [])
        undo_count = len(root.get("UndoHistory") or [])
        return game_state, played_count, undo_count
 
    gamestate_text = "".join(lines[:played_start])
    played_end = undo_start if undo_start is not None else len(lines)
    played_count = sum(1 for l in lines[played_start:played_end] if l.startswith("- "))
    undo_count = (
        sum(1 for l in lines[undo_start:] if l.startswith("- "))
        if undo_start is not None
        else 0
    )
 
    root = yaml.load(gamestate_text, Loader=yaml.CSafeLoader)
    return root["GameState"], played_count, undo_count



def yaml_to_numpy(path: str, num_players: int = 4) -> np.ndarray:
    game_state, _played_count, _undo_count = load_state_yaml_fast(path)
    feats = state_to_feature_vector(game_state, num_players=num_players)
    return np.asarray(feats, dtype=np.float16)



if __name__ == "__main__":
    vec = yaml_to_numpy("./input/r0_s0.yaml")
    print(f"Feature vector shape: {vec.shape}, dtype: {vec.dtype}")
    print(vec)