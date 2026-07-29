"""
Convert GameState YAML snapshots into a flat fixed-length (currently 1611) numeric feature vector.

Design notes
------------
- Only the `GameState` sub-tree is encoded (Settings, Turn, Board,
  ResourceBank, DevelopmentBank, Players). `PlayedActions` / `UndoHistory`
  are an event log of *arbitrary length*, not a fixed-size state snapshot,
  so they are intentionally excluded from the vector. If you want
  history-derived features (e.g. "turns played so far"), add them as
  scalar summaries (see `encode_extra_history_features`) rather than trying
  to flatten the whole log.
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
                f"[state_yaml_to_tensor] WARNING: unseen value {key!r} for "
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
        print(f"[state_yaml_to_tensor] WARNING: owner index {owner} out of range")
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
                f"[state_yaml_to_tensor] WARNING: unseen PortPrivileges flag "
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
    feats.append(float(turn["RoundCounter"]))

    last_roll = turn.get("LastRoll") or {}
    feats.append(float(last_roll.get("First") or 0))
    feats.append(float(last_roll.get("Second") or 0))

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


def encode_extra_history_features(root: Dict[str, Any]) -> List[float]:
    """Optional scalar summaries derived from the (variable-length) action
    log, kept separate from the fixed board/player encoding above. Safe to
    call even if PlayedActions are absent from `root`."""
    played_actions = root.get("PlayedActions") or []
    return [float(len(played_actions))]


# --------------------------------------------------------------------------
# Top-level conversion
# --------------------------------------------------------------------------

def state_to_feature_vector(
    root: Dict[str, Any], num_players: int = 4
) -> List[float]:
    """Convert a full parsed YAML document (with `GameState`,
    `PlayedActions` top-level keys) into a flat feature
    vector."""
    state = root["GameState"]

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
    feats.extend(encode_extra_history_features(root))
    return feats


def load_state_yaml(path: str) -> Dict[str, Any]:
    with open(path) as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def yaml_to_numpy(path: str, num_players: int = 4) -> np.ndarray:
    root = load_state_yaml(path)
    feats = state_to_feature_vector(root, num_players=num_players)
    return np.asarray(feats, dtype=np.float16)


if __name__ == "__main__":
    vec = yaml_to_numpy("./input/r0_s0.yaml")
    print(f"Feature vector shape: {vec.shape}, dtype: {vec.dtype}")
    print(vec)