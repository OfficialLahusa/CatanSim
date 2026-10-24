"""
Read the raw playout results the dataset collector writes and derive labels from them.

Each collector worker writes, in the dataset directory:
- `samples/w<worker>.csv`: one line per sample (columns below), with the offset and count of its playouts in
- `playouts/w<worker>.bin`: one 12-byte record per playout (PLAYOUT_DTYPE): the winner (-1 if the playout hit the
  round limit), the final VP of the 4 players, the agent kind of the 4 seats (AGENT_KINDS) and the round counter at the end.

Ended states have no playouts, their labels come from their result in the sample line (end_winner, end_vp).
Sibling groups: a sampled state (source "root") and the states after several moves of its acting player ("actor"),
all labeled with the same playout seeds. Every member names the group (the root's sample name) in sibling_group;
the move's action type is in "move", and "chance" marks moves whose random outcome (development card, robber steal)
was realized once. Labels derived here:
- win rates (what output/*.txt holds)
- placement distribution P(1st..4th) per player, ties among the losers split evenly over their places
- share of the final VP per player
- mean rounds until the end
So new labels need no new run, and the number of finished playouts stays available for weighting.

Usage: python playout_results.py <index.csv> <base path>
  writes placements.npy (n, 4, 4), vp_shares.npy (n, 4), rounds_left.npy (n,), finished_playouts.npy (n,),
  match_seeds.npy (n,), sibling_groups.npy (n,), actors.npy (n,) and chance_moves.npy (n,) aligned with the index
  entries, for entries of datasets with raw results (NaN / 0 / -1 otherwise)
"""

import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

PLAYER_COUNT = 4
AGENT_KINDS = ("random", "simple", "builder")

PLAYOUT_DTYPE = np.dtype([
    ("winner", "i1"),
    ("vp", "u1", (PLAYER_COUNT,)),
    ("agents", "u1", (PLAYER_COUNT,)),
    ("end_round", "<u2"),
    ("unused", "u1"),
])
assert PLAYOUT_DTYPE.itemsize == 12


@dataclass
class Sample:
    name: str
    match_seed: int         # group ID: samples of one match share it, also across runs
    match: int
    sample: int
    position: int           # actions played until the sampled state
    match_length: int       # actions of the whole match
    round: int
    terminal: bool
    playouts: np.ndarray    # PLAYOUT_DTYPE records, empty for terminal samples
    finished: int
    match_winner: int
    match_vp: np.ndarray
    match_rounds: int
    match_agents: str
    end_winner: int         # winner and final VP of an ended state (-1 / empty otherwise)
    end_vp: np.ndarray
    sibling_group: str      # name of the group's root sample, empty outside sibling groups
    source: str             # root, match, builder or random
    actor: int              # acting player whose moves the group compares, -1 outside sibling groups
    move: str
    chance: bool


def load_samples(dataset_dir: str) -> Dict[str, Sample]:
    """Returns every sample with raw results in the dataset by name (empty for datasets without them)."""
    samples: Dict[str, Sample] = {}

    for csv_path in sorted(glob.glob(os.path.join(dataset_dir, "samples", "w*.csv"))):
        worker = os.path.splitext(os.path.basename(csv_path))[0]
        records = np.fromfile(os.path.join(dataset_dir, "playouts", worker + ".bin"), dtype=PLAYOUT_DTYPE)

        with open(csv_path, newline="") as file:
            for row in csv.DictReader(file):
                offset, count = int(row["playout_offset"]), int(row["playouts"])
                if offset + count > len(records):
                    raise ValueError(f"Sample \"{row['name']}\" refers to playouts {offset}..{offset + count} of \"{worker}.bin\", which holds {len(records)}.")
                if row["name"] in samples:
                    raise ValueError(f"Sample \"{row['name']}\" appears more than once in \"{dataset_dir}\".")

                samples[row["name"]] = Sample(
                    name=row["name"],
                    match_seed=int(row["group"]),
                    match=int(row["match"]),
                    sample=int(row["sample"]),
                    position=int(row["position"]),
                    match_length=int(row["match_length"]),
                    round=int(row["round"]),
                    terminal=row["terminal"] == "1",
                    playouts=records[offset:offset + count],
                    finished=int(row["finished"]),
                    match_winner=int(row["match_winner"]),
                    match_vp=np.array([int(v) for v in row["match_vp"].split("/")]),
                    match_rounds=int(row["match_rounds"]),
                    match_agents=row["match_agents"],
                    end_winner=int(row["end_winner"]) if row["end_winner"] else -1,
                    end_vp=np.array([int(v) for v in row["end_vp"].split("/")]) if row["end_vp"] else np.zeros(0, dtype=int),
                    sibling_group=row["sibling_group"],
                    source=row["source"],
                    actor=int(row["actor"]) if row["actor"] else -1,
                    move=row["move"],
                    chance=row["chance"] == "1",
                )

    return samples


def load_match_seeds(dataset_dir: str) -> Dict[str, int]:
    """Returns the match seed (group ID) of every sample with raw results by name, reading only the sample lines."""
    seeds: Dict[str, int] = {}
    for csv_path in sorted(glob.glob(os.path.join(dataset_dir, "samples", "w*.csv"))):
        with open(csv_path, newline="") as file:
            for row in csv.DictReader(file):
                seeds[row["name"]] = int(row["group"])
    return seeds


def placements(winners: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """
    Placement distribution per game and player, shape (games, 4, 4) with [game, player, place].
    The winner takes 1st place; the others are ranked by final VP, and players with equal VP split their places evenly.
    """
    games = len(winners)
    result = np.zeros((games, PLAYER_COUNT, PLAYER_COUNT))

    # The winner ranks above everyone, whatever the VP
    score = vp.astype(np.float64)
    score[np.arange(games), winners] = np.inf

    for player in range(PLAYER_COUNT):
        better = (score > score[:, [player]]).sum(axis=1)
        tied = (score == score[:, [player]]).sum(axis=1)
        for share in range(PLAYER_COUNT):
            # Places better .. better + tied - 1 (0-based), each with 1 / tied
            in_range = (share >= better) & (share < better + tied)
            result[:, player, share] = np.where(in_range, 1.0 / tied, 0.0)

    return result


@dataclass
class Labels:
    win_rates: np.ndarray       # (4,)
    placements: np.ndarray      # (4 players, 4 places)
    vp_shares: np.ndarray       # (4,)
    rounds_left: float
    finished: int               # finished playouts, 1 for terminal samples


def derive_labels(sample: Sample) -> Labels:
    """Labels of a sample from its finished playouts, or from its result for an ended state. Raises without any."""
    if sample.terminal:
        winners = np.array([sample.end_winner])
        vp = sample.end_vp[None, :]
        rounds_left = np.array([0.0])
    else:
        finished = sample.playouts[sample.playouts["winner"] >= 0]
        if len(finished) == 0:
            raise ValueError(f"Sample \"{sample.name}\" has no finished playout.")
        winners = finished["winner"].astype(np.int64)
        vp = finished["vp"].astype(np.int64)
        rounds_left = finished["end_round"].astype(np.float64) - sample.round

    win_rates = np.bincount(winners, minlength=PLAYER_COUNT) / len(winners)
    vp_totals = vp.sum(axis=1, keepdims=True)
    vp_shares = np.divide(vp, vp_totals, out=np.full(vp.shape, 1.0 / PLAYER_COUNT), where=vp_totals > 0).mean(axis=0)

    return Labels(
        win_rates=win_rates,
        placements=placements(winners, vp).mean(axis=0),
        vp_shares=vp_shares,
        rounds_left=float(rounds_left.mean()),
        finished=len(winners),
    )


def build_label_arrays(base_path: str, index: List[list]) -> Dict[str, np.ndarray]:
    """Label arrays aligned with the index entries ([entry_idx, group_idx, subdir_name, entry_name])."""
    n = len(index)
    arrays = {
        "placements": np.full((n, PLAYER_COUNT, PLAYER_COUNT), np.nan, dtype=np.float32),
        "vp_shares": np.full((n, PLAYER_COUNT), np.nan, dtype=np.float32),
        "rounds_left": np.full((n,), np.nan, dtype=np.float32),
        "finished_playouts": np.zeros((n,), dtype=np.int32),
        "match_seeds": np.zeros((n,), dtype=np.uint64),
        "sibling_groups": np.full((n,), -1, dtype=np.int64),
        "actors": np.full((n,), -1, dtype=np.int8),
        "chance_moves": np.zeros((n,), dtype=bool),
    }
    group_ids: Dict[tuple, int] = {}

    samples_by_subdir = {subdir: load_samples(os.path.join(base_path, subdir)) for subdir in sorted({entry[2] for entry in index})}
    for entry_idx, _, subdir_name, entry_name in index:
        sample = samples_by_subdir[subdir_name].get(entry_name)
        if sample is None:
            continue
        labels = derive_labels(sample)
        arrays["placements"][entry_idx] = labels.placements
        arrays["vp_shares"][entry_idx] = labels.vp_shares
        arrays["rounds_left"][entry_idx] = labels.rounds_left
        arrays["finished_playouts"][entry_idx] = labels.finished
        arrays["match_seeds"][entry_idx] = sample.match_seed
        if sample.sibling_group:
            arrays["sibling_groups"][entry_idx] = group_ids.setdefault((subdir_name, sample.sibling_group), len(group_ids))
            arrays["actors"][entry_idx] = sample.actor
            arrays["chance_moves"][entry_idx] = sample.chance

    return arrays


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    from vectorize_indexed_dataset import load_index_file

    index = load_index_file(sys.argv[1])
    arrays = build_label_arrays(sys.argv[2], index)
    with_results = int((arrays["finished_playouts"] > 0).sum())
    print(f"{with_results} of {len(index)} entries have raw playout results.")
    for name, array in arrays.items():
        np.save(f"{name}.npy", array)
        print(f"Saved {name}.npy {array.shape}")
