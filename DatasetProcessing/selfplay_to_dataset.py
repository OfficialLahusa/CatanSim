"""
Turn self-play runs (SelfPlayCollector output) into the arrays train.py reads.

A run directory holds metadata.json, games.csv (one line per game: seed, result, where its save file is),
samples.csv (one line per recorded position: game, round, acting player, the search's mean result per player at the
root q0..q3, the row of its features and the root visits) and states_c*.f16 (float16 feature rows).

Writes into the output directory, one row per position of the games with a result (games stopped at the action
limit are left out):
- states.npy (n, features) float16
- root_q.npy (n, 4) float32: the search's value of the position
- winners.npy (n,) int8: the game's winner
- win_rates.npy (n, 4) float32: the training target, lambda * one-hot(winner) + (1 - lambda) * root_q
- placements.npy (n, 4, 4), vp_shares.npy (n, 4), rounds_left.npy (n,): the auxiliary labels from the game's end
- actors.npy (n,): the player deciding at the position
- match_seeds.npy (n,) uint64 and groups.npy (n,) int64: the game's seed, unique across runs
- idx_train.npy, idx_val.npy, idx_test.npy: split by game

Usage: python selfplay_to_dataset.py RUN [RUN ...] --output DIR [--target-lambda 0.5] [--val 0.1] [--test 0.0] [--seed 42]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from playout_results import PLAYER_COUNT, placements
from split_dataset import split_indices


def read_run(run_dir: str):
    """The positions of a run's games with a result, as a dict of arrays (states as (chunk, row) references)."""
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as file:
        feature_dim = json.load(file)["model"]["feature_dim"]

    games = {}
    with open(os.path.join(run_dir, "games.csv"), newline="") as file:
        for game in csv.DictReader(file):
            if game["finished"] == "1":
                games[game["game"]] = game

    columns = {name: [] for name in ("seed", "winner", "vp", "rounds_left", "actor", "q", "chunk", "row")}
    with open(os.path.join(run_dir, "samples.csv"), newline="") as file:
        for sample in csv.DictReader(file):
            game = games.get(sample["game"])
            if game is None:
                continue
            columns["seed"].append(int(game["seed"]))
            columns["winner"].append(int(game["winner"]))
            columns["vp"].append([int(game[f"vp{p}"]) for p in range(PLAYER_COUNT)])
            columns["rounds_left"].append(int(game["rounds"]) - int(sample["round"]))
            columns["actor"].append(int(sample["actor"]))
            columns["q"].append([float(sample[f"q{p}"]) for p in range(PLAYER_COUNT)])
            columns["chunk"].append(int(sample["chunk"]))
            columns["row"].append(int(sample["row"]))

    arrays = {
        "seed": np.array(columns["seed"], dtype=np.uint64),
        "winner": np.array(columns["winner"], dtype=np.int8),
        "vp": np.array(columns["vp"], dtype=np.float64).reshape(-1, PLAYER_COUNT),
        "rounds_left": np.array(columns["rounds_left"], dtype=np.float32),
        "actor": np.array(columns["actor"], dtype=np.int64),
        "q": np.array(columns["q"], dtype=np.float32).reshape(-1, PLAYER_COUNT),
        "chunk": np.array(columns["chunk"], dtype=np.int64),
        "row": np.array(columns["row"], dtype=np.int64),
    }
    return feature_dim, arrays, len(games)


def convert(runs, output_dir: str, target_lambda: float = 0.5, val: float = 0.1, test: float = 0.0, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    parts = [(run, *read_run(run)) for run in runs]
    feature_dims = {dim for _, dim, _, _ in parts}
    if len(feature_dims) != 1:
        raise ValueError(f"The runs have different feature layouts: {sorted(feature_dims)}")
    feature_dim = feature_dims.pop()
    n = sum(len(arrays["seed"]) for _, _, arrays, _ in parts)
    if n == 0:
        raise ValueError("The runs hold no positions of games with a result.")

    # States, copied chunk by chunk in the order of the positions
    states = np.lib.format.open_memmap(os.path.join(output_dir, "states.npy"), mode="w+", dtype=np.float16, shape=(n, feature_dim))
    start = 0
    for run, _, arrays, _ in parts:
        for chunk in np.unique(arrays["chunk"]):
            path = os.path.join(run, f"states_c{chunk:04d}.f16")
            rows = np.memmap(path, dtype="<f2", mode="r").reshape(-1, feature_dim)
            positions = np.flatnonzero(arrays["chunk"] == chunk)
            states[start + positions] = rows[arrays["row"][positions]]
        start += len(arrays["seed"])
    states.flush()
    del states

    def joined(name):
        return np.concatenate([arrays[name] for _, _, arrays, _ in parts])

    winners, vp, q = joined("winner"), joined("vp"), joined("q")
    one_hot = np.eye(PLAYER_COUNT, dtype=np.float32)[winners]
    vp_totals = vp.sum(axis=1, keepdims=True)
    match_seeds = joined("seed")
    labels = {
        "root_q": q,
        "winners": winners,
        "win_rates": (target_lambda * one_hot + (1 - target_lambda) * q).astype(np.float32),
        "placements": placements(winners.astype(np.int64), vp).astype(np.float32),
        "vp_shares": np.divide(vp, vp_totals, out=np.full(vp.shape, 1.0 / PLAYER_COUNT), where=vp_totals > 0).astype(np.float32),
        "rounds_left": joined("rounds_left"),
        "actors": joined("actor"),
        "match_seeds": match_seeds,
        "groups": match_seeds.view(np.int64),
    }
    for name, array in labels.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), array)

    # Split by game; without a test share everything that isn't validation is training
    groups = labels["groups"]
    if test > 0:
        train_idx, val_idx, test_idx = split_indices(groups, test, val, seed)
    else:
        unique = np.unique(groups)
        val_groups = np.random.default_rng(seed).choice(unique, size=int(round(len(unique) * val)), replace=False)
        in_val = np.isin(groups, val_groups)
        train_idx, val_idx, test_idx = np.flatnonzero(~in_val), np.flatnonzero(in_val), np.array([], dtype=np.int64)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        np.save(os.path.join(output_dir, f"idx_{name}.npy"), idx.astype(np.int64))

    games = sum(count for _, _, _, count in parts)
    print(f"{n} positions of {games} games with a result from {len(runs)} run(s): "
          f"{len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test, target lambda {target_lambda}")
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Turn self-play runs into the arrays train.py reads")
    parser.add_argument("runs", nargs="+", help="run directories (or glob patterns)")
    parser.add_argument("--output", required=True)
    # Weight of the game result against the search's value in the target
    parser.add_argument("--target-lambda", type=float, default=0.5)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    runs = sorted({path for pattern in args.runs for path in (glob.glob(pattern) or [pattern])})
    convert(runs, args.output, args.target_lambda, args.val, args.test, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
