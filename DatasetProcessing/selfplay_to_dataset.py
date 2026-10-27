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
If every run has the policy files of SelfPlayCollector (policy_counts.i32 and the rest), also the policy targets:
- policy_offsets.npy (n + 1,) int64: the moves of position i are rows offsets[i]:offsets[i + 1] of the arrays below
- policy_tokens.npy (moves, max_tokens) int16 and policy_weights.npy (moves, max_tokens) uint8: the encoded legal moves
  (token -1 unused), in the order of the root's children
- policy_targets.npy (moves,) float32: the share of the root visits per move
- policy_layout.json: the token ranges

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

    columns = {name: [] for name in ("seed", "winner", "vp", "rounds_left", "actor", "q", "chunk", "row", "line", "visits")}
    with open(os.path.join(run_dir, "samples.csv"), newline="") as file:
        for line, sample in enumerate(csv.DictReader(file)):
            game = games.get(sample["game"])
            if game is None:
                continue
            columns["line"].append(line)
            columns["visits"].append(sample["visits"])
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
        "line": np.array(columns["line"], dtype=np.int64),
        "visits": columns["visits"],
    }
    return feature_dim, arrays, len(games)


POLICY_FILES = ("policy_counts.i32", "policy_tokens.i16", "policy_weights.u8", "policy_layout.json")


def read_policy(run_dir: str, arrays: dict, max_tokens: int):
    """The encoded legal moves and visit shares of the run's kept positions (see the module docstring)"""
    counts = np.fromfile(os.path.join(run_dir, "policy_counts.i32"), dtype="<i4").astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)])
    tokens = np.fromfile(os.path.join(run_dir, "policy_tokens.i16"), dtype="<i2").reshape(-1, max_tokens)
    weights = np.fromfile(os.path.join(run_dir, "policy_weights.u8"), dtype=np.uint8).reshape(-1, max_tokens)
    if len(tokens) != starts[-1]:
        raise ValueError(f"{run_dir}: the policy files hold {len(tokens)} moves, the counts {starts[-1]}")

    lines = arrays["line"]
    if len(lines) and lines.max() >= len(counts):
        raise ValueError(f"{run_dir}: samples.csv has more lines than policy_counts.i32 entries")
    kept_counts = counts[lines]
    rows = np.concatenate([np.arange(starts[line], starts[line + 1]) for line in lines]) if len(lines) else np.array([], dtype=np.int64)
    targets = np.zeros(len(rows), dtype=np.float32)
    offset = 0
    for count, visits in zip(kept_counts, arrays["visits"]):
        pairs = [pair.split(":") for pair in visits.split()]
        moves = np.array([int(move) for move, _ in pairs], dtype=np.int64)
        if moves.size and moves.max() >= count:
            raise ValueError(f"{run_dir}: a visited move index lies beyond the position's {count} moves")
        values = np.array([float(v) for _, v in pairs], dtype=np.float64)
        targets[offset + moves] = values / values.sum()
        offset += count
    return kept_counts, tokens[rows], weights[rows], targets


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

    if all(os.path.exists(os.path.join(run, name)) for run, *_ in parts for name in POLICY_FILES):
        layouts = []
        for run, *_ in parts:
            with open(os.path.join(run, "policy_layout.json"), encoding="utf-8") as file:
                layouts.append(json.load(file))
        if any(layout != layouts[0] for layout in layouts):
            raise ValueError("The runs encode the moves differently (policy_layout.json)")
        max_tokens = layouts[0]["max_tokens"][0]
        policy = [read_policy(run, arrays, max_tokens) for run, _, arrays, _ in parts]
        counts = np.concatenate([part[0] for part in policy])
        np.save(os.path.join(output_dir, "policy_offsets.npy"), np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        for i, name in enumerate(("policy_tokens", "policy_weights", "policy_targets"), start=1):
            np.save(os.path.join(output_dir, f"{name}.npy"), np.concatenate([part[i] for part in policy]))
        with open(os.path.join(output_dir, "policy_layout.json"), "w", encoding="utf-8") as file:
            json.dump(layouts[0], file, indent=2)

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
