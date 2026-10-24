"""
Computes validation-set KL-divergence loss for two non-learned baselines,
for comparison against the trained model's val_loss:

1. Uniform baseline: always predicts [0.25, 0.25, 0.25, 0.25].
2. Victory-points baseline: predicts win probability proportional to each
   player's current total Victory Points (Settlement + City + DevCard +
   LongestRoad + LargestArmy points), Laplace-smoothed so it never assigns
   exactly zero probability to an outcome (which would blow up the KL when
   that outcome actually occurs in the label).

Uses the same reduction convention as training (mean per-sample KL
divergence over the batch/dataset -- see `reduction='batchmean'`
discussion) so the numbers here are directly comparable to your model's
logged val_loss.

Usage:
    python baseline_losses.py --states states.npy --labels labels.npy --val-idx val_idx.npy
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DatasetProcessing"))
from state_to_vector import FEATURE_DIM, PLAYER_BLOCK_STARTS, VICTORY_POINT_FIELDS  # noqa: E402

# Per-player Victory Points live at [start : start+5] within each player's
# block (SettlementPoints, CityPoints, DevelopmentCardPoints, LongestRoadPoints,
# LargestArmyPoints -- their sum is total VP). The offsets come from the
# vectorizer, so they follow changes of the encoding.
VP_FIELDS_PER_PLAYER = VICTORY_POINT_FIELDS
EXPECTED_VECTOR_DIM = FEATURE_DIM

VP_SMOOTHING_ALPHA = 1.0  # Laplace smoothing: avoids exact-zero probabilities
                          # (e.g. a player with 0 VP) and naturally falls back
                          # to uniform when all players have 0 VP (e.g. very
                          # early game, before any settlement is placed).


def kl_divergence(pred_probs: torch.Tensor, target_probs: torch.Tensor) -> float:
    """Mean per-sample KL(target || pred), matching reduction='batchmean'."""
    log_pred = torch.log(pred_probs.clamp_min(1e-12))
    return F.kl_div(log_pred, target_probs, reduction="batchmean").item()


def uniform_baseline(n: int, num_players: int = 4) -> torch.Tensor:
    return torch.full((n, num_players), 1.0 / num_players, dtype=torch.float32)


def victory_points_baseline(states: np.ndarray) -> torch.Tensor:
    vp = np.stack(
        [
            states[:, start : start + VP_FIELDS_PER_PLAYER].sum(axis=1)
            for start in PLAYER_BLOCK_STARTS
        ],
        axis=1,
    ).astype(np.float32)  # shape (N, num_players)

    smoothed = vp + VP_SMOOTHING_ALPHA
    probs = smoothed / smoothed.sum(axis=1, keepdims=True)
    return torch.from_numpy(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="dataset/states.npy")
    parser.add_argument("--labels", default="dataset/win_rates.npy")
    parser.add_argument("--val-idx", default="dataset/idx_val.npy")
    args = parser.parse_args()

    states = np.load(args.states, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    val_idx = np.load(args.val_idx)

    assert states.shape[1] == EXPECTED_VECTOR_DIM, (
        f"states.npy has feature dim {states.shape[1]}, expected "
        f"{EXPECTED_VECTOR_DIM} for the current encoding. Vectorize or migrate "
        "the dataset to the current encoding (DatasetProcessing/migrate_states_*.py) "
        "first, the player block offsets would not match."
    )

    val_states = np.asarray(states[val_idx], dtype=np.float32)
    val_labels = torch.from_numpy(np.asarray(labels[val_idx], dtype=np.float32))

    n = len(val_idx)
    print(f"Validation set size: {n}")

    uniform_pred = uniform_baseline(n)
    uniform_kl = kl_divergence(uniform_pred, val_labels)
    print(f"Uniform baseline        val_loss (KL, batchmean): {uniform_kl:.6f}")

    vp_pred = victory_points_baseline(val_states)
    vp_kl = kl_divergence(vp_pred, val_labels)
    print(f"Victory-points baseline val_loss (KL, batchmean): {vp_kl:.6f}")


if __name__ == "__main__":
    main()
