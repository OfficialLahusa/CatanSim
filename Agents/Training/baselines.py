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

import numpy as np
import torch
import torch.nn.functional as F

# Per-player Victory Points live at [start : start+5] within each player's
# 35-wide block (SettlementPoints, CityPoints, DevelopmentCardPoints,
# LongestRoadPoints, LargestArmyPoints -- their sum is total VP). Offsets
# below were computed via state_yaml_to_tensor.get_feature_layout() against
# the current encoding -- do not hand-edit; if the encoding changes,
# regenerate these via the snippet at the bottom of this file instead.
PLAYER_BLOCK_STARTS = [1470, 1505, 1540, 1575]
VP_FIELDS_PER_PLAYER = 5  # first 5 floats of each player block
EXPECTED_VECTOR_DIM = 1611

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
        f"{EXPECTED_VECTOR_DIM}. The hardcoded PLAYER_BLOCK_STARTS offsets "
        "above no longer match -- regenerate them (see snippet at bottom of "
        "this file) before trusting the VP baseline result."
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


# --------------------------------------------------------------------------
# If the feature encoding ever changes, regenerate PLAYER_BLOCK_STARTS with:
#
#   from state_yaml_to_tensor import (
#       load_state_yaml_fast, state_to_feature_vector_with_layout, get_feature_layout,
#   )
#   gs, played, undo = load_state_yaml_fast("path/to/any_state.yaml")
#   _, sections = state_to_feature_vector_with_layout(gs, 4, played, undo)
#   layout = get_feature_layout(sections)
#   print([layout[f"player_{i}"][0] for i in range(4)])
# --------------------------------------------------------------------------