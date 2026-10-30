"""
Scores value nets (ONNX, as exported by export_onnx.py) on a precision set: samples labeled with many playouts (e.g.
2000 instead of 300), collected separately from the training data. Precise labels make the remaining error mostly the
model's, so models can be told apart where the noisy validation loss can't.

Metrics per model:
- KL divergence from the precise win rates, overall and by game phase (the leader's victory points), next to the
  noise floor the finite playouts leave in the labels
- ranking within sibling groups (the states after different moves of one player): how often the model orders pairs of
  moves like the labels do, for all pairs and for clearer differences; how often it picks the labels' best move, and
  the win probability lost by its pick (regret)
- calibration by victory point deficit: mean predicted against mean actual win probability of the players behind the
  leader by 0, 1, 2, ... points
- for every model after the first: paired differences to the first one on the same samples, with 95% intervals

Matches whose effective seed (playout_results.effective_match_seed) also occurs in the training data are left out with
--exclude-seeds (match_seeds.npy of the training set): they can be identical to a training match.

Usage: python evaluate_precision.py --data <prepared precision set> [--exclude-seeds <prepared training set>/match_seeds.npy]
                                    name=model.onnx [name=model.onnx ...]
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DatasetProcessing"))
from state_to_vector import VIEW_FEATURE_DIM, _PLAYER_SIZE, _PLAYER_START  # noqa: E402

PLAYER_COUNT = 4
PHASES = ((0, 4, "leader 0-4 VP"), (5, 6, "leader 5-6 VP"), (7, 8, "leader 7-8 VP"), (9, 99, "leader 9+ VP"))
DEFICITS = ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+"))
RANKING_THRESHOLDS = (0.0, 0.01, 0.03)
EPSILON = 1e-7


def effective_seeds(match_seeds: np.ndarray) -> np.ndarray:
    """Vectorized playout_results.effective_match_seed: folded to 32 bits like Seeds.ToInt32, then the absolute value like Random(int)"""
    s = np.asarray(match_seeds, dtype=np.uint64)
    folded = ((s ^ (s >> np.uint64(32))) & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32).astype(np.int64)
    return np.where(folded == -(1 << 31), (1 << 31) - 1, np.abs(folded))


def victory_points(states: np.ndarray) -> np.ndarray:
    """Victory points per player (n, 4) from the base features: the sum of the trackers that open each player block"""
    blocks = np.asarray(states[:, _PLAYER_START:_PLAYER_START + PLAYER_COUNT * _PLAYER_SIZE], dtype=np.float32)
    blocks = blocks.reshape(len(states), PLAYER_COUNT, _PLAYER_SIZE)
    return np.rint(blocks[:, :, 0:5]).astype(np.int64).sum(axis=2)


def kl_divergence(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """KL(labels || predictions) per sample"""
    p = np.clip(labels, EPSILON, 1)
    q = np.clip(predictions, EPSILON, 1)
    return np.where(labels > 0, labels * (np.log(p) - np.log(q)), 0).sum(axis=1)


def noise_floor(labels: np.ndarray, playouts: np.ndarray) -> np.ndarray:
    """Expected KL a perfect model still has against labels from n playouts: (players with a chance - 1) / (2n)"""
    outcomes = np.maximum((labels > 0).sum(axis=1) - 1, 0)
    return np.where(playouts > 0, outcomes / (2.0 * np.maximum(playouts, 1)), 0)


def sibling_pairs(sibling_groups: np.ndarray, is_sibling: np.ndarray, actors: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """The rows of every sibling group (without its root, which is the state before the move) with the acting player"""
    groups: Dict[int, List[int]] = {}
    for row in np.flatnonzero((sibling_groups >= 0) & is_sibling):
        groups.setdefault(int(sibling_groups[row]), []).append(row)
    return [(np.array(rows), int(actors[rows[0]])) for rows in groups.values() if len(rows) >= 2]


def ranking_metrics(labels: np.ndarray, predictions: np.ndarray, groups: List[Tuple[np.ndarray, int]]) -> Dict[str, float]:
    """Pair accuracy by threshold on the label difference (ties in the prediction count half), best-move hits and regret"""
    agree = {t: 0.0 for t in RANKING_THRESHOLDS}
    pairs = {t: 0 for t in RANKING_THRESHOLDS}
    hits, regret = 0, 0.0
    for rows, actor in groups:
        label = labels[rows, actor]
        prediction = predictions[rows, actor]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                difference = label[i] - label[j]
                if difference == 0:
                    continue
                predicted = prediction[i] - prediction[j]
                score = 0.5 if predicted == 0 else float(np.sign(predicted) == np.sign(difference))
                for t in RANKING_THRESHOLDS:
                    if abs(difference) >= t:
                        agree[t] += score
                        pairs[t] += 1
        pick = int(np.argmax(prediction))
        hits += int(label[pick] == label.max())
        regret += float(label.max() - label[pick])

    metrics = {f"pair_accuracy_{t:g}": agree[t] / pairs[t] if pairs[t] else float("nan") for t in RANKING_THRESHOLDS}
    metrics.update({f"pairs_{t:g}": pairs[t] for t in RANKING_THRESHOLDS})
    metrics["best_move"] = hits / len(groups) if groups else float("nan")
    metrics["regret_pp"] = 100 * regret / len(groups) if groups else float("nan")
    return metrics


def calibration(labels: np.ndarray, predictions: np.ndarray, vp: np.ndarray) -> List[Tuple[str, int, float, float]]:
    """(deficit bucket, player entries, mean predicted, mean actual win probability) by points behind the leader"""
    deficit = vp.max(axis=1, keepdims=True) - vp
    rows = []
    for low, high, name in DEFICITS:
        mask = (deficit >= low) & (deficit <= high)
        if mask.any():
            rows.append((name, int(mask.sum()), float(predictions[mask].mean()), float(labels[mask].mean())))
    return rows


def paired_difference(values: np.ndarray, reference: np.ndarray, clusters: np.ndarray) -> Tuple[float, float]:
    """Mean difference and the half width of its 95% interval, with the samples of a match as one cluster"""
    difference = values - reference
    mean = difference.mean()
    _, cluster_index = np.unique(clusters, return_inverse=True)
    # Cluster-robust variance of the mean: the residuals of a match are summed before squaring
    residual_sums = np.bincount(cluster_index, weights=difference - mean)
    clusters_count = len(residual_sums)
    variance = (residual_sums ** 2).sum() / len(difference) ** 2 * clusters_count / max(clusters_count - 1, 1)
    return float(mean), float(1.96 * np.sqrt(variance))


def model_inputs(width: int, data: Dict[str, np.ndarray]) -> np.ndarray:
    """The features of a net of the given input width: a belief-aware net (the view layout) gets the views (views.npy,
    DatasetCollector --regenerate-views), nets of an older layout the start of the states (later blocks are appended)"""
    if width == VIEW_FEATURE_DIM:
        if "views" not in data:
            raise ValueError(f"A net of {width} features needs the views of the precision set (views.npy, DatasetCollector --regenerate-views).")
        return data["views"]
    if width > data["states"].shape[1]:
        raise ValueError(f"A net of {width} features doesn't fit the precision set's {data['states'].shape[1]}.")
    return data["states"][:, :width]


def run_model(path: str, data: Dict[str, np.ndarray], batch: int = 4096) -> np.ndarray:
    """Win probabilities of an ONNX value net (softmax of its "logits" output)"""
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    width = session.get_inputs()[0].shape[1]
    inputs = model_inputs(width if isinstance(width, int) else data["states"].shape[1], data)
    logits = np.concatenate([session.run(["logits"], {"state": np.asarray(inputs[i:i + batch], dtype=np.float32)})[0]
                             for i in range(0, len(inputs), batch)])
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def load_precision_set(data: str, exclude_seeds: Optional[str]) -> Dict[str, np.ndarray]:
    """The arrays of a prepared precision set, without samples lacking a label and matches that occur in the training data"""
    arrays = {name: np.load(os.path.join(data, f"{name}.npy"), mmap_mode="r")
              for name in ("states", "win_rates", "sibling_groups", "actors", "chance_moves", "match_seeds", "finished_playouts")}
    with open(os.path.join(data, "dataset_index.csv"), newline="") as file:
        names = [row["entry_name"] for row in csv.DictReader(file)]
    is_sibling = np.array(["_c" in name for name in names])

    keep = np.asarray(arrays["win_rates"]).sum(axis=1) > 0.99
    seeds = effective_seeds(arrays["match_seeds"])
    excluded = 0
    if exclude_seeds:
        training = np.unique(effective_seeds(np.load(exclude_seeds, mmap_mode="r")))
        overlap = np.isin(seeds, training)
        excluded = len(np.unique(seeds[overlap]))
        keep &= ~overlap

    # The belief-aware inputs, if the set has them (the metrics keep using the full states)
    views_path = os.path.join(data, "views.npy")
    if os.path.exists(views_path):
        arrays["views"] = np.load(views_path, mmap_mode="r")
    result = {name: np.asarray(array)[keep] for name, array in arrays.items()}
    result["states"] = result["states"].astype(np.float32)
    result["win_rates"] = result["win_rates"].astype(np.float64)
    result["is_sibling"] = is_sibling[keep]
    result["seeds"] = seeds[keep]
    print(f"Precision set: {int(keep.sum())} of {len(keep)} samples, {len(np.unique(result['seeds']))} matches"
          + (f" ({excluded} matches left out, they also occur in the training data)" if exclude_seeds else ""))
    return result


def evaluate(data: Dict[str, np.ndarray], models: List[Tuple[str, np.ndarray]]) -> None:
    labels = data["win_rates"]
    vp = victory_points(data["states"])
    leader = vp.max(axis=1)
    groups = sibling_pairs(data["sibling_groups"], data["is_sibling"], data["actors"])
    chance = data["chance_moves"].astype(bool)
    groups_no_chance = [(rows[~chance[rows]], actor) for rows, actor in groups if (~chance[rows]).sum() >= 2]
    floor = noise_floor(labels, data["finished_playouts"])
    print(f"{len(groups)} sibling groups ({len(groups_no_chance)} with at least two moves without chance)")
    print(f"Label noise floor (expected KL of a perfect model): {floor.mean():.5f}\n")

    kls = {name: kl_divergence(labels, predictions) for name, predictions in models}
    rankings = {name: ranking_metrics(labels, predictions, groups) for name, predictions in models}

    for name, predictions in models:
        kl, ranking = kls[name], rankings[name]
        print(f"== {name}")
        print(f"KL {kl.mean():.5f} (above the noise floor: {kl.mean() - floor.mean():.5f})")
        print("  by phase: " + ", ".join(f"{label} {kl[(leader >= low) & (leader <= high)].mean():.5f}"
                                         for low, high, label in PHASES if ((leader >= low) & (leader <= high)).any()))
        print("Ranking: " + ", ".join(f"pairs with |diff| >= {100 * t:g}pp {ranking[f'pair_accuracy_{t:g}']:.3f} ({ranking[f'pairs_{t:g}']})"
                                      for t in RANKING_THRESHOLDS))
        print(f"  best move {ranking['best_move']:.3f}, regret {ranking['regret_pp']:.2f}pp"
              + (f"; without chance moves: best move {ranking_metrics(labels, predictions, groups_no_chance)['best_move']:.3f}" if groups_no_chance else ""))
        print("Calibration by VP deficit (predicted / actual win probability):")
        for bucket, count, predicted, actual in calibration(labels, predictions, vp):
            print(f"  {bucket:>3} behind: {predicted:.4f} / {actual:.4f} ({count} player entries)")
        print()

    reference_name, _ = models[0]
    for name, _ in models[1:]:
        mean, half = paired_difference(kls[name], kls[reference_name], data["seeds"])
        print(f"{name} - {reference_name}: KL {mean:+.5f} +/- {half:.5f} (95%, by match), "
              f"pair accuracy {rankings[name]['pair_accuracy_0'] - rankings[reference_name]['pair_accuracy_0']:+.3f}, "
              f"best move {rankings[name]['best_move'] - rankings[reference_name]['best_move']:+.3f}, "
              f"regret {rankings[name]['regret_pp'] - rankings[reference_name]['regret_pp']:+.2f}pp")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score ONNX value nets on a precision set.")
    parser.add_argument("--data", required=True, help="prepared precision set (prepare_dataset.py output)")
    parser.add_argument("--exclude-seeds", default=None, help="match_seeds.npy of the training data, to leave out matches that occur there")
    parser.add_argument("models", nargs="+", help="name=path.onnx; the first one is the reference for the differences")
    args = parser.parse_args(argv)

    data = load_precision_set(args.data, args.exclude_seeds)
    models = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        models.append((name, run_model(path, data)))
    evaluate(data, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
