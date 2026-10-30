import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader, Sampler
from state_value_net import StateValueModule, PLAYER_COUNT
from augmentation import (BOARD_SYMMETRY_COUNT, NUMBER_PAIR_COUNT, is_rotatable, move_token_symmetries, rotate_features,
                          rotate_player_index, rotate_players, swap_number_pairs, transform_board)
# augmentation legt DatasetProcessing in den Suchpfad
from state_to_vector import FEATURE_DIM, VIEW_FEATURE_DIM  # noqa: E402

# Zusätzliche Label-Arrays aus DatasetProcessing/playout_results.py (gleiche Zeilenreihenfolge wie die States), alle optional
AUX_ARRAYS = ("placements", "vp_shares", "rounds_left", "sibling_groups", "actors")
# Policy-Targets aus selfplay_to_dataset.py: die legalen Züge pro Position als Tokens und ihr Anteil an den Root Visits
POLICY_ARRAYS = ("policy_offsets", "policy_tokens", "policy_weights", "policy_targets")


def read_policy_layout(directory):
    """Token-Layout der Policy-Targets (policy_layout.json) oder None, wenn das Verzeichnis keine hat"""
    if directory is None or not os.path.exists(os.path.join(directory, "policy_layout.json")):
        return None
    with open(os.path.join(directory, "policy_layout.json"), encoding="utf-8") as file:
        return json.load(file)


class StateDataset(Dataset):
    def __init__(self, states_path, labels_path, aux_dir=None, augment_rotation=False, augment_numbers=False, augment_board=False,
                 target_lambda=None, policy_temperature=1.0):
        self.states_path = states_path
        self.labels_path = labels_path
        # Self-Play-Daten (selfplay_to_dataset.py): Target λ·onehot(Sieger) + (1-λ)·Root-q statt der gespeicherten Labels,
        # wenn root_q.npy und winners.npy im aux_dir liegen; andere Daten behalten ihre Labels
        self.target_lambda = None
        # Visit-Anteile als Policy-Target hoch 1/T (T < 1 schärft die flachen Verteilungen der UCT-Suche), pro Position normiert
        self.policy_temperature = policy_temperature
        self.target_paths = {}
        if target_lambda is not None and aux_dir is not None:
            paths = {name: os.path.join(aux_dir, f"{name}.npy") for name in ("root_q", "winners")}
            if all(os.path.exists(path) for path in paths.values()):
                self.target_lambda = target_lambda
                self.target_paths = paths
        # Spieler zufällig zyklisch umbenennen (nur nach der Setup-Phase, siehe augmentation.py), für die Trainingsdaten
        self.augment_rotation = augment_rotation
        # Zahlenchip-Paare (2/12 bis 6/8) zufällig tauschen, in jeder Phase (siehe augmentation.py)
        self.augment_numbers = augment_numbers
        # Das Brett zufällig drehen und spiegeln (12 Symmetrien, braucht die Häfen als Features)
        self.augment_board = augment_board
        # Nur die Aux-Arrays, deren Datei existiert
        self.aux_paths = {}
        if aux_dir is not None:
            for name in AUX_ARRAYS:
                path = os.path.join(aux_dir, f"{name}.npy")
                if os.path.exists(path):
                    self.aux_paths[name] = path
        # Policy-Targets, wenn alle Arrays und das Token-Layout da sind; die Brett-Symmetrien permutieren auch die Tokens
        self.policy_layout = read_policy_layout(aux_dir)
        self.policy_paths = {}
        if self.policy_layout is not None and all(os.path.exists(os.path.join(aux_dir, f"{name}.npy")) for name in POLICY_ARRAYS):
            self.policy_paths = {name: os.path.join(aux_dir, f"{name}.npy") for name in POLICY_ARRAYS}
            self.token_symmetries = move_token_symmetries(self.policy_layout)
        self.max_tokens = self.policy_layout["max_tokens"][0] if self.policy_layout else 1
        self._open()

    def _open(self):
        # Memory mapped NP Arrays
        self.states = np.load(self.states_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')
        self.aux = {name: np.load(path, mmap_mode='r') for name, path in self.aux_paths.items()}
        self.targets = {name: np.load(path, mmap_mode='r') for name, path in self.target_paths.items()}
        self.policy = {name: np.load(path, mmap_mode='r') for name, path in self.policy_paths.items()}

    def __getstate__(self):
        # Wird beim Picklen des Datasets vor dem Senden an Worker aufgerufen
        # Deshalb erst nach der Übergabe die MemMaps anlegen, die werden sonst kopiert
        state = self.__dict__.copy()
        state["states"] = None
        state["labels"] = None
        state["aux"] = None
        state["targets"] = None
        state["policy"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In jedem Worker Process eigene MemMaps öffnen
        self._open()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {key: value[0] for key, value in self.__getitems__([idx]).items()}

    def __getitems__(self, indices):
        # Ganze Batches auf einmal (der DataLoader ruft das statt __getitem__ pro Sample auf): ein sortierter Zugriff pro
        # Array statt einer Python-Schleife über die Samples
        indices = np.asarray(indices, dtype=np.int64)
        n = len(indices)
        order = np.argsort(indices, kind="stable")
        restore = np.empty_like(order)
        restore[order] = np.arange(n)
        sorted_indices = indices[order]

        def take(array, dtype):
            return np.asarray(array[sorted_indices], dtype=dtype)[restore]

        # Im gespeicherten Typ (meist float16), das Modell rechnet sie in float32 um
        x = take(self.states, self.states.dtype)
        if self.target_lambda is None:
            y = take(self.labels, self.labels.dtype)
        else:
            one_hot = np.eye(PLAYER_COUNT, dtype=np.float32)[take(self.targets["winners"], np.int64)]
            y = (self.target_lambda * one_hot + (1 - self.target_lambda) * take(self.targets["root_q"], np.float32)).astype(np.float32)
        # Fehlende Labels als NaN bzw. -1, die Losses lassen solche Samples aus
        aux = self.aux
        placements = take(aux["placements"], np.float32) if "placements" in aux             else np.full((n, PLAYER_COUNT, PLAYER_COUNT), np.nan, dtype=np.float32)
        vp_shares = take(aux["vp_shares"], np.float32) if "vp_shares" in aux else np.full((n, PLAYER_COUNT), np.nan, dtype=np.float32)
        rounds_left = take(aux["rounds_left"], np.float32) if "rounds_left" in aux else np.full(n, np.nan, dtype=np.float32)
        sibling_groups = take(aux["sibling_groups"], np.int64) if "sibling_groups" in aux else np.full(n, -1, dtype=np.int64)
        actors = take(aux["actors"], np.int64) if "actors" in aux else np.full(n, -1, dtype=np.int64)

        # Die Labels pro Spieler wandern mit ihrem Spieler (torch.randint, damit jeder Worker eigene Zufallszahlen hat)
        if self.augment_rotation:
            ks = torch.randint(PLAYER_COUNT, (n,)).numpy()
            ks[~is_rotatable(x)] = 0
            for k in range(1, PLAYER_COUNT):
                rows = np.flatnonzero(ks == k)
                if rows.size:
                    x[rows] = rotate_features(x[rows], k)
                    y[rows] = rotate_players(y[rows], k)
                    placements[rows] = rotate_players(placements[rows], k, axis=1)
                    vp_shares[rows] = rotate_players(vp_shares[rows], k)
                    actors[rows] = rotate_player_index(actors[rows], k)

        if self.augment_numbers:
            x = swap_number_pairs(x, torch.randint(2 ** NUMBER_PAIR_COUNT, (n,)).numpy())
        ts = np.zeros(n, dtype=np.int64)
        if self.augment_board:
            ts = torch.randint(BOARD_SYMMETRY_COUNT, (n,)).numpy()
            for t in range(1, BOARD_SYMMETRY_COUNT):
                rows = np.flatnonzero(ts == t)
                if rows.size:
                    x[rows] = transform_board(x[rows], t)

        batch = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "placements": torch.from_numpy(placements),
            "vp_shares": torch.from_numpy(vp_shares),
            "rounds_left": torch.from_numpy(rounds_left),
            "sibling_groups": torch.from_numpy(sibling_groups),
            "actors": torch.from_numpy(np.maximum(actors, 0)),
        }
        batch.update(self._policy_batch(indices, ts))
        return batch

    def _policy_batch(self, indices, ts):
        """
        Die legalen Züge der Zeilen, aufgefüllt auf die größte Zuganzahl im Batch: Tokens (-1 = leer), ihre Gewichte, der
        Anteil an den Root Visits und eine Maske der echten Züge. Zeilen ohne Policy-Targets haben keine Züge.
        """
        n = len(indices)
        if not self.policy:
            return {"policy_tokens": torch.full((n, 0, self.max_tokens), -1, dtype=torch.int64),
                    "policy_weights": torch.zeros((n, 0, self.max_tokens)),
                    "policy_targets": torch.zeros((n, 0)), "policy_mask": torch.zeros((n, 0), dtype=torch.bool)}
        offsets = self.policy["policy_offsets"]
        starts, counts = offsets[indices], offsets[indices + 1] - offsets[indices]
        width = max(int(counts.max()), 1)
        total = int(counts.sum())
        row_of = np.repeat(np.arange(n), counts)
        slot = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        source = np.repeat(starts, counts) + slot
        # Sortiert aus den MemMaps lesen
        order = np.argsort(source, kind="stable")
        restore = np.empty_like(order)
        restore[order] = np.arange(total)

        def take(name, dtype):
            return np.asarray(self.policy[name][source[order]], dtype=dtype)[restore]

        tokens = np.full((n, width, self.max_tokens), -1, dtype=np.int64)
        weights = np.zeros((n, width, self.max_tokens), dtype=np.float32)
        targets = np.zeros((n, width), dtype=np.float32)
        mask = np.zeros((n, width), dtype=bool)
        moved = take("policy_tokens", np.int64)
        # Die Brett-Symmetrie der Zeile auch auf ihre Züge anwenden
        symmetry = np.broadcast_to(ts[row_of][:, None], moved.shape)
        valid = moved >= 0
        moved[valid] = self.token_symmetries[symmetry[valid], moved[valid]]
        tokens[row_of, slot] = moved
        weights[row_of, slot] = take("policy_weights", np.float32)
        shares = take("policy_targets", np.float32)
        if self.policy_temperature != 1.0:
            shares = shares ** (1.0 / self.policy_temperature)
            sums = np.zeros(n, dtype=np.float32)
            np.add.at(sums, row_of, shares)
            shares = shares / np.maximum(sums[row_of], 1e-12)
        targets[row_of, slot] = shares
        mask[row_of, slot] = True
        return {"policy_tokens": torch.from_numpy(tokens), "policy_weights": torch.from_numpy(weights),
                "policy_targets": torch.from_numpy(targets), "policy_mask": torch.from_numpy(mask)}


class MixedDataset(Dataset):
    """Mehrere StateDatasets hintereinander (globaler Index = Offset der Quelle + Zeile), batchweise gelesen"""
    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in datasets])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        return {key: value[0] for key, value in self.__getitems__([idx]).items()}

    def __getitems__(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        source = np.searchsorted(self.offsets, indices, side="right") - 1
        parts = [self.datasets[s].__getitems__(indices[source == s] - self.offsets[s]) for s in np.unique(source)]
        # Die Quellen können Labels in verschiedenen Typen speichern (v2 float16, Self-Play float32)
        batch = {}
        for key in parts[0]:
            dtype = parts[0][key].dtype
            for part in parts[1:]:
                dtype = torch.promote_types(dtype, part[key].dtype)
            tensors = [part[key].to(dtype) for part in parts]
            if key.startswith("policy_"):
                # Die Quellen füllen auf verschieden viele Züge auf (Daten ohne Policy auf keinen)
                # und auf verschieden viele Tokens pro Zug (Daten ohne Layout auf einen)
                width = max(t.shape[1] for t in tensors)
                tokens = max(t.shape[2] for t in tensors) if tensors[0].dim() == 3 else 0
                tensors = [F.pad(t, ([0, tokens - t.shape[2]] if t.dim() == 3 else []) + [0, width - t.shape[1]],
                                 value=-1 if key == "policy_tokens" else 0) for t in tensors]
            batch[key] = torch.cat(tensors)
        return batch


class MixSampler(Sampler):
    """
    Pro Epoche epoch_samples Trainings-Indizes, aus jeder Quelle ihr Anteil: ohne Zurücklegen, solange ihre Trainingsmenge
    reicht, sonst mit. Alles gemischt, damit jeder Batch die Quellen im Verhältnis enthält.
    """
    def __init__(self, indices, shares, epoch_samples):
        self.indices = [np.asarray(idx, dtype=np.int64) for idx in indices]
        shares = np.asarray(shares, dtype=np.float64)
        self.counts = np.round(shares / shares.sum() * epoch_samples).astype(np.int64)

    def __len__(self):
        return int(self.counts.sum())

    def __iter__(self):
        rng = np.random.default_rng(int(torch.randint(2 ** 31, (1,))))
        drawn = [rng.choice(idx, count, replace=count > len(idx)) for idx, count in zip(self.indices, self.counts) if count > 0]
        return iter(rng.permutation(np.concatenate(drawn)).tolist())


def widen_state_dict(state_dict, in_dim, new_mean=None, new_scale=None):
    """Ein Checkpoint mit weniger Input-Features für ein Modell mit in_dim (das Sicht-Layout hängt Features an das aktuelle
    an): die erste Schicht bekommt Gewicht 0 für die neuen Spalten, das Net rechnet auf den alten Features also wie vorher, die
    Input-Normalisierung bekommt für sie den gegebenen Mittelwert und die Skala (sonst 0 und 1). Gleich breit bleibt er, wie er ist."""
    weight = state_dict["model.net.0.weight"]
    old_dim = weight.shape[1]
    if old_dim > in_dim:
        raise ValueError(f"The checkpoint has {old_dim} input features, more than the model's {in_dim}")
    if old_dim == in_dim:
        return state_dict
    added = in_dim - old_dim
    widened = dict(state_dict)
    widened["model.net.0.weight"] = torch.cat([weight, torch.zeros(weight.shape[0], added, dtype=weight.dtype)], dim=1)
    if "model.input_mean" in state_dict:
        mean = torch.zeros(added) if new_mean is None else torch.as_tensor(new_mean, dtype=torch.float32)
        scale = torch.ones(added) if new_scale is None else torch.as_tensor(new_scale, dtype=torch.float32)
        widened["model.input_mean"] = torch.cat([state_dict["model.input_mean"], mean.to(state_dict["model.input_mean"].dtype)])
        widened["model.input_scale"] = torch.cat([state_dict["model.input_scale"], scale.to(state_dict["model.input_scale"].dtype)])
    return widened


def parse_source(spec):
    """'DIR' oder 'DIR@ANTEIL': ein vorbereitetes Verzeichnis (states.npy, win_rates.npy, idx_*.npy, Aux-Arrays)"""
    directory, share = spec, 1.0
    if "@" in spec:
        directory, share_text = spec.rsplit("@", 1)
        share = float(share_text)
    return directory, share


def keep_batch(batch):
    """collate_fn für Datasets mit __getitems__: der Batch ist schon fertig zusammengesetzt"""
    return batch


class GroupBatchSampler(Sampler):
    """
    Batches, die keine Sibling-Gruppe auseinanderreißen, damit der Ranking-Loss die Paare einer Gruppe im selben Batch sieht.
    Gruppen und einzelne Samples (Units) werden gemischt und der Reihe nach in Batches von etwa batch_size Samples geschnitten,
    jeweils an der nächsten Unit-Grenze.
    """
    def __init__(self, indices, groups, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.asarray(indices, dtype=np.int64)

        # Unit pro Sample: die Sibling-Gruppe, oder eine eigene für Samples ohne Gruppe
        unit = np.asarray(groups[self.indices], dtype=np.int64)
        singles = unit < 0
        unit[singles] = max(int(unit.max()), -1) + 1 + np.arange(int(singles.sum()))
        unique_units, self.unit_of = np.unique(unit, return_inverse=True)
        self.unit_count = unique_units.size

    def __len__(self):
        # Exact, since Lightning starts the validation once the stated number of batches is reached
        return max(1, -(-len(self.indices) // self.batch_size))

    def __iter__(self):
        keys = np.random.permutation(self.unit_count) if self.shuffle else np.arange(self.unit_count)
        order = np.argsort(keys[self.unit_of], kind="stable")
        indices, units = self.indices[order], self.unit_of[order]
        starts = np.flatnonzero(np.r_[True, units[1:] != units[:-1]])

        # Exactly len(self) batches: the i-th ends at the first unit start from i * n / batches on (a unit is only split
        # if units are as large as batches, which they aren't with sibling groups of a few samples)
        batches = len(self)
        end = 0
        for i in range(1, batches + 1):
            start = end
            if i == batches:
                end = len(indices)
            else:
                target = (i * len(indices)) // batches
                next_start = np.searchsorted(starts, target)
                end = starts[next_start] if next_start < len(starts) else len(indices)
                end = min(max(end, start + 1), len(indices) - (batches - i))
            yield indices[start:end].tolist()


def set_normalization(model, full_ds, train_idx, args):
    """Input-Normalisierung aus bis zu 200.000 Trainings-States, über die Sichten, die das Training durch die Augmentierung sieht"""
    in_dim = full_ds.states.shape[1]
    sample = np.sort(np.random.default_rng(0).choice(train_idx, min(len(train_idx), 200_000), replace=False))
    states = np.asarray(full_ds.states[sample], dtype=np.float32)
    if args.augment_rotation:
        # Statistik über die Sichten, die das Training sieht: jede rotierbare Zeile um ein zufälliges k gedreht
        ks = np.random.default_rng(1).integers(PLAYER_COUNT, size=len(states))
        ks[~is_rotatable(states)] = 0
        for k in range(1, PLAYER_COUNT):
            states[ks == k] = rotate_features(states[ks == k], k)
    if args.augment_numbers:
        states = swap_number_pairs(states, np.random.default_rng(2).integers(2 ** NUMBER_PAIR_COUNT, size=len(states)))
    if args.augment_board:
        ts = np.random.default_rng(3).integers(BOARD_SYMMETRY_COUNT, size=len(states))
        for t in range(1, BOARD_SYMMETRY_COUNT):
            states[ts == t] = transform_board(states[ts == t], t)
    scaled = model.set_input_normalization(states)
    print(f"Input normalization: {scaled} of {in_dim} features standardized, the binary ones unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="dataset/states.npy")
    # Die Feature-Datei der --data-Verzeichnisse: states.npy oder views.npy (Sicht des Sitzes mit Belief-Block, ViewVectorizer)
    parser.add_argument("--features", default="states.npy")
    parser.add_argument("--labels", default="dataset/win_rates.npy")
    parser.add_argument("--train-idx", default="dataset/idx_train.npy")
    parser.add_argument("--val-idx", default="dataset/idx_val.npy")
    # Verzeichnis mit den Arrays aus playout_results.py (placements.npy, vp_shares.npy, ...)
    parser.add_argument("--aux-dir", default=None)
    parser.add_argument("--placement-weight", type=float, default=0.0)
    parser.add_argument("--vp-share-weight", type=float, default=0.0)
    parser.add_argument("--rounds-weight", type=float, default=0.0)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    # Sibling-Ranking auf Gewinnwahrscheinlichkeit plus dieses Gewicht mal den Platzierungs-Score (braucht placements.npy)
    parser.add_argument("--ranking-placement-weight", type=float, default=0.0)
    # Inputs im Modell standardisieren (Statistik aus bis zu 200k Trainings-States)
    parser.add_argument("--normalize-inputs", action="store_true")
    # Trainingsdaten per zufälliger zyklischer Spielerrotation vervielfachen (siehe augmentation.py), Validation bleibt unverändert
    parser.add_argument("--augment-rotation", action="store_true")
    # Zusätzlich zufällig Zahlenchip-Paare tauschen (32 Sichten pro State)
    parser.add_argument("--augment-numbers", action="store_true")
    # Zusätzlich das Brett zufällig drehen und spiegeln (12 Sichten, nur mit dem aktuellen Layout mit Häfen)
    parser.add_argument("--augment-board", action="store_true")
    # Breite der ersten beiden Hidden Layer (die dritte hat die Hälfte)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Konstante Lernrate statt One-Cycle (Self-Play-Schleife: kleine Schritte pro Generation)
    parser.add_argument("--constant-lr", action="store_true")
    # Checkpoints in dieses Verzeichnis statt in das des W&B-Runs (die Self-Play-Schleife findet sie dort)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--devices", type=int, default=1)
    # bf16-mixed auf Ampere oder neuer, sonst 16-mixed
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=16)
    # Statt --states/--labels/--train-idx/--val-idx/--aux-dir: vorbereitete Verzeichnisse mit Anteil, z.B. Self-Play und v2
    # gemischt (--data selfplay_iter1@0.5 --data prepared_v2@0.5); validiert wird auf der ersten Quelle
    parser.add_argument("--data", action="append", default=None)
    # Trainings-Samples pro Epoche mit --data (Default: die Trainingsmenge der ersten Quelle geteilt durch ihren Anteil)
    parser.add_argument("--epoch-samples", type=int, default=None)
    # Target der Self-Play-Daten: λ·onehot(Sieger) + (1-λ)·Root-q (aus root_q.npy und winners.npy), Default: win_rates.npy
    parser.add_argument("--target-lambda", type=float, default=None)
    # Gewichte (und Input-Normalisierung) aus einem Checkpoint übernehmen und weitertrainieren, z.B. von E2 aus
    parser.add_argument("--init-from", default=None)
    # Policy Head auf die Visit-Verteilung der Self-Play-Daten (policy_*.npy aus selfplay_to_dataset.py); Daten ohne zählen nicht
    parser.add_argument("--policy-weight", type=float, default=0.0)
    parser.add_argument("--policy-dim", type=int, default=64)
    # Temperatur der Policy-Targets: Visit-Anteile hoch 1/T, T < 1 schärft
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    args = parser.parse_args()

    sources = [parse_source(spec) for spec in args.data] if args.data else []
    if sources:
        first = sources[0][0]
        args.states, args.labels = os.path.join(first, args.features), os.path.join(first, "win_rates.npy")
        args.train_idx, args.val_idx, args.aux_dir = os.path.join(first, "idx_train.npy"), os.path.join(first, "idx_val.npy"), first
        if args.ranking_weight > 0 and len(sources) > 1:
            raise SystemExit("--ranking-weight works with a single data source only")

    full_ds = StateDataset(args.states, args.labels, args.aux_dir, target_lambda=args.target_lambda, policy_temperature=args.policy_temperature)
    for name, weight in [("placements", args.placement_weight), ("vp_shares", args.vp_share_weight),
                         ("rounds_left", args.rounds_weight), ("sibling_groups", args.ranking_weight),
                         ("placements", args.ranking_placement_weight)]:
        if weight > 0 and name not in full_ds.aux_paths:
            raise SystemExit(f"--{name} loss needs {name}.npy in --aux-dir (from DatasetProcessing/playout_results.py)")
    if args.ranking_placement_weight > 0 and args.placement_weight == 0:
        # Das Ranking trainiert nur Unterschiede innerhalb der Gruppen, das absolute Niveau kommt vom Platzierungs-Loss
        raise SystemExit("--ranking-placement-weight needs --placement-weight > 0")
    print(f"Aux labels: {', '.join(full_ds.aux_paths) or 'none'}")
    if args.augment_board and full_ds.states.shape[1] not in (FEATURE_DIM, VIEW_FEATURE_DIM):
        raise SystemExit(f"--augment-board needs states with the ports as features ({FEATURE_DIM} or {VIEW_FEATURE_DIM} features, see migrate_states_1764_to_1965.py)")

    train_idx, val_idx = np.load(args.train_idx), np.load(args.val_idx)
    augment = args.augment_rotation or args.augment_numbers or args.augment_board
    train_ds = StateDataset(args.states, args.labels, args.aux_dir, augment_rotation=args.augment_rotation,
                            augment_numbers=args.augment_numbers, augment_board=args.augment_board,
                            target_lambda=args.target_lambda, policy_temperature=args.policy_temperature) if augment else full_ds
    loader_args = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=True, collate_fn=keep_batch)
    if len(sources) > 1:
        datasets = [train_ds] + [StateDataset(os.path.join(directory, args.features), os.path.join(directory, "win_rates.npy"), directory,
                                              augment_rotation=args.augment_rotation, augment_numbers=args.augment_numbers,
                                              augment_board=args.augment_board, target_lambda=args.target_lambda,
                                              policy_temperature=args.policy_temperature)
                                 for directory, _ in sources[1:]]
        if len({dataset.states.shape[1] for dataset in datasets}) != 1:
            raise SystemExit("The --data sources have different feature layouts")
        mixed = MixedDataset(datasets)
        indices = [train_idx] + [np.load(os.path.join(directory, "idx_train.npy")) for directory, _ in sources[1:]]
        indices = [idx + offset for idx, offset in zip(indices, mixed.offsets)]
        shares = [share for _, share in sources]
        epoch_samples = args.epoch_samples or int(len(train_idx) * sum(shares) / shares[0])
        sampler = MixSampler(indices, shares, epoch_samples)
        print("Training mix per epoch: " + ", ".join(f"{os.path.basename(os.path.normpath(directory))} {count}"
                                                     for (directory, _), count in zip(sources, sampler.counts)))
        train_loader = DataLoader(mixed, batch_size=args.batch_size, sampler=sampler, **loader_args)
        val_loader = DataLoader(torch.utils.data.Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False, **loader_args)
    elif args.ranking_weight > 0:
        # Sibling-Gruppen zusammenhalten, auch in der Validation (für die Ranking-Metriken)
        groups = full_ds.aux["sibling_groups"]
        train_loader = DataLoader(train_ds, batch_sampler=GroupBatchSampler(train_idx, groups, args.batch_size, shuffle=True), **loader_args)
        val_loader = DataLoader(full_ds, batch_sampler=GroupBatchSampler(val_idx, groups, args.batch_size, shuffle=False), **loader_args)
    else:
        train_loader = DataLoader(torch.utils.data.Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True, **loader_args)
        val_loader = DataLoader(torch.utils.data.Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False, **loader_args)

    in_dim = full_ds.states.shape[1]
    policy = {}
    if args.policy_weight > 0:
        if not full_ds.policy_paths:
            raise SystemExit("--policy-weight needs policy targets in the first --data source (selfplay_to_dataset.py of a run with policy files)")
        layout = full_ds.policy_layout
        policy = dict(policy_weight=args.policy_weight, policy_types=layout["move_types"][1], policy_vocabulary=layout["vocabulary"][1],
                      policy_dim=args.policy_dim)
    model = StateValueModule(in_dim=in_dim, lr=args.lr, constant_lr=args.constant_lr, placement_weight=args.placement_weight, vp_share_weight=args.vp_share_weight,
                             rounds_weight=args.rounds_weight, ranking_weight=args.ranking_weight, normalize_inputs=args.normalize_inputs,
                             ranking_placement_weight=args.ranking_placement_weight, hidden=args.hidden, **policy)
    if args.init_from:
        # Gleiche Architektur und Normalisierung wie der Checkpoint, sonst schlägt das Laden fehl; nur der Optimizer startet neu.
        # Ein neuer Policy Head fehlt im Checkpoint und startet zufällig
        # Nur Tensoren und einfache Typen: ein manipulierter Checkpoint kann so keinen Code ausführen
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        old_dim = state_dict["model.net.0.weight"].shape[1]
        if old_dim < in_dim:
            # Mehr Features als der Checkpoint (Sicht-Layout): die neuen Spalten starten mit Gewicht 0, ihre Normalisierung
            # kommt aus den Daten wie ohne Checkpoint
            new_mean = new_scale = None
            if "model.input_mean" in state_dict:
                set_normalization(model, full_ds, train_idx, args)
                new_mean = model.model.input_mean[old_dim:].numpy().copy()
                new_scale = model.model.input_scale[old_dim:].numpy().copy()
            state_dict = widen_state_dict(state_dict, in_dim, new_mean, new_scale)
            print(f"Widened the checkpoint from {old_dim} to {in_dim} input features")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected or any(not name.startswith("model.policy_") for name in missing):
            raise SystemExit(f"Checkpoint does not fit the model: missing {missing}, unexpected {unexpected}")
        print(f"Weights and input normalization from {args.init_from}" + (f" (new: {', '.join(missing)})" if missing else ""))
    elif args.normalize_inputs:
        set_normalization(model, full_ds, train_idx, args)

    logger = L.pytorch.loggers.WandbLogger(project="colonysim")
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="epoch={epoch}-step={step}-val_loss={val_loss:.6f}",
            auto_insert_metric_name=False,
        ),
        L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min"),
        L.pytorch.callbacks.LearningRateMonitor(),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=args.precision,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=callbacks,
        # Die Gruppen-Batches dürfen nicht durch einen DistributedSampler ersetzt werden
        use_distributed_sampler=args.ranking_weight == 0,
    )

    torch.set_float32_matmul_precision("high")

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
