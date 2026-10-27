import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader, Sampler
from state_value_net import StateValueModule, PLAYER_COUNT
from augmentation import (BOARD_SYMMETRY_COUNT, NUMBER_PAIR_COUNT, is_rotatable, rotate_features, rotate_player_index,
                          rotate_players, swap_number_pairs, transform_board)
# augmentation legt DatasetProcessing in den Suchpfad
from state_to_vector import FEATURE_DIM  # noqa: E402

# Zusätzliche Label-Arrays aus DatasetProcessing/playout_results.py (gleiche Zeilenreihenfolge wie die States), alle optional
AUX_ARRAYS = ("placements", "vp_shares", "rounds_left", "sibling_groups", "actors")


class StateDataset(Dataset):
    def __init__(self, states_path, labels_path, aux_dir=None, augment_rotation=False, augment_numbers=False, augment_board=False):
        self.states_path = states_path
        self.labels_path = labels_path
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
        self._open()

    def _open(self):
        # Memory mapped NP Arrays
        self.states = np.load(self.states_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')
        self.aux = {name: np.load(path, mmap_mode='r') for name, path in self.aux_paths.items()}

    def __getstate__(self):
        # Wird beim Picklen des Datasets vor dem Senden an Worker aufgerufen
        # Deshalb erst nach der Übergabe die MemMaps anlegen, die werden sonst kopiert
        state = self.__dict__.copy()
        state["states"] = None
        state["labels"] = None
        state["aux"] = None
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
        y = take(self.labels, self.labels.dtype)
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
        if self.augment_board:
            ts = torch.randint(BOARD_SYMMETRY_COUNT, (n,)).numpy()
            for t in range(1, BOARD_SYMMETRY_COUNT):
                rows = np.flatnonzero(ts == t)
                if rows.size:
                    x[rows] = transform_board(x[rows], t)

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "placements": torch.from_numpy(placements),
            "vp_shares": torch.from_numpy(vp_shares),
            "rounds_left": torch.from_numpy(rounds_left),
            "sibling_groups": torch.from_numpy(sibling_groups),
            "actors": torch.from_numpy(np.maximum(actors, 0)),
        }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="dataset/states.npy")
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
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--devices", type=int, default=1)
    # bf16-mixed auf Ampere oder neuer, sonst 16-mixed
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    full_ds = StateDataset(args.states, args.labels, args.aux_dir)
    for name, weight in [("placements", args.placement_weight), ("vp_shares", args.vp_share_weight),
                         ("rounds_left", args.rounds_weight), ("sibling_groups", args.ranking_weight),
                         ("placements", args.ranking_placement_weight)]:
        if weight > 0 and name not in full_ds.aux_paths:
            raise SystemExit(f"--{name} loss needs {name}.npy in --aux-dir (from DatasetProcessing/playout_results.py)")
    if args.ranking_placement_weight > 0 and args.placement_weight == 0:
        # Das Ranking trainiert nur Unterschiede innerhalb der Gruppen, das absolute Niveau kommt vom Platzierungs-Loss
        raise SystemExit("--ranking-placement-weight needs --placement-weight > 0")
    print(f"Aux labels: {', '.join(full_ds.aux_paths) or 'none'}")
    if args.augment_board and full_ds.states.shape[1] != FEATURE_DIM:
        raise SystemExit(f"--augment-board needs states with the ports as features ({FEATURE_DIM} features, see migrate_states_1764_to_1965.py)")

    train_idx, val_idx = np.load(args.train_idx), np.load(args.val_idx)
    augment = args.augment_rotation or args.augment_numbers or args.augment_board
    train_ds = StateDataset(args.states, args.labels, args.aux_dir, augment_rotation=args.augment_rotation,
                            augment_numbers=args.augment_numbers, augment_board=args.augment_board) if augment else full_ds
    loader_args = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=True, collate_fn=keep_batch)
    if args.ranking_weight > 0:
        # Sibling-Gruppen zusammenhalten, auch in der Validation (für die Ranking-Metriken)
        groups = full_ds.aux["sibling_groups"]
        train_loader = DataLoader(train_ds, batch_sampler=GroupBatchSampler(train_idx, groups, args.batch_size, shuffle=True), **loader_args)
        val_loader = DataLoader(full_ds, batch_sampler=GroupBatchSampler(val_idx, groups, args.batch_size, shuffle=False), **loader_args)
    else:
        train_loader = DataLoader(torch.utils.data.Subset(train_ds, train_idx), batch_size=args.batch_size, shuffle=True, **loader_args)
        val_loader = DataLoader(torch.utils.data.Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False, **loader_args)

    in_dim = full_ds.states.shape[1]
    model = StateValueModule(in_dim=in_dim, lr=args.lr, placement_weight=args.placement_weight, vp_share_weight=args.vp_share_weight,
                             rounds_weight=args.rounds_weight, ranking_weight=args.ranking_weight, normalize_inputs=args.normalize_inputs,
                             ranking_placement_weight=args.ranking_placement_weight, hidden=args.hidden)
    if args.normalize_inputs:
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

    logger = L.pytorch.loggers.WandbLogger(project="colonysim")
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
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
