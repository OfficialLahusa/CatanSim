import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader, Sampler
from state_value_net import StateValueModule, PLAYER_COUNT
from augmentation import is_rotatable, rotate_features, rotate_player_index, rotate_players

# Zusätzliche Label-Arrays aus DatasetProcessing/playout_results.py (gleiche Zeilenreihenfolge wie die States), alle optional
AUX_ARRAYS = ("placements", "vp_shares", "rounds_left", "sibling_groups", "actors")


class StateDataset(Dataset):
    def __init__(self, states_path, labels_path, aux_dir=None, augment_rotation=False):
        self.states_path = states_path
        self.labels_path = labels_path
        # Spieler zufällig zyklisch umbenennen (nur nach der Setup-Phase, siehe augmentation.py), für die Trainingsdaten
        self.augment_rotation = augment_rotation
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
        x = self.states[idx].astype(np.float32)
        y = np.asarray(self.labels[idx], dtype=np.float32)
        # Fehlende Labels als NaN bzw. -1, die Losses lassen solche Samples aus
        aux = self.aux
        placements = np.asarray(aux["placements"][idx], dtype=np.float32) if "placements" in aux \
            else np.full((PLAYER_COUNT, PLAYER_COUNT), np.nan, dtype=np.float32)
        vp_shares = np.asarray(aux["vp_shares"][idx], dtype=np.float32) if "vp_shares" in aux \
            else np.full((PLAYER_COUNT,), np.nan, dtype=np.float32)
        actor = int(aux["actors"][idx]) if "actors" in aux else -1

        # Die Labels pro Spieler wandern mit ihrem Spieler (torch.randint, damit jeder Worker eigene Zufallszahlen hat)
        if self.augment_rotation and is_rotatable(x):
            k = int(torch.randint(PLAYER_COUNT, (1,)))
            if k:
                x = rotate_features(x, k)
                y = rotate_players(y, k)
                placements = rotate_players(placements, k, axis=0)
                vp_shares = rotate_players(vp_shares, k)
                actor = rotate_player_index(actor, k)

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "placements": torch.from_numpy(np.ascontiguousarray(placements)),
            "vp_shares": torch.from_numpy(np.ascontiguousarray(vp_shares)),
            "rounds_left": torch.tensor(float(aux["rounds_left"][idx]) if "rounds_left" in aux else float("nan")),
            "sibling_groups": torch.tensor(int(aux["sibling_groups"][idx]) if "sibling_groups" in aux else -1),
            "actors": torch.tensor(max(actor, 0)),
        }


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
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--devices", type=int, default=1)
    # bf16-mixed auf Ampere oder neuer, sonst 16-mixed
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=8)
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

    train_idx, val_idx = np.load(args.train_idx), np.load(args.val_idx)
    train_ds = StateDataset(args.states, args.labels, args.aux_dir, augment_rotation=True) if args.augment_rotation else full_ds
    loader_args = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
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
                             ranking_placement_weight=args.ranking_placement_weight)
    if args.normalize_inputs:
        sample = np.sort(np.random.default_rng(0).choice(train_idx, min(len(train_idx), 200_000), replace=False))
        states = np.asarray(full_ds.states[sample], dtype=np.float32)
        if args.augment_rotation:
            # Statistik über die Sichten, die das Training sieht: jede rotierbare Zeile um ein zufälliges k gedreht
            rng = np.random.default_rng(1)
            for row in np.flatnonzero(is_rotatable(states)):
                states[row] = rotate_features(states[row], int(rng.integers(PLAYER_COUNT)))
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
