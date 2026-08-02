import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader, Subset


class StateDataset(Dataset):
    def __init__(self, states_path, labels_path):
        self.states_path = states_path
        self.labels_path = labels_path
        self._open()

    def _open(self):
        # Memory mapped NP Arrays
        self.states = np.load(self.states_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')

    def __getstate__(self):
        # Wird beim Picklen des Datasets vor dem Senden an Worker aufgerufen
        # Deshalb erst nach der Übergabe die MemMaps anlegen, die werden sonst kopiert
        state = self.__dict__.copy()
        state["states"] = None
        state["labels"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In jedem Worker Process eigene MemMaps öffnen
        self._open()  

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # from_numpy kopiert nicht, sondern teilt zugrundeliegenden Speicher
        x = torch.from_numpy(self.states[idx].astype(np.float32))
        y = torch.from_numpy(self.labels[idx].astype(np.float32))
        return x, y


class StateValueNet(nn.Module):
    def __init__(self, in_dim, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, int(hidden/2)), nn.LayerNorm(int(hidden/2)), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(int(hidden/2), 4)
        )

    def forward(self, x):
        return self.net(x)


class StateValueModule(L.LightningModule):
    def __init__(self, in_dim, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = StateValueNet(in_dim)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)

        # Log-softmax wird für KL-Divergenz erwartet
        # Batch-Mean Reduction => Innerhalb jedes Samples KL-Div summieren und dann Mean über Batch berechnen für Loss-Skalar
        loss = F.kl_div(F.log_softmax(logits, dim=-1), y, reduction='batchmean')
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            # Weight Decay als Regularisierung
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="dataset/states.npy")
    parser.add_argument("--labels", default="dataset/win_rates.npy")
    parser.add_argument("--train-idx", default="dataset/idx_train.npy")
    parser.add_argument("--val-idx", default="dataset/idx_val.npy")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--devices", type=int, default=1)
    # bf16-mixed auf Ampere oder neuer, sonst 16-mixed
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    full_ds = StateDataset(args.states, args.labels)
    train_ds = Subset(full_ds, np.load(args.train_idx))
    val_ds = Subset(full_ds, np.load(args.val_idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    in_dim = full_ds.states.shape[1]
    model = StateValueModule(in_dim=in_dim, lr=args.lr)

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
    )

    torch.set_float32_matmul_precision("high")

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()