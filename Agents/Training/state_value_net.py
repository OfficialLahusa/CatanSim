import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

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