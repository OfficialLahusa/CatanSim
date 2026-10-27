import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

PLAYER_COUNT = 4

# Runden bis Spielende werden für den Aux Head auf etwa [0, 1] skaliert
ROUNDS_LEFT_SCALE = 30.0

# Platzierungs-Score: Wert der Plätze 1 bis 4 auf [0, 1], gleich wie PlacementScore im MCTSAgent
PLACEMENT_SCORES = (1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0)


class StateValueNet(nn.Module):
    def __init__(self, in_dim, hidden=2048, aux_heads=False, normalize_inputs=False, policy_types=0, policy_vocabulary=0, policy_dim=64):
        super().__init__()
        # Optionale Standardisierung der Inputs (Mittelwert und Skala pro Feature aus den Trainingsdaten, siehe set_input_normalization),
        # Teil des Modells und damit auch des ONNX-Exports, die Vektorizer bleiben unskaliert
        if normalize_inputs:
            self.register_buffer("input_mean", torch.zeros(in_dim))
            self.register_buffer("input_scale", torch.ones(in_dim))
        else:
            self.input_mean = None
            self.input_scale = None
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, int(hidden/2)), nn.LayerNorm(int(hidden/2)), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(int(hidden/2), 4)
        )

        # Auxiliary Heads auf dem letzten Hidden Layer, nur fürs Training (forward und der ONNX-Export liefern nur die Value Logits):
        # Platzierungsverteilung (Logits pro Spieler und Platz), VP-Anteil (Logits pro Spieler) und Runden bis Spielende
        self.aux = nn.ModuleDict({
            "placements": nn.Linear(int(hidden/2), PLAYER_COUNT * PLAYER_COUNT),
            "vp_shares": nn.Linear(int(hidden/2), PLAYER_COUNT),
            "rounds_left": nn.Linear(int(hidden/2), 1),
        }) if aux_heads else None

        # Nur dann exportiert export_onnx die Platzierungen: mit Aux Heads für andere Losses wäre der Head untrainiert
        self.placements_trained = False

        # Policy Head (siehe Agents/Inference/ActionEncoding.cs): ein Query pro Zugtyp aus dem letzten Hidden Layer und eine
        # Embedding-Tabelle der Zug-Tokens; Logit eines Zugs = Query seines Typs · gewichtete Summe seiner Token-Embeddings
        self.policy_query = nn.Linear(int(hidden/2), policy_types * policy_dim) if policy_types > 0 else None
        self.policy_embeddings = nn.Embedding(policy_vocabulary, policy_dim) if policy_types > 0 else None
        self.policy_types, self.policy_dim = policy_types, policy_dim

    def _normalize(self, x):
        return x if self.input_mean is None else (x - self.input_mean) * self.input_scale

    def forward(self, x):
        return self.net(self._normalize(x))

    def forward_all(self, x):
        """Value Logits und, falls vorhanden, die Outputs der Aux Heads"""
        features = self.net[:-1](self._normalize(x))
        outputs = {"logits": self.net[-1](features)}
        if self.aux is not None:
            outputs["placements"] = self.aux["placements"](features).view(-1, PLAYER_COUNT, PLAYER_COUNT)
            outputs["vp_shares"] = self.aux["vp_shares"](features)
            outputs["rounds_left"] = self.aux["rounds_left"](features).squeeze(-1)
        if self.policy_query is not None:
            outputs["policy_query"] = self.policy_query(features).view(-1, self.policy_types, self.policy_dim)
        return outputs

    def policy_logits(self, query, tokens, weights):
        """Logits der Züge [batch, Züge] aus den Queries [batch, Typen, dim] und den Zug-Tokens [batch, Züge, Tokens]
        (-1 = leer) mit ihren Gewichten"""
        embedded = (self.policy_embeddings(tokens.clamp(min=0)) * (weights * (tokens >= 0))[..., None]).sum(dim=2)
        types = tokens[..., 0].clamp(min=0)
        selected = query.gather(1, types[..., None].expand(-1, -1, query.shape[-1]))
        return (selected * embedded).sum(dim=-1)


class StateValueModule(L.LightningModule):
    def __init__(self, in_dim, lr=1e-3, weight_decay=1e-4,
                 placement_weight=0.0, vp_share_weight=0.0, rounds_weight=0.0, ranking_weight=0.0, normalize_inputs=False,
                 ranking_placement_weight=0.0, hidden=2048, policy_weight=0.0, policy_types=0, policy_vocabulary=0, policy_dim=64):
        super().__init__()
        self.save_hyperparameters()
        aux_heads = placement_weight > 0 or vp_share_weight > 0 or rounds_weight > 0 or ranking_placement_weight > 0
        self.model = StateValueNet(in_dim, hidden=hidden, aux_heads=aux_heads, normalize_inputs=normalize_inputs,
                                   policy_types=policy_types if policy_weight > 0 else 0, policy_vocabulary=policy_vocabulary, policy_dim=policy_dim)
        self.model.placements_trained = placement_weight > 0

    def set_input_normalization(self, states):
        """Mittelwert und Skala pro Feature aus Trainings-States. Binäre Features (nur 0 und 1) bleiben unverändert,
        sonst würden seltene One-Hot-Features riesig; die übrigen werden auf Mittelwert 0 und Standardabweichung 1 gebracht."""
        states = np.asarray(states, dtype=np.float32)
        binary = np.all((states == 0) | (states == 1), axis=0)
        mean = np.where(binary, 0.0, states.mean(axis=0))
        scale = np.where(binary, 1.0, 1.0 / np.maximum(states.std(axis=0), 1e-3))
        self.model.input_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.model.input_scale.copy_(torch.from_numpy(scale.astype(np.float32)))
        return int((~binary).sum())

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        # States und Labels kommen im gespeicherten Typ (float16 halbiert den Transfer aus den Workern), gerechnet wird in float32
        x, y = batch["x"].float(), batch["y"].float()
        outputs = self.model.forward_all(x)
        logits = outputs["logits"]

        # Log-softmax wird für KL-Divergenz erwartet
        # Batch-Mean Reduction => Innerhalb jedes Samples KL-Div summieren und dann Mean über Batch berechnen für Loss-Skalar
        loss = F.kl_div(F.log_softmax(logits, dim=-1), y, reduction='batchmean')
        # val_loss bleibt die Value-KL, damit Runs mit und ohne Zusatz-Losses vergleichbar sind
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        total = loss

        hp = self.hparams
        if hp.placement_weight > 0:
            placement_loss = masked_kl(outputs["placements"], batch["placements"])
            self.log(f"{stage}_placement_loss", placement_loss, sync_dist=True)
            total = total + hp.placement_weight * placement_loss
        if hp.vp_share_weight > 0:
            vp_share_loss = masked_kl(outputs["vp_shares"], batch["vp_shares"])
            self.log(f"{stage}_vp_share_loss", vp_share_loss, sync_dist=True)
            total = total + hp.vp_share_weight * vp_share_loss
        if hp.rounds_weight > 0:
            rounds_loss = masked_rounds_loss(outputs["rounds_left"], batch["rounds_left"])
            self.log(f"{stage}_rounds_loss", rounds_loss, sync_dist=True)
            total = total + hp.rounds_weight * rounds_loss
        if hp.ranking_weight > 0:
            ranking_loss, accuracy, pairs = sibling_ranking(
                logits, y, batch["sibling_groups"], batch["actors"],
                outputs.get("placements"), batch["placements"], hp.get("ranking_placement_weight", 0.0))
            if pairs > 0:
                self.log(f"{stage}_ranking_loss", ranking_loss, sync_dist=True)
                self.log(f"{stage}_ranking_accuracy", accuracy, sync_dist=True)
            total = total + hp.ranking_weight * ranking_loss

        if hp.get("policy_weight", 0.0) > 0 and batch["policy_mask"].shape[1] > 0:
            policy_loss, accuracy = policy_cross_entropy(self.model, outputs["policy_query"], batch)
            self.log(f"{stage}_policy_loss", policy_loss, sync_dist=True)
            self.log(f"{stage}_policy_accuracy", accuracy, sync_dist=True)
            total = total + hp.policy_weight * policy_loss

        self.log(f"{stage}_total_loss", total, sync_dist=True)
        return total

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


def policy_cross_entropy(model, query, batch):
    """
    Kreuzentropie der Policy gegen den Anteil der Root Visits, Softmax nur über die legalen Züge (Maske); Zeilen ohne Züge
    (Daten ohne Policy-Targets) zählen nicht. Dazu der Anteil der Zeilen, in denen der meistbesuchte Zug auch das höchste
    Logit hat.
    """
    mask, targets = batch["policy_mask"], batch["policy_targets"].float()
    logits = model.policy_logits(query.float(), batch["policy_tokens"], batch["policy_weights"].float())
    logits = logits.masked_fill(~mask, float("-inf"))
    has_moves = mask.any(dim=1)
    log_probs = torch.where(mask, F.log_softmax(logits.masked_fill(~has_moves[:, None], 0.0), dim=-1), torch.zeros_like(logits))
    loss = _masked_mean(-(targets * log_probs).sum(dim=-1), has_moves)
    hits = (logits.argmax(dim=-1) == targets.argmax(dim=-1)).float()
    return loss, _masked_mean(hits, has_moves)


def masked_kl(logits, target):
    """KL-Divergenz zwischen Zielverteilung und Softmax über die letzte Dimension, gemittelt über alle Verteilungen.
    Samples ohne Label (NaN, z.B. aus Datensätzen ohne Rohergebnisse) zählen nicht. Maskiert statt indiziert, damit die
    GPU nicht pro Batch auf die CPU warten muss."""
    target = target.float()
    valid = ~torch.isnan(target).flatten(1).any(dim=1)
    target = torch.nan_to_num(target)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    kl = (torch.xlogy(target, target) - target * log_probs).sum(dim=-1)
    # Mittel pro Sample (Platzierungen: über die Spieler), dann über die Samples mit Label
    return _masked_mean(kl.flatten(1).mean(dim=1) if kl.dim() > 1 else kl, valid)


def _masked_mean(values, valid):
    """Mittelwert über die gültigen Einträge, 0 wenn keiner gültig ist"""
    return (values * valid).sum() / valid.sum().clamp(min=1)


def masked_rounds_loss(prediction, rounds_left):
    """Smooth L1 auf die skalierten Runden bis Spielende, ohne Samples ohne Label"""
    rounds_left = rounds_left.float()
    valid = ~torch.isnan(rounds_left)
    loss = F.smooth_l1_loss(prediction.float(), torch.nan_to_num(rounds_left) / ROUNDS_LEFT_SCALE, reduction="none")
    return _masked_mean(loss, valid)


def placement_score(placements):
    """Platzierungs-Score [..., Spieler] aus Platzierungsverteilungen [..., Spieler, 4 Plätze]"""
    return (placements * placements.new_tensor(PLACEMENT_SCORES)).sum(dim=-1)


def sibling_ranking(logits, win_rates, groups, actors, placement_logits=None, placement_labels=None, placement_weight=0.0):
    """
    Paarweiser Loss innerhalb der Sibling-Gruppen im Batch: die Differenz der vorhergesagten Werte des Spielers am Zug soll
    der Differenz der Labels entsprechen. Die Labels einer Gruppe stammen aus Playouts mit denselben Seeds, deshalb ist ihre
    Differenz deutlich genauer als die einzelnen Labels.
    Der Wert ist die Gewinnwahrscheinlichkeit plus placement_weight mal den Platzierungs-Score (wie die Utility im MCTSAgent),
    so hat das Ranking auch dann etwas zu lernen, wenn der Spieler am Zug weit zurückliegt und alle Gewinn-Labels 0 sind.
    Gibt Loss, Anteil der Paare mit richtiger Reihenfolge (bei mindestens 2 Punkten Unterschied im Label) und Anzahl Paare zurück.
    """
    in_group = groups >= 0
    if in_group.sum() < 2:
        return logits.sum() * 0.0, torch.tensor(0.0, device=logits.device), 0

    probs = F.softmax(logits[in_group].float(), dim=-1)
    labels = win_rates[in_group].float()
    if placement_weight > 0:
        probs = probs + placement_weight * placement_score(F.softmax(placement_logits[in_group].float(), dim=-1))
        labels = labels + placement_weight * placement_score(placement_labels[in_group].float())
    groups, actors = groups[in_group], actors[in_group].long()

    # Wert des Spielers am Zug, der zwischen den Zügen der Gruppe wählt
    predicted = probs.gather(1, actors[:, None]).squeeze(1)
    target = labels.gather(1, actors[:, None]).squeeze(1)

    same_group = (groups[:, None] == groups[None, :]).triu(diagonal=1)
    if not same_group.any():
        return logits.sum() * 0.0, torch.tensor(0.0, device=logits.device), 0

    predicted_diff = (predicted[:, None] - predicted[None, :])[same_group]
    target_diff = (target[:, None] - target[None, :])[same_group]
    loss = F.mse_loss(predicted_diff, target_diff)

    clear = target_diff.abs() >= 0.02
    accuracy = ((predicted_diff > 0) == (target_diff > 0))[clear].float().mean() if clear.any() else torch.tensor(0.0, device=logits.device)
    return loss, accuracy, int(same_group.sum())
