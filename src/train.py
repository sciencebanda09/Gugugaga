"""
Baby Voice Decoder - Training Pipeline
Full training loop with mixed precision, warmup cosine LR, early stopping,
differential LRs, balanced sampling, and CSV/checkpoint logging.
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lr        = base_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_scale", 1.0)
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._losses                        = []
        self._cr_preds, self._cr_labels     = [], []
        self._ag_preds, self._ag_labels     = [], []

    def update(self, loss, cr_preds, cr_labels, ag_preds, ag_labels):
        self._losses.append(loss)
        self._cr_preds.extend(cr_preds.cpu().numpy())
        self._cr_labels.extend(cr_labels.cpu().numpy())
        self._ag_preds.extend(ag_preds.cpu().numpy())
        self._ag_labels.extend(ag_labels.cpu().numpy())

    def compute(self) -> Dict:
        cr_acc  = accuracy_score(self._cr_labels, self._cr_preds)
        ag_acc  = accuracy_score(self._ag_labels, self._ag_preds)
        cr_f1   = f1_score(self._cr_labels, self._cr_preds, average="macro", zero_division=0)
        ag_f1   = f1_score(self._ag_labels, self._ag_preds, average="macro", zero_division=0)
        return {
            "loss":          np.mean(self._losses),
            "cry_reason_acc": cr_acc,
            "age_group_acc": ag_acc,
            "cry_reason_f1": cr_f1,
            "age_group_f1":  ag_f1,
            "mean_acc":      (cr_acc + ag_acc) / 2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval steps
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device, cfg) -> Dict:
    model.train()
    tracker = MetricsTracker()

    for batch in loader:
        spec = batch["spectrogram"].to(device)

        with autocast(enabled=cfg.use_amp and device == "cuda"):
            outputs = model(spec)
            losses  = criterion(outputs, {k: (v.to(device) if torch.is_tensor(v) else v)
                                          for k, v in batch.items() if k != "filepath"})

        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        mixed = batch.get("mixed", False)
        if not mixed:
            cr_preds = outputs["cry_reason_logits"].argmax(1)
            ag_preds = outputs["age_group_logits"].argmax(1)
            tracker.update(
                losses["total_loss"].item(),
                cr_preds, batch["cry_reason_label"].to(device),
                ag_preds, batch["age_group_label"].to(device),
            )

    return tracker.compute()


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> Dict:
    model.eval()
    tracker = MetricsTracker()

    for batch in loader:
        spec    = batch["spectrogram"].to(device)
        outputs = model(spec)
        losses  = criterion(outputs, {k: (v.to(device) if torch.is_tensor(v) else v)
                                       for k, v in batch.items() if k != "filepath"})

        cr_preds = outputs["cry_reason_logits"].argmax(1)
        ag_preds = outputs["age_group_logits"].argmax(1)
        tracker.update(
            losses["total_loss"].item(),
            cr_preds, batch["cry_reason_label"].to(device),
            ag_preds, batch["age_group_label"].to(device),
        )

    return tracker.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_metrics, cfg, filename: str):
    path = Path(cfg.checkpoint_dir) / filename
    torch.save({
        "epoch":             epoch,
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":           val_metrics["mean_acc"],
        "val_cry_acc":       val_metrics["cry_reason_acc"],
        "val_metrics":       val_metrics,
    }, path)
    print(f"  💾 Saved {path}")
