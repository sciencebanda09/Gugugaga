"""
Baby Voice Decoder — Colab Live Visualizer
-------------------------------------------
Drop this file into /content/baby_voice_decoder/
Then add  `from colab_visualizer import BabyColabVisualizer`  in your training cell.

Adapted from Bird Language ML's ColabVisualizer.
Shows:
  - Live mel-spectrograms of cry audio being trained on
  - Top-k cry reason predictions with emoji labels
  - Loss / accuracy / F1 curves
  - Step-level loss for current epoch
  - LR schedule
"""

import io
import time
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, HTML, clear_output
import torch


# ── Warm colormap evoking a baby monitor / night-vision vibe ──────────────
_BABYCMAP = LinearSegmentedColormap.from_list(
    "baby", ["#0a0010", "#1a0a3e", "#4a1a8e", "#c850a0", "#ff9060", "#fffbe6"], N=256
)

_CRY_EMOJIS = {
    "hunger":           "🍼",
    "pain":             "😢",
    "discomfort":       "😣",
    "tiredness":        "😴",
    "fear":             "😨",
    "needs_attention":  "🤗",
    "overstimulation":  "🌀",
    "gas_colic":        "😖",
    "boredom":          "😐",
    "happy_babbling":   "😊",
}


class BabyColabVisualizer:
    """
    Live training dashboard for Baby Voice Decoder inside Google Colab.

    Usage
    -----
    vis = BabyColabVisualizer(cry_reasons=CRY_REASONS, age_groups=AGE_GROUPS, total_epochs=40)

    # Inside training loop:
    train_metrics = vis.train_epoch_with_viz(
        model, train_loader, optimizer, criterion, scaler, device, cfg, epoch
    )
    val_metrics = vis.eval_epoch_with_viz(model, val_loader, criterion, device, epoch)
    vis.end_epoch(epoch, train_metrics, val_metrics, lr)
    """

    def __init__(self, cry_reasons, age_groups, total_epochs=40, update_every=10):
        self.cry_reasons    = cry_reasons
        self.age_groups     = age_groups
        self.total_epochs   = total_epochs
        self.update_every   = update_every

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_cry_acc": [], "val_cry_acc": [],
            "train_age_acc": [], "val_age_acc": [],
            "train_cry_f1":  [], "val_cry_f1": [],
            "lr": [],
        }

        self._step_losses    = []
        self._last_spec      = None      # (B, 3, H, W)
        self._last_cr_probs  = None      # (B, n_cry_reasons)
        self._last_ag_probs  = None      # (B, n_age_groups)
        self._last_labels    = {}
        self._epoch_start    = None
        self._step           = 0
        self._total_steps    = 0
        self._current_epoch  = 0
        self._best_val_acc   = 0.0
        self._phase          = "train"

        print("✅ BabyColabVisualizer ready — dashboard will appear during training 🍼")

    # ── Public API ────────────────────────────────────────────────────────────

    def train_epoch_with_viz(self, model, loader, optimizer, criterion, scaler, device, cfg, epoch):
        """Drop-in replacement for train_epoch() with live dashboard."""
        from torch.cuda.amp import autocast
        import torch.nn as nn
        from sklearn.metrics import f1_score, accuracy_score

        model.train()
        self._phase         = "train"
        self._epoch_start   = time.time()
        self._step          = 0
        self._total_steps   = len(loader)
        self._current_epoch = epoch
        self._step_losses   = []

        losses_list = []
        cr_preds_all, cr_labels_all = [], []
        ag_preds_all, ag_labels_all = [], []

        for step, batch in enumerate(loader):
            specs = batch["spectrogram"].to(device, non_blocking=True)

            with autocast(enabled=cfg.use_amp):
                outputs = model(specs)
                # Move labels to device for loss
                batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in batch.items() if k != "filepath"}
                losses  = criterion(outputs, batch_dev)

            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_val = losses["total_loss"].item()
            self._step_losses.append(loss_val)
            self._step = step + 1

            if not batch.get("mixed", False):
                cr_preds = outputs["cry_reason_logits"].argmax(dim=1)
                ag_preds = outputs["age_group_logits"].argmax(dim=1)
                losses_list.append(loss_val)
                cr_preds_all.extend(cr_preds.cpu().numpy())
                cr_labels_all.extend(batch["cry_reason_label"].numpy())
                ag_preds_all.extend(ag_preds.cpu().numpy())
                ag_labels_all.extend(batch["age_group_label"].numpy())

                with torch.no_grad():
                    self._last_spec     = specs[:4].cpu()
                    self._last_cr_probs = torch.softmax(outputs["cry_reason_logits"][:4], dim=1).cpu().numpy()
                    self._last_ag_probs = torch.softmax(outputs["age_group_logits"][:4],  dim=1).cpu().numpy()
                    self._last_labels   = {
                        "cry": batch["cry_reason_label"][:4].numpy(),
                        "age": batch["age_group_label"][:4].numpy(),
                    }

            if (step + 1) % self.update_every == 0:
                self._render_dashboard()

        return self._compute_metrics(losses_list, cr_preds_all, cr_labels_all,
                                                   ag_preds_all, ag_labels_all)

    @torch.no_grad()
    def eval_epoch_with_viz(self, model, loader, criterion, device, epoch):
        """Drop-in replacement for eval_epoch() with dashboard."""
        from sklearn.metrics import f1_score, accuracy_score

        model.eval()
        self._phase       = "val"
        self._step        = 0
        self._total_steps = len(loader)

        losses_list = []
        cr_preds_all, cr_labels_all = [], []
        ag_preds_all, ag_labels_all = [], []

        for batch in loader:
            specs      = batch["spectrogram"].to(device, non_blocking=True)
            cr_labels  = batch["cry_reason_label"].to(device)
            ag_labels  = batch["age_group_label"].to(device)

            outputs = model(specs)
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items() if k != "filepath"}
            losses  = criterion(outputs, batch_dev)

            cr_preds = outputs["cry_reason_logits"].argmax(dim=1)
            ag_preds = outputs["age_group_logits"].argmax(dim=1)

            losses_list.append(losses["total_loss"].item())
            cr_preds_all.extend(cr_preds.cpu().numpy())
            cr_labels_all.extend(cr_labels.cpu().numpy())
            ag_preds_all.extend(ag_preds.cpu().numpy())
            ag_labels_all.extend(ag_labels.cpu().numpy())
            self._step += 1

        return self._compute_metrics(losses_list, cr_preds_all, cr_labels_all,
                                                   ag_preds_all, ag_labels_all)

    def end_epoch(self, epoch, train_metrics, val_metrics, lr):
        """Call after both train + val. Updates history and renders final epoch summary."""
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["val_loss"].append(val_metrics["loss"])
        self.history["train_cry_acc"].append(train_metrics["cry_reason_acc"])
        self.history["val_cry_acc"].append(val_metrics["cry_reason_acc"])
        self.history["train_age_acc"].append(train_metrics["age_group_acc"])
        self.history["val_age_acc"].append(val_metrics["age_group_acc"])
        self.history["train_cry_f1"].append(train_metrics["cry_reason_f1"])
        self.history["val_cry_f1"].append(val_metrics["cry_reason_f1"])
        self.history["lr"].append(lr)

        if val_metrics["mean_acc"] > self._best_val_acc:
            self._best_val_acc = val_metrics["mean_acc"]

        self._phase = "summary"
        self._render_dashboard(final=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_metrics(self, losses, cr_preds, cr_labels, ag_preds, ag_labels):
        from sklearn.metrics import f1_score, accuracy_score
        if not losses:
            return {"loss": 0, "cry_reason_acc": 0, "age_group_acc": 0,
                    "cry_reason_f1": 0, "age_group_f1": 0, "mean_acc": 0}
        cr_acc = float(accuracy_score(cr_labels, cr_preds))
        ag_acc = float(accuracy_score(ag_labels, ag_preds))
        return {
            "loss":            float(np.mean(losses)),
            "cry_reason_acc":  cr_acc,
            "age_group_acc":   ag_acc,
            "cry_reason_f1":   float(f1_score(cr_labels, cr_preds, average="macro", zero_division=0)),
            "age_group_f1":    float(f1_score(ag_labels, ag_preds, average="macro", zero_division=0)),
            "mean_acc":        (cr_acc + ag_acc) / 2,
        }

    def _fig_to_b64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _render_dashboard(self, final=False):
        clear_output(wait=True)

        fig = plt.figure(figsize=(18, 10), facecolor="#0a0010")
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

        # ── Row 0: metric cards ───────────────────────────────────────────
        elapsed   = time.time() - self._epoch_start if self._epoch_start else 0
        step_loss = float(np.mean(self._step_losses[-20:])) if self._step_losses else 0
        eta_secs  = (elapsed / max(self._step, 1)) * (self._total_steps - self._step)
        pct_e     = self._step / max(self._total_steps, 1)
        pct_t     = (self._current_epoch + pct_e) / max(self.total_epochs, 1)

        cards = [
            ("Epoch",          f"{self._current_epoch+1} / {self.total_epochs}"),
            ("Phase",          self._phase.upper()),
            ("Step loss",      f"{step_loss:.4f}"),
            ("Best val acc",   f"{self._best_val_acc:.1%}"),
            ("Elapsed",        f"{int(elapsed//60)}m {int(elapsed%60)}s"),
            ("ETA epoch",      f"{int(eta_secs//60)}m {int(eta_secs%60)}s"),
            ("Epoch %",        f"{pct_e:.0%}"),
            ("Overall %",      f"{pct_t:.0%}"),
        ]
        bgs = ["#1a0a2e", "#0a1a1a", "#1a0a2e", "#2a0a1a",
               "#1a0a2e", "#0a1a1a", "#1a0a2e", "#2a1a0a"]

        ax0 = fig.add_subplot(gs[0, :])
        ax0.set_facecolor("#0a0010")
        ax0.axis("off")
        for i, ((lbl, val), bg) in enumerate(zip(cards, bgs)):
            x = 0.03 + i * 0.122
            ax0.add_patch(plt.Rectangle((x, 0.05), 0.11, 0.9, facecolor=bg,
                                         edgecolor="#330044", linewidth=0.8,
                                         transform=ax0.transAxes, clip_on=False))
            ax0.text(x + 0.055, 0.68, lbl, transform=ax0.transAxes,
                     ha="center", fontsize=7.5, color="#aa88cc")
            ax0.text(x + 0.055, 0.28, val, transform=ax0.transAxes,
                     ha="center", fontsize=11, color="#ffe8ff", fontweight="bold")

        # ── Row 1, Col 0: Loss curve ──────────────────────────────────────
        ax_loss = fig.add_subplot(gs[1, 0])
        self._style_ax(ax_loss)
        if self.history["train_loss"]:
            ep = range(1, len(self.history["train_loss"]) + 1)
            ax_loss.plot(ep, self.history["train_loss"], color="#88aaff", lw=1.5, label="train")
            ax_loss.plot(ep, self.history["val_loss"],   color="#ff8866", lw=1.5, label="val", ls="--")
            ax_loss.legend(fontsize=7, labelcolor="white", facecolor="#100020", edgecolor="#440044")
        ax_loss.set_title("Loss", color="#ddbbff", fontsize=9)

        # ── Row 1, Col 1: Cry reason accuracy + F1 ───────────────────────
        ax_acc = fig.add_subplot(gs[1, 1])
        self._style_ax(ax_acc)
        if self.history["train_cry_acc"]:
            ep = range(1, len(self.history["train_cry_acc"]) + 1)
            ax_acc.plot(ep, self.history["train_cry_acc"], color="#cc88ff", lw=1.5, label="cry acc train")
            ax_acc.plot(ep, self.history["val_cry_acc"],   color="#cc88ff", lw=1.5, label="cry acc val", ls="--")
            ax_acc.plot(ep, self.history["train_age_acc"], color="#88ffcc", lw=1.5, label="age acc train")
            ax_acc.plot(ep, self.history["val_age_acc"],   color="#88ffcc", lw=1.5, label="age acc val", ls="--")
            ax_acc.set_ylim(0, 1)
            ax_acc.legend(fontsize=6, labelcolor="white", facecolor="#100020", edgecolor="#440044")
        ax_acc.set_title("Accuracy", color="#ddbbff", fontsize=9)

        # ── Row 1, Col 2: Step-loss this epoch ───────────────────────────
        ax_step = fig.add_subplot(gs[1, 2])
        self._style_ax(ax_step)
        if self._step_losses:
            ax_step.plot(self._step_losses, color="#aa66ff", lw=1, alpha=0.5)
            if len(self._step_losses) > 10:
                ma = np.convolve(self._step_losses, np.ones(10)/10, mode="valid")
                ax_step.plot(range(9, len(self._step_losses)), ma, color="#ffaa44", lw=1.5)
        ax_step.set_title(f"Step loss  (epoch {self._current_epoch+1})", color="#ddbbff", fontsize=9)
        ax_step.set_xlabel("step", color="#886699", fontsize=7)

        # ── Row 1, Col 3: LR schedule ─────────────────────────────────────
        ax_lr = fig.add_subplot(gs[1, 3])
        self._style_ax(ax_lr)
        if self.history["lr"]:
            ax_lr.plot(range(1, len(self.history["lr"]) + 1),
                       self.history["lr"], color="#ff88cc", lw=1.5)
        ax_lr.set_title("Learning rate", color="#ddbbff", fontsize=9)
        ax_lr.set_xlabel("epoch", color="#886699", fontsize=7)

        # ── Row 2, Col 0-1: Mel-spectrograms ─────────────────────────────
        if self._last_spec is not None:
            n_show = min(2, self._last_spec.shape[0])
            for i in range(n_show):
                ax_sp = fig.add_subplot(gs[2, i])
                ax_sp.set_facecolor("#0a0010")
                spec_np = self._last_spec[i, 0].numpy()
                ax_sp.imshow(spec_np, aspect="auto", origin="lower",
                              cmap=_BABYCMAP, vmin=0, vmax=1)
                ax_sp.axis("off")
                true_cr = self.cry_reasons[self._last_labels["cry"][i]] \
                          if self._last_labels.get("cry") is not None else "?"
                true_ag = self.age_groups[self._last_labels["age"][i]] \
                          if self._last_labels.get("age") is not None else "?"
                emoji   = _CRY_EMOJIS.get(true_cr, "❓")
                ax_sp.set_title(
                    f"Sample {i+1}   {emoji}  {true_cr.replace('_',' ')}\n"
                    f"age: {true_ag.replace('_',' ')}",
                    color="#ddbbff", fontsize=7.5,
                )

            # ── Col 2: Top-5 cry reason bars ─────────────────────────────
            ax_cr = fig.add_subplot(gs[2, 2])
            self._style_ax(ax_cr)
            if self._last_cr_probs is not None:
                probs     = self._last_cr_probs[0]
                top5_idx  = np.argsort(probs)[-5:][::-1]
                top5_val  = probs[top5_idx]
                top5_lbl  = [
                    f"{_CRY_EMOJIS.get(self.cry_reasons[j],'❓')} {self.cry_reasons[j].replace('_',' ')[:16]}"
                    for j in top5_idx
                ]
                bar_colors = ["#c850a0" if j == self._last_labels["cry"][0] else "#7744aa"
                              for j in top5_idx]
                ax_cr.barh(range(5), top5_val, color=bar_colors, height=0.6)
                ax_cr.set_yticks(range(5))
                ax_cr.set_yticklabels(top5_lbl, fontsize=7.5, color="#ffe8ff")
                ax_cr.set_xlim(0, 1)
                ax_cr.set_title("Top-5 cry reasons (sample 1)", color="#ddbbff", fontsize=8)
                ax_cr.set_xlabel("confidence", color="#886699", fontsize=7)

            # ── Col 3: Age group bars ─────────────────────────────────────
            ax_ag = fig.add_subplot(gs[2, 3])
            self._style_ax(ax_ag)
            if self._last_ag_probs is not None:
                ag_probs   = self._last_ag_probs[0]
                ag_colors  = ["#88ffcc" if j == self._last_labels["age"][0] else "#334466"
                               for j in range(len(self.age_groups))]
                ax_ag.barh(range(len(self.age_groups)), ag_probs, color=ag_colors, height=0.6)
                ax_ag.set_yticks(range(len(self.age_groups)))
                ax_ag.set_yticklabels([g.replace("_", " ") for g in self.age_groups],
                                       fontsize=7, color="#ffe8ff")
                ax_ag.set_xlim(0, 1)
                ax_ag.set_title("Age group predictions", color="#ddbbff", fontsize=8)
                ax_ag.set_xlabel("confidence", color="#886699", fontsize=7)
        else:
            ax_w = fig.add_subplot(gs[2, :])
            ax_w.set_facecolor("#0a0010")
            ax_w.axis("off")
            ax_w.text(0.5, 0.5, "⏳  Waiting for first batch of crying babies…",
                      ha="center", va="center", color="#663355", fontsize=12,
                      transform=ax_w.transAxes)

        fig.suptitle("🍼  Baby Voice Decoder — Live Training Dashboard",
                     color="#ffe8ff", fontsize=13, fontweight="bold", y=0.98)

        img_b64 = self._fig_to_b64(fig)
        plt.close(fig)

        html = f"""
        <div style="background:#0a0010;padding:8px;border-radius:8px;font-family:monospace">
          <img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:6px"/>
          <div style="color:#664466;font-size:11px;margin-top:4px;text-align:right">
            Updated: step {self._step}/{self._total_steps} &nbsp;|&nbsp; {time.strftime('%H:%M:%S')}
          </div>
        </div>
        """
        display(HTML(html))

    def _style_ax(self, ax):
        ax.set_facecolor("#0d001a")
        ax.tick_params(colors="#886699", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#330044")
