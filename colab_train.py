# ═══════════════════════════════════════════════════════════════════════════════
# 🍼 Baby Voice Decoder — Colab Training Cell
# Paste this as a new cell (or replace your existing train cell)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 0. Install dependencies (run once) ────────────────────────────────────────
# !pip install timm librosa scikit-learn kaggle -q

# ── 1. Download the Kaggle dataset ────────────────────────────────────────────
# Set up Kaggle credentials first (upload kaggle.json or use secrets)
# !kaggle datasets download -d mennaahmed23/decoding-cries-baby -p /content/
# !unzip -q /content/decoding-cries-baby.zip -d /content/decoding-cries-baby

# ── 2. Upload baby_voice_decoder.zip from this project, then: ─────────────────
import shutil, os
# shutil.copy('/content/baby_voice_decoder.zip', '/content/')
# !unzip -q /content/baby_voice_decoder.zip -d /content/
os.chdir('/content/baby_voice_decoder')

# ── 3. Build metadata CSV from Kaggle dataset ─────────────────────────────────
# !python data/prepare_dataset.py --root /content/decoding-cries-baby

# ── 4. Imports ────────────────────────────────────────────────────────────────
import torch
import csv
from pathlib import Path
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

import config as cfg_module
from src.model import build_model, BabyVoiceLoss
from src.dataset import build_dataloaders
from src.train import WarmupCosineScheduler, save_checkpoint
from colab_visualizer import BabyColabVisualizer

# ── 5. Build everything ───────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'🖥️  Device: {device}')

train_loader, val_loader, test_loader = build_dataloaders(
    metadata_csv = cfg_module.train_cfg.metadata_csv,
    audio_cfg    = cfg_module.audio_cfg,
    aug_cfg      = cfg_module.aug_cfg,
    train_cfg    = cfg_module.train_cfg,
    cry_reasons  = cfg_module.CRY_REASONS,
    age_groups   = cfg_module.AGE_GROUPS,
)

model = build_model(cfg_module.model_cfg).to(device)

optimizer = AdamW([
    {"params": model.get_backbone_params(), "lr": cfg_module.train_cfg.learning_rate * 0.1, "lr_scale": 0.1},
    {"params": model.get_head_params(),     "lr": cfg_module.train_cfg.learning_rate,       "lr_scale": 1.0},
], weight_decay=cfg_module.train_cfg.weight_decay)

scheduler = WarmupCosineScheduler(
    optimizer,
    cfg_module.train_cfg.warmup_epochs,
    cfg_module.train_cfg.epochs,
    cfg_module.train_cfg.learning_rate,
)
criterion = BabyVoiceLoss(
    cry_reason_weight = cfg_module.train_cfg.cry_reason_loss_weight,
    age_group_weight  = cfg_module.train_cfg.age_group_loss_weight,
    label_smoothing   = cfg_module.train_cfg.label_smoothing,
    num_cry_reasons   = len(cfg_module.CRY_REASONS),
    num_age_groups    = len(cfg_module.AGE_GROUPS),
)
scaler = GradScaler(enabled=cfg_module.train_cfg.use_amp and device == 'cuda')

Path(cfg_module.train_cfg.checkpoint_dir).mkdir(exist_ok=True)
Path(cfg_module.train_cfg.log_dir).mkdir(exist_ok=True)

# ── 6. Create the visualizer ──────────────────────────────────────────────────
vis = BabyColabVisualizer(
    cry_reasons  = cfg_module.CRY_REASONS,
    age_groups   = cfg_module.AGE_GROUPS,
    total_epochs = cfg_module.train_cfg.epochs,
    update_every = 15,
)

# ── 7. Training loop ──────────────────────────────────────────────────────────
log_path   = f'{cfg_module.train_cfg.log_dir}/training_log.csv'
log_fields = ['epoch', 'lr', 'train_loss', 'train_cry_acc', 'train_age_acc',
              'train_cry_f1', 'val_loss', 'val_cry_acc', 'val_age_acc', 'val_cry_f1']

with open(log_path, 'w', newline='') as f:
    csv.DictWriter(f, fieldnames=log_fields).writeheader()

best_val_acc     = 0.0
patience_counter = 0

for epoch in range(cfg_module.train_cfg.epochs):
    lr = scheduler.step(epoch)

    train_metrics = vis.train_epoch_with_viz(
        model, train_loader, optimizer, criterion, scaler, device,
        cfg_module.train_cfg, epoch,
    )
    val_metrics = vis.eval_epoch_with_viz(
        model, val_loader, criterion, device, epoch,
    )
    vis.end_epoch(epoch, train_metrics, val_metrics, lr)

    with open(log_path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=log_fields).writerow({
            'epoch':          epoch + 1,
            'lr':             lr,
            'train_loss':     train_metrics['loss'],
            'train_cry_acc':  train_metrics['cry_reason_acc'],
            'train_age_acc':  train_metrics['age_group_acc'],
            'train_cry_f1':   train_metrics['cry_reason_f1'],
            'val_loss':       val_metrics['loss'],
            'val_cry_acc':    val_metrics['cry_reason_acc'],
            'val_age_acc':    val_metrics['age_group_acc'],
            'val_cry_f1':     val_metrics['cry_reason_f1'],
        })

    val_acc = val_metrics['mean_acc']
    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        patience_counter = 0
        save_checkpoint(model, optimizer, epoch + 1, val_metrics,
                        cfg_module.train_cfg, 'best_model.pth')
        print(f'  ✓ New best  val_acc={val_acc:.4f}')
    else:
        patience_counter += 1
        if patience_counter >= cfg_module.train_cfg.early_stopping_patience:
            print(f'⏹  Early stopping at epoch {epoch + 1}')
            break

    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch + 1, val_metrics,
                        cfg_module.train_cfg, f'checkpoint_epoch_{epoch+1}.pth')

print('\n✅ Training complete! 🍼')

# ── 8. Quick inference test ───────────────────────────────────────────────────
# from src.inference import BabyVoiceDecoder
# decoder = BabyVoiceDecoder('checkpoints/best_model.pth')
# result  = decoder.decode('/path/to/baby_cry.wav')
# decoder.print_result(result)
