"""
Baby Voice Decoder - Model Architecture
Multi-task CNN: EfficientNet-B0 backbone with dual classification heads.

  Head 1: Cry/vocalization reason  (10 classes — hunger, pain, tiredness…)
  Head 2: Age group                 (6 classes — newborn through toddler)

Architecture:
  Input (3, 128, ~188)               ← 3-channel mel-spectrogram (3 s at 16 kHz)
  ↓ EfficientNet-B0 backbone (pretrained ImageNet)
  ↓ Global Average Pooling   → (1280,)
  ↓ Dropout + BatchNorm
  ├─ Cry Reason Head         → softmax over 10 classes
  └─ Age Group Head          → softmax over 6 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict


class ClassificationHead(nn.Module):
    """FC → BN → GELU → Dropout → FC"""

    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        hidden = in_dim // 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BabyVoiceModel(nn.Module):
    """
    Multi-task baby vocalization classifier.

    Args:
        backbone:           timm model name
        pretrained:         load ImageNet weights
        cry_reason_classes: number of cry/vocalization reason classes
        age_group_classes:  number of age group classes
        dropout:            dropout probability in heads
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        cry_reason_classes: int = 10,
        age_group_classes: int = 6,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.backbone_name = backbone

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        embedding_dim = self.backbone.num_features

        self.shared_proj = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout * 0.5),
        )

        self.cry_reason_head = ClassificationHead(embedding_dim, cry_reason_classes, dropout)
        self.age_group_head  = ClassificationHead(embedding_dim, age_group_classes, dropout)

        # L2-normalized embedding for similarity search / retrieval
        self.embedding_proj = nn.Linear(embedding_dim, 256, bias=False)

        self._init_heads()

    def _init_heads(self):
        for module in [self.cry_reason_head, self.age_group_head, self.embedding_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: mel-spectrogram (B, 3, n_mels, time_frames)
        Returns:
            dict with cry_reason_logits, age_group_logits, embedding
        """
        features = self.backbone(x)
        features = self.shared_proj(features)

        cry_logits = self.cry_reason_head(features)
        age_logits = self.age_group_head(features)

        embedding = self.embedding_proj(features)
        embedding = F.normalize(embedding, p=2, dim=1)

        return {
            "cry_reason_logits": cry_logits,
            "age_group_logits":  age_logits,
            "embedding":         embedding,
            "features":          features,
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns probabilities (not logits)."""
        with torch.no_grad():
            out = self.forward(x)
        return {
            "cry_reason_probs": F.softmax(out["cry_reason_logits"], dim=-1),
            "age_group_probs":  F.softmax(out["age_group_logits"],  dim=-1),
            "embedding":        out["embedding"],
        }

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_head_params(self):
        return (
            list(self.cry_reason_head.parameters()) +
            list(self.age_group_head.parameters()) +
            list(self.shared_proj.parameters()) +
            list(self.embedding_proj.parameters())
        )


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class BabyVoiceLoss(nn.Module):
    """Multi-task cross-entropy with optional Mixup and label smoothing."""

    def __init__(
        self,
        cry_reason_weight: float = 0.75,
        age_group_weight: float  = 0.25,
        label_smoothing: float   = 0.1,
        num_cry_reasons: int     = 10,
        num_age_groups: int      = 6,
    ):
        super().__init__()
        self.cry_reason_weight = cry_reason_weight
        self.age_group_weight  = age_group_weight
        self.num_cry_reasons   = num_cry_reasons
        self.num_age_groups    = num_age_groups

        self.ce_cry = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_age = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _mixup_loss(self, logits, la, lb, lam, ce_fn):
        return lam * ce_fn(logits, la) + (1 - lam) * ce_fn(logits, lb)

    def forward(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        mixed = batch.get("mixed", False)

        if mixed:
            cr_a, cr_b, lam    = batch["cry_reason_label"]
            ag_a, ag_b, lam_ag = batch["age_group_label"]
            cry_loss = self._mixup_loss(outputs["cry_reason_logits"], cr_a, cr_b, lam,    self.ce_cry)
            age_loss = self._mixup_loss(outputs["age_group_logits"],  ag_a, ag_b, lam_ag, self.ce_age)
        else:
            cry_loss = self.ce_cry(outputs["cry_reason_logits"], batch["cry_reason_label"])
            age_loss = self.ce_age(outputs["age_group_logits"],  batch["age_group_label"])

        total = self.cry_reason_weight * cry_loss + self.age_group_weight * age_loss

        return {
            "total_loss":    total,
            "cry_loss":      cry_loss,
            "age_group_loss": age_loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg) -> BabyVoiceModel:
    model = BabyVoiceModel(
        backbone           = cfg.backbone,
        pretrained         = cfg.pretrained,
        cry_reason_classes = cfg.cry_reason_classes,
        age_group_classes  = cfg.age_group_classes,
        dropout            = cfg.dropout,
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.backbone} | Trainable params: {n/1e6:.2f}M")
    return model


def load_model(checkpoint_path: str, device, cfg) -> BabyVoiceModel:
    model = BabyVoiceModel(
        backbone           = cfg.backbone,
        pretrained         = False,
        cry_reason_classes = cfg.cry_reason_classes,
        age_group_classes  = cfg.age_group_classes,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded from epoch {ckpt.get('epoch', '?')}  "
          f"(val_acc: {ckpt.get('val_acc', 0):.3f})")
    return model
