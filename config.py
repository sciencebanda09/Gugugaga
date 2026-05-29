"""
Baby Voice Decoder - Central Configuration
Adapted from Bird Language ML for infant/toddler cry & speech classification.

Dataset: https://www.kaggle.com/datasets/mennaahmed23/decoding-cries-baby
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioConfig:
    sample_rate: int   = 16000        # standard for speech models
    duration: float    = 3.0          # seconds per chunk (cries are shorter than bird calls)
    n_fft: int         = 512
    hop_length: int    = 160          # 10 ms hop at 16 kHz
    n_mels: int        = 128
    f_min: float       = 80.0         # Hz — below fundamental frequency of infant cry
    f_max: float       = 8000.0       # Hz — covers formants and harmonics of cries
    power: float       = 2.0
    top_db: float      = 80.0


@dataclass
class AugmentationConfig:
    # Time-domain augmentations
    time_shift_prob: float   = 0.4
    time_shift_limit: float  = 0.1
    gain_prob: float         = 0.5
    gain_range: tuple        = (-6, 6)       # dB
    noise_prob: float        = 0.3           # lower — noise can mask cry nuances
    noise_snr_range: tuple   = (15, 35)      # higher SNR to preserve cry features

    # Spectrogram augmentations (SpecAugment)
    freq_mask_prob: float    = 0.4
    freq_mask_size: int      = 15
    time_mask_prob: float    = 0.4
    time_mask_size: int      = 20
    mixup_prob: float        = 0.2           # lower mixup — blending cry types is semantically odd
    mixup_alpha: float       = 0.3


@dataclass
class ModelConfig:
    backbone: str        = "efficientnet_b0"
    pretrained: bool     = True
    dropout: float       = 0.35

    # Multi-task heads
    cry_reason_classes: int = len([])        # filled below from CRY_REASONS
    age_group_classes: int  = len([])        # filled below from AGE_GROUPS


# ── What the baby might be communicating ─────────────────────────────────────
CRY_REASONS = [
    "hunger",
    "pain",
    "discomfort",
    "tiredness",
    "fear",
    "needs_attention",
    "overstimulation",
    "gas_colic",
    "boredom",
    "happy_babbling",
]

CRY_REASON_DESCRIPTIONS = {
    "hunger":           "Rhythmic, low-pitched cry building in intensity; may have sucking pauses",
    "pain":             "Sudden high-pitched shriek followed by long pause then repetition",
    "discomfort":       "Continuous whiny cry; may signal wet diaper, tight clothing, or temperature",
    "tiredness":        "Whiny, alternating cry, rubbing eyes; often accompanied by yawning cues",
    "fear":             "Sudden sharp cry triggered by loud noise, sudden movement, or stranger",
    "needs_attention":  "Short bursts of calling; stops when picked up, resumes if put down",
    "overstimulation":  "Fussy, escalating cry after prolonged activity or sensory input",
    "gas_colic":        "Sharp, intense crying episodes; legs pulling up toward abdomen",
    "boredom":          "Low-grade fussing that stops when new stimulation or toy is offered",
    "happy_babbling":   "Non-distress vocalization; cooing, babbling, laughter, social sounds",
}

CRY_REASON_EMOJIS = {
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

CRY_REASON_SUGGESTIONS = {
    "hunger":           "Try feeding — offer breast/bottle. Hunger cries build rhythmically.",
    "pain":             "Check for obvious causes (hair tourniquet, fever, injury). If persistent, consult a doctor.",
    "discomfort":       "Check diaper, clothing tightness, room temperature, and position.",
    "tiredness":        "Create a calm sleep environment; reduce stimulation and try rocking.",
    "fear":             "Soothe with gentle touch and a calm voice. Stay close for reassurance.",
    "needs_attention":  "Respond with eye contact, talking, and holding. Skin-to-skin contact helps.",
    "overstimulation":  "Move to a quiet, dim room. Reduce noise and visual clutter.",
    "gas_colic":        "Try bicycle legs, tummy time after feeds, or gentle belly massage.",
    "boredom":          "Introduce a new toy, change position, or try a short walk.",
    "happy_babbling":   "Enjoy it! Talk back, smile, and engage — this is language development.",
}

# ── Age groups (secondary classification) ────────────────────────────────────
AGE_GROUPS = [
    "newborn_0_1m",
    "infant_1_3m",
    "infant_3_6m",
    "infant_6_12m",
    "toddler_12_24m",
    "toddler_24_36m",
]

AGE_GROUP_DESCRIPTIONS = {
    "newborn_0_1m":    "0–1 month: mostly reflexive crying; very high fundamental frequency (~400–600 Hz)",
    "infant_1_3m":     "1–3 months: cry patterns beginning to differentiate; social smile emerges",
    "infant_3_6m":     "3–6 months: distinct hungry vs. pain cry; cooing begins",
    "infant_6_12m":    "6–12 months: babbling begins; cry has more intentional quality",
    "toddler_12_24m":  "12–24 months: first words; frustration cries become more complex",
    "toddler_24_36m":  "24–36 months: sentences emerging; emotional vocabulary expanding",
}


@dataclass
class TrainingConfig:
    # Data paths (update to your Kaggle dataset path)
    data_dir: str     = "data/audio"
    metadata_csv: str = "data/metadata.csv"
    val_split: float  = 0.15
    test_split: float = 0.10
    num_workers: int  = 2            # lower for Colab

    # Training
    epochs: int               = 40
    batch_size: int           = 32
    learning_rate: float      = 8e-4
    weight_decay: float       = 1e-4
    warmup_epochs: int        = 4
    label_smoothing: float    = 0.1

    # Mixed precision
    use_amp: bool             = True

    # Regularization
    gradient_clip: float             = 1.0
    early_stopping_patience: int     = 8

    # Checkpointing
    checkpoint_dir: str       = "checkpoints"
    save_best_only: bool      = True
    log_dir: str              = "logs"

    # Multi-task loss weights
    # Cry reason is the main task; age group is auxiliary
    cry_reason_loss_weight: float = 0.75
    age_group_loss_weight: float  = 0.25


@dataclass
class InferenceConfig:
    model_path: str             = "checkpoints/best_model.pth"
    confidence_threshold: float = 0.25
    top_k: int                  = 3
    chunk_overlap: float        = 0.5
    device: str                 = "auto"


# ── Singleton configs ─────────────────────────────────────────────────────────
audio_cfg  = AudioConfig()
aug_cfg    = AugmentationConfig()
train_cfg  = TrainingConfig()
infer_cfg  = InferenceConfig()
model_cfg  = ModelConfig(
    cry_reason_classes = len(CRY_REASONS),
    age_group_classes  = len(AGE_GROUPS),
)
