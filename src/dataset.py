"""
Baby Voice Decoder - Dataset & Audio Preprocessing

Handles loading audio, converting to mel-spectrograms, augmentations,
and building PyTorch DataLoaders for training and inference.

Expected CSV columns:
    filepath, cry_reason, age_group, [duration, source, ...]

The Kaggle dataset (mennaahmed23/decoding-cries-baby) ships with folders
named by label. Use data/prepare_dataset.py to build metadata.csv from it.
"""

import os
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
import librosa.effects

warnings.filterwarnings("ignore")

from config import AudioConfig, AugmentationConfig, TrainingConfig, CRY_REASONS, AGE_GROUPS


# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, cfg: AudioConfig) -> np.ndarray:
    """Load audio and resample to 16 kHz mono float32."""
    waveform, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    return waveform.astype(np.float32)


def pad_or_trim(waveform: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Pad (tile) or random-crop waveform to exactly cfg.duration seconds."""
    target = int(cfg.sample_rate * cfg.duration)
    if len(waveform) >= target:
        start = random.randint(0, len(waveform) - target)
        return waveform[start: start + target]
    repeats = (target // len(waveform)) + 1
    return np.tile(waveform, repeats)[:target]


def waveform_to_melspec(waveform: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """
    Raw waveform → log-mel spectrogram normalized to [0, 1].
    Output shape: (n_mels, time_frames)
    """
    spec = librosa.feature.melspectrogram(
        y          = waveform,
        sr         = cfg.sample_rate,
        n_fft      = cfg.n_fft,
        hop_length = cfg.hop_length,
        n_mels     = cfg.n_mels,
        fmin       = cfg.f_min,
        fmax       = cfg.f_max,
        power      = cfg.power,
    )
    log_spec = librosa.power_to_db(spec, ref=np.max, top_db=cfg.top_db)
    log_spec = (log_spec + cfg.top_db) / cfg.top_db  # → [0, 1]
    return log_spec.astype(np.float32)


def chunk_audio(waveform: np.ndarray, cfg: AudioConfig, overlap: float = 0.5) -> List[np.ndarray]:
    """
    Split long audio into overlapping chunks for inference.
    Returns list of fixed-length waveform chunks.
    """
    chunk_len = int(cfg.sample_rate * cfg.duration)
    step      = int(chunk_len * (1 - overlap))
    chunks    = []
    for start in range(0, len(waveform) - chunk_len + 1, step):
        chunks.append(waveform[start: start + chunk_len])
    if not chunks:
        chunks.append(pad_or_trim(waveform, cfg))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────────────────

class WaveformAugmenter:
    """Time-domain augmentations (pitch, gain, noise, time-shift)."""

    def __init__(self, cfg: AugmentationConfig, audio_cfg: AudioConfig):
        self.cfg       = cfg
        self.audio_cfg = audio_cfg

    def time_shift(self, w: np.ndarray) -> np.ndarray:
        shift = int(np.random.uniform(-self.cfg.time_shift_limit,
                                       self.cfg.time_shift_limit) * len(w))
        return np.roll(w, shift)

    def gain(self, w: np.ndarray) -> np.ndarray:
        db = np.random.uniform(*self.cfg.gain_range)
        return w * (10 ** (db / 20))

    def add_noise(self, w: np.ndarray) -> np.ndarray:
        snr_db  = np.random.uniform(*self.cfg.noise_snr_range)
        sig_pwr = np.mean(w ** 2) + 1e-10
        ns_pwr  = sig_pwr / (10 ** (snr_db / 10))
        return (w + np.random.randn(len(w)) * np.sqrt(ns_pwr)).astype(np.float32)

    def pitch_shift(self, w: np.ndarray) -> np.ndarray:
        """Slight random pitch shift — important because cry pitch is diagnostic."""
        n_steps = np.random.uniform(-1.5, 1.5)
        return librosa.effects.pitch_shift(w, sr=self.audio_cfg.sample_rate, n_steps=n_steps)

    def __call__(self, w: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return w
        if np.random.random() < self.cfg.time_shift_prob:
            w = self.time_shift(w)
        if np.random.random() < self.cfg.gain_prob:
            w = self.gain(w)
        if np.random.random() < self.cfg.noise_prob:
            w = self.add_noise(w)
        if np.random.random() < 0.3:           # pitch shift ~30% of the time
            w = self.pitch_shift(w)
        return np.clip(w, -1.0, 1.0)


class SpecAugment:
    """SpecAugment on mel-spectrogram tensor (1, n_mels, time)."""

    def __init__(self, cfg: AugmentationConfig):
        self.cfg = cfg

    def freq_mask(self, s: torch.Tensor) -> torch.Tensor:
        n   = s.shape[-2]
        sz  = random.randint(1, self.cfg.freq_mask_size)
        st  = random.randint(0, n - sz)
        s[..., st: st + sz, :] = 0.0
        return s

    def time_mask(self, s: torch.Tensor) -> torch.Tensor:
        n   = s.shape[-1]
        sz  = random.randint(1, min(self.cfg.time_mask_size, n - 1))
        st  = random.randint(0, n - sz)
        s[..., st: st + sz] = 0.0
        return s

    def __call__(self, s: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return s
        if random.random() < self.cfg.freq_mask_prob:
            s = self.freq_mask(s)
        if random.random() < self.cfg.time_mask_prob:
            s = self.time_mask(s)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BabyCryDataset(Dataset):
    """
    PyTorch Dataset for baby cry classification.

    CSV must have: filepath, cry_reason, age_group
    Optional:      duration, source, notes

    Outputs per sample:
        spectrogram:       (3, n_mels, time_frames)
        cry_reason_label:  int
        age_group_label:   int
        filepath:          str
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        audio_cfg:   AudioConfig,
        aug_cfg:     AugmentationConfig,
        cry_reasons: List[str],
        age_groups:  List[str],
        training:    bool = True,
    ):
        self.df           = df.reset_index(drop=True)
        self.audio_cfg    = audio_cfg
        self.aug_cfg      = aug_cfg
        self.training     = training
        self.reason2idx   = {r: i for i, r in enumerate(cry_reasons)}
        self.age2idx      = {a: i for i, a in enumerate(age_groups)}
        self.wave_aug     = WaveformAugmenter(aug_cfg, audio_cfg)
        self.spec_aug     = SpecAugment(aug_cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row      = self.df.iloc[idx]
        filepath = row["filepath"]

        try:
            waveform = load_audio(filepath, self.audio_cfg)
        except Exception:
            waveform = np.zeros(
                int(self.audio_cfg.sample_rate * self.audio_cfg.duration),
                dtype=np.float32,
            )

        waveform = pad_or_trim(waveform, self.audio_cfg)
        waveform = self.wave_aug(waveform, training=self.training)

        spec = waveform_to_melspec(waveform, self.audio_cfg)   # (n_mels, time)
        spec = torch.from_numpy(spec).unsqueeze(0)              # (1, n_mels, time)
        spec = self.spec_aug(spec, training=self.training)
        spec = spec.repeat(3, 1, 1)                             # (3, n_mels, time)

        return {
            "spectrogram":      spec,
            "cry_reason_label": torch.tensor(self.reason2idx.get(row["cry_reason"], 0), dtype=torch.long),
            "age_group_label":  torch.tensor(self.age2idx.get(row.get("age_group", AGE_GROUPS[0]), 0), dtype=torch.long),
            "filepath":         filepath,
        }

    def mixup_collate(self, batch):
        specs      = torch.stack([b["spectrogram"] for b in batch])
        cr_labels  = torch.stack([b["cry_reason_label"] for b in batch])
        ag_labels  = torch.stack([b["age_group_label"]  for b in batch])

        if self.training and random.random() < self.aug_cfg.mixup_prob:
            lam   = np.random.beta(self.aug_cfg.mixup_alpha, self.aug_cfg.mixup_alpha)
            idx   = torch.randperm(len(specs))
            specs = lam * specs + (1 - lam) * specs[idx]
            return {
                "spectrogram":      specs,
                "cry_reason_label": (cr_labels, cr_labels[idx], lam),
                "age_group_label":  (ag_labels,  ag_labels[idx], lam),
                "filepath":         [b["filepath"] for b in batch],
                "mixed":            True,
            }

        return {
            "spectrogram":      specs,
            "cry_reason_label": cr_labels,
            "age_group_label":  ag_labels,
            "filepath":         [b["filepath"] for b in batch],
            "mixed":            False,
        }


def build_dataloaders(
    metadata_csv: str,
    audio_cfg:    AudioConfig,
    aug_cfg:      AugmentationConfig,
    train_cfg:    TrainingConfig,
    cry_reasons:  List[str],
    age_groups:   List[str],
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    df = pd.read_csv(metadata_csv)
    df = df[df["cry_reason"].isin(cry_reasons)]
    if "age_group" not in df.columns:
        df["age_group"] = age_groups[0]   # default if not annotated
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n      = len(df)
    n_test = int(n * train_cfg.test_split)
    n_val  = int(n * train_cfg.val_split)

    test_df  = df.iloc[:n_test]
    val_df   = df.iloc[n_test: n_test + n_val]
    train_df = df.iloc[n_test + n_val:]

    train_ds = BabyCryDataset(train_df, audio_cfg, aug_cfg, cry_reasons, age_groups, training=True)
    val_ds   = BabyCryDataset(val_df,   audio_cfg, aug_cfg, cry_reasons, age_groups, training=False)
    test_ds  = BabyCryDataset(test_df,  audio_cfg, aug_cfg, cry_reasons, age_groups, training=False)

    # Balanced sampler for imbalanced cry-reason distribution
    reason2idx   = {r: i for i, r in enumerate(cry_reasons)}
    labels       = train_df["cry_reason"].map(reason2idx).values
    class_counts = np.bincount(labels, minlength=len(cry_reasons))
    class_wts    = 1.0 / (class_counts + 1e-6)
    sample_wts   = class_wts[labels]
    sampler      = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_wts).float(),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, sampler=sampler,
        num_workers=train_cfg.num_workers, pin_memory=True,
        collate_fn=train_ds.mixup_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )

    print(f"Dataset → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
