"""
Baby Voice Decoder - Inference
Run on a single audio file (or mic recording in Colab) and get a
human-readable "translation" of what your baby is trying to say.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    AudioConfig, InferenceConfig, CRY_REASONS, AGE_GROUPS,
    CRY_REASON_DESCRIPTIONS, CRY_REASON_EMOJIS, CRY_REASON_SUGGESTIONS,
    AGE_GROUP_DESCRIPTIONS, audio_cfg, infer_cfg, model_cfg,
)
from src.model import load_model
from src.dataset import load_audio, pad_or_trim, chunk_audio, waveform_to_melspec


def audio_to_tensor(waveform: np.ndarray, cfg: AudioConfig) -> torch.Tensor:
    """Single waveform → (1, 3, n_mels, time) tensor ready for model."""
    waveform = pad_or_trim(waveform, cfg)
    spec     = waveform_to_melspec(waveform, cfg)                 # (n_mels, time)
    spec_t   = torch.from_numpy(spec).unsqueeze(0).repeat(3, 1, 1)  # (3, n_mels, time)
    return spec_t.unsqueeze(0)                                    # (1, 3, n_mels, time)


class BabyVoiceDecoder:
    """
    High-level inference wrapper.

    Usage
    -----
    decoder = BabyVoiceDecoder("checkpoints/best_model.pth")
    result  = decoder.decode("baby_cry.wav")
    decoder.print_result(result)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device:          Optional[str] = None,
        confidence_threshold: float    = 0.25,
        top_k: int                     = 3,
    ):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device    = torch.device(device)
        self.threshold = confidence_threshold
        self.top_k     = top_k

        self.model = load_model(checkpoint_path, self.device, model_cfg)
        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────────

    def decode(self, audio_path: str) -> Dict:
        """
        Decode a baby audio file.
        Returns a dict with top-k cry reasons, confidence scores,
        descriptions, suggestions, and age group prediction.
        """
        waveform = load_audio(audio_path, audio_cfg)
        chunks   = chunk_audio(waveform, audio_cfg, overlap=infer_cfg.chunk_overlap)

        # Aggregate predictions over all chunks
        cr_probs_all = []
        ag_probs_all = []

        for chunk in chunks:
            tensor = audio_to_tensor(chunk, audio_cfg).to(self.device)
            with torch.no_grad():
                preds = self.model.predict(tensor)
            cr_probs_all.append(preds["cry_reason_probs"].cpu().numpy()[0])
            ag_probs_all.append(preds["age_group_probs"].cpu().numpy()[0])

        cr_probs = np.mean(cr_probs_all, axis=0)   # average over chunks
        ag_probs = np.mean(ag_probs_all, axis=0)

        # Top-k cry reasons
        top_k_idx = np.argsort(cr_probs)[::-1][: self.top_k]
        results   = []
        for idx in top_k_idx:
            reason = CRY_REASONS[idx]
            conf   = float(cr_probs[idx])
            if conf >= self.threshold:
                results.append({
                    "reason":      reason,
                    "confidence":  conf,
                    "emoji":       CRY_REASON_EMOJIS[reason],
                    "description": CRY_REASON_DESCRIPTIONS[reason],
                    "suggestion":  CRY_REASON_SUGGESTIONS[reason],
                })

        # Age group
        best_age_idx = int(np.argmax(ag_probs))
        age_group    = AGE_GROUPS[best_age_idx]

        return {
            "file":           audio_path,
            "top_reasons":    results,
            "all_cry_probs":  {CRY_REASONS[i]: float(cr_probs[i]) for i in range(len(CRY_REASONS))},
            "age_group":      age_group,
            "age_confidence": float(ag_probs[best_age_idx]),
            "age_description": AGE_GROUP_DESCRIPTIONS[age_group],
            "num_chunks":     len(chunks),
        }

    def print_result(self, result: Dict):
        """Pretty-print a decode() result to console."""
        print("\n" + "═" * 60)
        print(f"  🍼 Baby Voice Decoder — {Path(result['file']).name}")
        print("═" * 60)
        print(f"\n  👶 Estimated age: {result['age_group'].replace('_', ' ')}  "
              f"({result['age_confidence']*100:.0f}% confidence)")
        print(f"     {result['age_description']}\n")

        if not result["top_reasons"]:
            print("  ⚠️  No cry reason detected above confidence threshold.\n")
            return

        print("  🔍 Most likely reasons:\n")
        for i, r in enumerate(result["top_reasons"]):
            bar = "█" * int(r["confidence"] * 20) + "░" * (20 - int(r["confidence"] * 20))
            print(f"  {i+1}. {r['emoji']}  {r['reason'].replace('_', ' ').title()}")
            print(f"      [{bar}] {r['confidence']*100:.1f}%")
            print(f"      {r['description']}")
            print(f"      💡 {r['suggestion']}\n")

        print("═" * 60 + "\n")
