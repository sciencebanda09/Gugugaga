"""
Baby Voice Decoder - Dataset Preparation
Converts the Kaggle 'decoding-cries-baby' folder structure into metadata.csv
that the training pipeline expects.

Dataset URL: https://www.kaggle.com/datasets/mennaahmed23/decoding-cries-baby

Expected Kaggle folder layout after unzip:
    decoding-cries-baby/
        belly_pain/     *.wav / *.mp3
        burping/
        discomfort/
        hungry/
        tired/
        ...

Run in Colab after downloading:
    !python data/prepare_dataset.py --root /content/decoding-cries-baby
"""

import argparse
import os
import csv
from pathlib import Path

# Map Kaggle folder names → our CRY_REASONS labels
FOLDER_TO_LABEL = {
    # Common Kaggle folder names → our labels
    "belly_pain":       "gas_colic",
    "burping":          "gas_colic",
    "colic":            "gas_colic",
    "discomfort":       "discomfort",
    "hungry":           "hunger",
    "hunger":           "hunger",
    "tired":            "tiredness",
    "sleepy":           "tiredness",
    "pain":             "pain",
    "fear":             "fear",
    "scared":           "fear",
    "needs_attention":  "needs_attention",
    "attention":        "needs_attention",
    "boredom":          "boredom",
    "bored":            "boredom",
    "overstimulation":  "overstimulation",
    "stimulated":       "overstimulation",
    "happy":            "happy_babbling",
    "babbling":         "happy_babbling",
    "laugh":            "happy_babbling",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def prepare(root: str, output_csv: str = "data/metadata.csv"):
    root     = Path(root)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows     = []
    skipped  = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        label = FOLDER_TO_LABEL.get(folder.name.lower().strip())
        if label is None:
            # Try substring matching
            for key, val in FOLDER_TO_LABEL.items():
                if key in folder.name.lower():
                    label = val
                    break

        for audio_file in sorted(folder.rglob("*")):
            if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            if label:
                rows.append({
                    "filepath":   str(audio_file),
                    "cry_reason": label,
                    "age_group":  "infant_3_6m",  # default; update if dataset has age info
                    "source":     "kaggle_decoding_cries",
                })
            else:
                skipped.append(str(audio_file))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "cry_reason", "age_group", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote {len(rows)} rows → {out_path}")
    print(f"   Label distribution:")
    from collections import Counter
    counts = Counter(r["cry_reason"] for r in rows)
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // max(1, max(counts.values()) // 20))
        print(f"   {label:<22} {count:>5}  {bar}")
    if skipped:
        print(f"\n   ⚠️  Skipped {len(skipped)} files (unmapped folders):")
        for s in skipped[:5]:
            print(f"      {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",   required=True, help="Path to extracted Kaggle dataset folder")
    parser.add_argument("--output", default="data/metadata.csv")
    args = parser.parse_args()
    prepare(args.root, args.output)
