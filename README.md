```
██████╗  █████╗ ██████╗ ██╗   ██╗    ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗
██╔══██╗██╔══██╗██╔══██╗╚██╗ ██╔╝    ██║   ██║██╔═══██╗██║██╔════╝██╔════╝
██████╔╝███████║██████╔╝ ╚████╔╝     ██║   ██║██║   ██║██║██║     █████╗  
██╔══██╗██╔══██║██╔══██╗  ╚██╔╝      ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  
██████╔╝██║  ██║██████╔╝   ██║        ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗
╚═════╝ ╚═╝  ╚═╝╚═════╝    ╚═╝         ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝
                                                                             
██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ███████╗██████╗                   
██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗                  
██║  ██║█████╗  ██║     ██║   ██║██║  ██║█████╗  ██████╔╝                  
██║  ██║██╔══╝  ██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗                  
██████╔╝███████╗╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║                  
╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝                  
```

<div align="center">

🍼 **A deep learning system that decodes what your baby is trying to say**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/mennaahmed23/decoding-cries-baby)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Trained on infant and toddler cry audio — powered by EfficientNet-B0 on mel-spectrograms*

</div>

---

## 🧠 What It Does

Baby Voice Decoder listens to your baby's cry and classifies **why** they're crying and estimates their **age group** — all in real time, from raw audio.

---

## 🔍 What It Classifies

### Cry Reasons — 10 Classes

| Label | Meaning |
|---|---|
| 🍼 `hunger` | Rhythmic, building cry; sucking pauses |
| 😢 `pain` | Sudden high-pitched shriek with long pause |
| 😣 `discomfort` | Continuous whiny cry — wet diaper, clothing, temperature |
| 😴 `tiredness` | Whiny alternating cry, rubbing eyes |
| 😨 `fear` | Sudden sharp cry from noise or startling |
| 🤗 `needs_attention` | Short bursts; stops when picked up |
| 🌀 `overstimulation` | Escalating cry after prolonged stimulation |
| 😖 `gas_colic` | Sharp intense episodes; legs pulling up |
| 😐 `boredom` | Low-grade fussing; stops with new stimulation |
| 😊 `happy_babbling` | Non-distress: cooing, laughter, social sounds |

### Age Group — 6 Classes
`newborn_0_1m` · `infant_1_3m` · `infant_3_6m` · `infant_6_12m` · `toddler_12_24m` · `toddler_24_36m`

---

## 🏗️ Architecture

```
Input audio (any length)
    ↓ load + resample to 16 kHz mono
    ↓ chunk into 3-second windows (50% overlap)
    ↓ log-mel spectrogram (128 mels, 10 ms hop)
    ↓ normalize → (3, 128, 188) tensor
    ↓ EfficientNet-B0 backbone (ImageNet pretrained)
    ↓ Global Average Pooling → (1280,)
    ↓ BatchNorm + Dropout
    ├── Cry Reason Head  → 10 classes  (weight 0.75)
    └── Age Group Head   → 6 classes   (weight 0.25)
```

**Key design choices:**
- `sample_rate` **16 kHz** — standard for speech; infant voices don't exceed 8 kHz usefully
- `duration` **3s** — cry bouts are shorter than bird songs
- `f_max` **8 kHz** — covers all cry harmonics and formants
- **Pitch shift augmentation** (~30% prob) — cry pitch is highly diagnostic
- **Mixup probability 0.2** — blending different cry types is semantically meaningful in small doses
- **Loss weights:** 0.75 cry reason + 0.25 age group

---

## 🚀 Quick Start (Google Colab)

```python
# 1. Install dependencies
!pip install timm librosa scikit-learn -q

# 2. Download Kaggle dataset
!kaggle datasets download -d mennaahmed23/decoding-cries-baby -p /content/
!unzip -q /content/decoding-cries-baby.zip -d /content/decoding-cries-baby

# 3. Upload baby_voice_decoder.zip → unzip to /content/baby_voice_decoder/

# 4. Build metadata
!python data/prepare_dataset.py --root /content/decoding-cries-baby

# 5. Run the training cell from colab_train.py
```

---

## 🎯 Inference

```python
from src.inference import BabyVoiceDecoder

decoder = BabyVoiceDecoder("checkpoints/best_model.pth")
result  = decoder.decode("baby_audio.wav")
decoder.print_result(result)
```

**Example output:**
```
════════════════════════════════════════════════════════════
  🍼 Baby Voice Decoder — baby_audio.wav
════════════════════════════════════════════════════════════

  👶 Estimated age: infant 3–6m  (78% confidence)
     3–6 months: distinct hungry vs. pain cry; cooing begins

  🔍 Most likely reasons:

  1. 🍼  Hunger
      [████████████████░░░░] 82.3%
      Rhythmic, low-pitched cry building in intensity
      💡 Try feeding — offer breast/bottle.

  2. 😴  Tiredness
      [████░░░░░░░░░░░░░░░░] 11.1%
      Whiny alternating cry, rubbing eyes
      💡 Create a calm sleep environment; reduce stimulation.
```

---

## 📂 Project Structure

```
baby_voice_decoder/
├── config.py                ← all hyperparameters, labels, descriptions
├── colab_train.py           ← paste-into-Colab training cell
├── colab_visualizer.py      ← live training dashboard (mel-specs + predictions)
├── requirements.txt
├── data/
│   └── prepare_dataset.py   ← convert Kaggle folder structure → metadata.csv
└── src/
    ├── model.py             ← BabyVoiceModel + BabyVoiceLoss
    ├── dataset.py           ← BabyCryDataset + audio utils + augmentations
    ├── train.py             ← WarmupCosineScheduler + train/eval loops
    └── inference.py         ← BabyVoiceDecoder with pretty-print output
```

---

## 📦 Dataset

[Kaggle: Decoding Cries Baby](https://www.kaggle.com/datasets/mennaahmed23/decoding-cries-baby)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/sciencebanda09">sciencebanda09</a>
</div>
