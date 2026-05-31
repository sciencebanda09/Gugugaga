<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff6b9d,50:ffb347,100:a855f7&height=200&section=header&text=gugugaga&fontSize=80&fontColor=ffffff&fontAlignY=45&animation=fadeIn&stroke=ffffff&strokeWidth=1&desc=...i+said+what+i+said&descAlignY=70&descSize=22&descColor=ffffffaa&fontStyle=bold" width="100%"/>

<img src="https://readme-typing-svg.herokuapp.com?font=Comic+Sans+MS&size=16&duration=2500&pause=600&color=FF6B9D&center=true&vCenter=true&width=700&lines=my+baby+is+crying+AGAIN+😭;ok+but+WHY+though;neural+network+go+brrr;turns+out+it+was+hunger.+again.;it+is+ALWAYS+hunger." alt="Tagline" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mennaahmed23/decoding-cries-baby)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

![Classes](https://img.shields.io/badge/CRY_REASONS-10-ff6b9d?style=flat-square&labelColor=1a0a0f)
![Age Groups](https://img.shields.io/badge/AGE_GROUPS-6-a855f7?style=flat-square&labelColor=1a0a0f)
![Sample Rate](https://img.shields.io/badge/SAMPLE_RATE-16kHz-ffb347?style=flat-square&labelColor=1a0a0f)
![Sleep Lost](https://img.shields.io/badge/SLEEP_LOST-countless_hours-ff4466?style=flat-square&labelColor=1a0a0f)

</div>

---

## OK BUT WHAT IS THIS

Your baby is crying. You have no idea why. You have not slept in 72 hours. You have tried everything.

This model has tried everything too — except it did it on thousands of cry recordings and actually learned something.

**Baby Voice Decoder** listens to your baby's cry and classifies **why** they're crying and estimates their **age group** in real time, from raw audio. Powered by EfficientNet-B0 on log-mel spectrograms.

> It is literally a neural network that learned to understand babies before you did.
> Take that however you want.

---

## THE CRYING DICTIONARY

### Why Are They Crying — 10 Possibilities

*(spoiler: it's probably hunger)*

| Label | Translation | Vibe |
|---|---|---|
| `hunger` | FEED ME NOW | the classic. the original. never gets old (for them) |
| `pain` | OW OW OW OW | you will know. everyone will know. |
| `discomfort` | something is wrong but I won't tell you what | detective mode activated |
| `tiredness` | I am tired but I REFUSE to sleep | relatable honestly |
| `fear` | that noise scared me and now it's your problem | very fair |
| `needs_attention` | I saw you sit down. unacceptable. | they have cameras somehow |
| `overstimulation` | too much going on | same |
| `gas_colic` | my tummy hurts and I am going to be dramatic about it | valid |
| `boredom` | entertain me peasant | the audacity |
| `happy_babbling` | actually fine, just talking | plot twist |

### How Old Are They — 6 Age Groups

`newborn_0_1m` · `infant_1_3m` · `infant_3_6m` · `infant_6_12m` · `toddler_12_24m` · `toddler_24_36m`

---

## HOW THE MAGIC WORKS

```
your baby, crying at 3am
        |
        v  resample to 16 kHz mono  (the baby stays loud)
        |
        v  slice into 3-second chunks  (50% overlap, very thorough)
        |
        v  convert to log-mel spectrogram  (crying but make it art)
        |
        v  normalize  ->  (3, 128, 188) tensor  (now it's math)
        |
        v  EfficientNet-B0  (pretrained on ImageNet, repurposed for chaos)
        |
        v  Global Average Pooling  ->  (1280,)  (big number, small answer)
        |
        v  BatchNorm + Dropout  (the model is also tired)
        |
        +---> Cry Reason Head   ->  10 classes   (WHY)
        |
        +---> Age Group Head    ->   6 classes   (WHO, approximately)
```

**Design choices, explained simply:**

| Choice | Value | Why |
|---|---|---|
| Sample rate | 16 kHz | covers all the frequencies a baby can produce |
| Window | 3 seconds | cry bouts are short. unlike the nights. |
| `f_max` | 8 kHz | covers all harmonics and formants |
| Pitch shift augmentation | 30% probability | cry pitch is extremely diagnostic |
| Mixup probability | 0.2 | a little chaos in training = more robust model |
| Loss weights | 0.75 cry + 0.25 age | cry reason is the whole point |

---

## QUICK START — GOOGLE COLAB

*(because who has a GPU right now. not you. not me.)*

```python
# step 1: get the data
!pip install timm librosa scikit-learn -q
!kaggle datasets download -d mennaahmed23/decoding-cries-baby -p /content/
!unzip -q /content/decoding-cries-baby.zip -d /content/decoding-cries-baby

# step 2: upload baby_voice_decoder.zip and unzip it
# step 3: build metadata
!python data/prepare_dataset.py --root /content/decoding-cries-baby

# step 4: train (go make coffee. or sleep. you need it.)
# run the training cell from colab_train.py
```

---

## INFERENCE

```python
from src.inference import BabyVoiceDecoder

decoder = BabyVoiceDecoder("checkpoints/best_model.pth")
result  = decoder.decode("baby_audio.wav")
decoder.print_result(result)
```

**What it actually outputs:**

```
================================================================
  Baby Voice Decoder — baby_audio.wav
  (running at 3:47 AM, no judgment)
================================================================

  Estimated age:  infant 3-6m   (78% confidence)
  note: at this age they cry loudly and on purpose

  Most likely reasons:

  1.  Hunger                                         82.3%
      [################....]
      Rhythmic, low-pitched cry building in intensity
      -> Try feeding. It is always feeding.

  2.  Tiredness                                      11.1%
      [####................]
      Whiny alternating cry, rubbing eyes
      -> They are tired. They will not admit it.

  3.  Everything else                                 6.6%
      [#...................]
      unknown forces at work

================================================================
  good luck out there
================================================================
```

---

## PROJECT STRUCTURE

```
baby_voice_decoder/
├── config.py                <- hyperparameters, labels, all the important stuff
├── colab_train.py           <- paste into Colab, press run, wait
├── colab_visualizer.py      <- watch the model learn in real time
├── requirements.txt         <- the dependencies (fewer than raising a child)
├── data/
│   └── prepare_dataset.py   <- convert Kaggle folder -> metadata.csv
└── src/
    ├── model.py             <- EfficientNet backbone + dual heads
    ├── dataset.py           <- audio loading, augmentation, mel-spec pipeline
    ├── train.py             <- training loop + warmup cosine scheduler
    └── inference.py         <- load model, decode cry, print result
```

---

## DATASET

[Kaggle — Decoding Cries Baby](https://www.kaggle.com/datasets/mennaahmed23/decoding-cries-baby)

Thousands of infant and toddler cry recordings across 10 distress categories and 6 age groups. Collected so you don't have to.

---

## TECH STACK

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch |
| Backbone | EfficientNet-B0 via `timm` |
| Audio Processing | librosa |
| Spectrogram | Log-mel, 128 mels, 10 ms hop, 16 kHz |
| Augmentation | Pitch shift, time stretch, Mixup |
| Scheduler | WarmupCosine |
| Environment | Google Colab (free tier, chaotic, beloved) |

---

## LICENSE

MIT — use it, modify it, ship it, name your app whatever you want.

---

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Comic+Sans+MS&size=14&duration=3000&pause=1000&color=FF6B9D&center=true&vCenter=true&width=600&lines=built+by+sciencebanda09;who+also+did+not+sleep+enough;gugugaga" alt="footer text" />

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:a855f7,50:ff6b9d,100:ffb347&height=120&section=footer&fontSize=16&fontColor=ffffff&fontAlignY=65&animation=fadeIn" width="100%"/>

</div>
