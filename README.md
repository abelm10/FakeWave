kaggle dataset link
# Hindi Deepfake Audio Detector

Detects whether a Hindi voice clip is genuine or AI-generated (TTS / voice cloning).

## Setup

```bash
pip install torch torchaudio gradio
```

## 1. Add your data

```
data/
  real/   <- genuine Hindi recordings (.wav, .mp3, .flac)
  fake/   <- AI-generated / cloned Hindi audio
```

Aim for **at least 200–500 clips per class** to start; more is better. Keep clips
roughly 3–10 seconds. Try to match recording conditions between classes — if all
your real clips are phone recordings and all fake clips are studio-clean TTS output,
the model will learn "noisy = real" instead of learning actual deepfake artifacts.

### Where to get data
- **Real Hindi audio:** Common Voice Hindi (Mozilla), IndicTTS corpus, Kathbath,
  or your own recordings.
- **Fake Hindi audio:** generate samples with open TTS/voice-cloning models
  (e.g., Indic TTS models, XTTS, commercial TTS APIs) using varied speakers and texts.

## 2. Train

```bash
python train.py --data_dir data --epochs 20
```

Saves the best model to `detector.pt`.

## 3. Predict on a file

```bash
python predict.py sample.wav
```

## 4. Run the web app

```bash
python app.py
```

Opens a local page where you can upload or record audio and get a verdict.

## How it works

Each clip is converted to a log-mel spectrogram (a time–frequency image of the
audio). A small CNN learns to spot synthesis artifacts — unnatural harmonics,
overly smooth prosody, vocoder fingerprints — that distinguish generated speech
from real recordings. Long files are split into 4-second windows and the
predictions are averaged.

## Upgrading accuracy later

1. **More diverse fakes**: include several different TTS/cloning systems in your
   fake set, or the model will only detect the one system it saw.
2. **Pretrained backbones**: fine-tune `wav2vec2` (facebook/wav2vec2-base) or use
   anti-spoofing architectures like AASIST or RawNet2 — these are the current
   research standard and typically beat a small CNN.
3. **Report EER** (equal error rate) alongside accuracy — it's the standard metric
   in anti-spoofing benchmarks like ASVspoof.
