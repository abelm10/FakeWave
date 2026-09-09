"""
Run a trained FakeWave checkpoint on a single audio file.

Decoupled from train.py: the model architecture lives in model.py and the
preprocessing is preprocess.preprocess_clip() -- the exact same steps used
at training time (see preprocess.py), so inference can never silently drift
from what the model was actually trained on.

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --model_path detector.pt
"""

import argparse

import torch

from model import DeepfakeCNN
from preprocess import CONFIG, preprocess_clip

LABELS = ["real", "fake"]


def load_model(model_path="detector.pt", device="cpu"):
    model = DeepfakeCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(path, model=None, model_path="detector.pt", config=CONFIG, device="cpu"):
    """Run the exact training-time preprocessing pipeline on a single file
    and classify it.

    Returns (label, confidence, probs) where label is "real" or "fake",
    confidence is the model's probability for that label, and probs is
    {"real": p_real, "fake": p_fake}.
    """
    if model is None:
        model = load_model(model_path, device)
    logmel = preprocess_clip(path, config).unsqueeze(0).to(device)  # (1, 1, n_mels, time)
    with torch.no_grad():
        fake_prob = torch.sigmoid(model(logmel)).item()
    probs = {"real": 1.0 - fake_prob, "fake": fake_prob}
    label = "fake" if fake_prob > 0.5 else "real"
    return label, probs[label], probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--model_path", default="detector.pt")
    args = ap.parse_args()

    label, confidence, probs = predict(args.audio_path, model_path=args.model_path)
    verdict = "FAKE (AI-generated)" if label == "fake" else "REAL"
    print(f"Verdict: {verdict}  |  confidence: {confidence:.1%}")
    print(f"  p(real)={probs['real']:.3f}  p(fake)={probs['fake']:.3f}")


if __name__ == "__main__":
    main()
