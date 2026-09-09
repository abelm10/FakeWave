"""
Evaluate a trained detector on the held-out test set produced by prepare_data.py.

Reports accuracy, per-class precision/recall/F1, and a confusion matrix --
accuracy alone is misleading given the real/fake class imbalance.

Uses the same windowed-averaging inference as predict.py (CLIP_SECONDS windows,
averaged) rather than a single crop, so longer real clips aren't reduced to
just their first window.

Usage:
    python evaluate.py --data_dir data --model_path detector.pt
"""

import argparse
from pathlib import Path

import torch

from predict import preprocess
from train import DeepfakeCNN, AUDIO_EXTS, CLASS_DIRNAMES

CLASS_NAMES = ["real", "fake"]


def gather_holdout(data_dir):
    files, labels = [], []
    for label_name, label in [("real", 0), ("fake", 1)]:
        folder = Path(data_dir) / "holdout" / CLASS_DIRNAMES[label_name]
        if not folder.exists():
            raise SystemExit(f"{folder} not found. Run prepare_data.py then normalize_real.py first.")
        found = [p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        if not found:
            raise SystemExit(f"No holdout audio found in {folder}. Run prepare_data.py first.")
        files += found
        labels += [label] * len(found)
    return files, labels


def confusion_matrix(y_true, y_pred, n_classes=2):
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def precision_recall_f1(cm, n_classes=2):
    precision, recall, f1 = [], [], []
    for c in range(n_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes)) - tp
        fn = sum(cm[c][r] for r in range(n_classes)) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model_path", default="detector.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files, labels = gather_holdout(args.data_dir)
    print(f"Holdout set: {len(files)} clips ({labels.count(0)} real, {labels.count(1)} fake)")

    model = DeepfakeCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for path, label in zip(files, labels):
            specs = preprocess(path).to(device)
            fake_prob = torch.sigmoid(model(specs)).mean().item()
            y_true.append(label)
            y_pred.append(1 if fake_prob > 0.5 else 0)

    n = len(y_true)
    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / n
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(cm)

    print(f"\nAccuracy: {accuracy:.3f}\n")
    print(f"{'Class':<8}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<8}{precision[i]:>10.3f}{recall[i]:>10.3f}{f1[i]:>10.3f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = " " * 10 + "".join(f"{name:>10}" for name in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i][j]:>10}" for j in range(len(CLASS_NAMES)))
        print(f"{name:<10}{row}")


if __name__ == "__main__":
    main()
