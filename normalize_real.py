"""
Bring real clips' loudness/silence profile in line with the fake clips'.

The fake filenames spell out their processing chain (...norm...silence...) --
they were peak-normalized and silence-trimmed before hand-off. Real clips
never were. Measured on this dataset: fake peak amplitude ~0.98 (std 0.06)
vs real ~0.58 (std 0.16); fake silence ratio ~0.23 vs real ~0.50. A bare
"peak > 0.9" threshold alone scores ~96% at telling the classes apart --
this dwarfs the duration/rate/codec confounds and was silently driving most
of a detector's apparent accuracy.

This writes peak-normalized, silence-trimmed WAV copies alongside the
originals (non-destructive) into a sibling "*_norm" folder, mirroring folder
structure, for both the training pool (data/real) and the holdout pool
(data/holdout/real) so both stay on equal footing.

Usage:
    python normalize_real.py --data_dir data
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
TARGET_PEAK_RANGE = (0.90, 0.99)  # matches fake's observed peak stats (mean 0.98, std 0.06)
SILENCE_THRESHOLD_RATIO = 0.02    # fraction of a clip's peak amplitude counted as silence
SEED = 42


def trim_silence(wav, threshold_ratio=SILENCE_THRESHOLD_RATIO):
    peak = np.abs(wav).max()
    if peak < 1e-6:
        return wav
    idx = np.nonzero(np.abs(wav) > threshold_ratio * peak)[0]
    if idx.size == 0:
        return wav
    return wav[idx[0]: idx[-1] + 1]


def peak_normalize(wav, target_peak):
    peak = np.abs(wav).max()
    if peak < 1e-6:
        return wav
    return wav * (target_peak / peak)


def process_file(src, dest, rng):
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    wav = data.mean(axis=1)  # mono
    wav = trim_silence(wav)
    wav = peak_normalize(wav, rng.uniform(*TARGET_PEAK_RANGE))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), wav, sr, subtype="PCM_16")


def process_tree(src_root, dest_root, rng):
    files = [p for p in Path(src_root).rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    n_ok = n_fail = 0
    for src in files:
        dest = dest_root / src.relative_to(src_root).with_suffix(".wav")
        try:
            process_file(src, dest, rng)
            n_ok += 1
        except Exception as e:
            print(f"  WARNING: failed to process {src}: {e}")
            n_fail += 1
    return n_ok, n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)

    real_dir = data_dir / "real"
    if real_dir.exists():
        print(f"Normalizing training-pool real clips ({real_dir})...")
        n_ok, n_fail = process_tree(real_dir, data_dir / "real_norm", rng)
        print(f"  {n_ok} written to {data_dir / 'real_norm'}, {n_fail} failed")

    holdout_real = data_dir / "holdout" / "real"
    if holdout_real.exists():
        print(f"Normalizing holdout-pool real clips ({holdout_real})...")
        n_ok, n_fail = process_tree(holdout_real, data_dir / "holdout" / "real_norm", rng)
        print(f"  {n_ok} written to {data_dir / 'holdout' / 'real_norm'}, {n_fail} failed")


if __name__ == "__main__":
    main()
