"""
Scan the dataset, report what's in it, and carve out a fixed holdout set
for final evaluation (see evaluate.py).

Folder layout expected (files may be nested arbitrarily deep):
    data/
        real/   <- genuine Hindi audio clips
        fake/   <- AI-generated / cloned Hindi audio clips

After running, ~N clips per class are moved (not copied) into:
    data/holdout/real/
    data/holdout/fake/
train.py excludes data/holdout from training automatically.

Usage:
    python prepare_data.py --data_dir data --holdout_per_class 100 --seed 42
    python prepare_data.py --dry_run          # just print the report, move nothing
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def scan_class(folder):
    return [p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTS]


def probe(path):
    """Cheap metadata read (no full decode): (duration_seconds, sample_rate)."""
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate


def summarize(label, files):
    formats = Counter(p.suffix.lower() for p in files)
    sample_rates = Counter()
    durations = []
    corrupt = []

    for p in files:
        try:
            dur, sr = probe(p)
            durations.append(dur)
            sample_rates[sr] += 1
        except Exception as e:
            corrupt.append((p, str(e)))

    print(f"\n=== {label} ===")
    print(f"  total files:  {len(files)}")
    print(f"  formats:      {dict(formats)}")
    print(f"  sample rates: {dict(sample_rates)}")
    if durations:
        total_min = sum(durations) / 60
        print(
            f"  duration (s): min={min(durations):.2f}  max={max(durations):.2f}  "
            f"mean={sum(durations)/len(durations):.2f}  total={total_min:.1f} min"
        )
    if corrupt:
        print(f"  WARNING: {len(corrupt)} corrupt/unreadable file(s):")
        for p, err in corrupt:
            print(f"    - {p}: {err}")
    else:
        print("  no corrupt/unreadable files detected")

    return {p for p, _ in corrupt}


def make_holdout(holdout_dir, label, files, corrupt_paths, n, seed):
    usable = [p for p in files if p not in corrupt_paths]
    rng = random.Random(seed)
    chosen = rng.sample(usable, min(n, len(usable)))

    dest_dir = holdout_dir / label
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for src in chosen:
        dest = dest_dir / src.name
        if dest.exists():
            # names can collide once nested subfolders are flattened
            dest = dest_dir / f"{src.parent.name}_{src.name}"
        shutil.move(str(src), str(dest))
        moved.append(dest)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--holdout_per_class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true", help="report only, move nothing")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    class_dirs = {"real": data_dir / "real", "fake": data_dir / "fake"}

    all_files, all_corrupt = {}, {}
    for label, folder in class_dirs.items():
        if not folder.exists():
            raise SystemExit(f"Expected folder not found: {folder}")
        files = scan_class(folder)
        all_files[label] = files
        all_corrupt[label] = summarize(label, files)

    if args.dry_run:
        print("\nDry run: no files moved.")
        return

    holdout_dir = data_dir / "holdout"
    print(f"\nMoving up to {args.holdout_per_class} clips per class into {holdout_dir} (seed={args.seed})...")
    for label, files in all_files.items():
        moved = make_holdout(holdout_dir, label, files, all_corrupt[label], args.holdout_per_class, args.seed)
        print(f"  {label}: moved {len(moved)} clips -> {holdout_dir / label}")

    print("\nRemaining training pool:")
    for label, folder in class_dirs.items():
        remaining = scan_class(folder)
        print(f"  {label}: {len(remaining)} clips")


if __name__ == "__main__":
    main()
