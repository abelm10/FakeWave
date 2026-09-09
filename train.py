"""
Baseline trainer for the FakeWave symmetric preprocessing pipeline.

Loads the train/val split from preprocess.train_val_split() (or a
generator-capped balanced subset via --target_per_class), feeds it through
FakeWaveDataset (see preprocess.py for the pipeline itself and the
reasoning behind each step), and trains a small CNN baseline (model.py)
over log-mel spectrograms.

Usage:
    python train.py --epochs 12 --target_per_class 900
    python train.py --smoke_test          # ~100 clips/class, 5 epochs, CPU
"""

import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DeepfakeCNN
from preprocess import CONFIG, FakeWaveDataset, _split_df


def _subsample_per_class(df, n_per_class, seed):
    """Balanced subsample for --smoke_test, so a fast run isn't dominated by
    the fake:real class imbalance in the full manifest."""
    parts = [
        df[df["label"] == label].sample(min(n_per_class, (df["label"] == label).sum()), random_state=seed)
        for label in ["real", "fake"]
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _water_fill(capacities, target_total):
    """Distribute target_total units as evenly as possible across groups,
    each capped by its own capacity in `capacities` (dict: name -> max).
    Used to cap a class-balanced sample per generator/source so no single
    generator (e.g. a ~1000-clip MLAAD pool) crowds out a small one (e.g.
    the 13-clip teammate_bark pool) or dominates the sample."""
    remaining = dict(capacities)
    alloc = {k: 0 for k in capacities}
    budget = target_total
    active = [k for k, v in remaining.items() if v > 0]
    while budget > 0 and active:
        share = max(1, budget // len(active))
        progressed = False
        for k in list(active):
            take = min(share, remaining[k], budget)
            if take > 0:
                alloc[k] += take
                remaining[k] -= take
                budget -= take
                progressed = True
            if remaining[k] <= 0:
                active.remove(k)
            if budget <= 0:
                break
        if not progressed:
            break
    return alloc


def build_balanced_subset(df, target_per_class, max_per_generator, seed):
    """Sample a class-balanced subset, capped per source/generator so no
    single fake generator (or, trivially, the single real source) dominates
    training. Real has one source (kaggle_hindi); fake is capped per-
    generator via water-filling so e.g. the ~1000-clip MLAAD pools don't
    crowd out the 13-clip teammate_bark pool."""
    parts = []
    for label in ["real", "fake"]:
        sub = df[df["label"] == label]
        caps = {
            src: (min(len(grp), max_per_generator) if max_per_generator else len(grp))
            for src, grp in sub.groupby("source")
        }
        alloc = _water_fill(caps, target_per_class)
        for src, n in alloc.items():
            if n <= 0:
                continue
            grp = sub[sub["source"] == src]
            parts.append(grp.sample(n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_epoch(model, loader, device, loss_fn, opt=None):
    train_mode = opt is not None
    model.train(train_mode)
    total_loss, correct, total = 0.0, 0, 0
    for spec, y in loader:
        spec, y = spec.to(device), y.to(device)
        with torch.set_grad_enabled(train_mode):
            logits = model(spec)
            loss = loss_fn(logits, y)
            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=CONFIG["manifest_path"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model_path", default="detector.pt")
    ap.add_argument("--target_per_class", type=int, default=None,
                     help="build a balanced subset with ~this many clips/class, capped per generator")
    ap.add_argument("--max_per_generator", type=int, default=None,
                     help="optional hard cap per source/generator (default: even water-fill only)")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--smoke_test", action="store_true",
                     help="fast sanity run: ~100 clips/class, 5 epochs, no checkpoint saved")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    full_df = pd.read_csv(args.manifest)
    full_df = full_df[full_df["usage"] == "train"].reset_index(drop=True)  # never FoR by default

    if args.smoke_test:
        args.epochs = 5
        pool_df = _subsample_per_class(full_df, 100, seed=CONFIG["split_random_state"])
        train_df, val_df = _split_df(pool_df, CONFIG["val_fraction"], CONFIG["split_random_state"])
        print(f"[smoke test] subsampled to {len(train_df)} train / {len(val_df)} val clips")
    elif args.target_per_class:
        # NOTE: quick random split, not a proper stratified-by-generator
        # holdout -- see the TODO in preprocess.train_val_split().
        pool_df = build_balanced_subset(full_df, args.target_per_class, args.max_per_generator,
                                         seed=CONFIG["split_random_state"])
        train_df, val_df = _split_df(pool_df, CONFIG["val_fraction"], CONFIG["split_random_state"])
        counts = pool_df.groupby(["label", "source"]).size()
        print(f"[balanced subset] target={args.target_per_class}/class, "
              f"max_per_generator={args.max_per_generator}\n{counts}")
    else:
        train_df, val_df = _split_df(full_df, CONFIG["val_fraction"], CONFIG["split_random_state"])

    train_ds = FakeWaveDataset(train_df, random_crop=True)
    val_ds = FakeWaveDataset(val_df, random_crop=False)
    print(f"Train: {len(train_ds)} clips ({(train_df['label']=='real').sum()} real / "
          f"{(train_df['label']=='fake').sum()} fake)")
    print(f"Val:   {len(val_ds)} clips ({(val_df['label']=='real').sum()} real / "
          f"{(val_df['label']=='fake').sum()} fake)")

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if args.smoke_test else 6
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers)

    model = DeepfakeCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # loss weighting as a fallback safety net for any residual class imbalance
    # (a no-op when train_df is already balanced, as with --target_per_class)
    n_real = (train_df["label"] == "real").sum()
    n_fake = (train_df["label"] == "fake").sum()
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Best checkpoint = highest val acc, tie-broken by lowest val loss -- val
    # acc alone repeats often near 1.0 (small val set), so without the
    # tiebreak the FIRST epoch to reach a given accuracy wins even if a
    # later epoch reached the same accuracy with a better-calibrated (lower
    # loss) model.
    best_val_acc, best_val_loss = 0.0, float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_dl, device, loss_fn, opt)
        val_loss, val_acc = run_epoch(model, val_dl, device, loss_fn, opt=None)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")
        is_better = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if not args.smoke_test and is_better:
            best_val_acc, best_val_loss = val_acc, val_loss
            torch.save(model.state_dict(), args.model_path)
            print(f"  -> saved {args.model_path} (epoch {epoch})")

    if args.smoke_test:
        print("\nSmoke test complete (no checkpoint saved).")
    else:
        print(f"\nDone. Best val accuracy: {best_val_acc:.3f} (val loss {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
