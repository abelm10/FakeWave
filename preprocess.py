"""
FakeWave symmetric audio preprocessing pipeline.

Core principle: this pipeline applies IDENTICALLY to every clip regardless of
label. There is no `if label == "real": ...` anywhere in here. That is not a
style preference -- it's a direct response to a prior bug in this project
(see normalize_real.py / old train.py) where real and fake clips were put
through *different* loudness/silence processing, and a bare "peak > 0.9"
threshold on that mismatch alone scored ~96% "accuracy" at telling the
classes apart. Every step below exists to destroy a specific real/fake
confound (sample rate, codec, loudness, silence ratio, duration) rather than
to make the audio "sound better".

Preprocessing is NEVER baked into files on disk. FakeWaveDataset reads raw
files straight from data/raw/ (as pointed to by manifest.csv) and runs the
full pipeline at __getitem__ time, every time.

Pipeline order (fixed, see _run_pipeline):
    1. load_raw        -- decode via soundfile (torchaudio.load is broken for
                           some formats in this environment)
    2. resample         -- force every clip to the same sample rate
    3. mp3_roundtrip     -- force every clip through the same codec
    4. trim_silence      -- force every clip through the same VAD
    5. crop_or_pad        -- force every clip to the same duration
    6. rms_normalize       -- force every clip to the same loudness
    7. log_mel               -- final model input

Run `python preprocess.py --sample N` to sanity-check the pipeline visually
before trusting it inside training (see main()).
"""

import argparse
import io
import random
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch
import torchaudio

# --------------------------------------------------------------------------
# Config -- every tunable lives here, not scattered as magic numbers below.
# --------------------------------------------------------------------------
CONFIG = {
    "manifest_path": "data/manifest.csv",

    # step 2: resample. 16kHz is the standard speech-ML rate and is at or
    # below the native rate of every source in this dataset (22050/44100/
    # 48000/16000), so it's a safe common denominator with no upsampling.
    "target_sr": 16000,

    # step 3: mp3 roundtrip. Fixed bitrate (not randomized) -- this step
    # exists to equalize codec artifacts deterministically, not to add
    # train-time augmentation diversity.
    "mp3_bitrate_kbps": 128,

    # step 4: silence trim. Energy-based (short-time RMS in dB below the
    # clip's own peak), not a raw amplitude gate -- see trim_silence().
    "silence_top_db": 40.0,
    "silence_frame_ms": 25.0,
    "silence_hop_ms": 10.0,

    # step 5: crop/pad. FoR clips are EXACTLY 2.0s (an upstream splicing
    # signature); real clips are natively ~5s. 4.0s was chosen so the fixed
    # length never coincides with FoR's 2.0s fingerprint, while staying
    # close enough to real's natural length that little padding is needed.
    # Confirmed with project owner 2026-08-12.
    "target_duration_sec": 4.0,

    # step 6: RMS normalize. Target loudness in dBFS; -20 dBFS is a common
    # speech-level target. RMS (average energy), not peak, is used
    # deliberately -- see rms_normalize().
    "target_rms_dbfs": -20.0,

    # step 7: log-mel spectrogram.
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 64,

    # train/val split (see train_val_split)
    "val_fraction": 0.2,
    "split_random_state": 42,
}

LABEL_TO_INT = {"real": 0, "fake": 1}


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
def load_raw(path):
    """Step 1. Decode raw audio via soundfile and collapse to mono float32.

    WHY soundfile and not torchaudio.load: torchaudio.load needs
    torchcodec/ffmpeg for compressed formats and has been broken for some
    files in this dataset (mp3/mixed formats across sources); soundfile
    (libsndfile) decodes wav/mp3/flac/ogg directly and consistently.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = data.mean(axis=1).astype(np.float32)
    return wav, sr


def resample(wav, sr, target_sr):
    """Step 2. Resample to a single fixed rate for every clip.

    WHY: real clips arrive at 44.1/48kHz, MLAAD fakes at 22050Hz, FoR at
    16kHz. Left native, sample rate alone (and the distinct resampling/
    aliasing fingerprints each rate leaves) would trivially separate real
    from fake. Using scipy's polyphase resampler (not torchaudio, to avoid
    depending on torchaudio.load's decode path anywhere in this pipeline).
    """
    if sr == target_sr:
        return wav.astype(np.float32)
    frac = Fraction(target_sr, sr).limit_denominator(1000)
    out = scipy.signal.resample_poly(wav, frac.numerator, frac.denominator)
    return out.astype(np.float32)


def mp3_roundtrip(wav, sr, bitrate_kbps):
    """Step 3. Encode/decode through MP3 in memory, at a fixed bitrate.

    WHY: real clips are sourced as mp3, most fake generators emit wav --
    codec compression artifacts alone could be a shortcut signal. Routing
    every clip through the identical fixed-bitrate mp3 round-trip equalizes
    that, deliberately, in-memory (never written to disk).
    """
    compression_level = max(0.0, min(0.95, 1 - bitrate_kbps / 170))
    buf = io.BytesIO()
    sf.write(
        buf, wav, sr, format="MP3", subtype="MPEG_LAYER_III",
        compression_level=compression_level, bitrate_mode="CONSTANT",
    )
    buf.seek(0)
    out, out_sr = sf.read(buf, dtype="float32", always_2d=True)
    out = out.mean(axis=1).astype(np.float32)
    assert out_sr == sr
    # mp3 encode/decode can shift length by a frame or two; realign to input length
    if out.shape[0] < wav.shape[0]:
        out = np.pad(out, (0, wav.shape[0] - out.shape[0]))
    else:
        out = out[: wav.shape[0]]
    return out


def trim_silence(wav, sr, top_db, frame_ms, hop_ms):
    """Step 4. Energy-based (dB-below-peak) leading/trailing silence trim.

    WHY this method: previously, fake clips had been silence-trimmed
    upstream with a hard amplitude gate while real clips hadn't (fake
    silence ratio ~0.23 vs real ~0.50), which was itself part of the
    loudness/silence confound. A dB-below-the-clip's-own-peak threshold on
    short-time RMS energy (rather than a raw amplitude cutoff) adapts to
    each clip's own dynamic range and is applied identically, at load time,
    to both classes -- so no clip's silence ratio depends on where it came
    from.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    if wav.shape[0] <= frame_len:
        return wav

    n_frames = 1 + (wav.shape[0] - frame_len) // hop_len
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        frame = wav[start:start + frame_len]
        energies[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    peak_energy = energies.max()
    if peak_energy < 1e-8:
        return wav  # near-total silence; nothing to trim against

    threshold = peak_energy * (10 ** (-top_db / 20))
    active = np.nonzero(energies > threshold)[0]
    if active.size == 0:
        return wav

    start_sample = active[0] * hop_len
    end_sample = min(wav.shape[0], active[-1] * hop_len + frame_len)
    return wav[start_sample:end_sample]


def crop_or_pad(wav, sr, target_duration_sec, random_crop=False, rng=None):
    """Step 5. Force every clip to the same fixed length.

    WHY 4.0s specifically: FoR clips are EXACTLY 2.0s (an upstream splicing
    signature that was previously a confound source); MLAAD fakes range
    ~2-32s; real clips are natively ~5s. 4.0s avoids ever coinciding with
    FoR's fingerprint. Center-crop (not crop-from-start) when trimming, so
    a clip's onset -- disproportionately "clean" speech for TTS -- isn't
    systematically favored. `random_crop` is available for training-time
    variation but is applied the same way regardless of label, never
    class-conditionally.
    """
    target_len = int(round(target_duration_sec * sr))
    n = wav.shape[0]
    if n < target_len:
        return np.pad(wav, (0, target_len - n))
    if n == target_len:
        return wav
    if random_crop:
        source = rng if rng is not None else random
        start = source.randint(0, n - target_len)
    else:
        start = (n - target_len) // 2
    return wav[start:start + target_len]


def rms_normalize(wav, target_dbfs):
    """Step 6. Normalize loudness to a fixed RMS (average energy) target.

    WHY this was the worst confound: fake clips had been peak-normalized
    and silence-trimmed before hand-off (baked into files on disk, visible
    in their filenames); real clips never were. Measured previously: fake
    peak amplitude ~0.98 (std 0.06) vs real ~0.58 (std 0.16). A bare
    "peak > 0.9" threshold alone told the classes apart ~96% of the time --
    this dwarfed every other confound (duration/rate/codec) combined.
    RMS (average energy), not peak, is used here, and it is computed fresh
    at load time for every clip -- it can't drift out of sync with a
    class-specific offline step because there IS no offline step.
    """
    rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    if rms < 1e-8:
        return wav  # true silence; nothing to normalize
    target_rms = 10 ** (target_dbfs / 20)
    out = wav * (target_rms / rms)
    peak = np.abs(out).max()
    if peak > 1.0:
        out = out / peak  # guard rare clipping on very peaky/impulsive clips
    return out.astype(np.float32)


def _mel_transform(sr, n_fft, hop_length, n_mels):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )


def log_mel(wav, sr, n_fft, hop_length, n_mels):
    """Step 7. Log-mel spectrogram -- the model's actual input."""
    t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    mel = _mel_transform(sr, n_fft, hop_length, n_mels)(t)
    return torchaudio.transforms.AmplitudeToDB()(mel)  # (1, n_mels, time)


def _run_pipeline(path, config, random_crop=False, rng=None):
    """Steps 1-6, shared by the Dataset, the CLI plots, and the sanity checks."""
    raw_wav, raw_sr = load_raw(path)
    wav = resample(raw_wav, raw_sr, config["target_sr"])
    wav = mp3_roundtrip(wav, config["target_sr"], config["mp3_bitrate_kbps"])
    wav = trim_silence(
        wav, config["target_sr"], config["silence_top_db"],
        config["silence_frame_ms"], config["silence_hop_ms"],
    )
    wav = crop_or_pad(
        wav, config["target_sr"], config["target_duration_sec"],
        random_crop=random_crop, rng=rng,
    )
    wav = rms_normalize(wav, config["target_rms_dbfs"])
    return raw_wav, raw_sr, wav


def preprocess_clip(path, config=CONFIG, random_crop=False, rng=None):
    """Full pipeline, raw file -> final log-mel tensor. The Dataset's only entry point."""
    _, _, wav = _run_pipeline(path, config, random_crop=random_crop, rng=rng)
    return log_mel(wav, config["target_sr"], config["n_fft"], config["hop_length"], config["n_mels"])


def preprocess_stages(path, config=CONFIG, random_crop=False, rng=None):
    """Like preprocess_clip, but also returns the raw/intermediate stages, for plotting."""
    raw_wav, raw_sr, wav = _run_pipeline(path, config, random_crop=random_crop, rng=rng)
    return {
        "raw_wav": raw_wav,
        "raw_sr": raw_sr,
        "raw_logmel": log_mel(raw_wav, raw_sr, config["n_fft"], config["hop_length"], config["n_mels"]),
        "final_wav": wav,
        "final_sr": config["target_sr"],
        "logmel": log_mel(wav, config["target_sr"], config["n_fft"], config["hop_length"], config["n_mels"]),
    }


def _is_readable_audio(path):
    """Header-only probe (sf.info, no full decode) -- cheap enough to run over a manifest."""
    try:
        sf.info(str(path))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class FakeWaveDataset(torch.utils.data.Dataset):
    """Loads raw audio from a manifest subset and applies the pipeline above.

    The pipeline is identical for every row regardless of `label` -- that
    symmetry is the entire point of this class (see module docstring).

    By default, only rows with usage == "train" are used: the FoR clips
    (source == for_2sec_english) are tagged usage == "eval_only" in
    manifest.csv because they're English, not Hindi, and were a prior
    confound source (see CONFIG's target_duration_sec note). Pass
    eval_mode=True to explicitly include them for evaluation -- this should
    never happen in a training call.
    """

    def __init__(self, manifest_df, config=CONFIG, eval_mode=False, random_crop=False, seed=None):
        df = manifest_df.reset_index(drop=True)
        if not eval_mode:
            df = df[df["usage"] == "train"].reset_index(drop=True)

        if len(df):
            exists_mask = df["filepath"].apply(lambda p: Path(p).exists())
            n_missing = int(len(df) - exists_mask.sum())
            if n_missing:
                warnings.warn(f"FakeWaveDataset: {n_missing} manifest file(s) not found on disk; skipping them.")
            df = df[exists_mask].reset_index(drop=True)

        if len(df):
            # Cheap header-only probe (not a full decode) -- catches both
            # corrupt audio and non-audio rows accidentally in the manifest
            # (e.g. a generator's meta.csv scanned in alongside its clips) at
            # construction time, so a bad file can't crash a DataLoader
            # worker mid-epoch.
            readable_mask = df["filepath"].apply(_is_readable_audio)
            n_unreadable = int(len(df) - readable_mask.sum())
            if n_unreadable:
                warnings.warn(
                    f"FakeWaveDataset: {n_unreadable} manifest file(s) failed to open as audio "
                    "(corrupt or non-audio); skipping them."
                )
            df = df[readable_mask].reset_index(drop=True)

        self.df = df
        self.config = config
        self.random_crop = random_crop
        # Always a random.Random *instance* (never the bare `random` module):
        # DataLoader(num_workers>0) on Windows pickles the Dataset to send to
        # worker processes, and a module reference isn't picklable there.
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        logmel = preprocess_clip(row["filepath"], self.config, random_crop=self.random_crop, rng=self.rng)
        label = LABEL_TO_INT[row["label"]]
        return logmel, torch.tensor(label, dtype=torch.float32)


def _split_df(df, val_fraction, seed):
    """Shared random-split primitive: shuffle then cut. Used both by
    train_val_split() below and by train.py when it needs to split a
    pre-filtered (e.g. balanced-subset) DataFrame the same way."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(round((1 - val_fraction) * len(df)))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def train_val_split(manifest_path=None, val_fraction=None, seed=None, include_eval_only=False):
    """Quick random 80/20 split across the manifest, just enough to train tonight without leakage.

    TODO: this is NOT a proper holdout. It's a random split across the full
    pool, so e.g. near-duplicate lines/speakers from the same generator can
    land in both train and val. A proper stratified-by-generator holdout
    (disjoint by generator/speaker, with FoR kept strictly eval-only, not
    used for early stopping or model selection) still needs to be carved
    out separately before any reported numbers are trusted.
    """
    manifest_path = manifest_path or CONFIG["manifest_path"]
    val_fraction = CONFIG["val_fraction"] if val_fraction is None else val_fraction
    seed = CONFIG["split_random_state"] if seed is None else seed

    df = pd.read_csv(manifest_path)
    if not include_eval_only:
        df = df[df["usage"] == "train"].reset_index(drop=True)
    return _split_df(df, val_fraction, seed)


# --------------------------------------------------------------------------
# Checks: shape symmetry + RMS separability sanity check
# --------------------------------------------------------------------------
def check_shapes_match(manifest_path=None, config=CONFIG, seed=0):
    """Runs one real clip and one fake clip through the identical pipeline
    and asserts their output shapes match.

    This is the core promise of a symmetric pipeline made concrete: if
    shapes diverge by class, some class-conditional code path exists
    somewhere and must be found before training on it.
    """
    manifest_path = manifest_path or CONFIG["manifest_path"]
    df = pd.read_csv(manifest_path)
    df = df[df["usage"] == "train"]
    real_fp = df[df["label"] == "real"].sample(1, random_state=seed)["filepath"].iloc[0]
    fake_fp = df[df["label"] == "fake"].sample(1, random_state=seed)["filepath"].iloc[0]

    real_out = preprocess_clip(real_fp, config)
    fake_out = preprocess_clip(fake_fp, config)
    assert real_out.shape == fake_out.shape, (
        f"Shape mismatch: real {tuple(real_out.shape)} vs fake {tuple(fake_out.shape)} "
        "-- pipeline is not symmetric, investigate before training."
    )
    print(f"  OK: identical output shape {tuple(real_out.shape)} for real "
          f"({Path(real_fp).name}) and fake ({Path(fake_fp).name})")
    return True


def _best_threshold_accuracy(a, b):
    """Best accuracy a single threshold could achieve separating two 1-D samples."""
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    order = np.argsort(values)
    values, labels = values[order], labels[order]
    midpoints = (values[:-1] + values[1:]) / 2 if len(values) > 1 else np.array([])
    thresholds = np.concatenate([[values[0] - 1], midpoints, [values[-1] + 1]])
    best = 0.5
    for t in thresholds:
        pred = (values > t).astype(float)
        best = max(best, (pred == labels).mean(), (pred == 1 - labels).mean())
    return best


def sanity_check_rms(manifest_path=None, config=CONFIG, n_per_class=50, seed=0):
    """Reports post-normalization loudness stats for a balanced real/fake
    sample and flags if loudness is still separable between classes.

    This targets the exact confound class that previously hit ~96%
    "accuracy" from a bare peak threshold: if this flags red now, some
    class-conditional path re-broke the symmetry upstream of this check.
    RMS should land near-identical for both classes by construction (that's
    what step 6 does); peak amplitude and crest factor (peak/RMS) are also
    reported because RMS-matching alone does not equalize those, and they
    could still carry residual class signal.
    """
    manifest_path = manifest_path or CONFIG["manifest_path"]
    df = pd.read_csv(manifest_path)
    df = df[df["usage"] == "train"]

    results = {}
    for label in ["real", "fake"]:
        sub = df[df["label"] == label]
        sub = sub[sub["filepath"].apply(lambda p: Path(p).exists())]
        rows = sub.sample(min(n_per_class, len(sub)), random_state=seed)
        rms_vals, peak_vals, crest_vals = [], [], []
        for fp in rows["filepath"]:
            _, _, wav = _run_pipeline(fp, config)
            rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
            peak = float(np.abs(wav).max())
            rms_vals.append(rms)
            peak_vals.append(peak)
            crest_vals.append(peak / (rms + 1e-12))
        results[label] = {
            "rms": np.array(rms_vals), "peak": np.array(peak_vals), "crest": np.array(crest_vals),
        }

    print(f"  {'metric':<14}{'real mean±std':<22}{'fake mean±std':<22}")
    for metric in ["rms", "peak", "crest"]:
        r, f = results["real"][metric], results["fake"][metric]
        print(f"  {metric:<14}{r.mean():.4f} ± {r.std():.4f}       {f.mean():.4f} ± {f.std():.4f}")

    print()
    flag_threshold = 0.65
    for name, key in [("RMS", "rms"), ("peak amplitude", "peak"), ("crest factor", "crest")]:
        acc = _best_threshold_accuracy(results["real"][key], results["fake"][key])
        flag = "  <-- FLAGGED: still separable, investigate before training" if acc > flag_threshold else "  OK"
        print(f"  best-threshold separability on {name}: {acc:.3f}{flag}")
    if n_per_class < 100:
        print("  (note: best-threshold search over a small sample is optimistically biased -- "
              "re-run with a larger --rms_sample before trusting a FLAGGED result at this n)")

    return results


# --------------------------------------------------------------------------
# CLI: visual before/after sanity check
# --------------------------------------------------------------------------
def _plot_before_after(path, label, config, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = preprocess_stages(path, config)
    raw_wav, raw_sr = stages["raw_wav"], stages["raw_sr"]
    final_wav, final_sr = stages["final_wav"], stages["final_sr"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0, 0].plot(np.arange(len(raw_wav)) / raw_sr, raw_wav, linewidth=0.5)
    axes[0, 0].set_title(f"raw waveform ({label}, sr={raw_sr}, {len(raw_wav)/raw_sr:.2f}s)")
    axes[0, 0].set_xlabel("seconds")

    axes[0, 1].imshow(stages["raw_logmel"].squeeze(0).numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[0, 1].set_title("raw log-mel spectrogram")

    axes[1, 0].plot(np.arange(len(final_wav)) / final_sr, final_wav, linewidth=0.5)
    axes[1, 0].set_title(f"processed waveform (sr={final_sr}, {len(final_wav)/final_sr:.2f}s)")
    axes[1, 0].set_xlabel("seconds")
    axes[1, 0].set_ylim(-1.05, 1.05)

    axes[1, 1].imshow(stages["logmel"].squeeze(0).numpy(), origin="lower", aspect="auto", cmap="magma")
    axes[1, 1].set_title("processed log-mel spectrogram (model input)")

    fig.suptitle(Path(path).name)
    fig.tight_layout()
    out_path = out_dir / f"{label}_{Path(path).stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Visually sanity-check the FakeWave preprocessing pipeline.")
    ap.add_argument("--sample", type=int, default=6, help="clips per class to plot")
    ap.add_argument("--manifest", default=CONFIG["manifest_path"])
    ap.add_argument("--out_dir", default="scratch/preprocess_check")
    ap.add_argument("--rms_sample", type=int, default=100, help="clips per class for the RMS sanity check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Shape symmetry check")
    check_shapes_match(args.manifest, CONFIG, seed=args.seed)

    print(f"\n2. RMS/loudness separability sanity check ({args.rms_sample} clips/class)")
    sanity_check_rms(args.manifest, CONFIG, n_per_class=args.rms_sample, seed=args.seed)

    print(f"\n3. Saving before/after plots for {args.sample} clips/class -> {out_dir.resolve()}")
    df = pd.read_csv(args.manifest)
    df = df[df["usage"] == "train"]
    for label in ["real", "fake"]:
        sub = df[df["label"] == label]
        sub = sub[sub["filepath"].apply(lambda p: Path(p).exists())]
        rows = sub.sample(min(args.sample, len(sub)), random_state=args.seed)
        for fp in rows["filepath"]:
            try:
                out_path = _plot_before_after(fp, label, CONFIG, out_dir)
                print(f"  wrote {out_path}")
            except Exception as e:
                print(f"  WARNING: failed on {fp}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
