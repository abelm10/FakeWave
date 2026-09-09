"""
One-time backfill: add speaker_id, recording_id, condition to manifest.csv.

Run once (see main() at bottom). Investigation behind each rule is recorded
in project memory / the conversation that produced this script, not repeated
here -- this file is the mechanical implementation, not the argument for it.

Confidence per source (see conversation report for full reasoning):
  - MLAAD, 7/9 generators (indicf5, indri_tts_0_1, omnivoice,
    ringg_squirrel_tts_v1, veena, voxcpm2, voxtral): speaker_id from meta.csv's
    `reference_speaker` field -- CONFIDENT.
  - MLAAD, bark & xtts_v2: no reference_speaker field in meta.csv. Left
    UNGROUPED (unique placeholder per clip) per project owner's decision
    2026-08-18; the original-narrator name extracted from `original_file` is
    written to notes as a low-confidence proxy, not used for grouping.
  - MLAAD, all 9: recording_id = meta.csv's `original_file` field, used
    GLOBALLY (not scoped per generator) -- the same source text appearing
    under different generators (~1-2% pairwise overlap, confirmed by
    analysis) is a cross-generator near-duplicate leakage path, and sharing
    recording_id across generators closes it.
  - MLAAD rows with no meta.csv match (9 meta.csv-as-audio-row bugs, one per
    generator; bark's 10 undocumented clips): UNGROUPED, flagged in notes.
  - kaggle_hindi / kaggle_hindi_recovered: no speaker metadata exists at all.
    UNGROUPED (unique placeholder per clip) -- an UNQUANTIFIED leakage risk,
    not papered over.
  - teammate_bark: single shared speaker_id/recording_id group (13 clips,
    one voice, one batch).
  - for_2sec_english (FoR): usage=eval_only, excluded from the split
    entirely. Backfilled minimally (ungrouped placeholders) since grouping
    doesn't affect split integrity for eval-only data.
"""

from pathlib import Path

import pandas as pd

MANIFEST_PATH = "data/manifest.csv"
BACKUP_PATH = "data/manifest.csv.bak_pre_grouping_20260818"


def _narrator_from_original_file(original_file):
    """Extract '<gender>/<narrator_name>' from an MLAAD original_file path
    (e.g. '.../by_book/female/mary_ann/...' -> 'female/mary_ann'). Used only
    as a notes annotation for bark/xtts_v2, never for grouping -- see module
    docstring."""
    parts = Path(str(original_file)).parts
    if "by_book" in parts:
        i = parts.index("by_book")
        if i + 2 < len(parts):
            return f"{parts[i+1]}/{parts[i+2]}"
    return None


def backfill_mlaad(df):
    """Fill speaker_id/recording_id/condition for the 9 mlaad_hi_* sources
    by joining each generator's meta.csv on basename(filepath) == basename(path)."""
    gens_with_speaker = {
        "mlaad_hi_indicf5", "mlaad_hi_indri_tts_0_1", "mlaad_hi_omnivoice",
        "mlaad_hi_ringg_squirrel_tts_v1", "mlaad_hi_veena", "mlaad_hi_voxcpm2",
        "mlaad_hi_voxtral",
    }
    gens_no_speaker = {"mlaad_hi_bark", "mlaad_hi_xtts_v2"}

    for source in sorted(gens_with_speaker | gens_no_speaker):
        meta_path = Path("data/raw/fake") / source / "meta.csv"
        meta = pd.read_csv(meta_path, sep="|")
        meta["basename"] = meta["path"].apply(lambda p: Path(p).name)
        meta = meta.set_index("basename")

        mask = df["source"] == source
        idx = df.index[mask]
        basenames = df.loc[idx, "filepath"].apply(lambda p: Path(p).name)

        for i, bn in zip(idx, basenames):
            fp = df.at[i, "filepath"]
            if bn == "meta.csv":
                # known data-hygiene bug: manifest row points at meta.csv itself,
                # not an audio clip. Already filtered at Dataset-construction
                # time by FakeWaveDataset's readability probe; flag here too.
                # NOTE: use "not_a_clip", not "n/a" -- pandas' default NA-string
                # list includes "n/a" and silently round-trips it to real NaN
                # on the next read_csv, which then poisons any groupby/filter
                # keyed on this column (discovered 2026-08-18).
                df.at[i, "speaker_id"] = "not_a_clip"
                df.at[i, "recording_id"] = "not_a_clip"
                df.at[i, "condition"] = "not_a_clip"
                note = " | manifest row points at meta.csv, not an audio clip (known data-hygiene bug); not a real record"
                df.at[i, "notes"] = str(df.at[i, "notes"]) + note
                continue

            df.at[i, "condition"] = "synthetic_tts"

            if bn not in meta.index:
                # e.g. bark's 10 clips with no matching meta.csv row
                stem = Path(fp).stem
                df.at[i, "speaker_id"] = f"{source}::unknown::{stem}"
                df.at[i, "recording_id"] = f"{source}::unknown::{stem}"
                note = " | no meta.csv row found for this clip; speaker/recording metadata unavailable"
                df.at[i, "notes"] = str(df.at[i, "notes"]) + note
                continue

            row = meta.loc[bn]
            original_file = row["original_file"]
            df.at[i, "recording_id"] = original_file  # global key: shared across generators on purpose

            if source in gens_with_speaker:
                df.at[i, "speaker_id"] = f"{source}::{row['reference_speaker']}"
            else:
                stem = Path(fp).stem
                df.at[i, "speaker_id"] = f"{source}::unknown::{stem}"
                narrator = _narrator_from_original_file(original_file)
                if narrator:
                    note = f" | original narrator (unconfirmed voice-clone proxy, NOT used for grouping): {narrator}"
                    df.at[i, "notes"] = str(df.at[i, "notes"]) + note

    return df


def backfill_teammate_bark(df):
    mask = df["source"] == "teammate_bark"
    df.loc[mask, "speaker_id"] = "teammate_bark::voice1"
    df.loc[mask, "recording_id"] = "teammate_bark::session1"
    df.loc[mask, "condition"] = "synthetic_tts"
    return df


def backfill_kaggle_hindi(df, source_name):
    mask = df["source"] == source_name
    idx = df.index[mask]
    for i in idx:
        stem = Path(df.at[i, "filepath"]).stem
        df.at[i, "speaker_id"] = f"{source_name}::unknown::{stem}"
        df.at[i, "recording_id"] = f"{source_name}::unknown::{stem}"
        df.at[i, "condition"] = "clean_source"
    return df


def backfill_for(df):
    mask = df["source"] == "for_2sec_english"
    idx = df.index[mask]
    for i in idx:
        stem = Path(df.at[i, "filepath"]).stem
        df.at[i, "speaker_id"] = f"for_2sec_english::unknown::{stem}"
        df.at[i, "recording_id"] = f"for_2sec_english::unknown::{stem}"
        df.at[i, "condition"] = "synthetic_tts"
    return df


def build_recovered_rows():
    """New manifest rows for the 100 real clips recovered from the stale
    data/holdout/real/ directory (confirmed raw/unmodified 2026-08-18, see
    conversation) into data/raw/real/kaggle_hindi_recovered/."""
    files = sorted(Path("data/raw/real/kaggle_hindi_recovered").glob("*.mp3"))
    rows = []
    for f in files:
        stem = f.stem
        rows.append({
            "filepath": f"data/raw/real/kaggle_hindi_recovered/{f.name}",
            "label": "real",
            "source": "kaggle_hindi_recovered",
            "language": "hindi",
            "generator": None,
            "verified_hindi": "yes",
            "usage": "train",
            "notes": (
                "Real human speech; recovered from stale data/holdout/real/ "
                "(pre-raw/ reorg, orphaned by prepare_data.py/normalize_real.py "
                "workflow). Confirmed raw/unmodified: peak/RMS/format statistics "
                "match the current kaggle_hindi pool, nowhere near "
                "normalize_real.py's 0.90-0.99 target peak range; no real_norm "
                "artifact ever existed for these files. Reintegrated 2026-08-18."
            ),
            "speaker_id": f"kaggle_hindi_recovered::unknown::{stem}",
            "recording_id": f"kaggle_hindi_recovered::unknown::{stem}",
            "condition": "clean_source",
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(MANIFEST_PATH)
    assert Path(BACKUP_PATH).exists(), "backup missing -- aborting"

    for col in ["speaker_id", "recording_id", "condition"]:
        df[col] = pd.NA

    df = backfill_mlaad(df)
    df = backfill_teammate_bark(df)
    df = backfill_kaggle_hindi(df, "kaggle_hindi")
    df = backfill_for(df)

    recovered = build_recovered_rows()
    df = pd.concat([df, recovered], ignore_index=True)

    n_missing = df[["speaker_id", "recording_id", "condition"]].isna().any(axis=1).sum()
    assert n_missing == 0, f"{n_missing} rows missing new columns -- investigate before writing"

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {len(df)} rows to {MANIFEST_PATH}")
    print(df.groupby(["source"]).agg(
        n=("filepath", "size"),
        n_speaker_groups=("speaker_id", "nunique"),
        n_recording_groups=("recording_id", "nunique"),
    ))


if __name__ == "__main__":
    main()
