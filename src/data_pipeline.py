"""Multi-source data pipeline for roti-predictor.

Loads every CSV in data/raw/ (read-only — GOVERNANCE.md HC-4), each of which
may use a different schema (different column names for the same concept,
or missing columns entirely), standardizes them into one canonical schema,
handles missingness one column at a time with a decision that fits that
column's semantics (never one blanket strategy — GOVERNANCE.md HC-2), marks
rows that came from a source file needing schema normalization or that
carry unresolved conflicts, and writes data/processed/merged_clean.csv.

This is a standalone, single-file pipeline distinct from
src/roti_predictor/data_ingestion.py (which is scoped to the single known
daily_context.csv file). This script is the generalized version: it should
keep working if a second or third raw file with a different schema shows
up in data/raw/.

Run directly:
    python src/data_pipeline.py

Stops after writing merged_clean.csv and printing the summary — no
modeling here, by design.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "merged_clean.csv"

# ---------------------------------------------------------------------------
# 1. Schema normalization
# ---------------------------------------------------------------------------
# Canonical column name -> every raw column name (case-insensitive, spaces
# treated as underscores) that should map to it. Add to these lists as new
# source files reveal new naming conventions — do not add a new canonical
# column just because one file spells it differently.
#
# `day_name` / `day_type` are deliberately NOT in here: day-of-week is
# always *derived* from `date` (see derive_date_fields), never trusted from
# a raw column, because Phase 2 EDA found the raw day_name column in
# daily_context.csv didn't reliably match the real weekday (14/100 correct
# in the first 100 rows) — the same failure mode could hit a `day_type`
# column in a differently-sourced file, and there is no way to tell a
# trustworthy day-name column from an untrustworthy one except by
# recomputing from the date, so every source is treated the same way.
COLUMN_ALIASES = {
    "date": ["date", "log_date", "entry_date"],
    "meal_type": ["meal_type", "meal"],
    "workout_type": ["workout_type", "workout"],
    "workout_duration_min": ["workout_duration_min", "workout_minutes", "workout_duration"],
    "calories_burned": ["calories_burned", "calories"],
    "sleep_hours": ["sleep_hours", "sleep"],
    "mood_score": ["mood_score", "mood"],
    "hunger_level": ["hunger_level", "hunger"],
    "curry_type": ["curry_type", "curry"],
    "curry_richness": ["curry_richness", "richness"],
    "weight_trend": ["weight_trend", "trend"],
    "meal_prep_time_min": ["meal_prep_time_min", "prep_time_min", "prep_time"],
    "protein_target_g": ["protein_target_g", "protein_g", "protein"],
    "fibre_target_g": ["fibre_target_g", "fiber_target_g", "fibre_g", "fiber_g", "fibre"],
    "vitamin_focus": ["vitamin_focus", "vitamin"],
    "roti_count": ["roti_count", "rotis", "roti"],
}
CANONICAL_COLUMNS = list(COLUMN_ALIASES.keys())
TARGET_COLUMNS = ["roti_count", "protein_target_g", "fibre_target_g", "vitamin_focus"]


def _clean_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def normalize_schema(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, list[str]]:
    """Rename this source's columns to the canonical schema and report what
    had to change, so the caller can mark rows from this source as needing
    normalization vs. already-canonical."""
    notes: list[str] = []
    rename_map: dict[str, str] = {}
    lookup = {_clean_key(c): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        matched_raw = None
        for alias in aliases:
            if alias in lookup:
                matched_raw = lookup[alias]
                break
        if matched_raw is None:
            continue
        if matched_raw != canonical:
            notes.append(f"'{matched_raw}' -> '{canonical}'")
        rename_map[matched_raw] = canonical

    df = df.rename(columns=rename_map)

    missing_canonical = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    for col in missing_canonical:
        df[col] = np.nan
        notes.append(f"column '{col}' absent in {source_name} - filled as all-missing")

    df = df[CANONICAL_COLUMNS]  # canonical columns only; raw day_name/day_type/etc. dropped here
    return df, notes


def load_all_sources(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    schema_notes: dict[str, list[str]] = {}
    for path in csv_paths:
        raw = pd.read_csv(path)  # read-only — nothing in this script writes back to data/raw/ (HC-4)
        normalized, notes = normalize_schema(raw, path.name)
        normalized["source_file"] = path.name
        schema_notes[path.name] = notes
        frames.append(normalized)

    merged = pd.concat(frames, ignore_index=True)
    return merged, schema_notes


# ---------------------------------------------------------------------------
# 2. Date handling — always the first step, everything else depends on it
# ---------------------------------------------------------------------------

def derive_date_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Parse `date` (dayfirst, tolerant of format differences across
    sources) and derive day_of_week/is_weekend from it — never from a raw
    day_name/day_type column, see COLUMN_ALIASES comment above.

    date itself is the one column where a missing/unparseable value causes
    a row to be dropped rather than imputed or flagged-and-kept: there is
    no defensible way to invent a date, and every downstream step (the
    time-based split, day-of-week derivation, chronological sort) depends
    on it. This is the single deliberate drop in this pipeline; every other
    column is imputed or flagged, never dropped.
    """
    df = df.copy()
    # format="mixed" parses each value independently rather than inferring
    # ONE format from early rows and applying it to the whole column. That
    # inference is the default pd.to_datetime behavior without an explicit
    # format, and it's a real trap here: sources are concatenated (see
    # load_all_sources) before this runs, so if an early row happens to be
    # in a different format than the rest (e.g. one source's YYYY-MM-DD
    # next to another's DD/MM/YYYY), the wrong format silently "wins" for
    # every row and everything else in the column fails to parse — not a
    # hypothetical, this exact case dropped 220 of 221 rows in testing
    # before format="mixed" was added.
    parsed = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    n_dropped = int(parsed.isna().sum())

    # A handful of bad dates in real data is plausible (a typo, a corrupt
    # row) and is handled by the normal drop-and-report path below. Losing
    # most of the dataset to date parsing is never that — it means the
    # parse itself is broken, not the data, and continuing would silently
    # produce a near-empty (and therefore misleading) merged_clean.csv.
    if len(df) > 0 and n_dropped / len(df) > 0.5:
        bad_examples = df.loc[parsed.isna(), "date"].head(5).tolist()
        raise ValueError(
            f"{n_dropped} of {len(df)} rows failed date parsing ({n_dropped / len(df):.0%}) — "
            f"this looks like a parsing problem, not a few bad rows. Examples of unparsed "
            f"values: {bad_examples}. Investigate before re-running; refusing to silently "
            f"produce a near-empty output."
        )

    df = df.loc[parsed.notna()].copy()
    df["date"] = parsed.loc[parsed.notna()]
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    return df, n_dropped


# ---------------------------------------------------------------------------
# 3. Column-by-column missingness handling
# ---------------------------------------------------------------------------
# Each column gets its own decision, justified by what the column means and
# what "missing" plausibly means for it. A running `flags` dict collects
# which rows were touched per column so the summary and the data_quality
# column can report on it; a companion `<col>_missing` column marks the
# original gap explicitly per GOVERNANCE.md HC-2, whether or not the value
# was subsequently imputed.

def handle_missingness(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    imputed_counts: dict[str, int] = {}

    # meal_type — categorical, may be entirely absent from a source file
    # (e.g. a daily-granularity log has no concept of "which meal"). We
    # cannot guess breakfast/lunch/dinner/snack from other columns without
    # fabricating information, so missing values get an explicit sentinel
    # category rather than a guess, and the row is kept — lacking a meal
    # label doesn't invalidate the rest of the row's data.
    df["meal_type_missing"] = df["meal_type"].isna()
    imputed_counts["meal_type"] = int(df["meal_type_missing"].sum())
    df["meal_type"] = df["meal_type"].fillna("unspecified")

    # workout_type — categorical, and critically 'none' is already a valid,
    # real value (rest day), not a missingness marker. So a genuinely
    # missing value must NOT be imputed to 'none' (that would silently
    # assert "no workout happened" when the truth is "not logged" — a
    # meaningfully different claim). It gets its own 'unknown' sentinel.
    df["workout_type_missing"] = df["workout_type"].isna()
    imputed_counts["workout_type"] = int(df["workout_type_missing"].sum())
    df["workout_type"] = df["workout_type"].fillna("unknown")

    # workout_duration_min — numeric, semantically dependent on
    # workout_type. If workout_type is (originally) 'none', 0 minutes is
    # not a guess, it's the logical consequence of a rest day, so we fill
    # it directly. Otherwise (a workout happened or workout_type itself is
    # unknown) we don't know the duration, so we impute the column median
    # — a defensible central estimate — and flag it, rather than asserting
    # a specific number with false confidence.
    df["workout_duration_min_missing"] = df["workout_duration_min"].isna()
    rest_day_mask = df["workout_duration_min_missing"] & (df["workout_type"] == "none")
    df.loc[rest_day_mask, "workout_duration_min"] = 0
    remaining_mask = df["workout_duration_min"].isna()
    if remaining_mask.any():
        median_val = df["workout_duration_min"].median()
        df.loc[remaining_mask, "workout_duration_min"] = median_val
    imputed_counts["workout_duration_min"] = int(df["workout_duration_min_missing"].sum())

    # calories_burned — numeric, continuous, includes resting/basal burn so
    # it's fairly stable day to day even without a workout. Median
    # imputation is a reasonable default; dropping the row would lose
    # otherwise-good data over one soft signal.
    df["calories_burned_missing"] = df["calories_burned"].isna()
    imputed_counts["calories_burned"] = int(df["calories_burned_missing"].sum())
    df["calories_burned"] = df["calories_burned"].fillna(df["calories_burned"].median())

    # sleep_hours — numeric, physiological, no strong same-day predictor
    # available to condition on. Median imputation + flag; this is exactly
    # the "unlogged" case called out in the project brief.
    df["sleep_hours_missing"] = df["sleep_hours"].isna()
    imputed_counts["sleep_hours"] = int(df["sleep_hours_missing"].sum())
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())

    # mood_score, hunger_level — ordinal 1-5 self-reports. Median is a
    # robust central estimate for ordinal data (won't produce a non-integer
    # that misrepresents precision the way a mean might for a 1-5 scale
    # once rounded); flagged so downstream modeling can weight/exclude.
    for col in ["mood_score", "hunger_level"]:
        df[f"{col}_missing"] = df[col].isna()
        imputed_counts[col] = int(df[f"{col}_missing"].sum())
        df[col] = df[col].fillna(df[col].median())

    # curry_type, curry_richness — describe what was actually eaten. A gap
    # here plausibly means "meal skipped or details not logged" (the
    # project brief's other missingness example). We can't infer what curry
    # was eaten from other columns, so — same logic as meal_type — an
    # explicit 'unknown' sentinel, row kept (the day's roti_count can still
    # be a valid, useful label even if curry details are unlogged).
    for col in ["curry_type", "curry_richness"]:
        df[f"{col}_missing"] = df[col].isna()
        imputed_counts[col] = int(df[f"{col}_missing"].sum())
        df[col] = df[col].fillna("unknown")

    # weight_trend — categorical, but unlike curry/workout it's a *trend*:
    # by definition it changes slowly over days, not meal to meal. That
    # makes forward-fill (carry the last known trend forward) a materially
    # better estimate than an 'unknown' sentinel here — this is exactly the
    # "don't apply one blanket strategy" case: the same column type
    # (categorical) gets a different treatment because its real-world
    # semantics differ from curry_type/workout_type. Sorted by date within
    # each source first so "forward" means chronologically forward. Any
    # rows still missing after a forward-fill (i.e. missing at the very
    # start of a source's history, nothing earlier to carry from) fall back
    # to back-fill, then finally 'unknown' if a whole source never logged it.
    df["weight_trend_missing"] = df["weight_trend"].isna()
    imputed_counts["weight_trend"] = int(df["weight_trend_missing"].sum())
    df = df.sort_values(["source_file", "date"])
    df["weight_trend"] = df.groupby("source_file")["weight_trend"].ffill().bfill()
    df["weight_trend"] = df["weight_trend"].fillna("unknown")

    # meal_prep_time_min — numeric, needed downstream by Layer 2. May
    # correlate with meal_type (a snack preps faster than dinner), so
    # impute per meal_type group where possible for a tighter estimate,
    # falling back to the global median for a meal_type with no other
    # observed prep times at all.
    df["meal_prep_time_min_missing"] = df["meal_prep_time_min"].isna()
    imputed_counts["meal_prep_time_min"] = int(df["meal_prep_time_min_missing"].sum())
    group_median = df.groupby("meal_type")["meal_prep_time_min"].transform("median")
    df["meal_prep_time_min"] = df["meal_prep_time_min"].fillna(group_median)
    df["meal_prep_time_min"] = df["meal_prep_time_min"].fillna(df["meal_prep_time_min"].median())

    # roti_count, protein_target_g, fibre_target_g, vitamin_focus — the
    # Layer 1 TARGETS. These are never imputed under any circumstance:
    # inventing a label a model would then be trained on is fabricating
    # ground truth, which is the same category of mistake HC-1 forbids for
    # nutrition numbers, just one step upstream. A missing target is
    # flagged and the row is kept in the output (other columns may still be
    # useful for future analysis) but marked excluded_from_training so
    # supervised training code excludes it explicitly and auditably rather
    # than by accident.
    for col in TARGET_COLUMNS:
        df[f"{col}_missing"] = df[col].isna()
        imputed_counts[col] = 0  # never imputed, by design — see comment above
    df["excluded_from_training"] = df[[f"{c}_missing" for c in TARGET_COLUMNS]].any(axis=1)

    return df, imputed_counts


# ---------------------------------------------------------------------------
# 4. data_quality — marks rows tied to inconsistent logging, distinct from
#    per-column missingness (which is already fully captured by the
#    <col>_missing columns above).
# ---------------------------------------------------------------------------

def add_data_quality_column(df: pd.DataFrame, schema_notes: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    normalized_sources = {src for src, notes in schema_notes.items() if notes}

    dup_key = df.duplicated(subset=["date", "meal_type"], keep=False)

    missing_flag_cols = [c for c in df.columns if c.endswith("_missing")]
    any_missing = df[missing_flag_cols].any(axis=1) if missing_flag_cols else pd.Series(False, index=df.index)

    tags = []
    for i in df.index:
        row_tags = []
        if df.at[i, "source_file"] in normalized_sources:
            row_tags.append("schema_normalized_source")
        if dup_key.at[i]:
            row_tags.append("duplicate_date_meal_type")
        if any_missing.at[i]:
            row_tags.append("had_missing_values")
        tags.append(";".join(row_tags) if row_tags else "clean")
    df["data_quality"] = tags
    return df


# ---------------------------------------------------------------------------
# 5. Pipeline entry point
# ---------------------------------------------------------------------------

def run(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH):
    merged, schema_notes = load_all_sources(raw_dir)
    n_rows_loaded = len(merged)

    merged, n_dropped_bad_date = derive_date_fields(merged)
    merged, imputed_counts = handle_missingness(merged)
    merged = add_data_quality_column(merged, schema_notes)
    merged = merged.sort_values("date").reset_index(drop=True)  # GOVERNANCE.md HC-3: chronological, always

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    return merged, {
        "n_source_files": len(schema_notes),
        "schema_notes": schema_notes,
        "n_rows_loaded": n_rows_loaded,
        "n_dropped_bad_date": n_dropped_bad_date,
        "imputed_counts": imputed_counts,
        "n_excluded_from_training": int(merged["excluded_from_training"].sum()),
        "n_duplicate_conflicts": int(merged["data_quality"].str.contains("duplicate_date_meal_type").sum()),
    }


if __name__ == "__main__":
    df, stats = run()

    print(f"Loaded {stats['n_source_files']} source file(s) from {RAW_DIR}:")
    for src, notes in stats["schema_notes"].items():
        if notes:
            print(f"  - {src}: schema normalized ({len(notes)} change(s))")
            for note in notes:
                print(f"      * {note}")
        else:
            print(f"  - {src}: already matched canonical schema")

    print()
    print(f"Rows loaded (pre-clean): {stats['n_rows_loaded']}")
    print(f"Rows dropped (missing/unparseable date - the only column that can drop a row): {stats['n_dropped_bad_date']}")
    print(f"Rows in output: {len(df)}")
    print()

    print("Imputed cells by column (0 = never imputed by design, e.g. targets):")
    for col, n in stats["imputed_counts"].items():
        print(f"  - {col}: {n}")
    print()

    print(f"Rows excluded_from_training (missing a target column): {stats['n_excluded_from_training']}")
    print(f"Rows flagged duplicate_date_meal_type: {stats['n_duplicate_conflicts']}")
    print()

    print("data_quality value counts:")
    print(df["data_quality"].value_counts().to_string())
    print()
    print(f"Wrote {OUTPUT_PATH}")
