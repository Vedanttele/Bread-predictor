"""Data ingestion for roti-predictor (Phase 2).

Reads data/raw/daily_context.csv (read-only — GOVERNANCE.md HC-4), validates
it against the contract in data/SCHEMA.md, derives a reliable
day_of_week/is_weekend from `date` (the raw `day_name` column is
known-unreliable — see data/SCHEMA.md "Columns present in the raw file but
dropped at ingestion"), flags missingness explicitly per column
(GOVERNANCE.md HC-2) instead of dropping rows, sorts chronologically, and
writes the result to data/processed/daily_context.csv.

Run directly:
    python -m roti_predictor.data_ingestion
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "daily_context.csv"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "daily_context.csv"

# Every column that may legitimately contain a gap, per data/SCHEMA.md.
# None are observed missing in the current file, but the flagging logic
# below is generic on purpose so it holds for future data that does have
# real gaps (unlogged workouts, skipped meals).
NULLABLE_COLUMNS = [
    "workout_type",
    "workout_duration_min",
    "calories_burned",
    "sleep_hours",
    "mood_score",
    "hunger_level",
    "curry_type",
    "curry_richness",
    "weight_trend",
    "meal_prep_time_min",
    "protein_target_g",
    "fibre_target_g",
    "vitamin_focus",
    "roti_count",
]

TARGET_COLUMNS = ["roti_count", "protein_target_g", "fibre_target_g", "vitamin_focus"]

# Dropped rather than trusted — see data/SCHEMA.md for the mismatch
# analysis (day_name matches the real weekday in only 14/100 of the first
# 100 rows; the trailing-comma-induced 17th column is a partial workaround
# artifact, not a real data column).
RAW_COLUMNS_TO_DROP = ["day_name", "Unnamed: 16"]

# Known vocabularies as of Phase 2. Values outside these are not an error
# (real logging may expand a vocabulary over time) — just surfaced as a
# warning so drift is visible rather than silently absorbed.
KNOWN_CATEGORIES = {
    "workout_type": {"none", "cardio", "strength"},
    "curry_type": {"dal", "dry", "gravy"},
    "curry_richness": {"light", "heavy"},
    "weight_trend": {"maintaining", "losing"},
    "vitamin_focus": {"B12", "Iron", "Multivitamin", "Vitamin C", "Vitamin D"},
}
ORDINAL_RANGES = {"mood_score": (1, 5), "hunger_level": (1, 5)}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. Read-only — never writes back to data/raw/ (HC-4)."""
    if not path.exists():
        raise FileNotFoundError(f"Expected raw data at {path}. See data/raw/README.md.")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing_expected = set(NULLABLE_COLUMNS + ["date"]) - set(df.columns)
    if missing_expected:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing_expected)}")


def validate_values(df: pd.DataFrame) -> list[str]:
    """Non-fatal checks — return human-readable warnings rather than
    raising, since an out-of-contract value should be visible, not silently
    dropped or silently allowed to crash ingestion (GOVERNANCE.md HC-2 in
    spirit: surface anomalies explicitly)."""
    warnings: list[str] = []
    for col, known in KNOWN_CATEGORIES.items():
        unexpected = set(df[col].dropna().unique()) - known
        if unexpected:
            warnings.append(f"{col}: unexpected categories not in schema: {sorted(unexpected)}")
    for col, (lo, hi) in ORDINAL_RANGES.items():
        out_of_range = df[col].dropna()
        out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
        if len(out_of_range):
            warnings.append(f"{col}: {len(out_of_range)} value(s) outside expected [{lo},{hi}]")
    for col in ["workout_duration_min", "calories_burned", "sleep_hours", "meal_prep_time_min",
                "protein_target_g", "fibre_target_g", "roti_count"]:
        negative = df[col].dropna()
        negative = negative[negative < 0]
        if len(negative):
            warnings.append(f"{col}: {len(negative)} negative value(s)")
    return warnings


def derive_date_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `date` and derive day_of_week/is_weekend from it directly,
    dropping the unreliable raw day-name columns. See module docstring."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", dayfirst=True)
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    return df.drop(columns=[c for c in RAW_COLUMNS_TO_DROP if c in df.columns])


def flag_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `<col>_missing` boolean companion column for every nullable
    field (GOVERNANCE.md HC-2) instead of dropping rows with gaps. A no-op
    for the current file (no observed nulls) but runs unconditionally so
    future data with real gaps is handled identically."""
    df = df.copy()
    for col in NULLABLE_COLUMNS:
        df[f"{col}_missing"] = df[col].isna()
    return df


def flag_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows missing a target column can't be used for supervised training.
    They are flagged, not deleted (HC-2) — downstream training code decides
    whether/how to exclude them, and that exclusion stays auditable."""
    df = df.copy()
    df["excluded_from_training"] = df[[f"{c}_missing" for c in TARGET_COLUMNS]].any(axis=1)
    return df


def run(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> tuple[pd.DataFrame, list[str]]:
    df = load_raw(raw_path)
    validate_columns(df)
    warnings = validate_values(df)
    df = derive_date_fields(df)
    df = flag_missingness(df)
    df = flag_unusable_rows(df)
    df = df.sort_values("date").reset_index(drop=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df, warnings


if __name__ == "__main__":
    result, warnings = run()
    n_missing_total = int(sum(result[f"{c}_missing"].sum() for c in NULLABLE_COLUMNS))
    print(f"Ingested {len(result)} rows -> {PROCESSED_PATH}")
    print(f"Date range: {result['date'].min().date()} -> {result['date'].max().date()}")
    print(f"Total flagged-missing cells across all nullable columns: {n_missing_total}")
    print(f"Rows excluded from training (missing a target): {int(result['excluded_from_training'].sum())}")
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No validation warnings.")
