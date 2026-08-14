"""Tests for Phase 2 ingestion. Run with: pytest"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from roti_predictor.data_ingestion import (  # noqa: E402
    NULLABLE_COLUMNS,
    RAW_PATH,
    TARGET_COLUMNS,
    run,
)

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="data/raw/daily_context.csv not present")


def test_raw_file_is_never_modified():
    """GOVERNANCE.md HC-4: data/raw/ is read-only."""
    before = RAW_PATH.read_bytes()
    run()
    after = RAW_PATH.read_bytes()
    assert before == after


def test_row_count_preserved():
    """No row is dropped during ingestion (GOVERNANCE.md HC-2)."""
    raw_row_count = sum(1 for _ in RAW_PATH.open(encoding="utf-8")) - 1  # minus header
    df, _ = run()
    assert len(df) == raw_row_count


def test_missingness_flags_present_for_every_nullable_column():
    df, _ = run()
    for col in NULLABLE_COLUMNS:
        assert f"{col}_missing" in df.columns
        assert df[f"{col}_missing"].dtype == bool


def test_excluded_from_training_matches_target_missingness():
    df, _ = run()
    expected = df[[f"{c}_missing" for c in TARGET_COLUMNS]].any(axis=1)
    assert (df["excluded_from_training"] == expected).all()


def test_unreliable_raw_columns_are_dropped():
    df, _ = run()
    assert "day_name" not in df.columns
    assert "Unnamed: 16" not in df.columns


def test_day_of_week_derived_from_date():
    df, _ = run()
    assert "day_of_week" in df.columns
    assert "is_weekend" in df.columns
    recomputed = pd.to_datetime(df["date"]).dt.day_name()
    assert (df["day_of_week"] == recomputed).all()


def test_sorted_chronologically():
    df, _ = run()
    dates = pd.to_datetime(df["date"])
    assert dates.is_monotonic_increasing
