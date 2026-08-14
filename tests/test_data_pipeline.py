"""Tests for src/data_pipeline.py.

Exercises the multi-schema-normalization path with synthetic files in a
tmp_path raw dir (aliased column names, a missing meal_type column, a bad
date, missing targets, missing workout_type) since the one real file in
data/raw/ has no missingness to exercise those branches against. Also
checks the real data/raw/ directory is never modified.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_pipeline as dp  # noqa: E402

REAL_RAW_DIR = REPO_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Synthetic multi-schema fixture
# ---------------------------------------------------------------------------

SOURCE_A = """date,day_name,workout_type,workout_duration_min,calories_burned,sleep_hours,mood_score,hunger_level,curry_type,curry_richness,weight_trend,meal_prep_time_min,protein_target_g,fibre_target_g,vitamin_focus,roti_count
01/01/2025,Wednesday,cardio,30,2100,7.5,4,3,dal,light,maintaining,20,40,10,Iron,2
02/01/2025,Thursday,none,0,1900,,3,2,dry,heavy,maintaining,15,38,9.5,B12,1
03/01/2025,Friday,strength,45,2200,7.0,,4,gravy,heavy,,25,44,11,Vitamin C,3
not-a-date,Saturday,cardio,20,2000,8.0,5,3,dal,light,losing,10,20,5,B12,1
06/01/2025,Monday,cardio,,2150,7.1,4,3,dal,light,maintaining,20,,,,
"""

# File B uses a different schema on purpose: log_date (alias for date),
# day_type (weekday/weekend bucket, deliberately ignored), meal (alias for
# meal_type, and this file DOES have it, unlike file A), workout/calories
# aliases, and a genuinely missing workout_type.
SOURCE_B = """log_date,day_type,meal,workout,workout_duration_min,calories,sleep_hours,mood_score,hunger_level,curry_type,curry_richness,weight_trend,meal_prep_time_min,protein_target_g,fibre_target_g,vitamin_focus,roti_count
04/01/2025,weekend,lunch,,25,2050,6.5,3,3,dal,light,losing,20,39,9.8,Iron,2
05/01/2025,weekday,dinner,strength,50,2300,7.2,4,4,gravy,heavy,losing,30,45,11.5,Vitamin D,3
"""


@pytest.fixture
def synthetic_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "source_a.csv").write_text(SOURCE_A, encoding="utf-8")
    (raw_dir / "source_b.csv").write_text(SOURCE_B, encoding="utf-8")
    return raw_dir


def test_loads_multiple_sources_with_different_schemas(synthetic_raw_dir, tmp_path):
    df, stats = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    assert stats["n_source_files"] == 2
    # source_b needed alias normalization (log_date->date, meal->meal_type, etc.)
    assert stats["schema_notes"]["source_b.csv"]
    # source_a is missing meal_type entirely -> noted
    assert any("meal_type" in n for n in stats["schema_notes"]["source_a.csv"])


def test_bad_date_row_is_dropped_nothing_else_is(synthetic_raw_dir, tmp_path):
    df, stats = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    # 5 rows in source_a + 2 in source_b = 7 loaded, 1 bad date dropped -> 6 out
    assert stats["n_rows_loaded"] == 7
    assert stats["n_dropped_bad_date"] == 1
    assert len(df) == 6


def test_day_of_week_derived_from_date_not_raw_columns(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    assert "day_name" not in df.columns
    assert "day_type" not in df.columns
    recomputed = pd.to_datetime(df["date"]).dt.day_name()
    assert (df["day_of_week"] == recomputed).all()


def test_meal_type_imputed_as_unspecified_when_absent(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    a_rows = df[df["source_file"] == "source_a.csv"]
    assert (a_rows["meal_type"] == "unspecified").all()
    assert a_rows["meal_type_missing"].all()
    b_rows = df[df["source_file"] == "source_b.csv"]
    assert set(b_rows["meal_type"]) == {"lunch", "dinner"}
    assert not b_rows["meal_type_missing"].any()


def test_workout_type_missing_is_unknown_not_none(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    row = df[(df["source_file"] == "source_b.csv") & (df["meal_type"] == "lunch")].iloc[0]
    assert row["workout_type"] == "unknown"
    assert row["workout_type_missing"]


def test_workout_duration_zero_for_rest_day_median_otherwise(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    rest_day = df[(df["source_file"] == "source_a.csv") & (df["workout_type"] == "none")].iloc[0]
    assert rest_day["workout_duration_min"] == 0
    # 06/01/2025 has workout_type='cardio' (a real workout, not 'none') but
    # no logged duration -> must fall into the median-imputation branch,
    # not the rest-day-zero branch.
    had_workout_but_no_duration = df[
        (df["source_file"] == "source_a.csv") & (pd.to_datetime(df["date"]) == pd.Timestamp("2025-01-06"))
    ].iloc[0]
    assert had_workout_but_no_duration["workout_type"] == "cardio"
    assert had_workout_but_no_duration["workout_duration_min_missing"]
    assert pd.notna(had_workout_but_no_duration["workout_duration_min"])
    assert had_workout_but_no_duration["workout_duration_min"] != 0


def test_targets_never_imputed_and_excluded_from_training_flagged(synthetic_raw_dir, tmp_path):
    df, stats = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    for col in dp.TARGET_COLUMNS:
        assert stats["imputed_counts"][col] == 0
    excluded = df[df["excluded_from_training"]]
    assert len(excluded) >= 1
    assert excluded["roti_count"].isna().all() or excluded["roti_count_missing"].any()


def test_weight_trend_forward_filled_within_source(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    a_rows = df[df["source_file"] == "source_a.csv"].sort_values("date")
    # 03/01 had a blank weight_trend in the raw fixture -> should carry
    # forward 'maintaining' from 02/01 (the previous row chronologically).
    row_03_01 = a_rows[pd.to_datetime(a_rows["date"]) == pd.Timestamp("2025-01-03")].iloc[0]
    assert row_03_01["weight_trend"] == "maintaining"
    assert row_03_01["weight_trend_missing"]


def test_data_quality_column_marks_normalized_and_missing_rows(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    b_rows = df[df["source_file"] == "source_b.csv"]
    assert (b_rows["data_quality"].str.contains("schema_normalized_source")).all()


def test_output_sorted_chronologically(synthetic_raw_dir, tmp_path):
    df, _ = dp.run(raw_dir=synthetic_raw_dir, output_path=tmp_path / "out.csv")
    assert pd.to_datetime(df["date"]).is_monotonic_increasing


# ---------------------------------------------------------------------------
# Regression: mixed date formats across sources.
#
# pandas.to_datetime, given no explicit format, infers ONE format from
# early values and applies it to the whole column. Sources are concatenated
# before dates are parsed (see load_all_sources), so if one source's dates
# happen to sort alphabetically before another's and uses a different
# format, the wrong format silently "wins" and every row in the other
# format fails to parse. This is not hypothetical — it dropped 220 of 221
# real rows in production before format="mixed" was added to
# derive_date_fields. File names below are chosen so the ISO-format file
# sorts first (matching the real incident: app_actuals.csv < daily_context.csv).
# ---------------------------------------------------------------------------

ISO_DATES_SOURCE = """date,workout_type,calories_burned,sleep_hours,hunger_level,curry_type,curry_richness,meal_prep_time_min,protein_target_g,fibre_target_g,vitamin_focus,roti_count
2026-08-14,none,2100,7.0,3,dal,light,20,40.0,10.0,Iron,2
"""

SLASH_DATES_SOURCE = """date,workout_type,calories_burned,sleep_hours,hunger_level,curry_type,curry_richness,meal_prep_time_min,protein_target_g,fibre_target_g,vitamin_focus,roti_count
15/08/2025,cardio,2100,7.5,3,dal,light,20,40.0,10.0,Iron,2
16/08/2025,strength,2200,7.0,4,dry,heavy,15,42.0,9.5,B12,3
17/08/2025,none,1900,8.0,2,gravy,light,30,38.0,10.5,Vitamin C,2
"""


@pytest.fixture
def mixed_date_format_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # "a_iso..." sorts before "b_slash..." alphabetically, reproducing the
    # ordering that caused the original bug.
    (raw_dir / "a_iso_format.csv").write_text(ISO_DATES_SOURCE, encoding="utf-8")
    (raw_dir / "b_slash_format.csv").write_text(SLASH_DATES_SOURCE, encoding="utf-8")
    return raw_dir


def test_mixed_date_formats_across_sources_all_parse(mixed_date_format_raw_dir, tmp_path):
    df, stats = dp.run(raw_dir=mixed_date_format_raw_dir, output_path=tmp_path / "out.csv")
    # 1 ISO-format row + 3 slash-format rows = 4 total; none should be
    # dropped as unparseable.
    assert stats["n_rows_loaded"] == 4
    assert stats["n_dropped_bad_date"] == 0
    assert len(df) == 4


def test_mostly_unparseable_dates_raises_instead_of_silently_continuing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    garbage = """date,workout_type,calories_burned,sleep_hours,hunger_level,curry_type,curry_richness,meal_prep_time_min,protein_target_g,fibre_target_g,vitamin_focus,roti_count
not-a-date-1,none,2100,7.0,3,dal,light,20,40.0,10.0,Iron,2
not-a-date-2,none,2100,7.0,3,dal,light,20,40.0,10.0,Iron,2
15/08/2025,cardio,2100,7.5,3,dal,light,20,40.0,10.0,Iron,2
"""
    (raw_dir / "mostly_garbage.csv").write_text(garbage, encoding="utf-8")
    with pytest.raises(ValueError, match="date parsing"):
        dp.run(raw_dir=raw_dir, output_path=tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# Guard against touching the real data/raw/ directory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_RAW_DIR.exists(), reason="data/raw/ not present")
def test_real_raw_dir_never_modified(tmp_path):
    before = {p.name: p.read_bytes() for p in REAL_RAW_DIR.glob("*.csv")}
    dp.run(raw_dir=REAL_RAW_DIR, output_path=tmp_path / "out.csv")
    after = {p.name: p.read_bytes() for p in REAL_RAW_DIR.glob("*.csv")}
    assert before == after
