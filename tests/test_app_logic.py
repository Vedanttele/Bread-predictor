"""Tests for src/app_logic.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import app_logic as al  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"


def make_recipe(**overrides):
    base = {
        "name": "Test Dal", "cuisine": "Indian", "diet_type": "vegetarian",
        "main_ingredients": "lentils, tomato", "protein_g": 15.0, "fibre_g": 8.0,
        "calories": 220, "prep_time_min": 20, "vitamin_tags": ["Iron"],
        "pairs_with_roti": True, "suitable_meal": "dinner",
    }
    base.update(overrides)
    return base


SAMPLE_INPUTS = {
    "calories_burned": 2100, "hunger_level": 3, "sleep_hours": 7.0,
    "workout_type": "cardio", "curry_type": "dal", "curry_richness": "light",
    "meal_prep_time_min": 20,
}
SAMPLE_TARGET = {
    "roti_count": 3, "protein_target_g": 44.0, "fibre_target_g": 11.0, "vitamin_focus": "Iron",
}


def test_build_feature_row_has_exactly_the_model_features():
    row = al.build_feature_row(SAMPLE_INPUTS)
    assert list(row.columns) == al.NUMERIC_FEATURES + al.CATEGORICAL_FEATURES
    assert len(row) == 1


def test_lookup_recipe_finds_by_name():
    recipes = [make_recipe(name="A"), make_recipe(name="B")]
    assert al.lookup_recipe("B", recipes)["name"] == "B"


def test_lookup_recipe_raises_for_unknown_name():
    with pytest.raises(ValueError):
        al.lookup_recipe("Nonexistent", [make_recipe(name="A")])


def test_vitamin_focus_matched_true():
    recipe = make_recipe(vitamin_tags=["Iron", "Vitamin C"])
    assert al.vitamin_focus_matched("Iron", recipe) is True


def test_vitamin_focus_matched_false():
    recipe = make_recipe(vitamin_tags=["Vitamin D"])
    assert al.vitamin_focus_matched("Iron", recipe) is False


def test_build_log_row_actual_nutrition_comes_from_recipe_not_invented():
    recipe = make_recipe(name="Chana Masala", protein_g=15.5, fibre_g=10.5, vitamin_tags=["Iron", "Vitamin C"])
    row = al.build_log_row(
        date="2026-08-14", inputs=SAMPLE_INPUTS, target=SAMPLE_TARGET,
        recommended_recipe_name="Chana Masala", actual_roti_count=2, actual_recipe=recipe,
    )
    assert row["actual_protein_g"] == 15.5
    assert row["actual_fibre_g"] == 10.5
    assert row["actual_recipe_name"] == "Chana Masala"
    assert row["vitamin_match"] is True


def test_build_log_row_deltas_computed_correctly():
    recipe = make_recipe(protein_g=40.0, fibre_g=9.0)
    row = al.build_log_row(
        date="2026-08-14", inputs=SAMPLE_INPUTS, target=SAMPLE_TARGET,
        recommended_recipe_name=None, actual_roti_count=2, actual_recipe=recipe,
    )
    assert row["delta_roti_count"] == 2 - 3
    assert row["delta_protein_g"] == pytest.approx(40.0 - 44.0)
    assert row["delta_fibre_g"] == pytest.approx(9.0 - 11.0)


def test_build_log_row_vitamin_tags_joined_as_string():
    recipe = make_recipe(vitamin_tags=["Iron", "B12"])
    row = al.build_log_row(
        date="2026-08-14", inputs=SAMPLE_INPUTS, target=SAMPLE_TARGET,
        recommended_recipe_name=None, actual_roti_count=3, actual_recipe=recipe,
    )
    assert row["actual_vitamin_tags"] == "Iron;B12"


def test_append_and_load_log_roundtrip(tmp_path):
    log_path = tmp_path / "meal_log.csv"
    recipe = make_recipe(name="R1")
    row = al.build_log_row(
        date="2026-08-14", inputs=SAMPLE_INPUTS, target=SAMPLE_TARGET,
        recommended_recipe_name="R1", actual_roti_count=3, actual_recipe=recipe,
    )
    al.append_log_row(row, log_path=log_path)
    df = al.load_log(log_path)
    assert len(df) == 1
    assert df.iloc[0]["actual_recipe_name"] == "R1"


def test_append_log_row_appends_not_overwrites(tmp_path):
    log_path = tmp_path / "meal_log.csv"
    recipe = make_recipe(name="R1")
    row1 = al.build_log_row("2026-08-14", SAMPLE_INPUTS, SAMPLE_TARGET, "R1", 3, recipe)
    row2 = al.build_log_row("2026-08-15", SAMPLE_INPUTS, SAMPLE_TARGET, "R1", 2, recipe)
    al.append_log_row(row1, log_path=log_path)
    al.append_log_row(row2, log_path=log_path)
    df = al.load_log(log_path)
    assert len(df) == 2


def test_load_log_returns_empty_frame_with_columns_when_no_file(tmp_path):
    df = al.load_log(tmp_path / "does_not_exist.csv")
    assert df.empty
    assert list(df.columns) == al.LOG_COLUMNS


def test_load_log_sorted_chronologically(tmp_path):
    log_path = tmp_path / "meal_log.csv"
    recipe = make_recipe(name="R1")
    row_late = al.build_log_row("2026-08-16", SAMPLE_INPUTS, SAMPLE_TARGET, "R1", 3, recipe)
    row_early = al.build_log_row("2026-08-14", SAMPLE_INPUTS, SAMPLE_TARGET, "R1", 3, recipe)
    al.append_log_row(row_late, log_path=log_path)
    al.append_log_row(row_early, log_path=log_path)
    df = al.load_log(log_path)
    assert df["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Raw feedback (data/raw/app_actuals.csv) for retraining
# ---------------------------------------------------------------------------

def test_raw_feedback_row_uses_actual_recipe_not_target():
    recipe = make_recipe(name="Chana Masala", protein_g=15.5, fibre_g=10.5, vitamin_tags=["Iron", "Vitamin C"])
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, actual_roti_count=2, actual_recipe=recipe)
    # protein/fibre come from the eaten recipe, NOT SAMPLE_TARGET's 44.0/11.0
    assert row["protein_target_g"] == 15.5
    assert row["fibre_target_g"] == 10.5
    assert row["roti_count"] == 2


def test_raw_feedback_row_vitamin_focus_is_first_tag():
    recipe = make_recipe(vitamin_tags=["Iron", "B12"])
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, recipe)
    assert row["vitamin_focus"] == "Iron"


def test_raw_feedback_row_empty_tags_gives_empty_vitamin_focus():
    recipe = make_recipe(vitamin_tags=[])
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, recipe)
    assert row["vitamin_focus"] == ""


def test_raw_feedback_row_matches_expected_schema_columns():
    recipe = make_recipe()
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, recipe)
    assert set(row.keys()) == set(al.RAW_FEEDBACK_COLUMNS)


def test_raw_feedback_row_omits_uncollected_columns():
    """workout_duration_min, mood_score, weight_trend are never in the
    row — they're absent, not guessed."""
    recipe = make_recipe()
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, recipe)
    assert "workout_duration_min" not in row
    assert "mood_score" not in row
    assert "weight_trend" not in row


def test_append_raw_feedback_row_appends_not_overwrites(tmp_path):
    path = tmp_path / "app_actuals.csv"
    recipe = make_recipe()
    row1 = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, recipe)
    row2 = al.build_raw_feedback_row("2026-08-15", SAMPLE_INPUTS, 2, recipe)
    al.append_raw_feedback_row(row1, path=path)
    al.append_raw_feedback_row(row2, path=path)
    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df.columns) == al.RAW_FEEDBACK_COLUMNS


def test_append_raw_feedback_row_never_touches_daily_context_csv(tmp_path):
    daily_context = tmp_path / "daily_context.csv"
    daily_context.write_text("date,roti_count\n2025-08-15,3\n", encoding="utf-8")
    before = daily_context.read_bytes()

    app_actuals = tmp_path / "app_actuals.csv"
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, make_recipe())
    al.append_raw_feedback_row(row, path=app_actuals)

    assert daily_context.read_bytes() == before


REAL_DAILY_CONTEXT = REPO_ROOT / "data" / "raw" / "daily_context.csv"


@pytest.mark.skipif(not REAL_DAILY_CONTEXT.exists(), reason="data/raw/daily_context.csv not present")
def test_real_daily_context_csv_never_modified_by_raw_feedback(tmp_path):
    before = REAL_DAILY_CONTEXT.read_bytes()
    row = al.build_raw_feedback_row("2026-08-14", SAMPLE_INPUTS, 3, make_recipe())
    al.append_raw_feedback_row(row, path=tmp_path / "app_actuals.csv")
    after = REAL_DAILY_CONTEXT.read_bytes()
    assert before == after


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _sample_trend_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-08-01", "2026-08-02", "2026-08-03"]),
        "target_roti_count": [3, 2, 4],
        "actual_roti_count": [2, 2, 3],
        "target_protein_target_g": [42.0, 38.0, 50.0],
        "actual_protein_g": [15.5, 14.0, 17.0],
    })


def test_build_trend_chart_has_two_series_correct_colors():
    df = _sample_trend_df()
    fig = al.build_trend_chart(df, "target_roti_count", "actual_roti_count", "Rotis")
    assert len(fig.data) == 2
    assert fig.data[0].name == "Target"
    assert fig.data[0].marker.color == al.COLOR_TARGET
    assert fig.data[1].name == "Actual"
    assert fig.data[1].marker.color == al.COLOR_ACTUAL


def test_build_pct_of_target_chart_computes_correct_percentages():
    df = _sample_trend_df()
    fig = al.build_pct_of_target_chart(df, "target_protein_target_g", "actual_protein_g", "protein")
    # 15.5/42.0*100 = 36.9, 14.0/38.0*100 = 36.8, 17.0/50.0*100 = 34.0
    assert list(fig.data[0].y) == pytest.approx([36.9, 36.8, 34.0], abs=0.1)


def test_build_pct_of_target_chart_is_single_series_with_reference_line():
    df = _sample_trend_df()
    fig = al.build_pct_of_target_chart(df, "target_protein_target_g", "actual_protein_g", "protein")
    assert len(fig.data) == 1  # not a misleading two-bar comparison
    assert fig.data[0].marker.color == al.COLOR_ACTUAL
    # the 100% reference line is a layout shape, not a second data series
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].y0 == 100


def test_build_pct_of_target_chart_never_exceeds_reasonable_bounds_for_protein_data():
    """Regression guard for the actual scale mismatch this chart exists to
    fix: with real recipe_database.json values (max 19.0g protein) against
    real training-data targets (min 31.2g), % of target should always be
    well under 100% — if this ever shows >100%+ regularly, the underlying
    scale assumption this chart documents has changed and the chart choice
    should be revisited."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-01"]),
        "target_protein_target_g": [31.2],  # lowest observed target
        "actual_protein_g": [19.0],  # highest possible recipe protein
    })
    fig = al.build_pct_of_target_chart(df, "target_protein_target_g", "actual_protein_g", "protein")
    assert fig.data[0].y[0] < 100  # even the best case never reaches the target


# ---------------------------------------------------------------------------
# Real trained models
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODELS_DIR.exists() or not (MODELS_DIR / "vitamin_focus_model.joblib").exists(),
                     reason="trained models not present")
class TestWithRealModels:
    def test_load_models_returns_all_four(self):
        models = al.load_models()
        assert set(models.keys()) == set(al.TARGET_MODEL_FILES.keys())

    def test_predict_layer1_returns_complete_valid_output(self):
        models = al.load_models()
        row = al.build_feature_row(SAMPLE_INPUTS)
        result = al.predict_layer1(models, row)
        assert set(result.keys()) == {"roti_count", "protein_target_g", "fibre_target_g", "vitamin_focus"}
        assert al.ROTI_COUNT_BOUNDS[0] <= result["roti_count"] <= al.ROTI_COUNT_BOUNDS[1]
        assert isinstance(result["roti_count"], int)
        assert result["protein_target_g"] > 0
        assert result["fibre_target_g"] > 0
        assert isinstance(result["vitamin_focus"], str)

    def test_load_models_raises_clear_error_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing trained model"):
            al.load_models(models_dir=tmp_path)
