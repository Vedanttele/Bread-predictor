"""Tests for src/features.py. No model is trained anywhere in this suite,
consistent with the module itself."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import features as ft  # noqa: E402

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "merged_clean.csv"


def test_eta_squared_matches_hand_computed_example():
    # Two groups with no within-group variance and a clean between-group
    # gap -> all variance is "explained" by group membership, eta^2 == 1.
    groups = [np.array([1.0, 1.0, 1.0]), np.array([5.0, 5.0, 5.0])]
    grand_mean = np.concatenate(groups).mean()
    assert ft.eta_squared(groups, grand_mean) == pytest.approx(1.0)


def test_eta_squared_zero_when_groups_share_a_mean():
    groups = [np.array([1.0, 3.0]), np.array([1.0, 3.0])]
    grand_mean = np.concatenate(groups).mean()
    assert ft.eta_squared(groups, grand_mean) == pytest.approx(0.0)


def test_correlate_numeric_flags_zero_variance():
    feature = pd.Series([5.0] * 10)
    target = pd.Series(range(10), dtype=float)
    result = ft.correlate_numeric(feature, target)
    assert pd.isna(result["value"])
    assert "not assessable" in result["note"]


def test_correlate_numeric_detects_perfect_correlation():
    feature = pd.Series(range(20), dtype=float)
    target = pd.Series(range(20), dtype=float)
    result = ft.correlate_numeric(feature, target)
    assert result["value"] == pytest.approx(1.0)
    assert result["p_value"] < 0.001


def test_correlate_categorical_single_level_not_assessable():
    feature = pd.Series(["a"] * 10)
    target = pd.Series(range(10), dtype=float)
    result = ft.correlate_categorical(feature, target)
    assert pd.isna(result["value"])


def test_classify_verdict_thresholds():
    assert ft.classify_verdict("pearson_r", 0.6, 0.001) == "signal"
    assert ft.classify_verdict("pearson_r", 0.15, 0.2) == "weak signal"
    assert ft.classify_verdict("pearson_r", 0.05, 0.9) == "noise-like"
    assert ft.classify_verdict("pearson_r", 0.6, 0.5) == "weak signal"  # large effect but not significant
    assert ft.classify_verdict("eta_squared", 0.1, 0.001) == "signal"
    assert ft.classify_verdict("pearson_r", np.nan, np.nan) == "not assessable (no variance)"


def test_vif_higher_for_correlated_columns():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = x1 * 0.95 + rng.normal(scale=0.1, size=n)  # near-collinear with x1
    x3 = rng.normal(size=n)  # independent
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    vif = ft.variance_inflation_factors(X)
    assert vif["x1"] > 5
    assert vif["x2"] > 5
    assert vif["x3"] < 2


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="data/processed/merged_clean.csv not present")
class TestOnRealData:
    @pytest.fixture(scope="class")
    @classmethod
    def df(cls):
        return ft.load_data()

    def test_signal_table_runs_and_has_expected_shape(self, df):
        table = ft.build_signal_table(df, ft.PRIMARY_TARGET)
        expected_features = set(ft.NUMERIC_FEATURES + ft.CATEGORICAL_FEATURES + ft.BINARY_FEATURES)
        assert set(table["feature"]) == expected_features
        assert set(table["verdict"]) <= {"signal", "weak signal", "noise-like", "not assessable (no variance)"}

    def test_constant_missing_flags_are_not_assessable(self, df):
        table = ft.build_signal_table(df, ft.PRIMARY_TARGET)
        flag_rows = table[table["feature"].str.endswith("_missing")]
        assert (flag_rows["verdict"] == "not assessable (no variance)").all()

    def test_calories_and_workout_duration_flagged_as_collinear(self, df):
        collinearity = ft.check_multicollinearity(df)
        assert collinearity["r_calories_burned_vs_workout_duration_min"] > 0.5
        assert collinearity["vif_focused"]["workout_duration_min"] > 5

    def test_encode_for_linear_model_uses_drop_first(self, df):
        encoded = ft.encode_for_linear_model(df)
        n_workout_types = df["workout_type"].nunique()
        workout_dummy_cols = [c for c in encoded.columns if c.startswith("workout_type_")]
        assert len(workout_dummy_cols) == n_workout_types - 1  # drop_first

    def test_encode_for_tree_model_uses_integer_codes(self, df):
        encoded = ft.encode_for_tree_model(df)
        assert "workout_type_code" in encoded.columns
        assert pd.api.types.is_integer_dtype(encoded["workout_type_code"])

    def test_targets_present_in_both_encodings_and_not_used_as_features(self, df):
        linear = ft.encode_for_linear_model(df)
        tree = ft.encode_for_tree_model(df)
        for col in ft.TARGET_COLUMNS:
            assert col in linear.columns
            assert col in tree.columns
        # none of the target columns should have leaked into a dummy/code feature name
        feature_cols = [c for c in linear.columns if c not in ft.TARGET_COLUMNS + ["date", "source_file", "excluded_from_training"]]
        for target in ft.TARGET_COLUMNS:
            assert not any(target in c and c != target for c in feature_cols)

    def test_no_model_object_is_created(self, df):
        """Sanity guard: this module should expose only functions/constants,
        no fitted estimator (no attribute with a .predict method)."""
        for name in dir(ft):
            obj = getattr(ft, name)
            assert not hasattr(obj, "predict"), f"{name} looks like a fitted model — none should exist here"
