"""Tests for src/train_model.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_model as tm  # noqa: E402

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "merged_clean.csv"


def test_time_based_split_is_chronological_and_contiguous():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=100, freq="D"), "x": range(100)})
    train, test = tm.time_based_split(df, test_fraction=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train["date"].max() < test["date"].min()
    # contiguous: no gap, no overlap
    assert (test["date"].min() - train["date"].max()).days == 1


def test_time_based_split_does_not_shuffle():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=50, freq="D"), "x": range(50)})
    train, test = tm.time_based_split(df, test_fraction=0.2)
    assert train["x"].tolist() == sorted(train["x"].tolist())
    assert test["x"].tolist() == sorted(test["x"].tolist())


def test_heuristic_roti_count_examples():
    """User-approved reference formula — pin down its behavior so a future
    edit to the rules is a deliberate, visible change, not an accident."""
    rest_day_low_hunger_dal = pd.Series({
        "hunger_level": 2, "workout_type": "none", "curry_richness": "light", "curry_type": "dal",
    })
    assert tm.heuristic_roti_count(rest_day_low_hunger_dal) == 1  # 2 -1(hunger) -1(dal) = 0 -> clipped to 1

    hungry_workout_heavy_gravy = pd.Series({
        "hunger_level": 5, "workout_type": "cardio", "curry_richness": "heavy", "curry_type": "gravy",
    })
    assert tm.heuristic_roti_count(hungry_workout_heavy_gravy) == 5  # 2 +1 +1 +1 = 5

    average_day = pd.Series({
        "hunger_level": 3, "workout_type": "none", "curry_richness": "light", "curry_type": "gravy",
    })
    assert tm.heuristic_roti_count(average_day) == 2  # no adjustments trigger


def test_heuristic_slot_is_wired_to_the_named_function():
    assert tm.HEURISTIC_ROTI_COUNT_FN is tm.heuristic_roti_count


def test_select_best_picks_heuristic_when_it_genuinely_wins():
    """Synthetic case — on the real dataset ML currently wins, so this
    guards the untriggered branch: select_best must not have a hidden bias
    toward the ML candidates."""
    results = {
        "target": "roti_count",
        "candidates": {
            "linear": {"mae_raw": 0.9, "mae_rounded": 0.8},
            "gbm": {"mae_raw": 1.0, "mae_rounded": 0.95},
            "heuristic": {"mae_raw": 0.3, "mae_rounded": 0.3},
        },
    }
    best_name, _ = tm.select_best(results)
    assert best_name == "heuristic"


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="data/processed/merged_clean.csv not present")
class TestOnRealData:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        df = tm.load_data()
        return tm.time_based_split(df)

    def test_no_leakage_scaler_fit_on_train_only(self, data):
        train_df, test_df = data
        pipe = tm.build_pipeline("linear")
        X_train = train_df[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES]
        y_train = train_df["roti_count"]
        pipe.fit(X_train, y_train)
        scaler = pipe.named_steps["preprocess"].named_transformers_["num"]
        expected_mean = train_df[tm.NUMERIC_FEATURES].mean().to_numpy()
        assert np.allclose(scaler.mean_, expected_mean)
        # and NOT the full-dataset mean, which would indicate leakage
        full_df = pd.concat([train_df, test_df])
        full_mean = full_df[tm.NUMERIC_FEATURES].mean().to_numpy()
        assert not np.allclose(scaler.mean_, full_mean)

    def test_evaluate_target_roti_count_scores_heuristic(self, data):
        train_df, test_df = data
        results = tm.evaluate_target("roti_count", train_df, test_df)
        assert "pending" not in results["candidates"]["heuristic"]
        assert results["candidates"]["heuristic"]["mae_raw"] > 0
        assert "linear" in results["candidates"]
        assert "gbm" in results["candidates"]
        assert results["candidates"]["linear"]["mae_raw"] > 0

    def test_select_best_can_choose_any_candidate_on_real_data(self, data):
        train_df, test_df = data
        results = tm.evaluate_target("roti_count", train_df, test_df)
        best_name, _ = tm.select_best(results)
        assert best_name in ("linear", "gbm", "heuristic")

    def test_roti_count_rounded_predictions_within_bounds(self, data):
        train_df, test_df = data
        results = tm.evaluate_target("roti_count", train_df, test_df)
        pipe = results["candidates"]["linear"]["pipeline"]
        preds = pipe.predict(test_df[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES])
        rounded = np.clip(np.round(preds), *tm.ROTI_COUNT_BOUNDS)
        assert rounded.min() >= tm.ROTI_COUNT_BOUNDS[0]
        assert rounded.max() <= tm.ROTI_COUNT_BOUNDS[1]

    def test_save_and_load_model_roundtrip(self, data, tmp_path):
        train_df, test_df = data
        results = tm.evaluate_target("protein_target_g", train_df, test_df)
        best_name, best_entry = tm.select_best(results)
        original_dir = tm.MODELS_DIR
        try:
            tm.MODELS_DIR = tmp_path
            path = tm.save_model("protein_target_g", best_name, best_entry, results)
            assert path.exists()
            loaded = __import__("joblib").load(path)
            preds = loaded.predict(test_df[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES])
            assert len(preds) == len(test_df)
            metadata_path = tmp_path / "protein_target_g_model_metadata.json"
            assert metadata_path.exists()
        finally:
            tm.MODELS_DIR = original_dir

    def test_workout_duration_min_not_in_feature_set(self, data):
        """Documented exclusion (collinearity with calories_burned /
        workout_type) — guard against it silently creeping back in."""
        assert "workout_duration_min" not in tm.NUMERIC_FEATURES
        assert "workout_duration_min" not in tm.CATEGORICAL_FEATURES

    def test_targets_excluded_from_own_training_when_missing(self, data):
        train_df, test_df = data
        train_df = train_df.copy()
        train_df.loc[train_df.index[0], "roti_count_missing"] = True
        results = tm.evaluate_target("roti_count", train_df, test_df)
        assert results["n_train"] == len(train_df) - 1
