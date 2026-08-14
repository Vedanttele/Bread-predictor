"""Tests for src/train_vitamin_model.py."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_model as tm  # noqa: E402
import train_vitamin_model as tvm  # noqa: E402

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "merged_clean.csv"


@pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="data/processed/merged_clean.csv not present")
class TestOnRealData:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        df = tm.load_data()
        return tm.time_based_split(df)

    def test_beats_majority_class_baseline(self, data):
        train_df, test_df = data
        results = tvm.evaluate(train_df, test_df)
        best_name, best_entry = tvm.select_best(results)
        assert best_entry["accuracy"] > results["majority_class_baseline_accuracy"]

    def test_select_best_picks_higher_accuracy_candidate(self, data):
        train_df, test_df = data
        results = tvm.evaluate(train_df, test_df)
        best_name, best_entry = tvm.select_best(results)
        other = [v["accuracy"] for k, v in results["candidates"].items() if k != best_name]
        assert all(best_entry["accuracy"] >= o for o in other)

    def test_save_and_reload_model(self, data, tmp_path):
        train_df, test_df = data
        results = tvm.evaluate(train_df, test_df)
        best_name, best_entry = tvm.select_best(results)
        original_dir = tvm.MODELS_DIR
        try:
            tvm.MODELS_DIR = tmp_path
            path = tvm.save_model(best_name, best_entry, results)
            assert path.exists()
            loaded = __import__("joblib").load(path)
            preds = loaded.predict(test_df[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES])
            assert len(preds) == len(test_df)
            assert set(preds) <= set(tm.load_data()["vitamin_focus"].dropna().unique())
        finally:
            tvm.MODELS_DIR = original_dir
