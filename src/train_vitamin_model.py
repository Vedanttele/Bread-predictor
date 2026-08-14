"""Vitamin-focus classifier for roti-predictor.

Fills a gap: src/train_model.py only trained roti_count, protein_target_g,
and fibre_target_g — vitamin_focus (the 4th Layer 1 target) was left out.
Adding it here because src/app.py (the Streamlit dashboard) needs a
complete Layer 1 output to display "today's targets" and to feed Layer 2's
recipe_agent, which requires vitamin_focus.

Same feature set, same chronological split, same no-shuffling rule as
train_model.py — this module imports its constants/functions directly
rather than duplicating them. The only real difference is that
vitamin_focus is categorical (5-way: B12, Iron, Multivitamin, Vitamin C,
Vitamin D), so this is classification evaluated by accuracy, not
regression evaluated by MAE.

Run directly:
    python src/train_vitamin_model.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import train_model as tm

MODELS_DIR = tm.MODELS_DIR
TARGET = "vitamin_focus"


def build_pipeline(model_kind: str) -> Pipeline:
    if model_kind == "linear":
        # Multinomial logistic regression — the classification analogue of
        # the LinearRegression baseline in train_model.py.
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), tm.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), tm.CATEGORICAL_FEATURES),
        ])
        estimator = LogisticRegression(max_iter=2000)
    elif model_kind == "gbm":
        preprocessor = ColumnTransformer([
            ("num", "passthrough", tm.NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), tm.CATEGORICAL_FEATURES),
        ])
        estimator = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def evaluate(train_df, test_df) -> dict:
    # Exclude rows missing vitamin_focus itself — never impute a target,
    # same rule as train_model.py.
    train_usable = train_df[~train_df[f"{TARGET}_missing"]]
    test_usable = test_df[~test_df[f"{TARGET}_missing"]]
    X_train, y_train = train_usable[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES], train_usable[TARGET]
    X_test, y_test = test_usable[tm.NUMERIC_FEATURES + tm.CATEGORICAL_FEATURES], test_usable[TARGET]

    results = {"n_train": len(train_usable), "n_test": len(test_usable), "candidates": {}}
    # Majority-class baseline: the accuracy a model has to beat to be worth
    # anything at all, given vitamin_focus's class imbalance.
    majority_class_accuracy = y_test.value_counts(normalize=True).max() if len(y_test) else float("nan")
    results["majority_class_baseline_accuracy"] = majority_class_accuracy

    for kind in ["linear", "gbm"]:
        pipe = build_pipeline(kind)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results["candidates"][kind] = {"pipeline": pipe, "accuracy": accuracy_score(y_test, preds)}
    return results


def select_best(results: dict) -> tuple[str, dict]:
    best_name = max(results["candidates"], key=lambda k: results["candidates"][k]["accuracy"])
    return best_name, results["candidates"][best_name]


def save_model(best_name: str, best_entry: dict, results: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{TARGET}_model.joblib"
    joblib.dump(best_entry["pipeline"], model_path)

    metadata = {
        "target": TARGET,
        "selected_model": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "numeric_features": tm.NUMERIC_FEATURES,
        "categorical_features": tm.CATEGORICAL_FEATURES,
        "majority_class_baseline_accuracy": results["majority_class_baseline_accuracy"],
        "candidate_scores": {k: v["accuracy"] for k, v in results["candidates"].items()},
    }
    (MODELS_DIR / f"{TARGET}_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


if __name__ == "__main__":
    df = tm.load_data()
    train_df, test_df = tm.time_based_split(df)
    print(f"Loaded {len(df)} rows. Train: {len(train_df)}. Test: {len(test_df)}. No shuffling.")
    print()

    results = evaluate(train_df, test_df)
    best_name, best_entry = select_best(results)
    model_path = save_model(best_name, best_entry, results)

    print(f"=== vitamin_focus (n_train={results['n_train']}, n_test={results['n_test']}) ===")
    print(f"  majority-class baseline accuracy: {results['majority_class_baseline_accuracy']:.4f}")
    for name, c in results["candidates"].items():
        print(f"  {name:10s}: accuracy={c['accuracy']:.4f}")
    print(f"  -> selected: {best_name}, saved to {model_path.relative_to(tm.REPO_ROOT)}")
