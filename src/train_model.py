"""Layer 1 model training for roti-predictor.

Loads data/processed/merged_clean.csv, applies a single chronological
train/test cutoff (last 20% of days, no shuffling — GOVERNANCE.md HC-3),
trains a linear baseline and a gradient-boosted-trees model for
`roti_count`, `protein_target_g`, and `fibre_target_g`, compares
`roti_count` against a manual heuristic reference function, reports test
MAE for all candidates per target, and saves the best-performing model per
target to models/.

Feature set (see notebooks/feature_analysis_report.md for the full
signal-vs-noise analysis this is drawn from):
    numeric:     calories_burned, hunger_level, sleep_hours
    categorical: workout_type, curry_type, curry_richness
`workout_duration_min` is deliberately EXCLUDED despite showing signal on
its own (r=0.47) — it's highly collinear with calories_burned (r=0.86) and
had the highest VIF of any feature (9.25, borderline severe). calories_burned
is the stronger, cleaner signal (r=0.58) and workout_type already captures
the categorical "what kind of workout" information, so keeping
workout_duration_min alongside both would mostly add redundant variance
rather than new signal, and would destabilize the linear model's
coefficients. `mood_score`, `is_weekend`, `meal_prep_time_min`, `weight_trend`
were noise-like; `day_of_week` was weak and not significant (p=0.081);
`meal_type` is constant in the current data (no source logs it) — all
excluded.

Compares `roti_count` against a manual heuristic reference function
(see heuristic_roti_count() below): a baseline of 2, adjusted +/-1 for
hunger extremes, +1 for any workout, +1 for heavy curry, -1 for dal.
Rules reflect common-sense direction only, not fitted to this dataset..

Run directly:
    python src/train_model.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "merged_clean.csv"
MODELS_DIR = REPO_ROOT / "models"
REPORT_PATH = REPO_ROOT / "notebooks" / "model_training_report.md"

NUMERIC_FEATURES = ["calories_burned", "hunger_level", "sleep_hours"]
CATEGORICAL_FEATURES = ["workout_type", "curry_type", "curry_richness"]
TARGETS = ["roti_count", "protein_target_g", "fibre_target_g"]
TEST_FRACTION = 0.20
ROTI_COUNT_BOUNDS = (1, 5)  # observed range in data/SCHEMA.md / Phase 2 EDA

# --- Heuristic slot -----------------------------------------------------
# Manual reference formula (user-approved), built from plain common-sense
# rules rather than fit to this dataset's exact numbers — tuning the +1/-1
# sizes to minimize error here would turn it into a crude copy of whatever
# the ML models learn, which would defeat the point of an independent
# baseline. Only the *direction* of each rule (more hunger -> more rotis,
# a workout -> more rotis, richer curry -> more rotis, dal -> fewer) is a
# common-sense judgment call, not something mined from the data.
def heuristic_roti_count(row: pd.Series) -> float:
    count = 2  # baseline: an average meal

    # more/less hungry than usual -> eat more/fewer rotis
    if row["hunger_level"] >= 4:
        count += 1
    elif row["hunger_level"] <= 2:
        count -= 1

    # worked out -> burned more energy, eat more
    if row["workout_type"] != "none":
        count += 1

    # heavier/richer curry -> more roti for mopping it up
    if row["curry_richness"] == "heavy":
        count += 1

    # dal is often eaten lighter / with rice instead -> fewer rotis
    if row["curry_type"] == "dal":
        count -= 1

    return max(1, min(5, count))


# Signature: fn(row: pd.Series) -> float, taking a row of the ORIGINAL
# (unencoded) merged_clean.csv columns and returning a roti_count estimate.
# Wired into evaluate_target() below.
HEURISTIC_ROTI_COUNT_FN = heuristic_roti_count


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}. Run src/data_pipeline.py first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One chronological cutoff computed over the FULL sorted dataset,
    reused for every target. Filtering out a target's missing rows before
    splitting would let each target pick its own cutoff date, making
    cross-target comparisons date-inconsistent and defeating the point of
    a single held-out time window (GOVERNANCE.md HC-3). No shuffling
    anywhere in this function, deliberately — sklearn's train_test_split
    is not used here so shuffle=True can't be accidentally left on."""
    n_test = round(len(df) * test_fraction)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    return train_df, test_df


def build_pipeline(model_kind: str) -> Pipeline:
    if model_kind == "linear":
        # StandardScaler + OneHotEncoder(drop='first') are fit inside the
        # Pipeline, i.e. only on whatever data .fit() is called with (the
        # train split) — never on the full dataset. Fitting a scaler on
        # train+test would leak test-set mean/variance into training.
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ])
        estimator = LinearRegression()
    elif model_kind == "gbm":
        # Trees don't need scaling; ordinal-coded categories are fine since
        # a tree splits on thresholds rather than assuming a linear
        # relationship between the code and the target. Shallow depth and
        # modest n_estimators to reduce overfitting risk on ~176 train rows.
        preprocessor = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ])
        estimator = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def evaluate_target(target: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    # Exclude rows missing THIS target (never impute a target — see
    # data_pipeline.py). A no-op today (no missingness observed) but
    # required for correctness if a future data refresh has gaps.
    train_usable = train_df[~train_df[f"{target}_missing"]]
    test_usable = test_df[~test_df[f"{target}_missing"]]

    X_train, y_train = train_usable[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_usable[target]
    X_test, y_test = test_usable[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_usable[target]

    results = {"target": target, "n_train": len(train_usable), "n_test": len(test_usable), "candidates": {}}

    for kind in ["linear", "gbm"]:
        pipe = build_pipeline(kind)
        pipe.fit(X_train, y_train)
        raw_pred = pipe.predict(X_test)
        mae_raw = mean_absolute_error(y_test, raw_pred)
        entry = {"pipeline": pipe, "mae_raw": mae_raw}

        if target == "roti_count":
            rounded_pred = np.clip(np.round(raw_pred), *ROTI_COUNT_BOUNDS)
            entry["mae_rounded"] = mean_absolute_error(y_test, rounded_pred)
        results["candidates"][kind] = entry

    if target == "roti_count":
        if HEURISTIC_ROTI_COUNT_FN is not None:
            heuristic_pred = test_usable.apply(HEURISTIC_ROTI_COUNT_FN, axis=1).to_numpy(dtype=float)
            results["candidates"]["heuristic"] = {
                "pipeline": None,
                "mae_raw": mean_absolute_error(y_test, heuristic_pred),
                "mae_rounded": mean_absolute_error(y_test, np.clip(np.round(heuristic_pred), *ROTI_COUNT_BOUNDS)),
            }
        else:
            results["candidates"]["heuristic"] = {"pipeline": None, "mae_raw": None, "mae_rounded": None, "pending": True}

    return results


def select_best(results: dict) -> tuple[str, dict]:
    """Best = lowest test MAE among candidates with an actual number.
    A pending heuristic is never eligible to "win" by omission — it's
    excluded from selection until it has a real score, and that's called
    out explicitly wherever this result is reported."""
    scored = {
        name: c["mae_rounded"] if "mae_rounded" in c and c["mae_rounded"] is not None else c.get("mae_raw")
        for name, c in results["candidates"].items()
        if c.get("mae_raw") is not None
    }
    best_name = min(scored, key=scored.get)
    return best_name, results["candidates"][best_name]


def save_model(target: str, best_name: str, best_entry: dict, results: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{target}_model.joblib"
    joblib.dump(best_entry["pipeline"], model_path)

    metadata = {
        "target": target,
        "selected_model": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": results["n_train"],
        "n_test": results["n_test"],
        "test_fraction": TEST_FRACTION,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "candidate_scores": {
            name: {k: v for k, v in c.items() if k != "pipeline"}
            for name, c in results["candidates"].items()
        },
        "note": (
            "roti_count vs. heuristic comparison PENDING — heuristic formula not yet supplied."
            if target == "roti_count" and results["candidates"].get("heuristic", {}).get("pending")
            else None
        ),
    }
    metadata_path = MODELS_DIR / f"{target}_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def build_report(all_results: list[dict], selections: dict) -> str:
    lines = ["# Model training report", ""]
    lines.append("Generated by `src/train_model.py` from `data/processed/merged_clean.csv`. "
                  "Time-based split, last 20% of chronologically sorted days held out as test "
                  "(no shuffling — GOVERNANCE.md HC-3).")
    lines.append("")
    lines.append(f"Feature set: numeric={NUMERIC_FEATURES}, categorical={CATEGORICAL_FEATURES}. "
                  "`workout_duration_min` deliberately excluded — see module docstring "
                  "(collinear with calories_burned/workout_type, VIF=9.25).")
    lines.append("")

    for results in all_results:
        target = results["target"]
        best_name, best_entry = selections[target]
        lines.append(f"## {target}")
        lines.append("")
        lines.append(f"n_train={results['n_train']}, n_test={results['n_test']}")
        lines.append("")
        rows = []
        for name, c in results["candidates"].items():
            if c.get("pending"):
                rows.append({"model": name, "mae_raw": "PENDING", "mae_rounded": "PENDING"})
            else:
                row = {"model": name, "mae_raw": round(c["mae_raw"], 4)}
                if "mae_rounded" in c:
                    row["mae_rounded"] = round(c["mae_rounded"], 4)
                rows.append(row)
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")
        lines.append(f"**Selected: `{best_name}`** (saved to `models/{target}_model.joblib`)")
        if target == "roti_count" and results["candidates"].get("heuristic", {}).get("pending"):
            lines.append("")
            lines.append("**Heuristic comparison PENDING.** The manual heuristic formula was not "
                          "supplied — this selection is only between the linear and gradient-boosted "
                          "models. Re-run once the heuristic is wired in; it may beat both.")
        elif target == "roti_count":
            heuristic_mae = results["candidates"]["heuristic"]["mae_rounded"]
            ml_maes = {k: v["mae_rounded"] for k, v in results["candidates"].items() if k != "heuristic"}
            if heuristic_mae <= min(ml_maes.values()):
                lines.append("")
                lines.append(f"**The heuristic (MAE={heuristic_mae:.4f}) was NOT beaten by either ML model** "
                              f"(best ML MAE={min(ml_maes.values()):.4f}). Reporting this plainly, as instructed.")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    df = load_data()
    train_df, test_df = time_based_split(df)
    print(f"Loaded {len(df)} rows. Train: {len(train_df)} ({train_df['date'].min().date()} -> "
          f"{train_df['date'].max().date()}). Test: {len(test_df)} ({test_df['date'].min().date()} -> "
          f"{test_df['date'].max().date()}). No shuffling.")
    print()

    all_results = []
    selections = {}
    for target in TARGETS:
        results = evaluate_target(target, train_df, test_df)
        all_results.append(results)
        best_name, best_entry = select_best(results)
        selections[target] = (best_name, best_entry)
        model_path = save_model(target, best_name, best_entry, results)

        print(f"=== {target} (n_train={results['n_train']}, n_test={results['n_test']}) ===")
        for name, c in results["candidates"].items():
            if c.get("pending"):
                print(f"  {name:10s}: PENDING (formula not supplied)")
            elif "mae_rounded" in c:
                print(f"  {name:10s}: MAE(raw)={c['mae_raw']:.4f}  MAE(rounded)={c['mae_rounded']:.4f}")
            else:
                print(f"  {name:10s}: MAE={c['mae_raw']:.4f}")
        print(f"  -> selected: {best_name}, saved to {model_path.relative_to(REPO_ROOT)}")
        if target == "roti_count":
            if results["candidates"]["heuristic"].get("pending"):
                print("  NOTE: heuristic comparison is PENDING - formula not yet supplied.")
            else:
                heuristic_mae = results["candidates"]["heuristic"]["mae_rounded"]
                ml_maes = {k: v["mae_rounded"] for k, v in results["candidates"].items() if k != "heuristic"}
                if heuristic_mae <= min(ml_maes.values()):
                    print(f"  NOTE: heuristic (MAE={heuristic_mae:.4f}) was NOT beaten by ML "
                          f"(best ML MAE={min(ml_maes.values()):.4f}).")
        print()

    report = build_report(all_results, selections)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
