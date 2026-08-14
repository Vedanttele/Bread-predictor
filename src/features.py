"""Feature analysis for roti-predictor — Layer 1 pre-modeling step.

Reads data/processed/merged_clean.csv, measures how strongly each candidate
feature associates with `roti_count`, flags multicollinearity between
calories_burned/workout_type/workout_duration_min, encodes categoricals two
ways (one suited to linear models, one to tree models), and prints/saves a
feature-importance-style summary.

Deliberately NOT here: no predictive model is fit or saved. Every statistic
below (Pearson/Spearman correlation, one-way ANOVA / eta-squared, mutual
information, VIF via the correlation-matrix inverse) is a diagnostic used to
screen features, not a model with its own .predict() — that's Phase 3.

Run directly:
    python src/features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "merged_clean.csv"
REPORT_PATH = REPO_ROOT / "notebooks" / "feature_analysis_report.md"
LINEAR_ENCODED_PATH = REPO_ROOT / "data" / "processed" / "features_linear.csv"
TREE_ENCODED_PATH = REPO_ROOT / "data" / "processed" / "features_tree.csv"

PRIMARY_TARGET = "roti_count"
# All four are Layer 1 outputs — none of them is available as an input at
# inference time for a new day, so none of them may appear in any other
# target's feature list. Correlations among them are reported separately,
# labeled informational-only, never folded into the "feature signal" table.
TARGET_COLUMNS = ["roti_count", "protein_target_g", "fibre_target_g", "vitamin_focus"]

NUMERIC_FEATURES = [
    "workout_duration_min", "calories_burned", "sleep_hours",
    "mood_score", "hunger_level", "meal_prep_time_min",
]
CATEGORICAL_FEATURES = ["workout_type", "curry_type", "curry_richness", "weight_trend", "meal_type", "day_of_week"]
# is_weekend plus every *_missing flag that isn't about a target itself (a
# target's own missingness routes excluded_from_training, it isn't a
# predictor). Whether logging was skipped for some other field is a
# legitimate candidate signal (e.g. "didn't log sleep" might correlate with
# behavior) even though every flag is constant/False in the current data.
# day_of_week is excluded here: it's derived from `date` (always
# computable, see data_pipeline.py), so it has no *_missing companion.
NULLABLE_CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c != "day_of_week"]
NON_TARGET_MISSING_FLAGS = [f"{c}_missing" for c in NUMERIC_FEATURES + NULLABLE_CATEGORICAL_FEATURES]
BINARY_FEATURES = ["is_weekend"] + NON_TARGET_MISSING_FLAGS

# Effect-size thresholds — Cohen's (1988) conventions, the standard
# rule-of-thumb rather than an arbitrary cutoff:
#   |r|      < 0.10 negligible, 0.10-0.30 small/weak, 0.30-0.50 moderate, >= 0.50 large
#   eta^2    < 0.01 negligible, 0.01-0.06 small/weak, 0.06-0.14 moderate, >= 0.14 large
# "signal" here means moderate-or-larger AND statistically significant at
# p<0.05; "weak signal" means small effect size regardless of p (worth a
# second look, not worth building on alone); anything smaller, or a larger
# effect that isn't significant, is called noise-like given this sample size.
CORR_SIGNAL_THRESHOLD = 0.30
CORR_WEAK_THRESHOLD = 0.10
ETA2_SIGNAL_THRESHOLD = 0.06
ETA2_WEAK_THRESHOLD = 0.01
ALPHA = 0.05


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}. Run src/data_pipeline.py first.")
    return pd.read_csv(path, parse_dates=["date"])


# ---------------------------------------------------------------------------
# 1. Feature <-> target association (the "signal vs. noise" question)
# ---------------------------------------------------------------------------

def eta_squared(groups: list[np.ndarray], grand_mean: float) -> float:
    """Proportion of target variance explained by group membership — the
    categorical analogue of r^2, so it's comparable in spirit to a squared
    correlation coefficient."""
    all_values = np.concatenate(groups)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    if ss_total == 0:
        return float("nan")
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total


def correlate_numeric(feature: pd.Series, target: pd.Series) -> dict:
    valid = feature.notna() & target.notna()
    x, y = feature[valid], target[valid]
    if x.nunique() <= 1:
        return {"metric": "pearson_r", "value": np.nan, "p_value": np.nan,
                "spearman_r": np.nan, "note": "zero variance in current data — not assessable"}
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return {"metric": "pearson_r", "value": pearson_r, "p_value": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p, "note": ""}


def correlate_categorical(feature: pd.Series, target: pd.Series) -> dict:
    valid = feature.notna() & target.notna()
    x, y = feature[valid], target[valid]
    levels = x.unique()
    if len(levels) <= 1:
        return {"metric": "eta_squared", "value": np.nan, "p_value": np.nan,
                "spearman_r": np.nan, "note": "single category in current data — not assessable"}
    groups = [y[x == lvl].to_numpy(dtype=float) for lvl in levels]
    f_stat, p_value = stats.f_oneway(*groups)
    eta2 = eta_squared(groups, y.mean())
    return {"metric": "eta_squared", "value": eta2, "p_value": p_value,
            "spearman_r": np.nan, "note": f"F={f_stat:.2f}, {len(levels)} categories"}


def classify_verdict(metric: str, value: float, p_value: float) -> str:
    if pd.isna(value):
        return "not assessable (no variance)"
    threshold = CORR_SIGNAL_THRESHOLD if metric == "pearson_r" else ETA2_SIGNAL_THRESHOLD
    weak_threshold = CORR_WEAK_THRESHOLD if metric == "pearson_r" else ETA2_WEAK_THRESHOLD
    magnitude = abs(value)
    significant = pd.notna(p_value) and p_value < ALPHA
    if magnitude >= threshold and significant:
        return "signal"
    if magnitude >= weak_threshold:
        return "weak signal"
    return "noise-like"


def mutual_information_ranking(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.Series:
    """Non-parametric dependency estimate (sklearn.feature_selection —
    a screening statistic, not a fitted predictive model) that can catch
    non-linear relationships Pearson/eta^2 would miss. Categorical columns
    are integer-coded purely so the estimator can consume them; that coding
    is local to this computation, not the encoding used for actual modeling
    (see section 3)."""
    work = df[feature_cols].copy()
    is_discrete = []
    for col in feature_cols:
        if work[col].dtype == object or work[col].dtype == bool or col in CATEGORICAL_FEATURES:
            work[col] = pd.Categorical(work[col]).codes
            is_discrete.append(True)
        else:
            work[col] = work[col].fillna(work[col].median())
            is_discrete.append(False)
    y = df[target_col]
    mi = mutual_info_regression(work.values, y.values, discrete_features=is_discrete, random_state=0)
    return pd.Series(mi, index=feature_cols, name="mutual_info")


def build_signal_table(df: pd.DataFrame, target_col: str = PRIMARY_TARGET) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        feat = df[col].astype(float) if df[col].dtype == bool else df[col]
        result = correlate_numeric(feat, df[target_col])
        rows.append({
            "feature": col, "feature_type": "numeric" if col in NUMERIC_FEATURES else "binary",
            "metric": result["metric"], "value": result["value"], "p_value": result["p_value"],
            "spearman_r": result.get("spearman_r", np.nan), "note": result["note"],
        })
    for col in CATEGORICAL_FEATURES:
        result = correlate_categorical(df[col], df[target_col])
        rows.append({
            "feature": col, "feature_type": "categorical",
            "metric": result["metric"], "value": result["value"], "p_value": result["p_value"],
            "spearman_r": np.nan, "note": result["note"],
        })
    table = pd.DataFrame(rows)
    table["verdict"] = table.apply(lambda r: classify_verdict(r["metric"], r["value"], r["p_value"]), axis=1)

    mi_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    mi_cols = [c for c in mi_cols if df[c].nunique(dropna=True) > 1]  # MI undefined for constant columns
    mi = mutual_information_ranking(df, mi_cols, target_col)
    table = table.merge(mi.rename("mutual_info"), left_on="feature", right_index=True, how="left")

    table["abs_value"] = table["value"].abs()
    return table.sort_values("abs_value", ascending=False).drop(columns="abs_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Multicollinearity — calories_burned vs. workout_type / workout_duration
# ---------------------------------------------------------------------------

def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF_i = i-th diagonal of the inverse correlation matrix of the
    (standardized) numeric design matrix. This is closed-form linear
    algebra on the correlation matrix — a diagnostic statistic, not a
    fitted regression model. pinv (pseudo-inverse) is used instead of inv
    so a near-singular matrix degrades to very large VIFs rather than
    crashing. VIF > 5 is a common "worth a look" threshold, > 10 is
    generally considered severe multicollinearity."""
    corr = X.corr()
    inv = np.linalg.pinv(corr.values)
    return pd.Series(np.diag(inv), index=X.columns, name="VIF")


def check_multicollinearity(df: pd.DataFrame) -> dict:
    # Direct pairwise check, exactly what was asked: calories_burned vs
    # workout_duration_min (numeric-numeric), and workout_type
    # (categorical) vs each of them via eta^2.
    r_cal_dur, p_cal_dur = stats.pearsonr(df["calories_burned"], df["workout_duration_min"])

    wt_groups_cal = [df.loc[df["workout_type"] == lvl, "calories_burned"].to_numpy(dtype=float)
                      for lvl in df["workout_type"].unique()]
    eta2_workouttype_calories = eta_squared(wt_groups_cal, df["calories_burned"].mean())

    wt_groups_dur = [df.loc[df["workout_type"] == lvl, "workout_duration_min"].to_numpy(dtype=float)
                      for lvl in df["workout_type"].unique()]
    eta2_workouttype_duration = eta_squared(wt_groups_dur, df["workout_duration_min"].mean())

    # Focused VIF block: the numeric features plus workout_type dummy-coded
    # (drop_first — see section 3 for why linear-style encoding drops a
    # level) — directly answers "is workout_type/workout_duration_min
    # redundant with calories_burned once combined in one feature set".
    focused = df[NUMERIC_FEATURES].copy()
    focused = pd.concat([focused, pd.get_dummies(df["workout_type"], prefix="workout_type", drop_first=True, dtype=float)], axis=1)
    vif_focused = variance_inflation_factors(focused)

    # Full VIF across every numeric + one-hot(drop_first) categorical
    # feature, for completeness beyond just the workout/calories trio.
    full = df[NUMERIC_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        full = pd.concat([full, pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)], axis=1)
    vif_full = variance_inflation_factors(full)

    return {
        "r_calories_burned_vs_workout_duration_min": r_cal_dur,
        "p_calories_burned_vs_workout_duration_min": p_cal_dur,
        "eta2_workout_type_vs_calories_burned": eta2_workouttype_calories,
        "eta2_workout_type_vs_workout_duration_min": eta2_workouttype_duration,
        "vif_focused": vif_focused,
        "vif_full": vif_full,
    }


# ---------------------------------------------------------------------------
# 3. Encoding — two variants, because the right encoding depends on the
#    model family, which hasn't been chosen yet (that's the point of this
#    step: produce both, decide later).
# ---------------------------------------------------------------------------

def encode_for_linear_model(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot, drop_first=True. Linear models need a level dropped per
    categorical column — keeping every level makes the design matrix
    exactly rank-deficient (the classic 'dummy variable trap'), which is a
    much harder failure than ordinary multicollinearity. Numeric features
    are intentionally left unscaled here: a scaler (e.g. StandardScaler)
    must be *fit* only on the training split, and the time-based train/test
    cutoff isn't decided until Phase 3 (GOVERNANCE.md HC-3) — fitting one
    on the full dataset now would leak test-set statistics into training.
    """
    out = df[["date", "source_file"] + NUMERIC_FEATURES].copy()
    out["is_weekend"] = df["is_weekend"].astype(int)
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        out = pd.concat([out, dummies], axis=1)
    for col in TARGET_COLUMNS:
        out[col] = df[col]
    out["excluded_from_training"] = df["excluded_from_training"]
    return out


def encode_for_tree_model(df: pd.DataFrame) -> pd.DataFrame:
    """Integer/ordinal category codes rather than one-hot. Tree-based
    models split on thresholds over a single column, so they don't need
    (or benefit from) the dummy-variable expansion linear models require —
    one-hot would only add unnecessary width and dilute each split's
    signal across many sparse binary columns. No scaling needed either;
    trees are invariant to monotonic transforms of a feature.
    """
    out = df[["date", "source_file"] + NUMERIC_FEATURES].copy()
    out["is_weekend"] = df["is_weekend"].astype(int)
    for col in CATEGORICAL_FEATURES:
        out[f"{col}_code"] = pd.Categorical(df[col]).codes
    for col in TARGET_COLUMNS:
        out[col] = df[col]
    out["excluded_from_training"] = df["excluded_from_training"]
    return out


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------

def target_target_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Informational only: correlations AMONG the four Layer 1 targets.
    None of these may be used as a feature for predicting another —
    they're all unavailable at inference time for a new day — this is
    purely to understand how the targets relate to each other."""
    numeric_targets = ["roti_count", "protein_target_g", "fibre_target_g"]
    corr = df[numeric_targets].corr()
    eta2_vitamin_roti = eta_squared(
        [df.loc[df["vitamin_focus"] == lvl, "roti_count"].to_numpy(dtype=float) for lvl in df["vitamin_focus"].unique()],
        df["roti_count"].mean(),
    )
    return corr, eta2_vitamin_roti


def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown()


def build_report(signal_table: pd.DataFrame, collinearity: dict, target_corr: pd.DataFrame, eta2_vitamin_roti: float) -> str:
    lines = ["# Feature analysis report", ""]
    lines.append("Generated by `src/features.py` from `data/processed/merged_clean.csv`. "
                  "No model is trained here — see module docstring.")
    lines.append("")

    lines.append(f"## Feature signal vs. `{PRIMARY_TARGET}`")
    lines.append("")
    lines.append("Verdict thresholds: Cohen's (1988) conventions — `signal` = moderate-or-larger "
                  "effect size (|r|>=0.30 or eta2>=0.06) AND p<0.05; `weak signal` = small effect "
                  "(|r|>=0.10 or eta2>=0.01) regardless of significance; otherwise `noise-like`. "
                  "`mutual_info` is a non-linear cross-check (sklearn.feature_selection, a screening "
                  "statistic, not a fitted model) — worth a second look if it disagrees with the verdict.")
    lines.append("")
    display_cols = ["feature", "feature_type", "metric", "value", "p_value", "spearman_r", "mutual_info", "verdict", "note"]
    lines.append(md_table(signal_table[display_cols].round(4)))
    lines.append("")

    signal_feats = signal_table[signal_table["verdict"] == "signal"]["feature"].tolist()
    weak_feats = signal_table[signal_table["verdict"] == "weak signal"]["feature"].tolist()
    noise_feats = signal_table[signal_table["verdict"] == "noise-like"]["feature"].tolist()
    not_assessable = signal_table[signal_table["verdict"] == "not assessable (no variance)"]["feature"].tolist()
    lines.append(f"- **Signal:** {', '.join(signal_feats) if signal_feats else '(none)'}")
    lines.append(f"- **Weak signal:** {', '.join(weak_feats) if weak_feats else '(none)'}")
    lines.append(f"- **Noise-like:** {', '.join(noise_feats) if noise_feats else '(none)'}")
    lines.append(f"- **Not assessable (constant in current data):** {', '.join(not_assessable) if not_assessable else '(none)'}")
    lines.append("")

    lines.append("## Multicollinearity — calories_burned vs. workout_type / workout_duration_min")
    lines.append("")
    lines.append(f"- Pearson r(calories_burned, workout_duration_min) = "
                  f"{collinearity['r_calories_burned_vs_workout_duration_min']:.3f} "
                  f"(p={collinearity['p_calories_burned_vs_workout_duration_min']:.2e})")
    lines.append(f"- eta^2(workout_type -> calories_burned) = {collinearity['eta2_workout_type_vs_calories_burned']:.3f}")
    lines.append(f"- eta^2(workout_type -> workout_duration_min) = {collinearity['eta2_workout_type_vs_workout_duration_min']:.3f}")
    lines.append("")
    lines.append("**VIF — numeric features + workout_type (dummy-coded):**")
    lines.append("")
    lines.append(md_table(collinearity["vif_focused"].round(2).rename("VIF").to_frame()))
    lines.append("")
    lines.append(f"- VIF(workout_duration_min) = {collinearity['vif_focused'].get('workout_duration_min', float('nan')):.2f}, "
                  f"VIF(calories_burned) = {collinearity['vif_focused'].get('calories_burned', float('nan')):.2f}. "
                  + ("Both under 5 — no severe redundancy despite the two being related; a model can "
                     "reasonably use both." if collinearity['vif_focused'].get('workout_duration_min', 0) < 5
                     and collinearity['vif_focused'].get('calories_burned', 0) < 5 else
                     "At least one is elevated (>=5) — consider dropping or combining calories_burned and "
                     "workout_duration_min rather than feeding both to a linear model."))
    lines.append("")
    lines.append("**Full VIF — all numeric + one-hot(drop_first) categorical features:**")
    lines.append("")
    lines.append(md_table(collinearity["vif_full"].round(2).rename("VIF").to_frame()))
    lines.append("")

    lines.append("## Target-target correlations (informational only — never usable as input features)")
    lines.append("")
    lines.append("All four columns below are Layer 1 outputs. None may appear in another target's "
                  "feature list — none is available at inference time for a new day.")
    lines.append("")
    lines.append(md_table(target_corr.round(3)))
    lines.append("")
    lines.append(f"- eta^2(vitamin_focus -> roti_count) = {eta2_vitamin_roti:.3f}")
    lines.append("")

    lines.append("## Encoding")
    lines.append("")
    lines.append(f"- `{LINEAR_ENCODED_PATH.relative_to(REPO_ROOT)}` — one-hot, drop_first=True. "
                  "Numeric features left unscaled (see encode_for_linear_model docstring — scaling "
                  "must wait for Phase 3's train/test cutoff to avoid leakage).")
    lines.append(f"- `{TREE_ENCODED_PATH.relative_to(REPO_ROOT)}` — integer category codes, no scaling.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()

    signal_table = build_signal_table(df, PRIMARY_TARGET)
    collinearity = check_multicollinearity(df)
    target_corr, eta2_vitamin_roti = target_target_correlations(df)

    linear_df = encode_for_linear_model(df)
    tree_df = encode_for_tree_model(df)
    LINEAR_ENCODED_PATH.parent.mkdir(parents=True, exist_ok=True)
    linear_df.to_csv(LINEAR_ENCODED_PATH, index=False)
    tree_df.to_csv(TREE_ENCODED_PATH, index=False)

    report = build_report(signal_table, collinearity, target_corr, eta2_vitamin_roti)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Loaded {len(df)} rows from {PROCESSED_PATH.relative_to(REPO_ROOT)}")
    print()
    print(f"=== Feature signal vs. {PRIMARY_TARGET} (Cohen's-convention verdicts) ===")
    print(signal_table[["feature", "feature_type", "metric", "value", "p_value", "mutual_info", "verdict"]]
          .round(4).to_string(index=False))
    print()
    print("=== Multicollinearity: calories_burned vs workout_type/workout_duration_min ===")
    print(f"Pearson r(calories_burned, workout_duration_min) = "
          f"{collinearity['r_calories_burned_vs_workout_duration_min']:.3f} "
          f"(p={collinearity['p_calories_burned_vs_workout_duration_min']:.2e})")
    print(f"eta^2(workout_type -> calories_burned)      = {collinearity['eta2_workout_type_vs_calories_burned']:.3f}")
    print(f"eta^2(workout_type -> workout_duration_min) = {collinearity['eta2_workout_type_vs_workout_duration_min']:.3f}")
    print("VIF (numeric + workout_type dummies):")
    print(collinearity["vif_focused"].round(2).to_string())
    print()
    print(f"Wrote {LINEAR_ENCODED_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {TREE_ENCODED_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print()
    print("No model was trained. Stopping here.")
