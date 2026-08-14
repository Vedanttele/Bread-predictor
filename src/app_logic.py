"""Pure, testable logic for the Streamlit dashboard (src/app.py).

Kept separate from app.py because Streamlit scripts execute top-to-bottom
and are awkward to unit test directly — everything that isn't a widget call
lives here instead, so it can be tested the same way as the rest of this
project's src/ modules.

Log file: logs/meal_log.csv — one row per logged day. Every "actual"
nutrition value in a row is looked up from recipe_database.json (never
typed/estimated), per GOVERNANCE.md HC-1 and the project's Phase 5 design
decision (dropdown-select the eaten recipe, not free-entry numbers).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go

# Validated categorical palette (light mode), slots 1 & 2 — see the
# project's dataviz skill reference. This pair is documented as passing
# every adjacent-pair colorblind/contrast check in this fixed order;
# never reassign which series gets which slot.
COLOR_TARGET = "#2a78d6"   # slot 1, blue
COLOR_ACTUAL = "#eb6834"   # slot 2, orange
COLOR_GRIDLINE = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
LOG_PATH = REPO_ROOT / "logs" / "meal_log.csv"

# Feedback loop into retraining (GOVERNANCE.md HC-4, amended): this is the
# ONE controlled exception to "data/raw/ is read-only" — a separate file
# from the manually-curated data/raw/daily_context.csv, append-only, never
# edited or deleted by code, so the original file is never touched.
# src/data_pipeline.py already merges every CSV in data/raw/, so a row
# appended here is automatically picked up by the next retraining run.
RAW_FEEDBACK_PATH = REPO_ROOT / "data" / "raw" / "app_actuals.csv"
RAW_FEEDBACK_COLUMNS = [
    "date", "workout_type", "calories_burned", "sleep_hours", "hunger_level",
    "curry_type", "curry_richness", "meal_prep_time_min",
    "protein_target_g", "fibre_target_g", "vitamin_focus", "roti_count",
]

NUMERIC_FEATURES = ["calories_burned", "hunger_level", "sleep_hours"]
CATEGORICAL_FEATURES = ["workout_type", "curry_type", "curry_richness"]
TARGET_MODEL_FILES = {
    "roti_count": "roti_count_model.joblib",
    "protein_target_g": "protein_target_g_model.joblib",
    "fibre_target_g": "fibre_target_g_model.joblib",
    "vitamin_focus": "vitamin_focus_model.joblib",
}
ROTI_COUNT_BOUNDS = (1, 5)

LOG_COLUMNS = [
    "date", "logged_at_utc",
    "calories_burned", "hunger_level", "sleep_hours",
    "workout_type", "curry_type", "curry_richness", "meal_prep_time_min",
    "target_roti_count", "target_protein_target_g", "target_fibre_target_g", "target_vitamin_focus",
    "recommended_recipe_name",
    "actual_roti_count", "actual_recipe_name", "actual_protein_g", "actual_fibre_g",
    "actual_vitamin_tags", "vitamin_match",
    "delta_roti_count", "delta_protein_g", "delta_fibre_g",
]


def load_models(models_dir: Path = MODELS_DIR) -> dict:
    missing = [f for f in TARGET_MODEL_FILES.values() if not (models_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained model file(s) in {models_dir}: {missing}. "
            "Run src/train_model.py and src/train_vitamin_model.py first."
        )
    return {target: joblib.load(models_dir / fname) for target, fname in TARGET_MODEL_FILES.items()}


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """inputs must contain every key in NUMERIC_FEATURES + CATEGORICAL_FEATURES."""
    row = {col: inputs[col] for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    return pd.DataFrame([row])


def predict_layer1(models: dict, feature_row: pd.DataFrame) -> dict:
    """Runs all four trained models and returns a complete Layer 1 output,
    the shape recipe_agent.recommend_recipe() expects. roti_count is
    rounded and clipped to the observed valid range, same as train_model.py."""
    roti_raw = models["roti_count"].predict(feature_row)[0]
    roti_count = int(min(max(round(roti_raw), ROTI_COUNT_BOUNDS[0]), ROTI_COUNT_BOUNDS[1]))
    return {
        "roti_count": roti_count,
        "protein_target_g": round(float(models["protein_target_g"].predict(feature_row)[0]), 1),
        "fibre_target_g": round(float(models["fibre_target_g"].predict(feature_row)[0]), 1),
        "vitamin_focus": str(models["vitamin_focus"].predict(feature_row)[0]),
    }


def lookup_recipe(recipe_name: str, recipes: list[dict]) -> dict:
    match = next((r for r in recipes if r["name"] == recipe_name), None)
    if match is None:
        raise ValueError(f"'{recipe_name}' is not in the recipe database.")
    return match


def vitamin_focus_matched(target_vitamin_focus: str, actual_recipe: dict) -> bool:
    return target_vitamin_focus in actual_recipe.get("vitamin_tags", [])


def build_log_row(
    date: str, inputs: dict, target: dict, recommended_recipe_name: str | None,
    actual_roti_count: int, actual_recipe: dict,
) -> dict:
    """Every actual_* nutrition value comes from `actual_recipe`, i.e. a
    real recipe_database.json record — never a typed/estimated number."""
    match = vitamin_focus_matched(target["vitamin_focus"], actual_recipe)
    return {
        "date": date,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "calories_burned": inputs["calories_burned"],
        "hunger_level": inputs["hunger_level"],
        "sleep_hours": inputs["sleep_hours"],
        "workout_type": inputs["workout_type"],
        "curry_type": inputs["curry_type"],
        "curry_richness": inputs["curry_richness"],
        "meal_prep_time_min": inputs["meal_prep_time_min"],
        "target_roti_count": target["roti_count"],
        "target_protein_target_g": target["protein_target_g"],
        "target_fibre_target_g": target["fibre_target_g"],
        "target_vitamin_focus": target["vitamin_focus"],
        "recommended_recipe_name": recommended_recipe_name,
        "actual_roti_count": actual_roti_count,
        "actual_recipe_name": actual_recipe["name"],
        "actual_protein_g": actual_recipe["protein_g"],
        "actual_fibre_g": actual_recipe["fibre_g"],
        "actual_vitamin_tags": ";".join(actual_recipe.get("vitamin_tags", [])),
        "vitamin_match": match,
        "delta_roti_count": actual_roti_count - target["roti_count"],
        "delta_protein_g": round(actual_recipe["protein_g"] - target["protein_target_g"], 1),
        "delta_fibre_g": round(actual_recipe["fibre_g"] - target["fibre_target_g"], 1),
    }


def append_log_row(row: dict, log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=LOG_COLUMNS)
    write_header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=write_header, index=False)


def load_log(log_path: Path = LOG_PATH) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(log_path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_raw_feedback_row(date: str, inputs: dict, actual_roti_count: int, actual_recipe: dict) -> dict:
    """A row matching data/raw/daily_context.csv's schema, for retraining.

    Built entirely from what actually happened, not from Layer 1's
    prediction — that's what makes it a usable label: `roti_count` is the
    actually-eaten count, `protein_target_g`/`fibre_target_g` are the
    actually-eaten recipe's real, verified values (never the earlier
    target), and `vitamin_focus` takes that recipe's first vitamin_tags
    entry (its primary tag) since the raw schema holds one label per day
    but a recipe can carry several tags — a disclosed simplification, not
    a fabricated value (the tag really is one of the recipe's real tags).

    Columns the CLI doesn't collect (workout_duration_min, mood_score,
    weight_trend) are intentionally omitted rather than guessed —
    src/data_pipeline.py already treats an absent column as all-missing
    for that source, which is the honest outcome here.
    """
    vitamin_tags = actual_recipe.get("vitamin_tags", [])
    return {
        "date": date,
        "workout_type": inputs["workout_type"],
        "calories_burned": inputs["calories_burned"],
        "sleep_hours": inputs["sleep_hours"],
        "hunger_level": inputs["hunger_level"],
        "curry_type": inputs["curry_type"],
        "curry_richness": inputs["curry_richness"],
        "meal_prep_time_min": inputs["meal_prep_time_min"],
        "protein_target_g": actual_recipe["protein_g"],
        "fibre_target_g": actual_recipe["fibre_g"],
        "vitamin_focus": vitamin_tags[0] if vitamin_tags else "",
        "roti_count": actual_roti_count,
    }


def append_raw_feedback_row(row: dict, path: Path = RAW_FEEDBACK_PATH) -> None:
    """Append-only, and only to app_actuals.csv — this function never
    writes to daily_context.csv or any other file in data/raw/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=RAW_FEEDBACK_COLUMNS)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def build_trend_chart(df: pd.DataFrame, target_col: str, actual_col: str, y_title: str) -> go.Figure:
    """Grouped bar, target vs actual, one metric per chart — never combines
    two different units on one axis (roti_count, grams of protein, and
    grams of fibre each get their own chart). Colors are the validated
    slot-1/slot-2 pair; legend always shown for the 2 series; hover is
    Plotly's native default (satisfies "hover layer by default" for a
    trend chart)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df[target_col], name="Target",
        marker_color=COLOR_TARGET, marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df[actual_col], name="Actual",
        marker_color=COLOR_ACTUAL, marker_line_width=0,
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor=COLOR_SURFACE,
        paper_bgcolor=COLOR_SURFACE,
        font_color=COLOR_TEXT_SECONDARY,
        yaxis_title=y_title,
        xaxis=dict(showgrid=False, linecolor=COLOR_AXIS),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRIDLINE, linecolor=COLOR_AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.25,
    )
    return fig


def build_pct_of_target_chart(df: pd.DataFrame, target_col: str, actual_col: str, label: str) -> go.Figure:
    """Percent of target met — for a metric where "actual" and "target"
    are NOT directly comparable quantities (see protein_target_g below),
    a grouped-bar chart of the raw numbers is actively misleading: it
    visually implies the two bars should be roughly equal, when they
    structurally can't be. This shows actual/target as a single-series
    bar with a dashed reference line at 100%, which is honest regardless
    of the underlying scale gap — a real percentage rather than a
    fabricated equivalence.

    Used for protein_target_g specifically: recipe_database.json's
    protein_g tops out at 19.0g (a single dish), while protein_target_g
    ranges 31.2-52.0g in the training data (very plausibly a whole-day or
    whole-meal goal, not one dish's contribution) — no recipe can ever
    reach even the lowest observed target, so a grams-vs-grams comparison
    always shows the same large gap regardless of how good the pick was.
    fibre_target_g doesn't have this problem (recipe range 3-14g overlaps
    target range 8-14g almost completely) and keeps the grouped-bar chart.
    """
    pct = (df[actual_col] / df[target_col] * 100).round(1)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=pct, name=f"% of {label} target met",
        marker_color=COLOR_ACTUAL, marker_line_width=0,
    ))
    fig.add_hline(y=100, line_dash="dash", line_color=COLOR_TARGET, line_width=2,
                  annotation_text="Target (100%)", annotation_position="top left",
                  annotation_font_color=COLOR_TARGET)
    fig.update_layout(
        plot_bgcolor=COLOR_SURFACE,
        paper_bgcolor=COLOR_SURFACE,
        font_color=COLOR_TEXT_SECONDARY,
        yaxis_title=f"% of {label} target",
        xaxis=dict(showgrid=False, linecolor=COLOR_AXIS),
        yaxis=dict(showgrid=True, gridcolor=COLOR_GRIDLINE, linecolor=COLOR_AXIS),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.25,
    )
    return fig
