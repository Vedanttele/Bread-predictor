"""Simple interactive CLI for roti-predictor.

Asks for today's context, runs it through the trained Layer 1 models
(src/app_logic.py), then Layer 2 (src/recipe_agent.py), and prints the
recommendation. Then optionally lets you log what you actually ate.

That log is appended to data/raw/app_actuals.csv — a NEW file, separate
from the manually-curated data/raw/daily_context.csv, which this script
never touches. src/data_pipeline.py already merges every CSV in data/raw/,
so a logged entry is automatically included the next time you retrain
(src/train_model.py / src/train_vitamin_model.py). See GOVERNANCE.md HC-4
for why this is the one controlled, append-only exception to
"data/raw/ is read-only".

Note: `mood` is intentionally not asked for here — Phase 2/3's feature
analysis found it has no real signal (r=-0.04, not significant) and it was
excluded from the trained model's feature set. Asking for it would not
affect the prediction. `meal_type` (breakfast/lunch/dinner/snack) IS asked
for, but it's a Layer 2 filter against the recipe database's suitable_meal
field, not a Layer 1 model feature — the trained models were never given a
meal-type column (the raw data is one row per day, not per meal).

Run:
    python src/cli.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_logic as al  # noqa: E402
from recipe_agent import DEFAULT_DB_PATH, load_recipes, recommend_recipe  # noqa: E402

WORKOUT_TYPES = ["none", "cardio", "strength"]
CURRY_TYPES = ["dal", "dry", "gravy"]
CURRY_RICHNESS = ["light", "heavy"]
MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack", "skip"]


# ---------------------------------------------------------------------------
# Small prompt helpers — each validates and re-asks on bad input.
# ---------------------------------------------------------------------------

def prompt_choice(question: str, options: list[str], default: str) -> str:
    options_str = "/".join(o if o != default else o.upper() for o in options)
    while True:
        raw = input(f"{question} [{options_str}]: ").strip().lower()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"  Please enter one of: {', '.join(options)}")


def prompt_int(question: str, default: int, min_value: int, max_value: int) -> int:
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if not (min_value <= value <= max_value):
            print(f"  Please enter a number between {min_value} and {max_value}.")
            continue
        return value


def prompt_float(question: str, default: float, min_value: float, max_value: float) -> float:
    while True:
        raw = input(f"{question} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if not (min_value <= value <= max_value):
            print(f"  Please enter a number between {min_value} and {max_value}.")
            continue
        return value


def prompt_date(question: str, default: date) -> str:
    """Returns the date as DD/MM/YYYY — the format data/SCHEMA.md documents
    for data/raw/daily_context.csv. Writing anything else here would
    reintroduce the exact mixed-date-format bug that once made
    src/data_pipeline.py silently drop 220 of 221 rows (see its
    derive_date_fields docstring) — accept ISO input since that's what
    people type, but always normalize before it ever reaches a CSV."""
    while True:
        raw = input(f"{question} [YYYY-MM-DD, default {default.isoformat()}]: ").strip()
        if not raw:
            parsed = default
        else:
            try:
                parsed = date.fromisoformat(raw)
            except ValueError:
                print("  Please enter a date as YYYY-MM-DD.")
                continue
        return parsed.strftime("%d/%m/%Y")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


def prompt_recipe_choice(recipes: list[dict], default_name: str | None) -> dict:
    names = sorted(r["name"] for r in recipes)
    print("\nRecipes:")
    for i, name in enumerate(names, start=1):
        marker = " (recommended)" if name == default_name else ""
        print(f"  {i:2d}. {name}{marker}")
    default_index = names.index(default_name) + 1 if default_name in names else 1
    while True:
        raw = input(f"Which recipe did you actually eat? [1-{len(names)}] [{default_index}]: ").strip()
        if not raw:
            chosen_index = default_index
        else:
            try:
                chosen_index = int(raw)
            except ValueError:
                print("  Please enter a number from the list.")
                continue
        if not (1 <= chosen_index <= len(names)):
            print(f"  Please enter a number between 1 and {len(names)}.")
            continue
        return al.lookup_recipe(names[chosen_index - 1], recipes)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def collect_context() -> tuple[dict, str | None]:
    """Returns (inputs, meal_type). inputs has exactly the 6 keys the
    trained models use; meal_type is separate (Layer 2 filter only, may
    be None)."""
    print("=== Today's context ===")
    workout_type = prompt_choice("Workout type", WORKOUT_TYPES, default="none")
    calories_burned = prompt_int("Calories burned", default=2100, min_value=1000, max_value=4000)
    sleep_hours = prompt_float("Sleep (hours)", default=7.0, min_value=0.0, max_value=14.0)
    hunger_level = prompt_int("Hunger level (1-5)", default=3, min_value=1, max_value=5)
    curry_type = prompt_choice("Curry type", CURRY_TYPES, default="dal")
    curry_richness = prompt_choice("Curry richness", CURRY_RICHNESS, default="light")
    meal_prep_time_min = prompt_int("Minutes available to cook", default=20, min_value=0, max_value=240)
    meal_type_raw = prompt_choice("Meal type (for recipe matching; 'skip' for no filter)", MEAL_TYPES, default="skip")
    meal_type = None if meal_type_raw == "skip" else meal_type_raw

    inputs = {
        "workout_type": workout_type, "calories_burned": calories_burned, "sleep_hours": sleep_hours,
        "hunger_level": hunger_level, "curry_type": curry_type, "curry_richness": curry_richness,
        "meal_prep_time_min": meal_prep_time_min,
    }
    return inputs, meal_type


def print_targets(target: dict) -> None:
    print("\n=== Today's targets ===")
    print(f"  Roti count:      {target['roti_count']}")
    print(f"  Protein target:  {target['protein_target_g']:.1f} g")
    print(f"  Fibre target:    {target['fibre_target_g']:.1f} g")
    print(f"  Vitamin focus:   {target['vitamin_focus']}")


def get_recommendation(target: dict, meal_prep_time_min: int, meal_type: str | None) -> dict | None:
    print("\n=== Recipe recommendation ===")
    try:
        result = recommend_recipe(target, meal_prep_time_min, meal_type=meal_type)
    except Exception as e:  # noqa: BLE001 — surface any API/auth error plainly, don't crash the CLI
        print(f"Couldn't reach Claude: {e}")
        print("If this is an authentication error, set ANTHROPIC_API_KEY or run "
              "`ant auth login` in a terminal, then try again.")
        return None

    if result["status"] == "no_match":
        print(f"No match: {result['reason']}")
        return None

    recipe = result["chosen_recipe"]
    print(f"Chosen: {recipe['name']} ({recipe['cuisine']})")
    print(f"  {result['explanation']}")
    print(f"  Protein: {recipe['protein_g']:.1f} g | Fibre: {recipe['fibre_g']:.1f} g | "
          f"Calories: {recipe['calories']} | Prep time: {recipe['prep_time_min']} min")
    return result


def log_actual(inputs: dict, recipes: list[dict], recommendation: dict | None) -> None:
    print("\n=== Log what you actually ate ===")
    if not prompt_yes_no("Log what you actually ate?", default=True):
        return

    log_date = prompt_date("Date", default=date.today())
    actual_roti_count = prompt_int("Rotis actually eaten", default=0, min_value=0, max_value=10)
    default_recipe_name = recommendation["chosen_recipe"]["name"] if recommendation and recommendation["status"] == "ok" else None
    actual_recipe = prompt_recipe_choice(recipes, default_recipe_name)

    row = al.build_raw_feedback_row(log_date, inputs, actual_roti_count, actual_recipe)
    al.append_raw_feedback_row(row)
    print(f"\nLogged to {al.RAW_FEEDBACK_PATH.relative_to(al.REPO_ROOT)} "
          f"- will be included next time you run src/data_pipeline.py and retrain.")


def main() -> None:
    inputs, meal_type = collect_context()

    try:
        models = al.load_models()
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)

    feature_row = al.build_feature_row(inputs)
    target = al.predict_layer1(models, feature_row)
    print_targets(target)

    recommendation = get_recommendation(target, inputs["meal_prep_time_min"], meal_type)

    recipes = load_recipes(DEFAULT_DB_PATH)
    log_actual(inputs, recipes, recommendation)


if __name__ == "__main__":
    main()
