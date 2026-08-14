"""Layer 2 recipe recommendation agent for roti-predictor.

Input: Layer 1 output (roti_count, protein_target_g, fibre_target_g,
vitamin_focus) plus meal_prep_time_min supplied by the user. Filters
recipe_database.json down to recipes that fit the time budget, ranks the
survivors by closeness to the nutrition targets and vitamin_focus match,
and asks Claude to pick one of the top 3 and explain the choice in plain
language.

Hard constraint (GOVERNANCE.md HC-1): Claude never invents a nutrition
number. It only ever sees the pre-filtered/pre-ranked candidates' real
fields straight from recipe_database.json, and its choice is constrained
via output_config.format + a JSON-schema enum to one of those candidates'
exact names — it is structurally unable to name a recipe it wasn't shown,
let alone a value that isn't in the database.

If no recipe fits meal_prep_time_min, that is reported explicitly
(status: "no_match") and the API is never called — no bad match is forced.

Run directly:
    python src/recipe_agent.py
(uses a couple of built-in example Layer 1 outputs; see main() below.
 Requires ANTHROPIC_API_KEY or an `ant auth login` profile to actually
 call the API — see the module docstring in the repo's claude-api skill
 notes for credential resolution.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipe_db import DEFAULT_DB_PATH, load_raw  # noqa: E402

MODEL = "claude-opus-5"
TOP_N = 3
# Vitamin match bonus subtracted from nutrition distance when the
# recipe's vitamin_tags include Layer 1's requested vitamin_focus.
# Sized roughly to a typical single-nutrient gap (protein/fibre diffs in
# this database run from under a gram to ~15g) so a vitamin match can tip
# a close ranking without completely overriding nutrition fit. A tunable
# design choice, not a value derived from the data.
VITAMIN_MATCH_BONUS = 5.0

# Fields shown to Claude and used to build the prompt/schema — every one
# of these is a real, verified value straight from recipe_database.json.
CANDIDATE_FIELDS = [
    "name", "cuisine", "main_ingredients", "protein_g", "fibre_g", "calories",
    "prep_time_min", "vitamin_tags", "pairs_with_roti", "suitable_meal",
]


def load_recipes(path: Path = DEFAULT_DB_PATH) -> list[dict]:
    data = load_raw(path)
    return data["recipes"]


def filter_by_prep_time(recipes: list[dict], meal_prep_time_min: int) -> list[dict]:
    return [r for r in recipes if r["prep_time_min"] <= meal_prep_time_min]


def filter_by_meal_type(recipes: list[dict], meal_type: str | None) -> list[dict]:
    """meal_type is which meal this is for (breakfast/lunch/dinner/snack) —
    a Layer 2 filter against the database's suitable_meal field, not a
    Layer 1 model feature (the trained models were never given a meal-type
    column; the raw daily data is one row per day, not per meal). A recipe
    tagged 'both' always qualifies. None/empty means no filtering."""
    if not meal_type:
        return recipes
    return [r for r in recipes if r["suitable_meal"] in (meal_type, "both")]


def score_recipe(recipe: dict, protein_target_g: float, fibre_target_g: float, vitamin_focus: str) -> float:
    """Lower is better. Nutrition distance is the sum of absolute gaps to
    the two targets (both fields are in grams, roughly comparable scale,
    so a raw sum is a simple, defensible heuristic — not normalized or
    weighted beyond that). A vitamin_focus match subtracts a flat bonus
    (see VITAMIN_MATCH_BONUS)."""
    nutrition_distance = abs(recipe["protein_g"] - protein_target_g) + abs(recipe["fibre_g"] - fibre_target_g)
    vitamin_match = vitamin_focus in recipe.get("vitamin_tags", [])
    return nutrition_distance - (VITAMIN_MATCH_BONUS if vitamin_match else 0.0)


def rank_candidates(
    recipes: list[dict], protein_target_g: float, fibre_target_g: float,
    vitamin_focus: str, top_n: int = TOP_N,
) -> list[dict]:
    return sorted(
        recipes, key=lambda r: score_recipe(r, protein_target_g, fibre_target_g, vitamin_focus),
    )[:top_n]


def build_candidate_summary(recipe: dict) -> dict:
    return {k: recipe[k] for k in CANDIDATE_FIELDS if k in recipe}


def build_prompt(layer1_output: dict, meal_prep_time_min: int, candidates: list[dict]) -> str:
    candidate_json = json.dumps([build_candidate_summary(c) for c in candidates], indent=2)
    return f"""A meal-planning model predicted these targets for today's meal:
- roti_count: {layer1_output['roti_count']}
- protein_target_g: {layer1_output['protein_target_g']}
- fibre_target_g: {layer1_output['fibre_target_g']}
- vitamin_focus: {layer1_output['vitamin_focus']}

The user has {meal_prep_time_min} minutes to cook.

Here are the top {len(candidates)} candidate recipes, already filtered to fit the
time budget and ranked by closeness to the targets above. Every field below
is a real, verified value from the recipe database:

{candidate_json}

Pick exactly one of these {len(candidates)} recipes and explain the choice in
plain, friendly language for the person eating it. Reference only the
numbers given above (protein_g, fibre_g, calories, prep_time_min,
vitamin_tags, etc. from the candidate list) — do not invent, estimate, or
state a number that isn't explicitly present above, and do not claim a
nutrition fact about any recipe you weren't given."""


def build_output_schema(candidates: list[dict]) -> dict:
    """chosen_recipe_name is constrained to an enum of the exact candidate
    names — Claude cannot return a name outside this list."""
    return {
        "type": "object",
        "properties": {
            "chosen_recipe_name": {"type": "string", "enum": [c["name"] for c in candidates]},
            "explanation": {"type": "string"},
        },
        "required": ["chosen_recipe_name", "explanation"],
        "additionalProperties": False,
    }


def ask_claude_to_choose(
    layer1_output: dict, meal_prep_time_min: int, candidates: list[dict],
    client: anthropic.Anthropic | None = None,
) -> dict:
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": build_output_schema(candidates)}},
        messages=[{"role": "user", "content": build_prompt(layer1_output, meal_prep_time_min, candidates)}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def recommend_recipe(
    layer1_output: dict, meal_prep_time_min: int, meal_type: str | None = None,
    db_path: Path = DEFAULT_DB_PATH, client: anthropic.Anthropic | None = None,
) -> dict:
    """Full Layer 2 pipeline.

    meal_type (optional): breakfast/lunch/dinner/snack — filters against
    the database's suitable_meal field (a recipe tagged 'both' always
    qualifies). None means don't filter on it.

    Returns either:
      {"status": "no_match", "reason": "..."} — no recipe fits the
      constraints; the API is never called and no bad match is forced, or
      {"status": "ok", "chosen_recipe": {...full db record...},
       "explanation": "...", "candidates_considered": [...top N...]}
    """
    recipes = load_recipes(db_path)
    fitting = filter_by_meal_type(filter_by_prep_time(recipes, meal_prep_time_min), meal_type)

    if not fitting:
        fastest = min(r["prep_time_min"] for r in recipes)
        reason = f"No recipe in the database has prep_time_min <= {meal_prep_time_min}"
        reason += f" and suitable_meal matching '{meal_type}'" if meal_type else ""
        reason += f". Fastest available recipe takes {fastest} minutes."
        return {"status": "no_match", "reason": reason}

    candidates = rank_candidates(
        fitting, layer1_output["protein_target_g"], layer1_output["fibre_target_g"],
        layer1_output["vitamin_focus"],
    )
    result = ask_claude_to_choose(layer1_output, meal_prep_time_min, candidates, client=client)

    chosen_name = result["chosen_recipe_name"]
    # Guaranteed to be found: chosen_name is enum-constrained to exactly
    # the names in `candidates` by build_output_schema().
    chosen_recipe = next(c for c in candidates if c["name"] == chosen_name)

    return {
        "status": "ok",
        "chosen_recipe": chosen_recipe,
        "explanation": result["explanation"],
        "candidates_considered": [build_candidate_summary(c) for c in candidates],
    }


if __name__ == "__main__":
    # Illustrative Layer 1 outputs — in real use these come from the
    # trained models in models/roti_count_model.joblib etc., not hardcoded.
    example_ok = {
        "roti_count": 3, "protein_target_g": 44.0, "fibre_target_g": 11.0, "vitamin_focus": "Iron",
    }
    example_no_match = dict(example_ok)

    print("=== Example 1: normal case ===")
    result = recommend_recipe(example_ok, meal_prep_time_min=25)
    if result["status"] == "no_match":
        print(f"No match: {result['reason']}")
    else:
        print(f"Chosen: {result['chosen_recipe']['name']}")
        print(f"Explanation: {result['explanation']}")
        print(f"(considered {len(result['candidates_considered'])} candidates)")

    print()
    print("=== Example 2: no recipe fits the time budget ===")
    result = recommend_recipe(example_no_match, meal_prep_time_min=5)
    print(json.dumps(result, indent=2))
