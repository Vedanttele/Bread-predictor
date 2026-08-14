"""Recipe database loader/validator for roti-predictor (Layer 2 input).

Loads recipe_database/recipe_database.json, checks it for missing fields,
duplicate names, and out-of-range/unexpected values, and reports every
issue as a flag. Nothing in this module edits, corrects, or regenerates a
single value in the file — GOVERNANCE.md HC-1 in spirit: this data is
hand-maintained, code only reads and reports on it.

Run directly:
    python src/recipe_db.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "recipe_database" / "recipe_database.json"
REPORT_PATH = REPO_ROOT / "recipe_database" / "validation_report.md"

REQUIRED_FIELDS = [
    "name", "cuisine", "diet_type", "main_ingredients", "protein_g", "fibre_g",
    "calories", "prep_time_min", "vitamin_tags", "pairs_with_roti", "suitable_meal",
]

# Vocabularies observed as of the current file (see
# recipe_database/recipe_database.schema.json). A value outside these is
# not necessarily wrong — the database can legitimately grow — so it's
# reported as a warning to review, not an error.
EXPECTED_CUISINES = {"Indian", "European", "Mexican"}
EXPECTED_DIET_TYPES = {"vegetarian"}  # the whole database is described as vegetarian
EXPECTED_SUITABLE_MEALS = {"breakfast", "lunch", "dinner", "snack", "both"}
# Must match src/data_pipeline.py's vitamin_focus vocabulary — Layer 2 joins
# Layer 1's predicted vitamin_focus against this field, so drift here would
# silently break that match rather than error loudly.
EXPECTED_VITAMIN_TAGS = {"Iron", "B12", "Vitamin C", "Vitamin D", "Multivitamin"}

# Generous per-serving sanity bounds for a home-style vegetarian dish — NOT
# tuned to this dataset's observed min/max, deliberately wider so a
# legitimately unusual (but real) recipe doesn't get flagged just for being
# at the edge of what's currently in the file.
PROTEIN_G_RANGE = (0, 60)
FIBRE_G_RANGE = (0, 30)
CALORIES_RANGE = (0, 1500)
PREP_TIME_MIN_RANGE = (0, 240)
PROTEIN_KCAL_PER_G = 4  # standard macro conversion, used only for a self-consistency check


def flag(severity: str, recipe: str | None, field: str | None, message: str) -> dict:
    return {"severity": severity, "recipe": recipe, "field": field, "message": message}


def load_raw(path: Path = DEFAULT_DB_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}.")
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from e


def validate_top_level(data: dict) -> list[dict]:
    flags = []
    if "recipes" not in data or not isinstance(data["recipes"], list):
        raise ValueError("Top-level 'recipes' key missing or not a list — cannot validate further.")
    if "recipe_count" in data and data["recipe_count"] != len(data["recipes"]):
        flags.append(flag("warning", None, "recipe_count",
                           f"top-level recipe_count={data['recipe_count']} does not match "
                           f"actual len(recipes)={len(data['recipes'])}"))
    if "note" not in data:
        flags.append(flag("info", None, "note", "no top-level sourcing/methodology note present"))
    return flags


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_recipe(recipe: dict, index: int) -> list[dict]:
    flags = []
    name = recipe.get("name") if isinstance(recipe, dict) else None
    label = name if name else f"<recipe at index {index}, no name>"

    for field_name in REQUIRED_FIELDS:
        if field_name not in recipe or recipe[field_name] is None:
            flags.append(flag("error", label, field_name, "missing required field"))
    # Nothing further to check on fields that are entirely absent.
    present = {k: v for k, v in recipe.items() if k in REQUIRED_FIELDS}

    if "name" in present and (not isinstance(present["name"], str) or not present["name"].strip()):
        flags.append(flag("error", label, "name", "name must be a non-empty string"))

    if "main_ingredients" in present and (not isinstance(present["main_ingredients"], str) or not present["main_ingredients"].strip()):
        flags.append(flag("warning", label, "main_ingredients", "empty or non-string main_ingredients"))

    if "cuisine" in present:
        if not isinstance(present["cuisine"], str):
            flags.append(flag("error", label, "cuisine", "cuisine must be a string"))
        elif present["cuisine"] not in EXPECTED_CUISINES:
            flags.append(flag("warning", label, "cuisine",
                               f"unexpected cuisine '{present['cuisine']}' (expected one of {sorted(EXPECTED_CUISINES)})"))

    if "diet_type" in present:
        if not isinstance(present["diet_type"], str):
            flags.append(flag("error", label, "diet_type", "diet_type must be a string"))
        elif present["diet_type"] not in EXPECTED_DIET_TYPES:
            flags.append(flag("warning", label, "diet_type",
                               f"unexpected diet_type '{present['diet_type']}' (database is described as vegetarian-only)"))

    if "suitable_meal" in present:
        if not isinstance(present["suitable_meal"], str):
            flags.append(flag("error", label, "suitable_meal", "suitable_meal must be a string"))
        elif present["suitable_meal"] not in EXPECTED_SUITABLE_MEALS:
            flags.append(flag("warning", label, "suitable_meal",
                               f"unexpected suitable_meal '{present['suitable_meal']}' (expected one of {sorted(EXPECTED_SUITABLE_MEALS)})"))

    if "pairs_with_roti" in present and not isinstance(present["pairs_with_roti"], bool):
        flags.append(flag("error", label, "pairs_with_roti", "pairs_with_roti must be true/false"))

    if "vitamin_tags" in present:
        tags = present["vitamin_tags"]
        if not isinstance(tags, list) or not tags:
            flags.append(flag("error", label, "vitamin_tags", "vitamin_tags must be a non-empty list"))
        else:
            unexpected = [t for t in tags if t not in EXPECTED_VITAMIN_TAGS]
            if unexpected:
                flags.append(flag("warning", label, "vitamin_tags",
                                   f"unexpected vitamin tag(s) {unexpected} not in Layer 1's vocabulary "
                                   f"{sorted(EXPECTED_VITAMIN_TAGS)} — Layer 2 matching would silently miss these"))

    for field_name, bounds in [
        ("protein_g", PROTEIN_G_RANGE), ("fibre_g", FIBRE_G_RANGE),
        ("calories", CALORIES_RANGE), ("prep_time_min", PREP_TIME_MIN_RANGE),
    ]:
        if field_name not in present:
            continue
        value = present[field_name]
        if not _is_number(value):
            flags.append(flag("error", label, field_name, f"{field_name} must be numeric, got {type(value).__name__}"))
        elif not (bounds[0] <= value <= bounds[1]):
            flags.append(flag("warning", label, field_name,
                               f"{field_name}={value} is outside the sanity range {bounds} — review, not auto-corrected"))

    if _is_number(present.get("protein_g")) and _is_number(present.get("calories")):
        protein_kcal = present["protein_g"] * PROTEIN_KCAL_PER_G
        if protein_kcal > present["calories"]:
            flags.append(flag("warning", label, "protein_g/calories",
                               f"protein alone implies {protein_kcal:.0f} kcal ({present['protein_g']}g x {PROTEIN_KCAL_PER_G}), "
                               f"which exceeds the stated total calories ({present['calories']}) — macro/calorie mismatch"))

    return flags


def find_duplicate_names(recipes: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for i, r in enumerate(recipes):
        name = r.get("name")
        if isinstance(name, str) and name.strip():
            groups[name.strip().lower()].append((i, name))
    flags = []
    for key, entries in groups.items():
        if len(entries) > 1:
            indices = [i for i, _ in entries]
            names = [n for _, n in entries]
            flags.append(flag("error", "/".join(names), "name",
                               f"duplicate name (case/whitespace-insensitive match) at indices {indices}"))
    return flags


def check_cross_field_patterns(recipes: list[dict]) -> list[dict]:
    """Soft, informational observations — not errors — patterns that are
    plausibly intentional but worth a human glance."""
    flags = []
    by_cuisine_pairs = defaultdict(set)
    for r in recipes:
        cuisine, pairs = r.get("cuisine"), r.get("pairs_with_roti")
        if cuisine is not None and isinstance(pairs, bool):
            by_cuisine_pairs[cuisine].add(pairs)
    perfectly_separated = all(len(v) == 1 for v in by_cuisine_pairs.values()) and len(by_cuisine_pairs) > 1
    if perfectly_separated:
        summary = {c: next(iter(v)) for c, v in by_cuisine_pairs.items()}
        flags.append(flag("info", None, "pairs_with_roti/cuisine",
                           f"pairs_with_roti is fully determined by cuisine with no exceptions: {summary} - "
                           "plausible (roti is Indian bread) but worth confirming no fusion dish was intended "
                           "to break the pattern"))
    return flags


def validate_database(path: Path = DEFAULT_DB_PATH) -> tuple[dict, list[dict]]:
    data = load_raw(path)
    flags = list(validate_top_level(data))
    recipes = data["recipes"]
    for i, recipe in enumerate(recipes):
        flags.extend(validate_recipe(recipe, i))
    flags.extend(find_duplicate_names(recipes))
    flags.extend(check_cross_field_patterns(recipes))
    return data, flags


def build_report(data: dict, flags: list[dict]) -> str:
    lines = ["# Recipe database validation report", ""]
    lines.append(f"Generated by `src/recipe_db.py` from `recipe_database/recipe_database.json` "
                 f"({len(data.get('recipes', []))} recipes). Read-only — no value in the file was "
                 f"modified or regenerated by this script.")
    lines.append("")
    if "note" in data:
        lines.append(f"> File's own sourcing note: {data['note']}")
        lines.append("")

    by_severity = {"error": [], "warning": [], "info": []}
    for f in flags:
        by_severity[f["severity"]].append(f)

    lines.append(f"## Summary: {len(by_severity['error'])} error(s), "
                 f"{len(by_severity['warning'])} warning(s), {len(by_severity['info'])} info note(s)")
    lines.append("")

    for severity, title in [("error", "Errors (structural — should be fixed)"),
                             ("warning", "Warnings (worth reviewing)"),
                             ("info", "Info (observational only)")]:
        lines.append(f"### {title}")
        lines.append("")
        if not by_severity[severity]:
            lines.append("- (none)")
        else:
            for f in by_severity[severity]:
                where = f"**{f['recipe']}**" if f["recipe"] else "(database-level)"
                field_part = f" [`{f['field']}`]" if f["field"] else ""
                lines.append(f"- {where}{field_part}: {f['message']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    data, flags = validate_database()
    n_errors = sum(1 for f in flags if f["severity"] == "error")
    n_warnings = sum(1 for f in flags if f["severity"] == "warning")
    n_info = sum(1 for f in flags if f["severity"] == "info")

    print(f"Loaded {len(data['recipes'])} recipes from recipe_database/recipe_database.json")
    print(f"{n_errors} error(s), {n_warnings} warning(s), {n_info} info note(s)")
    print()
    for f in flags:
        where = f["recipe"] if f["recipe"] else "(database-level)"
        field_part = f"[{f['field']}] " if f["field"] else ""
        print(f"[{f['severity'].upper():7s}] {where}: {field_part}{f['message']}")

    report = build_report(data, flags)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print()
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    print()
    print("No value in recipe_database.json was modified. Stopping here.")
