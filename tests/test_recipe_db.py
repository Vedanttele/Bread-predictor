"""Tests for src/recipe_db.py.

The real recipe_database.json is clean (0 errors, 0 warnings), so synthetic
fixtures exercise the branches it doesn't: missing fields, duplicate names,
out-of-range values, unexpected categories, and the macro/calorie check.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import recipe_db as rdb  # noqa: E402

REAL_DB_PATH = REPO_ROOT / "recipe_database" / "recipe_database.json"


def make_recipe(**overrides):
    base = {
        "name": "Test Dal", "cuisine": "Indian", "diet_type": "vegetarian",
        "main_ingredients": "lentils, tomato", "protein_g": 12.0, "fibre_g": 6.0,
        "calories": 220, "prep_time_min": 20, "vitamin_tags": ["Iron"],
        "pairs_with_roti": True, "suitable_meal": "dinner",
    }
    base.update(overrides)
    return base


def test_valid_recipe_has_no_flags():
    flags = rdb.validate_recipe(make_recipe(), 0)
    assert flags == []


def test_missing_required_field_is_an_error():
    recipe = make_recipe()
    del recipe["protein_g"]
    flags = rdb.validate_recipe(recipe, 0)
    assert any(f["severity"] == "error" and f["field"] == "protein_g" for f in flags)


def test_negative_protein_is_out_of_range_warning():
    flags = rdb.validate_recipe(make_recipe(protein_g=-5), 0)
    assert any(f["severity"] == "warning" and f["field"] == "protein_g" for f in flags)


def test_absurdly_high_calories_is_out_of_range_warning():
    flags = rdb.validate_recipe(make_recipe(calories=5000), 0)
    assert any(f["severity"] == "warning" and f["field"] == "calories" for f in flags)


def test_value_within_wide_sanity_bounds_is_not_flagged_even_if_unusual_for_dataset():
    # 55g protein is well above anything in the real file (max ~19g) but
    # still inside the generous sanity range -> should NOT be flagged.
    flags = rdb.validate_recipe(make_recipe(protein_g=55), 0)
    assert not any(f["field"] == "protein_g" for f in flags)


def test_unexpected_cuisine_is_a_warning_not_an_error():
    flags = rdb.validate_recipe(make_recipe(cuisine="Thai"), 0)
    matches = [f for f in flags if f["field"] == "cuisine"]
    assert len(matches) == 1
    assert matches[0]["severity"] == "warning"  # not fatal, database can grow


def test_non_vegetarian_diet_type_flagged():
    flags = rdb.validate_recipe(make_recipe(diet_type="non-vegetarian"), 0)
    assert any(f["field"] == "diet_type" for f in flags)


def test_unexpected_vitamin_tag_flagged():
    flags = rdb.validate_recipe(make_recipe(vitamin_tags=["Zinc"]), 0)
    assert any(f["field"] == "vitamin_tags" and "Zinc" in f["message"] for f in flags)


def test_empty_vitamin_tags_is_an_error():
    flags = rdb.validate_recipe(make_recipe(vitamin_tags=[]), 0)
    assert any(f["severity"] == "error" and f["field"] == "vitamin_tags" for f in flags)


def test_pairs_with_roti_wrong_type_is_an_error():
    flags = rdb.validate_recipe(make_recipe(pairs_with_roti="yes"), 0)
    assert any(f["severity"] == "error" and f["field"] == "pairs_with_roti" for f in flags)


def test_protein_calorie_mismatch_flagged():
    # 30g protein * 4 kcal/g = 120 kcal, exceeds the stated 100 total.
    flags = rdb.validate_recipe(make_recipe(protein_g=30, calories=100), 0)
    assert any("macro/calorie mismatch" in f["message"] for f in flags)


def test_duplicate_names_case_and_whitespace_insensitive():
    recipes = [make_recipe(name="Chana Masala"), make_recipe(name=" chana masala ")]
    flags = rdb.find_duplicate_names(recipes)
    assert len(flags) == 1
    assert flags[0]["severity"] == "error"


def test_no_duplicates_when_names_distinct():
    recipes = [make_recipe(name="Chana Masala"), make_recipe(name="Rajma Curry")]
    assert rdb.find_duplicate_names(recipes) == []


def test_cross_field_pattern_flags_perfect_correlation():
    recipes = [
        make_recipe(cuisine="Indian", pairs_with_roti=True),
        make_recipe(cuisine="Indian", pairs_with_roti=True),
        make_recipe(cuisine="Mexican", pairs_with_roti=False),
    ]
    flags = rdb.check_cross_field_patterns(recipes)
    assert len(flags) == 1
    assert flags[0]["severity"] == "info"


def test_cross_field_pattern_silent_when_pattern_broken():
    recipes = [
        make_recipe(cuisine="Indian", pairs_with_roti=True),
        make_recipe(cuisine="Indian", pairs_with_roti=False),  # Indian now has both values -> not fully determined
        make_recipe(cuisine="Mexican", pairs_with_roti=False),
    ]
    assert rdb.check_cross_field_patterns(recipes) == []


def test_recipe_count_mismatch_flagged(tmp_path):
    db = {"recipe_count": 5, "recipes": [make_recipe()]}
    flags = rdb.validate_top_level(db)
    assert any(f["field"] == "recipe_count" for f in flags)


def test_missing_recipes_key_raises():
    with pytest.raises(ValueError):
        rdb.validate_top_level({"note": "x"})


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rdb.load_raw(tmp_path / "does_not_exist.json")


def test_load_raw_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        rdb.load_raw(bad)


# ---------------------------------------------------------------------------
# Real file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_DB_PATH.exists(), reason="recipe_database/recipe_database.json not present")
class TestOnRealData:
    def test_real_file_has_no_errors(self):
        data, flags = rdb.validate_database()
        errors = [f for f in flags if f["severity"] == "error"]
        assert errors == []

    def test_real_file_has_24_recipes(self):
        data, flags = rdb.validate_database()
        assert len(data["recipes"]) == 24

    def test_real_file_never_modified(self):
        before = REAL_DB_PATH.read_bytes()
        rdb.validate_database()
        after = REAL_DB_PATH.read_bytes()
        assert before == after

    def test_real_file_flags_pairs_with_roti_cuisine_pattern(self):
        data, flags = rdb.validate_database()
        assert any(f["field"] == "pairs_with_roti/cuisine" for f in flags)
