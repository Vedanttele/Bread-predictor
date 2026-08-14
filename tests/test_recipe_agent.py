"""Tests for src/recipe_agent.py. The Claude API is always mocked here —
no test in this file makes a real network call or requires credentials."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import recipe_agent as agent  # noqa: E402

REAL_DB_PATH = REPO_ROOT / "recipe_database" / "recipe_database.json"


def make_recipe(**overrides):
    base = {
        "name": "Test Dal", "cuisine": "Indian", "diet_type": "vegetarian",
        "main_ingredients": "lentils, tomato", "protein_g": 15.0, "fibre_g": 8.0,
        "calories": 220, "prep_time_min": 20, "vitamin_tags": ["Iron"],
        "pairs_with_roti": True, "suitable_meal": "dinner",
    }
    base.update(overrides)
    return base


def fake_anthropic_client(chosen_recipe_name: str, explanation: str = "Because it fits well."):
    """A MagicMock standing in for anthropic.Anthropic() — returns a
    canned response shaped like the real SDK's Message object."""
    client = MagicMock()
    response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps({"chosen_recipe_name": chosen_recipe_name, "explanation": explanation})
    response.content = [text_block]
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_filter_by_meal_type_matches_exact():
    recipes = [make_recipe(name="Breakfast One", suitable_meal="breakfast"),
               make_recipe(name="Dinner One", suitable_meal="dinner")]
    result = agent.filter_by_meal_type(recipes, "breakfast")
    assert [r["name"] for r in result] == ["Breakfast One"]


def test_filter_by_meal_type_both_always_qualifies():
    recipes = [make_recipe(name="Both One", suitable_meal="both"),
               make_recipe(name="Dinner One", suitable_meal="dinner")]
    result = agent.filter_by_meal_type(recipes, "lunch")
    assert [r["name"] for r in result] == ["Both One"]


def test_filter_by_meal_type_none_means_no_filter():
    recipes = [make_recipe(name="A", suitable_meal="breakfast"), make_recipe(name="B", suitable_meal="dinner")]
    assert agent.filter_by_meal_type(recipes, None) == recipes
    assert agent.filter_by_meal_type(recipes, "") == recipes



def test_filter_by_prep_time_excludes_recipes_over_budget():
    recipes = [make_recipe(name="Fast", prep_time_min=10), make_recipe(name="Slow", prep_time_min=45)]
    result = agent.filter_by_prep_time(recipes, meal_prep_time_min=20)
    assert [r["name"] for r in result] == ["Fast"]


def test_filter_by_prep_time_is_inclusive_at_the_boundary():
    recipes = [make_recipe(name="Exact", prep_time_min=20)]
    result = agent.filter_by_prep_time(recipes, meal_prep_time_min=20)
    assert len(result) == 1


def test_filter_by_prep_time_empty_when_nothing_fits():
    recipes = [make_recipe(prep_time_min=30)]
    assert agent.filter_by_prep_time(recipes, meal_prep_time_min=10) == []


# ---------------------------------------------------------------------------
# Scoring / ranking
# ---------------------------------------------------------------------------

def test_score_recipe_zero_distance_matching_vitamin():
    recipe = make_recipe(protein_g=40.0, fibre_g=10.0, vitamin_tags=["Iron"])
    score = agent.score_recipe(recipe, protein_target_g=40.0, fibre_target_g=10.0, vitamin_focus="Iron")
    assert score == -agent.VITAMIN_MATCH_BONUS


def test_score_recipe_no_vitamin_match_no_bonus():
    recipe = make_recipe(protein_g=40.0, fibre_g=10.0, vitamin_tags=["Vitamin C"])
    score = agent.score_recipe(recipe, protein_target_g=40.0, fibre_target_g=10.0, vitamin_focus="Iron")
    assert score == 0.0


def test_score_recipe_sums_absolute_gaps():
    recipe = make_recipe(protein_g=30.0, fibre_g=5.0, vitamin_tags=[])
    score = agent.score_recipe(recipe, protein_target_g=40.0, fibre_target_g=10.0, vitamin_focus="Iron")
    assert score == pytest.approx(10.0 + 5.0)


def test_rank_candidates_prefers_closer_nutrition_match():
    close = make_recipe(name="Close", protein_g=44.0, fibre_g=11.0, vitamin_tags=["Iron"])
    far = make_recipe(name="Far", protein_g=10.0, fibre_g=2.0, vitamin_tags=["Iron"])
    ranked = agent.rank_candidates([far, close], protein_target_g=44.0, fibre_target_g=11.0, vitamin_focus="Iron")
    assert ranked[0]["name"] == "Close"


def test_rank_candidates_vitamin_match_can_win_a_close_tiebreak():
    matches = make_recipe(name="Matches", protein_g=38.0, fibre_g=9.0, vitamin_tags=["Iron"])
    no_match = make_recipe(name="NoMatch", protein_g=40.0, fibre_g=10.0, vitamin_tags=["Vitamin C"])
    ranked = agent.rank_candidates(
        [no_match, matches], protein_target_g=40.0, fibre_target_g=10.0, vitamin_focus="Iron",
    )
    assert ranked[0]["name"] == "Matches"  # small nutrition gap (3) beats no-match's 0 once bonus applies


def test_rank_candidates_respects_top_n():
    recipes = [make_recipe(name=f"R{i}", protein_g=float(i)) for i in range(10)]
    ranked = agent.rank_candidates(recipes, protein_target_g=0, fibre_target_g=0, vitamin_focus="Iron", top_n=3)
    assert len(ranked) == 3


# ---------------------------------------------------------------------------
# Prompt / schema construction — the leakage guardrails
# ---------------------------------------------------------------------------

def test_output_schema_enum_is_exactly_the_candidate_names():
    candidates = [make_recipe(name="A"), make_recipe(name="B"), make_recipe(name="C")]
    schema = agent.build_output_schema(candidates)
    assert schema["properties"]["chosen_recipe_name"]["enum"] == ["A", "B", "C"]
    assert schema["required"] == ["chosen_recipe_name", "explanation"]
    assert schema["additionalProperties"] is False


def test_prompt_contains_only_candidate_data_not_full_database():
    shown = make_recipe(name="Shown Recipe")
    hidden = make_recipe(name="Hidden Recipe", protein_g=999.0)
    layer1 = {"roti_count": 2, "protein_target_g": 40.0, "fibre_target_g": 10.0, "vitamin_focus": "Iron"}
    prompt = agent.build_prompt(layer1, meal_prep_time_min=20, candidates=[shown])
    assert "Shown Recipe" in prompt
    assert "Hidden Recipe" not in prompt
    assert "999.0" not in prompt


def test_prompt_instructs_against_inventing_numbers():
    candidates = [make_recipe()]
    layer1 = {"roti_count": 2, "protein_target_g": 40.0, "fibre_target_g": 10.0, "vitamin_focus": "Iron"}
    prompt = agent.build_prompt(layer1, meal_prep_time_min=20, candidates=candidates)
    assert "do not invent" in prompt.lower()


def test_candidate_summary_only_includes_declared_fields():
    recipe = make_recipe(name="X")
    summary = agent.build_candidate_summary(recipe)
    assert set(summary.keys()) == set(agent.CANDIDATE_FIELDS)


# ---------------------------------------------------------------------------
# End-to-end pipeline (mocked client)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REAL_DB_PATH.exists(), reason="recipe_database/recipe_database.json not present")
class TestPipelineOnRealData:
    LAYER1 = {"roti_count": 3, "protein_target_g": 44.0, "fibre_target_g": 11.0, "vitamin_focus": "Iron"}

    def test_no_match_never_calls_the_api(self):
        client = fake_anthropic_client("shouldn't matter")
        result = agent.recommend_recipe(self.LAYER1, meal_prep_time_min=1, client=client)
        assert result["status"] == "no_match"
        assert "reason" in result
        client.messages.create.assert_not_called()

    def test_meal_type_filter_applied_end_to_end(self, monkeypatch):
        # 'both'-tagged recipes always qualify regardless of the requested
        # meal_type, so a genuine no-match-on-meal_type case needs data
        # where nothing is tagged the requested type or 'both' — construct
        # that explicitly rather than relying on the real DB's exact mix.
        synthetic = [make_recipe(name="Dinner Only", suitable_meal="dinner", prep_time_min=10)]
        monkeypatch.setattr(agent, "load_recipes", lambda path=None: synthetic)

        client = fake_anthropic_client("shouldn't matter")
        result = agent.recommend_recipe(
            self.LAYER1, meal_prep_time_min=240, meal_type="breakfast", client=client,
        )
        assert result["status"] == "no_match"
        assert "breakfast" in result["reason"]
        client.messages.create.assert_not_called()

    def test_meal_type_both_recipe_qualifies_for_any_requested_meal(self, monkeypatch):
        synthetic = [make_recipe(name="Flex Meal", suitable_meal="both", prep_time_min=10,
                                  protein_g=44.0, fibre_g=11.0, vitamin_tags=["Iron"])]
        monkeypatch.setattr(agent, "load_recipes", lambda path=None: synthetic)

        client = fake_anthropic_client("Flex Meal")
        result = agent.recommend_recipe(
            self.LAYER1, meal_prep_time_min=240, meal_type="breakfast", client=client,
        )
        assert result["status"] == "ok"
        assert result["chosen_recipe"]["name"] == "Flex Meal"

    def test_meal_type_none_is_backward_compatible(self):
        recipes = agent.load_recipes()
        fitting = agent.filter_by_prep_time(recipes, 25)
        top3 = agent.rank_candidates(fitting, 44.0, 11.0, "Iron")
        client = fake_anthropic_client(top3[0]["name"])
        result = agent.recommend_recipe(self.LAYER1, meal_prep_time_min=25, client=client)
        assert result["status"] == "ok"

    def test_ok_path_returns_chosen_recipe_from_real_db(self):
        recipes = agent.load_recipes()
        fitting = agent.filter_by_prep_time(recipes, 25)
        top3 = agent.rank_candidates(fitting, 44.0, 11.0, "Iron")
        client = fake_anthropic_client(top3[0]["name"])

        result = agent.recommend_recipe(self.LAYER1, meal_prep_time_min=25, client=client)
        assert result["status"] == "ok"
        assert result["chosen_recipe"]["name"] == top3[0]["name"]
        assert result["explanation"]
        assert len(result["candidates_considered"]) == agent.TOP_N

    def test_api_call_uses_expected_model_and_structured_output(self):
        recipes = agent.load_recipes()
        fitting = agent.filter_by_prep_time(recipes, 25)
        top3 = agent.rank_candidates(fitting, 44.0, 11.0, "Iron")
        client = fake_anthropic_client(top3[0]["name"])

        agent.recommend_recipe(self.LAYER1, meal_prep_time_min=25, client=client)
        _, kwargs = client.messages.create.call_args
        assert kwargs["model"] == agent.MODEL
        assert kwargs["output_config"]["format"]["type"] == "json_schema"

    def test_all_candidates_fit_the_time_budget(self):
        recipes = agent.load_recipes()
        meal_prep_time_min = 20
        fitting = agent.filter_by_prep_time(recipes, meal_prep_time_min)
        top3 = agent.rank_candidates(fitting, 44.0, 11.0, "Iron")
        assert all(c["prep_time_min"] <= meal_prep_time_min for c in top3)

    def test_recipe_db_never_modified_by_the_pipeline(self):
        before = REAL_DB_PATH.read_bytes()
        recipes = agent.load_recipes()
        fitting = agent.filter_by_prep_time(recipes, 30)
        top3 = agent.rank_candidates(fitting, 44.0, 11.0, "Iron")
        client = fake_anthropic_client(top3[0]["name"])
        agent.recommend_recipe(self.LAYER1, meal_prep_time_min=30, client=client)
        after = REAL_DB_PATH.read_bytes()
        assert before == after
