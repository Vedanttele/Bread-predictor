"""Streamlit dashboard for roti-predictor.

Two tabs:
  - "Today": enter today's context, get Layer 1's predicted targets and
    Layer 2's recipe recommendation, then log what you actually ate.
  - "Dashboard": target-vs-actual trends over every logged day.

All business logic lives in src/app_logic.py (testable independently of
Streamlit); this file is UI wiring only. See app_logic.py's docstring for
the log schema and the "never invent nutrition numbers" rule it enforces.

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from datetime import date, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_logic as al  # noqa: E402
from recipe_agent import DEFAULT_DB_PATH, load_recipes, recommend_recipe  # noqa: E402

st.set_page_config(page_title="Roti Predictor", page_icon="🫓", layout="wide")

WORKOUT_TYPES = ["none", "cardio", "strength"]
CURRY_TYPES = ["dal", "dry", "gravy"]
CURRY_RICHNESS = ["light", "heavy"]


@st.cache_resource
def get_models():
    return al.load_models()


@st.cache_data
def get_recipes():
    return load_recipes(DEFAULT_DB_PATH)


st.title("🫓 Roti Predictor")

tab_today, tab_dashboard = st.tabs(["Today", "Dashboard"])

# ---------------------------------------------------------------------------
# Tab: Today
# ---------------------------------------------------------------------------
with tab_today:
    st.subheader("Today's context")
    with st.form("context_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            calories_burned = st.number_input("Calories burned", min_value=1000, max_value=4000, value=2100, step=10)
            workout_type = st.selectbox("Workout type", WORKOUT_TYPES)
        with col2:
            sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
            curry_type = st.selectbox("Curry type", CURRY_TYPES)
        with col3:
            hunger_level = st.slider("Hunger level", min_value=1, max_value=5, value=3)
            curry_richness = st.selectbox("Curry richness", CURRY_RICHNESS)
        meal_prep_time_min = st.number_input("Minutes available to cook", min_value=0, max_value=240, value=20, step=5)
        submitted = st.form_submit_button("Predict my targets")

    if submitted:
        try:
            models = get_models()
        except FileNotFoundError as e:
            st.error(str(e))
            models = None

        if models is not None:
            inputs = {
                "calories_burned": calories_burned, "hunger_level": hunger_level, "sleep_hours": sleep_hours,
                "workout_type": workout_type, "curry_type": curry_type, "curry_richness": curry_richness,
                "meal_prep_time_min": meal_prep_time_min,
            }
            feature_row = al.build_feature_row(inputs)
            target = al.predict_layer1(models, feature_row)
            st.session_state["inputs"] = inputs
            st.session_state["target"] = target
            st.session_state["recommendation"] = None  # cleared until we ask Claude below

    if "target" in st.session_state:
        target = st.session_state["target"]
        st.subheader("Today's targets")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Roti count", target["roti_count"])
        m2.metric("Protein target", f"{target['protein_target_g']:.1f} g")
        m3.metric("Fibre target", f"{target['fibre_target_g']:.1f} g")
        m4.metric("Vitamin focus", target["vitamin_focus"])

        st.subheader("Recipe recommendation")
        if st.button("Get recipe recommendation"):
            with st.spinner("Asking Claude to pick a recipe..."):
                try:
                    result = recommend_recipe(target, st.session_state["inputs"]["meal_prep_time_min"])
                    st.session_state["recommendation"] = result
                except Exception as e:  # noqa: BLE001 — surface any API/auth error plainly, don't crash the app
                    st.error(
                        f"Couldn't reach Claude: {e}\n\n"
                        "If this is an authentication error, set ANTHROPIC_API_KEY or run "
                        "`ant auth login` in a terminal, then try again."
                    )
                    st.session_state["recommendation"] = None

        recommendation = st.session_state.get("recommendation")
        if recommendation is not None:
            if recommendation["status"] == "no_match":
                st.warning(recommendation["reason"])
            else:
                recipe = recommendation["chosen_recipe"]
                st.success(f"**{recipe['name']}** ({recipe['cuisine']})")
                st.write(recommendation["explanation"])
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                rcol1.metric("Protein", f"{recipe['protein_g']:.1f} g")
                rcol2.metric("Fibre", f"{recipe['fibre_g']:.1f} g")
                rcol3.metric("Calories", recipe["calories"])
                rcol4.metric("Prep time", f"{recipe['prep_time_min']} min")

        st.divider()
        st.subheader("Log what you actually ate")
        recipes = get_recipes()
        recipe_names = sorted(r["name"] for r in recipes)
        recommended_name = None
        if recommendation is not None and recommendation["status"] == "ok":
            recommended_name = recommendation["chosen_recipe"]["name"]
        default_index = recipe_names.index(recommended_name) if recommended_name in recipe_names else 0

        with st.form("actual_form"):
            log_date = st.date_input("Date", value=date.today())
            actual_roti_count = st.number_input(
                "Rotis actually eaten", min_value=0, max_value=10, value=target["roti_count"], step=1,
            )
            actual_recipe_name = st.selectbox("Recipe actually eaten", recipe_names, index=default_index)
            log_submitted = st.form_submit_button("Save log entry")

        if log_submitted:
            actual_recipe = al.lookup_recipe(actual_recipe_name, recipes)
            row = al.build_log_row(
                date=log_date.isoformat(),
                inputs=st.session_state["inputs"],
                target=target,
                recommended_recipe_name=recommended_name,
                actual_roti_count=actual_roti_count,
                actual_recipe=actual_recipe,
            )
            al.append_log_row(row)
            st.success(
                f"Logged. Roti delta: {row['delta_roti_count']:+d}, "
                f"protein delta: {row['delta_protein_g']:+.1f} g, "
                f"fibre delta: {row['delta_fibre_g']:+.1f} g, "
                f"vitamin match: {'yes' if row['vitamin_match'] else 'no'}."
            )

# ---------------------------------------------------------------------------
# Tab: Dashboard
# ---------------------------------------------------------------------------
with tab_dashboard:
    log_df = al.load_log()
    if log_df.empty:
        st.info("No logged days yet — log a meal in the **Today** tab to start building this dashboard.")
    else:
        st.subheader("Most recent day")
        latest = log_df.iloc[-1]
        protein_pct = latest["actual_protein_g"] / latest["target_protein_target_g"] * 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Roti count", int(latest["actual_roti_count"]), delta=int(latest["delta_roti_count"]))
        c2.metric("Protein (this recipe)", f"{protein_pct:.0f}% of target",
                  help="A single recipe's protein vs. your daily/meal protein target aren't the same "
                       "scale (see the note below the trend chart) — shown as % of target, not grams.")
        c3.metric("Fibre", f"{latest['actual_fibre_g']:.1f} g", delta=f"{latest['delta_fibre_g']:+.1f} g")
        c4.metric("Vitamin match", "✅" if latest["vitamin_match"] else "❌")

        st.subheader("Trends: target vs. actual")
        st.plotly_chart(
            al.build_trend_chart(log_df, "target_roti_count", "actual_roti_count", "Rotis"),
            width='stretch',
        )

        st.plotly_chart(
            al.build_pct_of_target_chart(log_df, "target_protein_target_g", "actual_protein_g", "protein"),
            width='stretch',
        )
        st.caption(
            "Shown as % of target, not grams: recipe_database.json's recipes top out at 19.0g protein "
            "(a single dish), while protein_target_g ranges 31.2-52.0g in the training data — very "
            "plausibly a whole-day or whole-meal goal, not one dish's contribution. No single recipe "
            "can reach that on its own, so a grams-vs-grams chart always showed the same large gap "
            "regardless of how good the pick was. See docs/project_report.md for the full note."
        )

        st.plotly_chart(
            al.build_trend_chart(log_df, "target_fibre_target_g", "actual_fibre_g", "Fibre (g)"),
            width='stretch',
        )

        st.subheader("Vitamin focus: target vs. what you actually ate")
        vitamin_view = log_df[["date", "target_vitamin_focus", "actual_recipe_name", "actual_vitamin_tags", "vitamin_match"]].copy()
        vitamin_view["vitamin_match"] = vitamin_view["vitamin_match"].map({True: "✅", False: "❌"})
        st.dataframe(vitamin_view, width='stretch', hide_index=True)

        st.subheader("Full log")
        st.dataframe(log_df, width='stretch', hide_index=True)
