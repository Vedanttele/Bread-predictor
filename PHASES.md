# Phase plan

Rule for every session: work one phase at a time, then stop and ask before
starting the next one. Do not chain phases in a single session.

- [x] **Phase 1 — Scaffolding + data contract**
  Folder structure, `data/SCHEMA.md` (raw feature schema, target schema,
  missingness rules, split rule), `recipe_database.schema.json`, README.
  No modeling, no real data ingestion.

- [x] **Phase 2 — Data ingestion + EDA**
  Real data landed in `data/raw/daily_context.csv` (220 days,
  2025-08-15 → 2026-03-22, one row/day). Ingestion
  (`src/roti_predictor/data_ingestion.py`) validates against the schema,
  derives a reliable `day_of_week`/`is_weekend` from `date` (raw `day_name`
  is unreliable — see `data/SCHEMA.md`), flags missingness explicitly per
  column (currently all zero — none observed in this file), and writes
  `data/processed/daily_context.csv`. EDA
  (`notebooks/phase2_eda.py` → `notebooks/phase2_eda_report.md`) covers
  target distributions and feature/target relationships. Tests in
  `tests/test_data_ingestion.py` (7 passing), including a checksum guard
  that `data/raw/` is never modified. `data/SCHEMA.md` updated to match the
  real schema (one row/day, not one row/meal).

- [x] **Phase 3 — Layer 1 ML**
  Feature signal analysis (`src/features.py`,
  `notebooks/feature_analysis_report.md`) found `calories_burned`,
  `hunger_level`, `curry_richness`, `workout_type`, `curry_type` as signal;
  `workout_duration_min` tested as signal too but was excluded (VIF=9.25,
  collinear with `calories_burned`/`workout_type`). Chronological 176/44
  train/test split (`src/train_model.py`, `src/train_vitamin_model.py`).
  Linear regression beat gradient boosting on `roti_count` (MAE 0.3864
  rounded), `protein_target_g` (MAE 2.5891g), `fibre_target_g` (MAE
  1.4034g); GBM won `vitamin_focus` (56.82% vs. 34.09% majority-class
  baseline). `roti_count`'s linear model also beat a manual heuristic
  baseline (0.3864 vs. 0.9773 MAE). Full numbers: `docs/project_report.md`.

- [x] **Phase 4 — recipe_database.json (real data)**
  24 real, verified recipes supplied (`recipe_database/recipe_database.json`
  — Indian 9, European 8, Mexican 7). Validated by `src/recipe_db.py`:
  0 errors, 0 warnings (`recipe_database/validation_report.md`). No
  LLM-generated nutrition values entered this file at any point.

- [x] **Phase 5 — Layer 2 retrieval agent**
  `src/recipe_agent.py` takes Layer 1 output + `meal_prep_time_min` (+
  optional `meal_type`), filters/ranks `recipe_database.json` in plain
  code, and asks Claude to pick one of the top 3 real candidates —
  constrained via a JSON-schema enum to exactly those 3 names, so it
  cannot name or invent a value outside what it was shown. Returns an
  explicit `no_match` (API never called) if nothing fits.

- [x] **Phase 6 — Integration + interfaces**
  Layer 1 → Layer 2 wired end-to-end behind two interfaces: a Streamlit
  dashboard (`src/app.py`) and a CLI (`src/cli.py`), both built on shared
  logic in `src/app_logic.py`. Logging what you actually ate feeds back
  into retraining via `data/raw/app_actuals.csv` (a documented, narrow
  exception to `GOVERNANCE.md` HC-4), merged automatically by
  `src/data_pipeline.py` on the next training run. 119 tests passing
  across the whole pipeline.

Phases beyond this list will be added here if/when they come up — not
assumed in advance.
