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

- [ ] **Phase 3 — Layer 1 ML**
  Missingness handling (explicit flags/imputation, decided from what Phase 2
  actually shows), time-based train/test split, baseline tabular model(s)
  for `roti_count`, `protein_target_g`, `fibre_target_g`, `vitamin_focus`,
  evaluation, error analysis.

- [ ] **Phase 4 — recipe_database.json (real data)**
  User supplies/verifies the real recipe entries against the Phase 1 schema.
  No LLM-generated nutrition values enter this file at any point.

- [ ] **Phase 5 — Layer 2 retrieval agent**
  Takes Layer 1 output + `meal_prep_time_min`, retrieves best-matching
  recipe from `recipe_database.json`, explains the match. Nutrition numbers
  are always read from the database, never computed by the LLM.

- [ ] **Phase 6 — Integration + evaluation**
  Wire Layer 1 → Layer 2 end-to-end, define what "good recommendation"
  means, test on held-out (chronologically later) days.

Phases beyond this list (packaging, UI, daily logging tool, etc.) will be
added here if/when they come up — not assumed in advance.
