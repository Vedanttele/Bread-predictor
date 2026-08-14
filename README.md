# roti-predictor

A personal AI project that predicts how many Indian rotis (Bread) to eat per meal — and
recommends a matching recipe — based on daily context (workout, sleep, mood,
curry type, hunger).

## Architecture

Two layers, deliberately kept separate:

**Layer 1 — ML (tabular, not deep learning).**
Input: daily context features (workout, sleep, mood, curry type, hunger, etc.).
Output: `roti_count`, `protein_target_g`, `fibre_target_g`, `vitamin_focus`.
Dataset is small, so this stays classic tabular ML (e.g. gradient-boosted
trees / linear models), not a neural net.

**Layer 2 — AI/agent (retrieval, not generation).**
Input: Layer 1's output + `meal_prep_time_min`.
Behavior: retrieves the best-matching recipe from `recipe_database/` and
explains the match in natural language.
It does **not** invent nutrition numbers — see hard constraints below.

```
daily context ──▶ [Layer 1: ML model] ──▶ roti_count, protein_target_g,
                                            fibre_target_g, vitamin_focus
                                                    │
                          meal_prep_time_min ───────┤
                                                    ▼
                                    [Layer 2: retrieval agent]
                                                    │
                                                    ▼
                              matched recipe + explanation
                            (nutrition numbers straight from
                             recipe_database, never invented)
```

## Hard constraints

Foundational requirements — no invented nutrition numbers, explicit
missingness handling, time-based splits only, `data/raw/` is read-only, one
phase per session — live in **[`GOVERNANCE.md`](./GOVERNANCE.md)**, the
single source of truth for them. Update that file (not this section) when a
foundational requirement changes.

## Project layout

```
roti-predictor/
├── GOVERNANCE.md            # foundational requirements — single source of truth
├── data/
│   ├── raw/                 # read-only real data (user-provided), see data/raw/README.md
│   ├── processed/           # derived/cleaned data, safe to regenerate
│   └── SCHEMA.md            # data contract: raw schema, targets, missingness, split rule
├── recipe_database/
│   ├── recipe_database.schema.json   # JSON Schema for a recipe entry
│   └── recipe_database.example.json  # illustrative FAKE entries only — not real data
├── src/roti_predictor/
│   ├── layer1_ml/           # tabular model: features -> roti_count/protein/fibre/vitamin
│   └── layer2_agent/        # retrieval + explanation over recipe_database.json
├── notebooks/                # exploratory analysis
├── tests/
└── PHASES.md                 # phase plan and current status
```

## Status

See `PHASES.md` for the phase plan. Current phase: **Phase 1 — scaffolding
and data contract** (this commit). No model code, no data ingestion, and no
real data yet — that starts in Phase 2, once real files are in `data/raw/`.
