# roti-predictor — Project Report

A personal project that predicts how many Indian rotis to eat per meal and
recommends a matching recipe, based on daily context (workout, sleep,
hunger, curry choice). Built as two deliberately separate layers: a small
tabular ML model for the numeric prediction, and a retrieval-grounded LLM
agent for the recipe explanation — chosen specifically so the model never
has to invent a nutrition fact.

Every number in this report is taken directly from this repo's own
generated reports (`notebooks/*.md`, `models/*_metadata.json`,
`recipe_database/validation_report.md`) or from commands re-run against
the current code while writing this report — nothing here is estimated.

## Architecture

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

**Layer 1 — tabular ML, not deep learning.** Six context features in,
four targets out (`roti_count`, `protein_target_g`, `fibre_target_g`,
`vitamin_focus`). Two candidate model families were trained and compared
per target — plain linear/logistic regression and gradient-boosted trees
— with the lower-error candidate saved. Deliberately kept to classic,
low-parameter tabular models given the dataset size (see Limitations).

**Layer 2 — retrieval, not generation.** Takes Layer 1's output plus
`meal_prep_time_min`, filters and ranks `recipe_database.json` by
deterministic code (not an LLM call), and only then asks Claude to pick
one of the top 3 candidates and explain the choice — with the recipe
database's real values as the only numbers in the prompt.

A CLI (`src/cli.py`) and a Streamlit dashboard (`src/app.py`) sit on top
of both layers for daily use, plus a feedback loop
(`data/raw/app_actuals.csv`) that lets what you actually ate flow back
into future retraining without ever touching the original logged dataset.

## Dataset

- **220 rows**, one row per day, `2025-08-15` → `2026-03-22`, no calendar
  gaps.
- **Zero missing values** in the current file — all 14 nullable columns
  are flagged and tracked (`*_missing` companion columns), but every flag
  is currently `False`. The missingness-handling code path exists and is
  tested, but is untested *by real gaps* — see Limitations.
- **Chronological train/test split**, never random: 176 train days / 44
  test days (last 20%), per `GOVERNANCE.md` HC-3.
- `roti_count` distribution: 1→23, 2→67, 3→93, 4→31, 5→6 (n=220).
- `vitamin_focus` distribution: Vitamin C→73, Vitamin D→54, B12→39,
  Iron→29, Multivitamin→25.

## Layer 1 results (test set, n=44, chronological hold-out)

| Target | Linear | GBM | Selected |
|---|---:|---:|---|
| `roti_count` (MAE, rounded to valid 1–5 range) | **0.3864** | 0.4545 | linear |
| `roti_count` (MAE, raw prediction) | 0.4776 | 0.4992 | — |
| `protein_target_g` (MAE, grams) | **2.5891** | 2.6417 | linear |
| `fibre_target_g` (MAE, grams) | **1.4034** | 1.707 | linear |
| `vitamin_focus` (accuracy, 5-way) | 0.5000 | **0.5682** | gbm |

Linear models won 3 of 4 targets — unsurprising with 176 training rows;
gradient boosting has more capacity than the data supports here. For
`vitamin_focus` (categorical, 5 classes), GBM was compared against a
majority-class baseline of **0.3409** accuracy — both trained candidates
clear it, GBM by the wider margin.

### Heuristic comparison (`roti_count`)

A manual, hand-written reference rule was built and compared against the
trained models on the same held-out 44 days — deliberately *not* tuned to
this dataset's numbers (only the direction of each rule — more hunger, a
workout, richer curry → more rotis — reflects common-sense reasoning, not
fitted coefficients), so it's a genuine independent baseline rather than a
copy of what the model already learned:

| Model | MAE (rounded) |
|---|---:|
| Linear regression | **0.3864** |
| Gradient boosting | 0.4545 |
| Manual heuristic | 0.9773 |

Both trained models clearly beat the heuristic — roughly 2.5× lower error
for the selected linear model. This result was reported as-is regardless
of which way it came out; the comparison harness prints `PENDING` rather
than a fabricated number if a heuristic isn't supplied, specifically so a
result like "ML doesn't actually help" couldn't get silently glossed over.

## What features turned out to matter

Full statistical detail in `notebooks/feature_analysis_report.md`. Verdicts
use Cohen's (1988) effect-size conventions: `signal` = moderate-or-larger
effect **and** p<0.05; `weak signal` = small effect regardless of
significance; otherwise `noise-like`.

| Feature | Metric | Value | p-value | Verdict |
|---|---|---:|---:|---|
| `calories_burned` | Pearson r | 0.5818 | <0.001 | **signal** |
| `hunger_level` | Pearson r | 0.5253 | <0.001 | **signal** |
| `workout_duration_min` | Pearson r | 0.4683 | <0.001 | **signal**, but excluded (see below) |
| `curry_richness` | η² | 0.2596 | <0.001 | **signal** |
| `workout_type` | η² | 0.1832 | <0.001 | **signal** |
| `curry_type` | η² | 0.1810 | <0.001 | **signal** |
| `sleep_hours` | Pearson r | 0.1697 | 0.0117 | weak signal |
| `day_of_week` | η² | 0.0510 | 0.0811 | weak signal (not significant) |
| `is_weekend` | Pearson r | -0.0711 | 0.2935 | noise-like |
| `meal_prep_time_min` | Pearson r | -0.0549 | 0.4174 | noise-like |
| `mood_score` | Pearson r | -0.0447 | 0.5092 | noise-like |
| `weight_trend` | η² | 0.0004 | 0.7788 | noise-like |
| `meal_type` | — | — | — | not assessable — constant in this data (no source logs it) |

**`workout_duration_min` was excluded from the final feature set despite
testing as signal**, because it's collinear with `calories_burned` and
`workout_type`:

- Pearson r(`calories_burned`, `workout_duration_min`) = **0.859**
  (p=2.33e-65)
- η²(`workout_type` → `calories_burned`) = 0.643; η²(`workout_type` →
  `workout_duration_min`) = 0.820
- VIF(`workout_duration_min`) = **9.25**, VIF(`calories_burned`) = 4.99 —
  the first crosses the conventional severe-multicollinearity threshold
  (>10 is generally considered severe, ≥5 worth a second look)

**Final feature set used by every trained model:**
numeric = `calories_burned`, `hunger_level`, `sleep_hours`;
categorical = `workout_type`, `curry_type`, `curry_richness`.

## Layer 2: retrieval grounding, and why it can't hallucinate a number

Layer 2 (`src/recipe_agent.py`) never lets Claude produce a nutrition
value. The pipeline, in order:

1. **Filter** `recipe_database.json`'s 24 recipes to those with
   `prep_time_min <= meal_prep_time_min` (and optionally matching
   `suitable_meal`). All filtering is plain code — no model call yet.
2. **Rank** the survivors by
   `score = |protein_g − protein_target_g| + |fibre_g − fibre_target_g| − (5.0 if vitamin_focus matches else 0)`,
   lower is better — again, plain code.
3. **Only the top 3** candidates' real, already-verified database fields
   (name, cuisine, protein_g, fibre_g, calories, prep_time_min,
   vitamin_tags, etc.) are put in the prompt. Nothing else from the
   24-recipe database is ever shown to the model.
4. Claude is asked to pick **one of those three** and explain the choice.
   The response is constrained via `output_config.format` (a JSON schema)
   with `chosen_recipe_name` as an **enum of exactly the 3 candidate
   names** — this is a structural guarantee, not a prompt instruction:
   the API will not return a name outside that enum, so the model cannot
   name a recipe it wasn't shown, let alone a nutrition number that isn't
   already in the database.
5. If no recipe satisfies the time (and meal-type) constraint, the
   pipeline returns an explicit `no_match` result and **the API is never
   called** — no recipe is forced.

The only thing Claude actually generates is the free-text `explanation`
field, and the prompt explicitly instructs it to reference only the
numbers already present in the candidate list. Every nutrition number a
user ever sees traces back to a value that was in `recipe_database.json`
before the prompt was built — the retrieval step happens before
generation, not the other way around.

## Known limitations

- **Dataset size: 220 days, one household.** Small by ML standards — the
  gradient-boosted models losing to plain linear regression on 3 of 4
  targets is itself evidence of this (GBM has more capacity than 176
  training rows can support without overfitting).
- **Uncertain data provenance.** Phase 2 EDA found the raw `day_name`
  column matched the actual weekday computed from `date` in only 14 of
  100 rows for the first ~100 days — near what random assignment would
  produce. That's documented in `data/SCHEMA.md` and doesn't affect any
  feature actually used (day-of-week is recomputed from `date`, never
  trusted from the raw column), but it's a signal the underlying log may
  be partly or fully synthetic rather than an authentic daily diary,
  which matters for how much to trust the learned patterns generalizing
  to a real future.
- **No real missingness has been exercised yet.** The flagging/imputation
  machinery (`GOVERNANCE.md` HC-2) is built and unit-tested against
  synthetic gaps, but the actual dataset has zero missing values — the
  real-world case it was designed for (unlogged workouts, skipped meals)
  hasn't happened yet in this data.
- **`vitamin_focus` accuracy is modest.** 56.82% on a 5-way classification
  beats the 34.09% majority-class baseline meaningfully, but is far from
  reliable — expect it to be wrong close to half the time.
- **Recipe database nutrition values are calculated, not lab-verified.**
  Per the database's own sourcing note: protein/fibre are derived from
  USDA-equivalent ingredient data scaled to the recipe, not measured.
  `src/recipe_db.py`'s validation pass (0 errors, 0 warnings across all 24
  entries) checks internal consistency, not real-world accuracy.
- **Recipe database is small: 24 recipes, 3 cuisines** (Indian 9,
  European 8, Mexican 7). Layer 2 can only be as good as its candidate
  pool — a bad day for available recipes matching the time budget can
  leave only 1-2 real candidates for the model to choose from.
- **`protein_target_g` and recipe protein are on different scales — found
  and fixed.** `protein_target_g` ranges 31.2-52.0g in the training data;
  every recipe in `recipe_database.json` tops out at 19.0g (min 4.0g,
  mean 13.1g) — no single dish can ever reach even the lowest historical
  target, very plausibly because the target reflects a whole-day or
  whole-meal goal while a recipe is one dish, and no per-roti nutrition
  data exists to fill the gap without inventing a number. The ranking
  math in `src/recipe_agent.py` isn't actually broken by this — when a
  target is unreachable by any candidate, minimizing the distance
  correctly reduces to "prefer more protein," which is the right
  behavior — but the dashboard's original grams-vs-grams chart made it
  look broken (always the same large gap, regardless of how good the
  pick was). Fixed by showing protein as % of target
  (`build_pct_of_target_chart` in `src/app_logic.py`) instead of a
  misleading two-bar comparison; documented in `score_recipe`'s
  docstring. `fibre_target_g` doesn't have this problem — recipe fibre
  spans 3-14g, nearly the full 8-14g target range — and keeps the
  grams-vs-grams chart.
- **One unresolved data pattern**: `pairs_with_roti` is `True` for every
  Indian recipe and `False` for every European/Mexican one, with no
  exceptions — plausible, but flagged as an info note
  (`recipe_database/validation_report.md`) rather than confirmed.
- **The heuristic baseline is self-authored**, not an independently
  sourced or expert-validated one — the comparison shows the trained
  models beat a simple, disclosed set of if/else rules, not necessarily
  the best possible non-ML approach.
- **The retraining feedback loop is wired but not yet exercised** with
  real accumulated usage: `data/raw/app_actuals.csv` is written by
  `src/cli.py` and picked up by `src/data_pipeline.py`, but no model in
  this repo has yet been retrained on real logged-actual data.
- **No automated retraining or monitoring.** Retraining is a manual,
  explicit step (`python src/train_model.py`, etc.) — there's no
  scheduled pipeline or drift detection.

## Engineering notes

- 119 tests passing across the pipeline (ingestion, feature analysis,
  model training, recipe validation, the recommendation agent, and the
  app logic layer), including checksum guards that raw data files are
  never modified and regression tests for a real bug this project hit
  (mixed date formats across merged raw sources silently dropping 220 of
  221 rows — see `GOVERNANCE.md` HC-4).
- Every hard constraint (no invented numbers, explicit missingness,
  chronological splits only, read-only raw data, one phase at a time) is
  centralized in `GOVERNANCE.md`, not scattered across code comments.
