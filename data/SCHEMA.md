# Data contract

This is the agreed shape of the data. Section 1 was updated in Phase 2 to
match the real file (`data/raw/daily_context.csv`) once it arrived — the
original Phase 1 draft assumed one row per meal-occasion with a `meal_type`
column; the real data is one row per **day** instead. Any further change to
this contract should be a deliberate edit to this file, not something that
happens silently inside a script or notebook.

## 1. Raw daily-context data (`data/raw/daily_context.csv`)

One row = one day (not one meal — there is no `meal_type` column; each day
has a single set of context values and a single `roti_count`/nutrition
target outcome).

| Column                  | Type                 | Description                                                        | Missingness observed |
|--------------------------|----------------------|----------------------------------------------------------------------|---|
| `date`                   | date, `DD/MM/YYYY`   | Calendar date. Required, used for the time-based split. Parse with `dayfirst=True`. | None |
| `workout_type`           | categorical           | `none` \| `cardio` \| `strength`. `none` is a real, valid category (rest day) — not a stand-in for "unlogged". | None |
| `workout_duration_min`    | integer               | Minutes worked out. `0` when `workout_type == 'none'`, consistent throughout. | None |
| `calories_burned`         | integer               | Daily calories burned (includes resting/basal — nonzero even on `workout_type == 'none'` days). | None |
| `sleep_hours`             | float                 | Hours slept.                                                        | None |
| `mood_score`               | ordinal (1-5)         | Self-reported mood.                                                  | None |
| `hunger_level`             | ordinal (1-5)         | Self-reported hunger.                                                | None |
| `curry_type`               | categorical            | `dal` \| `dry` \| `gravy`.                                           | None |
| `curry_richness`           | categorical            | `light` \| `heavy`.                                                  | None |
| `weight_trend`             | categorical            | `maintaining` \| `losing`.                                           | None |
| `meal_prep_time_min`       | integer                | Minutes available to cook. Also consumed directly by Layer 2 at inference time. | None |
| `protein_target_g`         | float (target)         | Historical actual/prescribed protein target for the day. **Layer 1 label.** | None |
| `fibre_target_g`           | float (target)         | Historical actual/prescribed fibre target for the day. **Layer 1 label.** | None |
| `vitamin_focus`            | categorical (target)   | `B12` \| `Iron` \| `Multivitamin` \| `Vitamin C` \| `Vitamin D`. **Layer 1 label.** | None |
| `roti_count`               | integer (target)       | Actual number of rotis eaten. **Primary Layer 1 label.**             | None |

### Columns present in the raw file but dropped at ingestion

- **`day_name`** — unreliable. Checked against the weekday computed from
  `date`: matches in only 14/100 rows for the first ~100 rows, and for the
  remaining ~120 rows the column was replaced with a generic `weekday`/
  `weekend` bucket while the real weekday value leaked into a stray 17th
  CSV field (see next item). Decision: drop `day_name`, derive
  `day_of_week` / `is_weekend` from `date` during ingestion instead, since
  `date` itself is reliable and this recomputation is deterministic and
  auditable.
- **Unnamed 17th column** — the raw CSV header ends with a trailing comma
  (`...,roti_count,`), producing a phantom extra field. For the last ~120
  rows it happens to contain the correct weekday name (a side effect of
  whatever produced the `day_name` bug above); for the first ~100 rows it's
  empty. Dropped at ingestion — `day_of_week` is recomputed from `date`
  instead of relied upon from either raw column.

This is a known data-generation quirk in the source file, not something
ingestion silently papers over — it's documented here and called out again
in the Phase 2 EDA notes so it isn't forgotten.

## 2. Missingness — observed vs. designed-for

**Observed in the current file: none.** Every field is populated for all
220 rows (2025-08-15 → 2026-03-22, no calendar gaps). `workout_type='none'`
is a legitimate rest-day value, not a null.

The project brief anticipates future data with real gaps (unlogged
workouts, skipped meals). Per `GOVERNANCE.md` HC-2, ingestion still builds
the missingness-flagging machinery generically (a `*_missing` boolean
companion column for every nullable field) so it's ready for such data —
for this file, those flags are simply all-`False`. Nothing here is dropped
silently regardless of whether it's this file or a future one.

Imputation strategy (mean/median/mode/"unknown" category/model-based) is
deferred to Phase 3 and will be decided against whatever data actually has
gaps at that point — moot for the current file since there are none.

## 3. Layer 1 targets (already present as historical labels in raw data)

Unlike the original Phase 1 assumption that these were purely model
*output*, the raw file already contains them as historical actuals per day
— this is what makes supervised training possible:

| Column               | Type              | Notes |
|-----------------------|-------------------|-------|
| `roti_count`           | integer           | Primary target. |
| `protein_target_g`     | float             | Historical target for that day. |
| `fibre_target_g`       | float             | Historical target for that day. |
| `vitamin_focus`        | categorical        | 5-way: `B12`, `Iron`, `Multivitamin`, `Vitamin C`, `Vitamin D`. |

At inference time (Phase 3+), Layer 1 predicts these four for a new day
from the context features above (everything in §1 except the targets
themselves and `meal_prep_time_min`, which belongs to Layer 2 — see below).

`protein_target_g` / `fibre_target_g` are targets to search for, not
nutrition facts about a specific recipe — those numbers still come only
from `recipe_database.json` once Layer 2 selects a recipe (HC-1).

## 4. Train/test split

- Chronological only, per `GOVERNANCE.md` HC-3. Sort by `date`, take an
  early contiguous block as train, a later contiguous block as test.
- Exact cutoff decided in Phase 3. With 220 days total (2025-08-15 to
  2026-03-22), a plausible default is holding out the last ~4-6 weeks as
  test — to be confirmed in Phase 3 once model requirements are clearer.

## 5. `recipe_database.json`

Schema lives in `recipe_database/recipe_database.schema.json`. Each entry
has verified nutrition facts (protein, fibre, calories), prep time, and
tags (curry type compatibility, vitamin focus). Hand-maintained/verified —
no pipeline in this repo generates or edits its nutrition values (HC-1).
