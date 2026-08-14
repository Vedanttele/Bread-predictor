# data/raw/ — read-only, with one controlled exception

This folder holds the real, manually logged daily-context data. It is
**read-only from code's perspective** — with one narrow, documented
exception (below). Cleaned/derived data goes in `data/processed/` instead.

## What goes here

Any number of CSV files matching the general shape in `../SCHEMA.md` §1
(one row per day of logging). Different files are allowed to use different
column names for the same concept (e.g. `date` vs `log_date`, `day_name`
vs `day_type`) or to omit a column entirely (e.g. no `meal_type`) —
`src/data_pipeline.py` normalizes each file's schema and merges them into
`data/processed/merged_clean.csv`. Real missingness (unlogged workouts,
skipped meals, etc.) is expected and fine — do not pre-fill or clean it
before dropping it in here; that handling happens explicitly in ingestion
code, not by hand in the raw file.

**Date format:** `DD/MM/YYYY`. Don't introduce a different format in a new
file — `data_pipeline.py` tolerates mixed formats across sources
(`format="mixed"`), but there's no reason to rely on that when one
convention is easy to keep.

## The one exception: `app_actuals.csv`

`src/cli.py`'s "log what you actually ate" step appends to
`data/raw/app_actuals.csv` — see `GOVERNANCE.md` HC-4. It is:
- a **separate file** from the manually-curated `daily_context.csv`, which
  no code ever writes to
- **append-only** — existing rows are never edited or deleted
- written by exactly one function, `append_raw_feedback_row` in
  `src/app_logic.py`

This exists so logging a real day's actual roti count and recipe feeds
back into the next retraining run (`src/data_pipeline.py` picks it up
automatically), without touching the original dataset.

## Currently

- `daily_context.csv` — 220 rows, 2025-08-15 → 2026-03-22, one row/day.
- `app_actuals.csv` — created by `src/cli.py` the first time you log a day;
  absent until then.
