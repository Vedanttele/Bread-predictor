# Governance

Single source of truth for this project's foundational requirements the
things that must hold true regardless of which phase is in progress. If a
new foundational requirement comes up mid-project, it gets added here (with
a changelog entry), not buried in a conversation or scattered across other
docs. Other docs (`README.md`, `data/SCHEMA.md`) should describe *how*
things are built; this file governs the non-negotiables they must satisfy.

Each rule has a stable ID (`HC-n`) so it can be referenced from code
comments, PRs, or other docs without re-stating it.

## Hard constraints

- **HC-1 — No invented nutrition numbers.**
  All protein/fibre/calorie values shown to the user must come from
  `recipe_database/recipe_database.json`, which is manually verified, not
  generated. Layer 2 may select and explain a recipe; it may never compute,
  estimate, or let an LLM fabricate a nutrition value. An entry with
  `verified` not `true` must never be surfaced.

- **HC-2 — Missingness is explicit, never silent.**
  Real data has unlogged workouts, skipped meals, etc. Rows with missing
  values are never dropped without a documented reason; nullable columns get
  explicit missingness flags at ingestion time. See `data/SCHEMA.md` §3.

- **HC-3 — Time-based splits only.**
  This is sequential daily data. Train/test splits are always chronological
  (train on earlier dates, test on later ones). Random shuffling or k-fold
  with random assignment is never used it leaks future context into
  training. See `data/SCHEMA.md` §4.

- **HC-4 — `data/raw/` is read-only, with one controlled exception.**
  Real logged data lives there. Code reads from it; nothing writes to,
  cleans, or reformats files in place there. Derived data goes in
  `data/processed/`.
  Exception: `src/cli.py`'s "log what you actually ate" step appends to
  `data/raw/app_actuals.csv` a separate file from the manually-curated
  `daily_context.csv`, which no code ever writes to. The append path is
  the *only* writer of `app_actuals.csv`, is append-only (never edits or
  deletes existing rows), and this is documented so the exception stays
  narrow and visible rather than quietly eroding the rule. See
  `src/app_logic.py`'s `append_raw_feedback_row` and `data/raw/README.md`.

  This exception exposed a real bug worth remembering: `src/data_pipeline.py`
  concatenates every file in `data/raw/` before parsing dates, and
  `pandas.to_datetime` without an explicit format infers ONE format from
  early rows and silently misparses the rest if a second source uses a
  different one this dropped 220 of 221 rows the first time
  `app_actuals.csv` (ISO dates) met `daily_context.csv` (DD/MM/YYYY).
  Fixed with `format="mixed"` plus a hard failure if parsing drops more
  than half a batch see `derive_date_fields` in `data_pipeline.py`.

- **HC-5 — One phase per session.**
  Work proceeds one phase at a time per `PHASES.md`. Ask before starting the
  next phase rather than chaining multiple phases into one session.

## How to amend this file

1. Add or edit the rule under **Hard constraints**, keeping IDs stable
   (never reuse a retired ID mark it "retired" instead of deleting, so
   history stays legible).
2. Add a row to the **Changelog** below.
3. If the change affects other docs (README's constraints summary,
   `data/SCHEMA.md`, `PHASES.md`), update those in the same pass so nothing
   drifts out of sync.

## Changelog

| Date       | Change                                                   |
|------------|-----------------------------------------------------------|
| 2026-08-14 | Initial governance file created: HC-1 through HC-5.        |
| 2026-08-14 | HC-4 amended: documented `data/raw/app_actuals.csv` (via `src/cli.py`) as the one controlled, append-only exception. |
