# Tasks

Track intended work for this project. Add newest entries at the top.

## 2026-05-01 00:00 - Apply Sparklines Across Dashboard Tables

- Goal: Apply 90-day and 1-year price sparklines to all reachable ticker/instrument tables.
- Scope: `app/PerformanceTable.py` and reachable view modules with `st.dataframe` ticker tables.
- Assumptions: Aggregate tables without ticker/instrument rows should not get sparklines; deprecated pages outside current navigation are lower priority.
- Alignment: User asked to apply sparklines to all tables and to move the Today universe filter away from the left rail.
- Plan: Add/reuse small view-local sparkline helpers, include both sparkline columns near instrument identifiers, and collapse Today universe controls into a top expander.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile ...`; `uv run ruff check --select E9,F63,F7,F82 ...` across updated reachable views.
- Status: completed.

## 2026-05-01 00:00 - Add One-Year Sparklines

- Goal: Add 1-year price context alongside 90-day sparklines.
- Scope: `app/PerformanceTable.py`, `app/views/PullbackScanner.py`, plus changelog tracking.
- Assumptions: Both tactical 90-day and contextual 1-year trends should appear in actionable performance tables.
- Alignment: User confirmed adding `Price (1y)` makes sense.
- Plan: Reuse `add_sparkline_column(..., col_name="Price (1y)", days=365)` and place it next to `Price (90d)`.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/PullbackScanner.py`.
- Status: completed.

## 2026-05-01 00:00 - Add Sparklines To Action Tables

- Goal: Add recent price sparklines to actionable performance tables.
- Scope: `app/PerformanceTable.py`, `app/views/PullbackScanner.py`, plus changelog tracking.
- Assumptions: Prioritize reachable/actionable tables over deprecated pages and non-instrument aggregate tables.
- Alignment: User asked to add price sparklines to every table with performance context.
- Plan: Reuse existing `add_sparkline_column()` and `LineChartColumn`, adding `Price (90d)` near instrument descriptions.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/PullbackScanner.py`.
- Status: completed.

## 2026-05-01 00:00 - Today Action List Tab

- Goal: Add a dedicated daily action queue tab and include stocks in the default Performance universe.
- Scope: `app/PerformanceTable.py`, plus task/changelog tracking.
- Assumptions: The action list should use its own compact universe filter, allow duplicate tickers across buckets, and keep detailed Action Screens in Performance.
- Alignment: Continued from `HANDOFF.md`; no blockers or new user questions recorded.
- Plan: Add local action-list builder/renderer, add a first `Today` tab with controls and action counts, sort by action priority then score, and update Performance instrument defaults.
- Test-first approach: No Streamlit UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Status: completed.

## 2026-05-01 00:00 - Focus Pullback Candidates On Best Setups

- Goal: Reduce noisy Pullback Candidates and default the view to only the highest-quality setups.
- Scope: `app/views/PullbackScanner.py`, plus task/changelog tracking.
- Assumptions: The user wants fewer, more actionable candidates by default while preserving access to all matches.
- Alignment: User accepted the proposed best-only quality-score approach and asked for implementation/documentation.
- Plan: Add Best only/All matches mode, default Best only, require intact 126-day trend by default, add max drawdown and bounce controls, compute quality score, rank and cap displayed candidates.
- Test-first approach: No UI snapshot tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/PullbackScanner.py`.
- Status: completed.

## 2026-05-01 00:00 - Merge Scanner Tabs And Fix Performance Screens

- Goal: Consolidate Puke Detector into Performance, merge Laggard Breakout into Pullback, and make Performance capitulation/high-vol breakout screens produce useful candidates.
- Scope: `app/PerformanceTable.py`, `app/views/PullbackScanner.py`, plus task/changelog tracking.
- Assumptions: Puke and Laggard should be removed from top-level tabs; their best ideas should remain available in Performance/Pullback; empty strict screens should fall back to ranked watchlists.
- Alignment: User explicitly requested implementing the proposed tab consolidation and fixing empty Performance action screens.
- Plan: Remove standalone Puke/Laggard navigation, add laggard breakout candidates to Pullback, add fallback rankings for capitulation and high-vol breakout screens.
- Test-first approach: No UI snapshot tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/PullbackScanner.py`.
- Status: completed.

## 2026-05-01 00:00 - Tab Filter Left Rail Standardization

- Goal: Move tab-level filters/settings out of top-of-page rows and into narrow tab-local left rails, matching the Pullback Scanner layout.
- Scope: View tabs with top-level filters in `app/views/`, plus task/changelog tracking.
- Assumptions: Preserve existing filter behavior and thresholds; only layout should change; left rail should use a narrow `1:4` ratio against main content.
- Alignment: User explicitly requested every tab with filters should use the same left-column pattern instead of top filters.
- Plan: Inventory filter widgets, convert eligible tabs to `settings_col`/`content_col`, verify syntax-critical checks, and stage changed files.
- Test-first approach: No UI snapshot tests exist; use compile and syntax-critical ruff checks across changed view files.
- Verify: `uv run python -m py_compile ...`; `uv run ruff check --select E9,F63,F7,F82 ...`.
- Status: completed.

## 2026-05-01 00:00 - Performance Tab Action Screens

- Goal: Move useful volatility/capitulation screens into the main Performance tab and add upside high-volatility breakout candidates.
- Scope: `app/PerformanceTable.py`, plus task/changelog tracking.
- Assumptions: The Performance tab should keep the main table but add action-oriented screens beneath it; breakout candidates are instruments with high short-term volatility and positive 1W/1M momentum.
- Alignment: User likes the Volatility Spike Heatmap and Capitulation Candidates, does not need the rest of the Puke Detector charts there, and wants ETFs exploding up on high volatility surfaced as breakout candidates.
- Plan: Add a compact action-screen renderer, reuse the Performance dataframe, include volatility spike, capitulation, and high-volatility breakout tabs.
- Test-first approach: No Streamlit UI tests exist; use syntax/lint checks and manual code review of required columns and empty states.
- Verify: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check app/PerformanceTable.py`.
- Status: completed.

## 2026-05-01 00:00 - Pullback Scanner Actionability Updates

- Goal: Make the Pullback Scanner include stocks, move settings out of the main page flow, add downside reversal risk warnings, and make recovery output more actionable.
- Scope: `app/views/PullbackScanner.py` only, plus this task/changelog tracking.
- Assumptions: Stocks should be included by default; downside reversals are bearish watch/risk-warning candidates; the underperformer scatter is less useful than a table with explicit action buckets.
- Alignment: User confirmed stock inclusion, downside framing as both bearish watch and risk warning, and accepted replacing the scatter with a more actionable view.
- Plan: Move controls to sidebar, use shared fund-type filter with stocks by default, add downside reversal criteria, replace recovery scatter with action buckets, clarify benchmark-relative labels.
- Test-first approach: Automated UI tests are not present; verification starts with syntax/lint checks and code review of the Streamlit render path.
- Verify: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check app/views/PullbackScanner.py`.
- Status: completed.
