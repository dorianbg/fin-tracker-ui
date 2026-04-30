# Tasks

Track intended work for this project. Add newest entries at the top.

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
