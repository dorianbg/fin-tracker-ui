# Changelog

Track meaningful changes made in this project. Add newest entries at the top.

## 2026-05-01 00:00 - Tab Filter Left Rail Standardization

- Changed: Converted filter/settings areas across filtered tabs to a narrow left rail with main content in a wider right column.
- Why: Top-level filters consumed vertical space and felt inconsistent; the Pullback Scanner left rail pattern worked better.
- How: Applied `st.columns([1, 4], gap="large")` to tabs with top filters, preserving existing filtering logic while moving controls into `settings_col` and outputs into `content_col`.
- Verified: `uv run python -m py_compile` across updated view files; `uv run ruff check --select E9,F63,F7,F82` across updated view files.
- Follow-up: Full lint still has pre-existing non-syntax issues in a few files; not changed as part of layout work.

## 2026-05-01 00:00 - Pullback Scanner Left Rail Settings

- Changed: Replaced the settings expander with a narrow in-page left controls rail and wide right results column.
- Why: The expander still consumed too much vertical space; the user wanted tab-local settings that behave like a left sidebar without using Streamlit's global sidebar.
- How: Used `st.columns([1, 4], gap="large")` in `app/views/PullbackScanner.py`, rendering controls in the narrow column and all scanner results in the wide column.
- Verified: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check app/views/PullbackScanner.py`.
- Follow-up: Adjust column ratio if the left rail still feels too wide/narrow in the live app.

## 2026-05-01 00:00 - Pullback Scanner Settings Placement Fix

- Changed: Moved Pullback Scanner controls out of `st.sidebar` into a tab-local settings expander with two columns.
- Why: Streamlit sidebars are global, so the Pullback Scanner controls leaked into the app sidebar across tabs.
- How: Replaced the sidebar context in `app/views/PullbackScanner.py` with an in-page `Scanner & Recovery Settings` expander.
- Verified: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check app/views/PullbackScanner.py`.
- Follow-up: None.

## 2026-05-01 00:00 - Performance Tab Action Screens

- Changed: Added an `Action Screens` section to the Performance tab with volatility spikes, capitulation candidates, and high-volatility upside breakout candidates.
- Why: The main Performance page should surface the most useful actionable volatility screens directly, including ETFs/funds exploding higher on unusual volatility.
- How: Added `render_action_screens()` in `app/PerformanceTable.py`, deriving `vol_ratio`, capitulation severity, and breakout score from the already-loaded performance dataframe.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check app/PerformanceTable.py`.
- Follow-up: Manual Streamlit render check is useful to validate layout placement under the main table.

## 2026-05-01 00:00 - Pullback Scanner Actionability Updates

- Changed: Updated Pullback Scanner controls, universe defaults, reversal sections, and recovery watchlist output.
- Why: The page needed to include stocks, reduce vertical clutter, surface downside risks, clarify recovery metrics, and provide more actionable insights.
- How: Moved scanner/recovery settings to `st.sidebar`, reused shared fund-type filtering with `eq` and `stock` selected by default, added a downside reversal/risk-warning section, replaced the underperformer scatter with action buckets, and renamed displayed recovery excess columns.
- Verified: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check app/views/PullbackScanner.py`.
- Follow-up: Manual Streamlit render check is still useful to validate layout and table ergonomics with live data.
