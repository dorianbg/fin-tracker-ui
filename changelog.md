# Changelog

## 2026-05-12 00:00 - Improve Today Trade Simulation Validity

- Changed: Updated the Today trade simulation to enter on the next available trading day after a signal and to report benchmark-relative returns.
- Why: Same-day execution can overstate signal quality, and absolute returns are less useful without comparison to `DEFAULT_BENCHMARK`.
- How: Included benchmark price history in backtest data, shifted entries from signal date to next ticker trading row, added `Signal Date`, `Benchmark Return`, `Relative Return`, and an `Avg rel` summary metric.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Commit: ab9c23865f0ffa6388ee1462233c3301bae93a2d

## 2026-05-02 00:00 - Add Nuclear And Uranium Theme Universe

- Changed: Added/updated the requested uranium and nuclear ETFs/stocks in `resources/instrument_info.csv` with nuclear sub-theme labels.
- Why: Uranium and nuclear energy are being tracked as a national-security-oriented investment theme.
- How: Added missing tickers, updated existing `NLR`, `URA`, `URNM`, `CCJ`, `DNN`, `UEC`, and `BHP` theme labels, and used one row per ticker.
- Verified: Python CSV validation confirmed all requested tickers are present and there are no duplicate tickers.
- Follow-up: Some symbols such as `XE`, `ISOU`, and `RYCEF` may need ticker-source validation during the next pipeline fetch if Yahoo Finance does not support those exact symbols.

## 2026-05-02 00:00 - Make Today Decisions Investable

- Changed: Added explicit `Decision`, `Entry Rule`, `Invalidation`, `Exit Plan`, `Review Trigger`, and `Backtest Edge` columns to the Today action list.
- Why: The prior `Watch` labels were screening buckets, not investment criteria.
- How: Preserved underlying signal names for filtering/backtests, but mapped signals into action states such as `Buy Candidate`, `Wait For Reclaim`, `Avoid Until Bounce`, `Trim Candidate`, and `Risk Review` based on confirmation and risk conditions.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: Use simulation results to calculate a real `Backtest Edge` value instead of static explanatory text.

## 2026-05-02 - Batch yfinance Pipeline Downloads

- Changed: the pipeline now batches missing ticker history into one `yf.download()` call and splits the result back per ticker/job before the existing insert/backup flow.
- Why: the old path created one `yf.Ticker(...).history()` request per ticker and slept 3 seconds after every ticker, making data pulls unnecessarily slow.
- How: added `download_jobs()` and `_normalize_yfinance_df()`, passed pre-downloaded per-ticker data into `execute_job()`, and kept the single-ticker fallback for direct use.
- Verified: `uv run python -m py_compile pipeline/executor.py` passes. A no-fetch pipeline smoke run could not complete because `duckdb.db` is currently locked by another Python process.
- Commit: Uncommitted

Track meaningful changes made in this project. Add newest entries at the top.

## 2026-05-02 00:00 - Replace Fixed Horizon Signal Backtest

- Changed: Replaced the Today fixed-forward-return signal backtest with a trigger-based trade simulation for long-entry signals.
- Why: Fixed holding periods are too crude for investable decisions; exits should depend on trend failure, bounce failure, profit protection, and drawdown stops.
- How: Added `simulate_signal_trades()` to enter historical signal days, avoid overlapping trades per ticker, and exit on state-based rules with explicit exit reasons.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: Add benchmark-relative trade results, parameter presets per signal, and portfolio-level sizing once the signal-level simulation looks sensible.

## 2026-05-02 00:00 - Add Today Signal Backtest

- Changed: Added a `Signal Backtest` section to the Today tab for historical validation of Today action-list rules.
- Why: The action list should become more automated and evidence-based before any portfolio automation is added.
- How: Factored Today signal construction into `build_signal_candidates()`, loaded historical performance rows plus prices, computed 5/10/21/63 trading-day forward returns, and displayed win rate/return stats with recent historical samples.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: Use the backtest results to weight/rank Today candidates by empirical signal edge, then consider portfolio-level automation.

## 2026-05-01 00:00 - Apply Sparklines Across Dashboard Tables

- Changed: Extended `Price (90d)` and `Price (1y)` sparklines to reachable ticker/instrument tables in Daily Summary, Today's Crossings, Relative Strength, Rotation, Factors, and Allocator views; moved the Today universe filter into a collapsed top expander.
- Why: Recent and 1-year price shape should be visible wherever table rows represent instruments, while Today should reserve horizontal space for the action table.
- How: Reused `add_sparkline_column()` with `LineChartColumn` configs and skipped aggregate/non-ticker tables where sparklines do not apply.
- Verified: `uv run python -m py_compile app/PerformanceTable.py app/views/DailySummary.py app/views/TodaysCrossings.py app/views/RelativeStrength.py app/views/RotationStrategies.py app/views/FactorDashboard.py app/views/PortfolioAllocator.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/DailySummary.py app/views/TodaysCrossings.py app/views/RelativeStrength.py app/views/RotationStrategies.py app/views/FactorDashboard.py app/views/PortfolioAllocator.py app/views/PullbackScanner.py`.
- Follow-up: If tables become too wide, make `Price (1y)` optional per tab.

## 2026-05-01 00:00 - Add One-Year Sparklines

- Changed: Added `Price (1y)` sparklines next to `Price (90d)` in the Today, Performance, and Pullback performance tables.
- Why: The 90-day view shows tactical momentum, while the 1-year view adds trend and drawdown/recovery context.
- How: Reused `add_sparkline_column(..., col_name="Price (1y)", days=365)` and added matching `LineChartColumn` configs.
- Verified: `uv run python -m py_compile app/PerformanceTable.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/PullbackScanner.py`.
- Follow-up: If tables feel too wide in Streamlit, make the 1-year sparkline optional or move it to detail views.

## 2026-05-01 00:00 - Add Today Action List Tab

- Changed: Added a first-class `Today` tab for `🎯 Today’s Action List`, balanced table rows across selected action types, and included `stock` in the default Performance instrument categories.
- Why: The Performance tab should surface the strongest daily decision candidates without requiring inspection across several scanner tabs.
- How: Added `build_action_list()`, `render_today_action_list()`, and `render_today_tab()` in `app/PerformanceTable.py`, synthesizing Buy Watch, Breakout Watch, Trim Watch, Short Monitor, and Capitulation Watch candidates from the performance dataframe with a compact Today universe filter and per-action row balancing.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: Tune scoring weights and duplicate handling after reviewing live candidate counts in Streamlit.

## 2026-05-01 00:00 - Add Sparklines To Action Tables

- Changed: Added `Price (90d)` sparklines to Today, Performance action screens, and Pullback performance tables.
- Why: Performance tables are easier to scan when recent price shape is visible next to each instrument.
- How: Reused `add_sparkline_column()` and Streamlit `LineChartColumn` on actionable tables that include tickers.
- Verified: `uv run python -m py_compile app/PerformanceTable.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/PullbackScanner.py`.
- Follow-up: Add sparklines to lower-priority/deprecated tables only if those pages become reachable again.

## 2026-05-01 00:00 - Remove Drawdowns Tab

- Changed: Removed the Drawdowns tab from the main app navigation.
- Why: User found it redundant now that drawdown context is available in the Performance tab.
- How: Removed DrawdownAnalysis import, tab variable, tab label, and render block from `app/PerformanceTable.py`.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: `app/views/DrawdownAnalysis.py` remains in the repo but is no longer reachable from the main tab navigation.

## 2026-05-01 00:00 - Add Leaders Weakening Monitor

- Changed: Added `Leaders Weakening` to Daily Summary as the inverse of Laggard Awakenings.
- Why: User wanted a monitor for long-term leaders that are starting to lose short-term relative momentum.
- How: Added an expander that finds 1Y outperformers vs the default benchmark with negative 1W relative strength, ranked by `weakening_score`.
- Verified: `uv run python -m py_compile app/views/DailySummary.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/DailySummary.py`.
- Follow-up: Tune the 10% 1Y outperformance threshold if it is too strict or too noisy.

## 2026-05-01 00:00 - Expand Pullback Short Target Monitor

- Changed: Renamed `Downside Reversal / Risk Warnings` to `Short Target Monitor` and expanded the table with scoring and monitoring context.
- Why: The section is more useful as a bearish watchlist for failed bounces, stop-loss review, hedging, or short setup monitoring.
- How: Added `short_priority_score`, `monitor_reason`, 1M return, drawdown, and explanatory text in `app/views/PullbackScanner.py`.
- Verified: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/PullbackScanner.py`.
- Follow-up: Tune score weights after seeing real candidates.

## 2026-05-01 00:00 - Focus Pullback Candidates On Best Setups

- Changed: Added a default `Best only` mode to Pullback Candidates with quality scoring and tighter filters.
- Why: The previous pullback list was too broad and surfaced too many low-quality candidates.
- How: Added controls for mode, top-N count, max 52-week drawdown, and bounce requirement; defaulted 126-day trend requirement on; ranked by `quality_score = trend strength + controlled pullback depth + bounce score - drawdown/breakdown penalties`.
- Verified: `uv run python -m py_compile app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/PullbackScanner.py`.
- Follow-up: Tune score weights after seeing live candidates.

## 2026-05-01 00:00 - Reorder Tabs By Actionability

- Changed: Reordered main app tabs so action-oriented scanner/workflow tabs appear first, followed by context and portfolio/tool tabs.
- Why: User wanted tab order to match the prior actionability ranking.
- How: Updated the `st.tabs(...)` ordering and matching render blocks in `app/PerformanceTable.py`.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: None.

## 2026-05-01 00:00 - Remove Thematic Dashboard Tab

- Changed: Removed the Thematic dashboard from the main app tab navigation.
- Why: User decided the Thematic dashboard was not useful enough to keep as a primary tab.
- How: Removed Thematic import, tab variable, tab label, and render block from `app/PerformanceTable.py`.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: `app/views/ThematicDashboard.py` remains in the repo but is no longer reachable from the main tab navigation.

## 2026-05-01 00:00 - Fix Recovery Duplicate Column Error

- Changed: Deduplicated Pullback recovery column selection and benchmark rows before excess return calculations.
- Why: If recovery and underperformance lookbacks resolved to the same return column, pandas returned duplicate columns and assigning `excess_short` failed.
- How: Built recovery columns with `dict.fromkeys(...)` and deduplicated benchmark rows by ticker in `app/views/PullbackScanner.py`.
- Verified: `uv run python -m py_compile app/views/PullbackScanner.py app/views/RotationStrategies.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/PullbackScanner.py app/views/RotationStrategies.py`.
- Follow-up: None.

## 2026-05-01 00:00 - Merge Scanner Tabs And Fix Performance Screens

- Changed: Removed standalone Puke Detector and Laggard Breakout tabs, added laggard breakout candidates into Pullback, and made Performance capitulation/high-vol breakout screens fall back to ranked watchlists.
- Why: Scanner tabs should be more consolidated and actionable; empty Performance screens were not useful.
- How: Updated `app/PerformanceTable.py` navigation and action-screen filters; added a `Laggard Breakout Candidates` section to `app/views/PullbackScanner.py` using existing recovery benchmark-relative calculations.
- Verified: `uv run python -m py_compile app/PerformanceTable.py app/views/PullbackScanner.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/PullbackScanner.py`.
- Follow-up: Manual UI check should confirm the remaining tab order and candidate counts feel right with live data.

## 2026-05-01 00:00 - Fix Today's Crossings Duplicate Ticker Error

- Changed: Deduplicated latest/yesterday rows by ticker before indexing in Today's Crossings.
- Why: Duplicate ticker rows caused `.loc[ticker, col]` to return a Series, which made `pd.isna(...)` raise an ambiguous truth-value error.
- How: Added `drop_duplicates("ticker")` before `set_index("ticker")` in `app/views/TodaysCrossings.py`.
- Verified: `uv run python -m py_compile app/views/TodaysCrossings.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/TodaysCrossings.py`.
- Follow-up: If duplicate tickers reflect upstream instrument data issues, clean them at source separately.

## 2026-05-01 00:00 - Remove Breakout Scanner Tab

- Changed: Removed the Breakout Scanner tab from the main Streamlit app navigation.
- Why: User requested removing the `🚀 Breakout Scanner` page.
- How: Removed the Breakout Scanner import, tab variable, tab label, and render block from `app/PerformanceTable.py`.
- Verified: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Follow-up: `app/views/BreakoutScanner.py` remains in the repo but is no longer reachable from the main tab navigation.

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
