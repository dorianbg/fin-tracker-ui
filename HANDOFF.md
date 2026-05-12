# Handoff: Fin Tracker UI Actionability, Sparklines, And Backtesting

## 1. Goal / Current Task

We are improving the Streamlit dashboard so it is more actionable and less cluttered.

Current active direction:

- Turn the dashboard from a collection of screens into a daily decision workflow.
- Use `Today` as the primary entry tab.
- Make instrument tables easier to scan by showing both short-term and 1-year price shape.
- Replace vague watchlists and fixed-horizon signal checks with explicit entry/exit rules and trade simulation.

Most recent task completed:

- Improved `Trade Simulation Backtest` validity by shifting historical entries to the next available trading day after the signal and adding benchmark-relative returns.

## 2. Completed So Far

### Navigation / Tab Consolidation

- Added `Today` as the first top-level tab.
- Removed these tabs from main navigation because their useful signals were merged elsewhere or were redundant:
  - `Breakout`
  - `Puke Detector`
  - `Laggard Breakout`
  - `Thematic`
  - `Drawdowns`
- Current top-level tab order in `app/PerformanceTable.py`:
  1. `Today`
  2. `Performance`
  3. `Pullback`
  4. `Crossings`
  5. `Daily Summary`
  6. `Relative Strength`
  7. `Rotation`
  8. `Cross-Asset`
  9. `Factors`
  10. `Charts`
  11. `Correlation`
  12. `Allocator`

### Layout / Filter Placement

- Converted tab filters/settings to tab-local left rail pattern using `st.columns([1, 4], gap="large")` where appropriate.
- Important decision: do not use `st.sidebar` for tab-specific controls because Streamlit sidebars are global and leak across tabs.
- `Today` originally used a left rail only for `Universe`, but this wasted horizontal space.
- `Today` now puts `Universe` inside a collapsed top expander.

### Today Tab

- Added `Today` tab via `render_today_tab()` in `app/PerformanceTable.py`.
- `Today` loads performance data from `di.perf_tbl` with:
  - `vol_adjust=False`
  - `show_returns=True`
  - `returns_cols=di.selectable_returns`
  - `fund_types=instrument_categories`
- `Today` default universe is `default=["eq", "stock", "commod"]`.
- `Today` renders:
  - `🎯 Today’s Action List`
  - `Trade Simulation Backtest`
- `Today` action table is balanced across selected action types rather than allowing the first priority bucket to fill the whole table.
- `Today` table includes both `Price (90d)` and `Price (1y)` sparklines.

### Today Action List / Signal Logic

- Added `ACTION_PRIORITY` in `app/PerformanceTable.py`:
  - `Buy Watch`: 1
  - `Breakout Watch`: 2
  - `Trim Watch`: 3
  - `Short Monitor`: 4
  - `Capitulation Watch`: 5
- Added `build_signal_candidates(df: pd.DataFrame) -> pd.DataFrame`.
- Added `build_action_list(df: pd.DataFrame) -> pd.DataFrame` for display formatting.
- Added `render_today_action_list(df: pd.DataFrame)`.
- Duplicate tickers across action buckets are allowed.

Current signal criteria:

- `Buy Watch`
  - `ma_252 > 0`
  - `ma_126 > 0`
  - `ma_21 < 0`
  - `drawdown_52w >= -20`
  - `r_1d > 0 or r_1w > 0`
  - Score column: `quality_score`
  - Why: `Strong trend pullback with early bounce`

- `Breakout Watch`
  - `r_1w > 0`
  - `r_1mo > 0`
  - `vol_ratio >= 1.1`
  - If available, prefer candidates with `ma_21 > 0` when that confirmed subset is non-empty.
  - Score column: `breakout_score = vol_ratio * r_1w`
  - Why: `Upside move with elevated volatility`

- `Capitulation Watch`
  - `drawdown_52w < 0`
  - `vol_ratio >= 1.2`
  - `drawdown_52w <= -10`
  - Score column: `capitulation_score = -drawdown_52w * vol_ratio`
  - Why: `High stress / drawdown candidate`

- `Trim Watch`
  - Benchmark is `config.DEFAULT_BENCHMARK` / imported `DEFAULT_BENCHMARK`.
  - `r_1y - benchmark_r_1y >= 10`
  - `r_1w - benchmark_r_1w < 0`
  - Score column: `trim_score = -(relative_1w) * relative_1y`
  - Why: `Long-term leader losing short-term relative strength`

- `Short Monitor`
  - `ma_252 > 0`
  - `ma_21 < 0`
  - `ma_63 < 0`
  - `r_1w < 0`
  - Score column: `short_priority_score`
  - Why: `Long-term trend intact but short/intermediate trend rolling over`

Important decision:

- User pushed back that these are still vague watchlists, not investable rules.
- Next useful work should convert Today rows from `Watch` language into explicit decision states such as `Buy Candidate`, `Wait`, `Avoid`, `Trim`, `Risk Review`, with entry/invalidation text.

### Trade Simulation Backtest

- Replaced fixed-horizon forward return backtest with trigger-based trade simulation.
- Current backtest functions in `app/PerformanceTable.py`:
  - `load_signal_backtest_data(fund_types: tuple[str, ...], years: int) -> pd.DataFrame`
  - `simulate_signal_trades(hist_df, signal_name, max_hold_days, failed_bounce_days) -> pd.DataFrame`
  - `render_signal_backtest(fund_types: list[str])`
- Current simulation is signal-level, not portfolio-level.
- Simulates only long-entry signals:
  - `Buy Watch`
  - `Breakout Watch`
  - `Capitulation Watch`
- `Trim Watch` and `Short Monitor` are intentionally excluded from long-entry simulation because they are risk/exit signals.
- It enters on historical signal dates.
- It avoids overlapping trades on the same ticker by skipping new entries until the previous simulated trade exits.
- Exit rules currently implemented:
  - `Hard drawdown stop`: `drawdown_52w <= -25`
  - `Failed bounce`: for `Buy Watch`, after `failed_bounce_days`, still `ma_21 < 0`
  - `MA63 trend stop`: for `Buy Watch` / `Breakout Watch`, `ma_63 < -3`
  - `Profit protection`: for `Breakout Watch`, return is above `8%` and `ma_21 < 0`
  - `No bounce confirmation`: for `Capitulation Watch`, after `failed_bounce_days`, `r_1w <= 0`
  - `Max hold`: fallback cap only, not primary exit logic
- Backtest controls:
  - `Signal`: `Buy Watch`, `Breakout Watch`, `Capitulation Watch`
  - `Bounce timeout`: `5`, `10`, `15`, `21` trading days
  - `Max hold cap`: `63`, `126`, `252` trading days
  - `History`: `1y`, `2y`, `3y`, `5y`
- Backtest output:
  - Trades
  - Win rate
  - Average return
  - Average relative return vs `DEFAULT_BENCHMARK`
  - Median return
  - Average hold
  - Worst return
  - Exit reason bar chart
  - Recent simulated trades table with `Signal Date`, next-day `Entry Date`, `Benchmark Return`, `Relative Return`, `Price (90d)`, and `Price (1y)`
- Latest implementation detail:
  - `load_signal_backtest_data()` now includes `DEFAULT_BENCHMARK` in the price fetch even when the selected universe would otherwise exclude it.
  - `simulate_signal_trades()` keeps the historical signal date but uses the next available ticker row as the simulated entry, reducing same-day lookahead bias.
  - Benchmark-relative return is calculated over the simulated entry/exit window when benchmark prices are available.

### Performance Tab

- Stocks are included by default in Performance via `default=["eq", "stock", "commod"]`.
- Added `Action Screens` under the main Performance table:
  - `📊 Volatility Spikes`
  - `💀 Capitulation`
  - `🚀 High-Vol Breakouts`
- Capitulation now falls back to ranked relative stress leaders if strict candidates are empty.
- High-Vol Breakouts now falls back to strongest positive 1W/1M movers when strict high-vol breakouts are empty.
- Performance main table includes `Price (90d)` and `Price (1y)` sparklines.
- Performance action-screen tables include `Price (90d)` and `Price (1y)` where instrument rows are shown.

### Pullback Tab

- Stocks are included by default in Pullback via `fund_type_sidebar(default=["eq", "stock"], key="pullback_fund_types")`.
- Added `Best only` pullback mode, default enabled.
- Added `Top pullbacks to show`, default `20`.
- Defaulted `Require above 126-day MA` to `True`.
- Added best-only filters:
  - `Best-only max 52W drawdown (%)`, default `-20%`
  - `Best-only require bounce`, default `True`
- Added `quality_score` for pullbacks.
- Renamed/expanded downside section to `🎯 Short Target Monitor`.
- Short Target Monitor includes:
  - `short_priority_score`
  - `monitor_reason`
  - `r_1mo`
  - `drawdown_52w`
- Added `🚀 Laggard Breakout Candidates` inside Pullback, replacing standalone Laggard Breakout tab.
- Fixed duplicate-column bug in Pullback recovery logic by deduplicating recovery column selection with `dict.fromkeys(...)`.
- Pullback tables now include `Price (90d)` and `Price (1y)` sparklines where ticker rows are shown.

### Daily Summary

- Added `⚠️ Leaders Weakening (Long-term Outperformers Losing Momentum)`.
- Logic:
  - Uses `config.DEFAULT_BENCHMARK`.
  - Finds 1Y outperformers vs benchmark by at least 10%.
  - Flags those with negative 1W relative strength vs benchmark.
  - Ranks by `weakening_score = (-rs_1w) * (r_1y - bm_1y)`.
- Existing `🔄 Laggard Awakenings` remains and is valued by the user.
- Daily Summary ticker/instrument tables now include `Price (90d)` and `Price (1y)` sparklines where applicable.
- Aggregate/non-ticker tables were intentionally skipped.

### Today's Crossings

- Fixed ambiguous truth-value error by deduplicating latest/yesterday rows by ticker before setting the index.
- Ticker/instrument tables now include `Price (90d)` and `Price (1y)` sparklines.

### Relative Strength / Rotation / Factors / Allocator

- Added `Price (90d)` and `Price (1y)` sparklines to reachable ticker/instrument tables:
  - Relative Strength ranking table
  - Rotation current picks table
  - Factor heatmap table rows for factor/benchmark ETFs
  - Allocator regional valuation ETF table
- Aggregate tables without tickers were skipped.

### Sparkline Decisions

- Streamlit `LineChartColumn` cannot render OHLC/candles.
- Current implementation uses normalized close/price history only.
- Candles would require OHLC data in the pipeline/export and a different UI approach.
- Decision: keep table sparklines as line charts for now.
- Added both:
  - `Price (90d)` for tactical/short-term shape
  - `Price (1y)` for regime/trend/drawdown context
- Helper reused: `add_sparkline_column(df, col_name="Price (90d)", days=90)` and `add_sparkline_column(df, col_name="Price (1y)", days=365)` from `app/data.py`.

## 3. Current State Of Relevant Files / Variables / Decisions

### Important Files

- `app/PerformanceTable.py`
  - Main app entry point and tab navigation.
  - Contains Today tab, signal/action-list logic, trade simulation backtest, Performance table, and Performance action screens.
  - Key functions:
    - `build_signal_candidates(df)`
    - `build_action_list(df)`
    - `render_today_action_list(df)`
    - `load_signal_backtest_data(fund_types, years)`
    - `simulate_signal_trades(hist_df, signal_name, max_hold_days, failed_bounce_days)`
    - `render_signal_backtest(fund_types)`
    - `render_today_tab()`
    - `render_action_screens(df)`
  - Key constant:
    - `ACTION_PRIORITY`

- `app/data.py`
  - Provides `get_conn()`, `create_query()`, `get_data()`, `load_latest_perf()`, `load_prices()`, and `add_sparkline_column()`.
  - `add_sparkline_column()` normalizes price history to start at 100.
  - `get_sparkline_data()` fetches from `di.px_tbl` and caches data.

- `app/views/PullbackScanner.py`
  - Contains best-only pullback filtering/scoring.
  - Contains Short Target Monitor.
  - Contains Laggard Breakout Candidates.
  - Contains local `_add_sparkline_columns()` and `_sparkline_config()` helpers.

- `app/views/DailySummary.py`
  - Contains Market Pulse, Biggest Movers, Opportunity Radar, Laggard Awakenings, Leaders Weakening, Z-score alerts, MA crossover summary.
  - Contains local `_add_sparkline_columns()` and `_sparkline_config()` helpers.

- `app/views/TodaysCrossings.py`
  - Contains crossing/high/z-score/mover screens.
  - Deduplicates ticker rows before index lookup.
  - Contains local `_add_sparkline_columns()` and `_sparkline_config()` helpers.

- `app/views/RelativeStrength.py`
  - RS ranking table now has both sparklines.

- `app/views/RotationStrategies.py`
  - Current picks table now has both sparklines.

- `app/views/FactorDashboard.py`
  - Factor returns heatmap includes both sparklines for ETF rows.

- `app/views/PortfolioAllocator.py`
  - Regional valuation ETF table includes both sparklines.

- `task.md` and `changelog.md`
  - Updated with entries for Today tab, sparklines, signal backtest, and trade simulation.

- `HANDOFF.md`
  - This file. Should be updated again before ending future substantial work.

### Important Constants / Data Columns

- `DEFAULT_BENCHMARK` from `app/config.py` is currently `VWRP`.
- `FUND_TYPE_OPTIONS` from `app/config.py` is used for category filters.
- `table_height` from `app/config.py` is used for Performance table height.
- Performance table/query columns come from `app/duckdb_importer.py` constants:
  - `perf_desc_cols_start`
  - `perf_z_score_cols`
  - `perf_vol_cols`
  - `perf_mavg_cols`
  - `perf_returns_cols`
  - `perf_desc_cols_end`
  - `perf_rownames_cols`
- Historical trade simulation depends on:
  - Performance rows in `di.perf_tbl`
  - Price rows in `di.px_tbl`
  - `ticker`, `date`, `price`, moving average columns, return columns, volatility columns, drawdown columns

### Current Verification Commands

Use `uv`.

Most recent broad verification command:

```bash
uv run python -m py_compile app/PerformanceTable.py app/views/DailySummary.py app/views/TodaysCrossings.py app/views/RelativeStrength.py app/views/RotationStrategies.py app/views/FactorDashboard.py app/views/PortfolioAllocator.py app/views/PullbackScanner.py
uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/DailySummary.py app/views/TodaysCrossings.py app/views/RelativeStrength.py app/views/RotationStrategies.py app/views/FactorDashboard.py app/views/PortfolioAllocator.py app/views/PullbackScanner.py
```

Both passed after the latest changes.

## 4. What Remains To Be Done

### Immediate Next Useful Work

1. Make Today action list more investable.
   - Rename vague states like `Buy Watch` into decision states.
   - Suggested states:
     - `Buy Candidate`
     - `Wait For Reclaim`
     - `Avoid`
     - `Trim Candidate`
     - `Risk Review`
   - Add columns:
     - `Decision`
     - `Entry Rule`
     - `Invalidation`
     - `Exit Plan`
     - `Review Trigger`
     - `Backtest Edge`

2. Improve trade simulation validity.
   - Next-day execution and benchmark-relative returns have been added.
   - Add CAGR/expectancy/profit factor style metrics.
   - Add max adverse excursion / max favorable excursion if enough price path data is available.

3. Add per-signal parameter presets.
   - `Buy Watch` should have different exits than `Breakout Watch` and `Capitulation Watch`.
   - Current UI exposes shared `Bounce timeout` and `Max hold cap`; this may be too generic.

4. Decide whether to use trade simulation results to rank Today.
   - Current Today ranking still uses raw signal score and fixed action priority.
   - More automated version should weight candidates by historical signal edge.

5. Consider portfolio-level automation later.
   - Only after signal-level simulation is credible.
   - Would need allocation rules, cash handling, max positions, rebalance frequency, and transaction assumptions.

### Nice-To-Have Later

- Add a row-selection detail panel with full-size chart for selected instruments.
- Add OHLC/candlestick support only if pipeline/export is extended to include OHLC.
- Make `Price (1y)` optional if tables feel too wide.
- Move duplicated sparkline helper patterns into a shared utility only if duplication becomes painful.
- Add UI smoke tests or lightweight data tests if the project adopts a test framework.

## 5. Blockers / Open Questions

### Blockers

- No hard blockers.

### Open Questions

- Should trade simulation enter on same-day close, next-day close, or next-day open? Current implementation effectively uses signal-day `price` from available data.
- What should the primary investable universe be by default: ETFs only, stocks only, or mixed `eq + stock + commod`?
- Should `Capitulation Watch` be treated as a long-entry signal, or only as an alert requiring extra confirmation?
- Should `Trim Watch` and `Short Monitor` become explicit exit overlays for existing positions instead of independent actions?
- What are acceptable transaction costs/slippage assumptions if portfolio-level automation is added?
- Should `Today` prioritize fewer high-confidence candidates over broad coverage of every signal bucket?

## Current Git / Staging Notes

- The repo has many pre-existing dirty/untracked files. Do not revert unrelated changes.
- Relevant files staged/modified during this work include:
  - `HANDOFF.md`
  - `app/PerformanceTable.py`
  - `app/views/DailySummary.py`
  - `app/views/TodaysCrossings.py`
  - `app/views/PullbackScanner.py`
  - `app/views/RelativeStrength.py`
  - `app/views/RotationStrategies.py`
  - `app/views/FactorDashboard.py`
  - `app/views/PortfolioAllocator.py`
  - `task.md`
  - `changelog.md`
- Do not amend, reset, or discard unless explicitly requested.
