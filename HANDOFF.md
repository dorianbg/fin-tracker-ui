# Handoff: Fin Tracker UI Actionability Work

## 1. Goal / Current Task

We are improving the Streamlit dashboard to make it more actionable and less cluttered.

Current active task to continue next:

- Add a `🎯 Today’s Action List` to the top of the `Performance` tab.
- Include stocks by default in the Performance universe.
- The action list should synthesize the strongest signals into a daily decision queue instead of requiring the user to inspect several tabs.

## 2. Completed So Far

### Navigation / Tab Consolidation

- Removed these tabs from main navigation because their useful signals were merged elsewhere or were redundant:
  - `Breakout`
  - `Puke Detector`
  - `Laggard Breakout`
  - `Thematic`
  - `Drawdowns`
- Reordered remaining tabs by actionability:
  1. `Performance`
  2. `Pullback`
  3. `Crossings`
  4. `Daily Summary`
  5. `Relative Strength`
  6. `Rotation`
  7. `Cross-Asset`
  8. `Factors`
  9. `Charts`
  10. `Correlation`
  11. `Allocator`

### Layout Standardization

- Converted tab filters/settings to a tab-local left rail pattern using `st.columns([1, 4], gap="large")`.
- Important decision: do not use `st.sidebar` for tab-specific controls because Streamlit sidebars are global and leak across tabs.
- Pullback now uses a narrow left settings column and wide right content column.

### Performance Tab

- Added `Action Screens` under the main Performance table:
  - `📊 Volatility Spikes`
  - `💀 Capitulation`
  - `🚀 High-Vol Breakouts`
- Capitulation now falls back to ranked relative stress leaders if strict candidates are empty.
- High-Vol Breakouts now falls back to strongest positive 1W/1M movers when strict high-vol breakouts are empty.
- Drawdown context remains available in the main Performance table and action screens.

### Pullback Tab

- Stocks are included by default in Pullback via `fund_type_sidebar(default=["eq", "stock"], key="pullback_fund_types")`.
- Added `Best only` pullback mode, default enabled.
- Added `Top pullbacks to show`, default `20`.
- Defaulted `Require above 126-day MA` to `True`.
- Added best-only filters:
  - `Best-only max 52W drawdown (%)`, default `-20%`
  - `Best-only require bounce`, default `True`
- Added `quality_score` for pullbacks:
  - Rewards positive `ma_252` and `ma_126`.
  - Rewards controlled pullback depth, capped at 15%.
  - Adds bounce score if `r_1d > 0` or `r_1w > 0`.
  - Penalizes drawdowns deeper than 20%.
  - Penalizes breakdowns below `ma_63` and `ma_126`.
- Renamed/expanded downside section to `🎯 Short Target Monitor`.
- Short Target Monitor now includes:
  - `short_priority_score`
  - `monitor_reason`
  - `r_1mo`
  - `drawdown_52w`
- Added `🚀 Laggard Breakout Candidates` inside Pullback, replacing the standalone Laggard Breakout tab.
- Fixed duplicate-column bug in Pullback recovery logic by deduplicating recovery column selection with `dict.fromkeys(...)`.

### Daily Summary

- Added `⚠️ Leaders Weakening (Long-term Outperformers Losing Momentum)`.
- Logic:
  - Uses `config.DEFAULT_BENCHMARK`.
  - Finds 1Y outperformers vs benchmark by at least 10%.
  - Flags those with negative 1W relative strength vs benchmark.
  - Ranks by `weakening_score = (-rs_1w) * (r_1y - bm_1y)`.
- Existing `🔄 Laggard Awakenings` remains and is highly valued by the user.

### Bug Fixes

- Fixed `Today's Crossings` ambiguous truth-value error by deduplicating latest/yesterday rows by ticker before setting the index.
- Fixed Pullback recovery duplicate column error when long and short lookbacks map to the same return column.

## 3. Current State Of Relevant Files / Variables / Decisions

### Important Files

- `app/PerformanceTable.py`
  - Main app tab navigation and Performance tab implementation.
  - Contains `render_action_screens(df)` for volatility/capitulation/high-vol breakout drilldowns.
  - Needs the next feature: `render_today_action_list(df)` and helper `build_action_list(df)`.
- `app/views/PullbackScanner.py`
  - Contains best-only pullback filtering/scoring.
  - Contains Short Target Monitor.
  - Contains Laggard Breakout Candidates.
- `app/views/DailySummary.py`
  - Contains Laggard Awakenings and Leaders Weakening.
- `app/views/TodaysCrossings.py`
  - Deduplicates ticker rows before index lookup.
- `task.md` and `changelog.md`
  - Project tracking files, updated throughout this work.

### Current Performance Tab Filter Default

- Currently still defaults to `default=["eq", "commod"]` in `app/PerformanceTable.py`.
- User explicitly asked to add stocks by default.
- Next implementation should change it to:
  - `default=["eq", "stock", "commod"]`

### Decision: Today’s Action List Should Respect Performance Filter

- The new action list should use the same filtered dataframe as the main Performance table.
- Therefore, if the user changes `Instrument Category`, the action list updates with the same universe.
- Stocks should be included by default through the Performance filter change above.

### Planned Today’s Action List Buckets

Add a new `🎯 Today’s Action List` section near the top of `Performance`, before the main dataframe.

Suggested columns:

- `Action`
- `Instrument`
- `Ticker`
- `Why`
- `Score`
- `1D`
- `1W`
- `1M`
- `MA21`
- `MA63`
- `MA126`
- `MA252`
- `Drawdown`
- `Vol Ratio`

Signal buckets:

1. `Buy Watch`
   - Quality pullbacks in intact uptrends.
   - Rules:
     - `ma_252 > 0`
     - `ma_126 > 0`
     - `ma_21 < 0`
     - `drawdown_52w >= -20`
     - `r_1d > 0 or r_1w > 0`
   - Score similar to Pullback `quality_score`.
   - Why: `Strong trend pullback with early bounce`.

2. `Breakout Watch`
   - Upside volatility expansion.
   - Rules:
     - `r_1w > 0`
     - `r_1mo > 0`
     - `vol_ratio >= 1.1`
     - Prefer `ma_21 > 0` when available.
   - Score: `vol_ratio * r_1w`.
   - Why: `Upside move with elevated volatility`.

3. `Capitulation Watch`
   - Downside stress candidates.
   - Rules:
     - `drawdown_52w < 0`
     - strict flag if `vol_ratio >= 1.2 and drawdown_52w <= -10`
   - Score: `(-drawdown_52w) * vol_ratio`.
   - Why: `High stress / drawdown candidate`.

4. `Trim Watch`
   - Leaders weakening, same spirit as Daily Summary.
   - Rules:
     - benchmark = `config.DEFAULT_BENCHMARK`
     - `r_1y - benchmark_r_1y >= 10`
     - `r_1w - benchmark_r_1w < 0`
   - Score: `-(relative_1w) * relative_1y`.
   - Why: `Long-term leader losing short-term relative strength`.

5. `Short Monitor`
   - Mirror of Pullback Short Target Monitor.
   - Rules:
     - `ma_252 > 0`
     - `ma_21 < 0`
     - `ma_63 < 0`
     - prefer `r_1w < 0`
   - Score similar to Pullback `short_priority_score`:
     - weak 21/63 structure + negative 1W + volatility - positive 252 buffer.
   - Why: `Long-term trend intact but short/intermediate trend rolling over`.

Recommended action priority:

1. `Buy Watch`
2. `Breakout Watch`
3. `Trim Watch`
4. `Short Monitor`
5. `Capitulation Watch`

Duplicate handling decision:

- Allow duplicate tickers across action buckets initially.
- Reason: one ticker can legitimately be both `Capitulation Watch` and `Buy Watch`, and hiding one signal loses context.

### Verification Convention

- User requested commands should use `uv`.
- Use:
  - `uv run python -m py_compile app/PerformanceTable.py`
  - `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`

## 4. What Remains To Be Done

### Immediate Next Implementation

1. In `app/PerformanceTable.py`, change Performance default instrument categories from:
   - `default=["eq", "commod"]`
   to:
   - `default=["eq", "stock", "commod"]`

2. Add helper functions near `render_action_screens(df)`:
   - `build_action_list(df: pd.DataFrame) -> pd.DataFrame`
   - `render_today_action_list(df: pd.DataFrame)`

3. Call `render_today_action_list(df)` after `df = filter_dataframe(df, modify=True)` and before the main Performance dataframe render.

4. Add lightweight controls inside the action-list section:
   - `Action filter`: multiselect default all actions.
   - `Max rows`: selectbox or slider default 30.

5. Render summary metrics above the action table:
   - count of each action type.

6. Render the combined action dataframe sorted by:
   - action priority
   - score descending

7. Update `task.md` and `changelog.md`.

8. Run `uv` verification and stage changes.

### Nice-To-Have Later

- Consider moving duplicated scoring formulas into helper functions shared between Performance and Pullback.
- For now, keep implementation local to avoid over-abstraction.
- Manual Streamlit check should confirm table ergonomics and candidate counts.

## 5. Blockers / Open Questions

No blockers.

Open questions / tuning items:

- Exact score weights may need tuning after seeing live candidates.
- Whether `Capitulation Watch` should be lower priority by default, or surfaced more prominently during market stress.
- Whether duplicate tickers across action buckets should remain allowed. Current decision: allow duplicates initially.
- Whether the new action list should replace the existing `Action Screens` eventually. Current decision: keep both; action list is summary, action screens are drilldowns.

## Current Git / Staging Notes

There are many pre-existing unrelated dirty/untracked files in the repo. Do not revert them.

Relevant files staged from this work include, among others:

- `app/PerformanceTable.py`
- `app/views/PullbackScanner.py`
- `app/views/DailySummary.py`
- `app/views/TodaysCrossings.py`
- `task.md`
- `changelog.md`

`HANDOFF.md` was created as this handoff document and should be staged if desired.
