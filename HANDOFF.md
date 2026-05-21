# Handoff — RAAM Strategy Calibration + Historical Allocation Fit

**Date:** 2026-05-17  
**Repo:** `/Users/dbg/code/fin-tracker-ui`  
**Current branch/worktree:** existing project worktree, dirty  
**Last known committed baseline before this handoff:** `cf2c8de Add RAAM strategy and allocator tooling`  
**Current source changes after that commit:** uncommitted  

---

## 1. Goal / Current Task

We are building and calibrating the **RAAM Strategy** Streamlit dashboard tab: a dynamic asset-allocation model inspired by 3Fourteen Research's *Real Asset Allocation: The World Has Changed*.

Current focus:
- Make the RAAM tab faster and easier to use.
- Improve model fit against official/historical RAA allocation screenshots supplied by the user.
- Move the model closer to official RAA behavior by using official-like proxies, slower/risk-adjusted trend scoring, sleeve constraints, and more historical validation points.
- Keep the dashboard useful for live allocation, diagnostics, backtesting, trend visualization, and model-vs-official comparisons.

The user also asked how to increase allocation correlation further. Current answer: load official proxies first, then add calibration/grid search, allocation smoothing, sleeve-level ranking, and explicit optimization against official snapshots.

---

## 2. Completed So Far

### 2.1 Previously Committed Work

Committed baseline:

```text
cf2c8de Add RAAM strategy and allocator tooling
```

That commit included:
- `app/views/RAAMStrategy.py` initial RAAM dashboard module.
- `app/PerformanceTable.py` RAAM tab integration.
- `resources/instrument_info.csv` initial missing RAAM proxy additions.
- Streamlit/Pandas compatibility upgrade to `streamlit>=1.57.0`, `pandas>=3.0.0,<4`.
- Replacement of deprecated `use_container_width=True` with `width="stretch"` under `app/`.
- Replacement of Pandas 3-incompatible `Styler.applymap(...)` with `Styler.map(...)`.
- `uv.lock` regenerated.

### 2.2 Performance Optimization Completed

File changed: `app/views/RAAMStrategy.py`

Completed speedup:
- Removed `scipy.stats.linregress` from the hot trend loop.
- Replaced it with equivalent NumPy OLS math in `_regression_metrics()`.
- Reused precomputed log-price series instead of recomputing `np.log(...)` for every asset/date.

Measured impact:
- Trend computation before: approximately `5.34s`.
- Trend computation after: approximately `1.87s`.
- Full model compute path before chart rendering: roughly `3.5s` on current data.

Verification:
- `python -m compileall -q app` passed.
- `import views.RAAMStrategy` passed.
- RAAM tab rendered in Streamlit smoke tests.

### 2.3 RAAM Default Load Behavior Changed

File changed: `app/views/RAAMStrategy.py`

Previous behavior:
- RAAM had a `Load RAAM Strategy` checkbox defaulting to unchecked because Streamlit tabs execute eagerly.
- This avoided RAAM computation when using other tabs.

Current behavior:
- Checkbox renamed to `Run RAAM Strategy`.
- It defaults to checked.
- Users no longer need to click to see RAAM.
- Users can uncheck it if they want non-RAAM tabs to load without running the model.

Important decision:
- Because Streamlit `st.tabs` execute eagerly, default-on RAAM means the `Today` tab and other tabs pay RAAM compute/render cost on initial app load.
- Proper future fix is route/page-level navigation so only the active page executes.

### 2.4 Trend Visualization Added

File changed: `app/views/RAAMStrategy.py`

Added visible trend visuals under `Trend Signals`:
- `Latest Trend Rank vs Breadth` scatter.
- `12-Month Trend Rank Heatmap`.

Existing trend detail remains:
- `Trend Signals` table.
- `Trend Diagnostics` per-window regression direction grid.
- `Trend Score History` expander.

### 2.5 Historical Allocation Validation Added

File changed: `app/views/RAAMStrategy.py`

Added `Historical Allocation Check` section.

It compares model weights against screenshot-derived official RAA allocations:
- Uses nearest model rebalance date on or before the target snapshot date.
- Shows summary metrics: correlation, mean absolute delta, max absolute delta.
- Shows asset-level model vs official details.
- Shows grouped bar chart for selected snapshot.

Initial screenshot file used:
- `resources/Historical allocations.jpg`

Additional screenshot files added by user and currently untracked:
- `resources/2025-05.jpg`
- `resources/2025-06.jpg`
- `resources/2025-09.jpg`
- `resources/2026-01.jpg`

### 2.6 Historical Allocation Snapshots Transcribed

File changed: `app/views/RAAMStrategy.py`

Current `HISTORICAL_RAA_ALLOCATIONS` includes these snapshot dates:
- `2025-01-30`
- `2025-02-27`
- `2025-03-31`
- `2025-04-30`
- `2025-05-30`
- `2025-06-30`
- `2025-07-31`
- `2025-08-29`
- `2025-11-28`
- `2025-12-31`
- `2026-01-30`

Source images and visible official columns:
- `Historical allocations.jpg`: Jan 30, Feb 27, Mar 31 official from the April RAA positioning screenshot.
- `2025-05.jpg`: Apr 30 official, plus Mar 31 and Feb 28 columns.
- `2025-06.jpg`: May 30 official, plus Apr 30 and Mar 31 columns.
- `2025-09.jpg`: Aug 29 official, plus Jul 31 and Jun 30 columns.
- `2026-01.jpg`: Jan 30 official, plus Dec 31 and Nov 28 columns.

Notes:
- Values were manually transcribed from screenshots.
- Some image text is low-resolution; if precise production calibration is required, recheck transcription.

### 2.7 Official-Like Proxy Support Added

Files changed:
- `app/views/RAAMStrategy.py`
- `resources/instrument_info.csv`

Official screenshot proxies mentioned:
- `LQD`, `USRT`, `SDY`, `NOBL`, `IEV`, `BITO`, `GBTC`, `JNK`, `CTA`, `GSG`

Added to `resources/instrument_info.csv`:

```csv
LQD,iShares iBoxx $ Investment Grade Corporate Bond ETF,USD,bonds-corp,,Bond - Corporate
USRT,iShares Core U.S. REIT ETF,USD,eq,,Sector - Real Estate
SDY,SPDR S&P Dividend ETF,USD,eq,,Factor - Dividend
NOBL,ProShares S&P 500 Dividend Aristocrats ETF,USD,eq,,Factor - Dividend
IEV,iShares Europe ETF,USD,eq,,Equity - Europe
BITO,ProShares Bitcoin Strategy ETF,USD,eq,,Commodity - Digital
GBTC,Grayscale Bitcoin Trust ETF,USD,eq,,Commodity - Digital
JNK,SPDR Bloomberg High Yield Bond ETF,USD,bonds-corp,,Bond - High Yield
CTA,Simplify Managed Futures Strategy ETF,USD,eq,,Alternatives - Managed Futures
```

`GSG` was already present in `instrument_info.csv`.

Current RAAM proxy preferences with fallbacks:

| Asset | Preferred | Fallbacks |
|---|---|---|
| Bitcoin | `BITO` | `GBTC`, `IBIT` |
| Commodities | `GSG` | `PDBC` |
| Energy | `XLE` | none |
| Gold | `GLD` | none |
| Managed Futures | `CTA` | `DBMF` |
| Miners | `PICK` | none |
| Real Estate | `USRT` | `VNQ` |
| Dividend Payers | `NOBL` | `SDY`, `SCHD` |
| EM ex-China | `EMXC` | none |
| Europe | `IEV` | `VGK` |
| Japan | `EWJ` | none |
| Nasdaq | `QQQ` | none |
| US Large Cap | `SPY` | none |
| US Small Cap | `IWM` | none |
| Corporate Bonds | `LQD` | `VCIT` |
| EM Bonds | `EMB` | none |
| High Yield | `JNK` | `HYG` |
| Long-Term Treasuries | `TLT` | none |
| T-Bills | `BIL` | none |
| TIPS | `TIP` | none |

Current implementation details:
- `_all_proxies()` returns preferred proxies plus fallbacks plus benchmark proxies.
- `_select_proxy(cfg, columns)` chooses the first available proxy from preferred/fallback list.
- Trend, HRP, dynamic backtest, static RAAM backtest, and trend diagnostics now use `_select_proxy(...)`.

Important data state:
- At last verification, only `GSG` was available in exported app data among the new official proxies.
- The other official proxies are now in `instrument_info.csv` but need `make pipeline && make export` before the Streamlit app can use them.
- Until then, RAAM uses fallbacks.

### 2.8 Trend Engine Calibration Changes

File changed: `app/views/RAAMStrategy.py`

Changed default `WINDOW_WEIGHTS` to slower weighting:

```python
WINDOW_WEIGHTS = {21: 0.12, 42: 0.16, 63: 0.20, 126: 0.22, 189: 0.16, 252: 0.14}
```

Previous short windows were too dominant and caused a sharp defensive flip in March 2025.

Changed `_regression_metrics()` return tuple:
- Now returns `(slope, r_squared, slope_tstat, residual_zscore)`.
- Previously third return value was `predicted_last`.
- Downstream code only used ignored `_` for the third item except trend strength.

Changed trend strength:
- Uses `slope_tstat * r2` for windows `63`, `126`, `252`.
- Previously used raw `slope * r2`.

Current `TREND_BLEND` remains:

```python
TREND_BLEND = (0.50, 0.50, 0.00)
```

Meaning:
- 50% breadth z-score.
- 50% risk-adjusted strength z-score.
- Mean reversion disabled.

### 2.9 Sleeve Constraints Added

File changed: `app/views/RAAMStrategy.py`

Added broad bucket/sleeve bounds:

```python
BUCKET_BOUNDS = {
    "Alternatives": (0.13, 0.30),
    "Equities": (0.34, 0.60),
    "Fixed Income": (0.20, 0.45),
}
```

Implemented `_apply_bucket_bounds(grp)`:
- Runs after cap-safe normalization in `_normalize_and_enforce()`.
- Nudges bucket weights into broad official-like ranges.
- Re-runs capped normalization after bucket adjustments.

Important caveat:
- These are broad inferred constraints, not confirmed official RAA rules.
- User said they can provide sleeve-level constraints later.

### 2.10 Expanded Allocation Check Results

Last check ran after adding slower/risk-adjusted trend and sleeve constraints, before loading most new official proxies.

Available official proxies in exported data at that time:

```text
['GSG']
```

Model data range/check:

```text
weights shape: (2926, 8)
first weight date: 2012-05-31
latest weight date: 2026-05-13
latest weight sum: approximately 1.0
latest buckets:
  Alternatives: 28.2%
  Equities: 50.7%
  Fixed Income: 21.1%
```

Expanded official allocation fit:

| Snapshot | Model Date | Corr | MAD | Max Abs Delta |
|---|---:|---:|---:|---:|
| 2025-01-30 | 2024-12-31 | 0.877 | 1.86% | 4.93% |
| 2025-02-27 | 2025-01-31 | 0.928 | 1.36% | 4.60% |
| 2025-03-31 | 2025-03-31 | 0.518 | 3.07% | 13.30% |
| 2025-04-30 | 2025-04-30 | 0.462 | 2.92% | 10.22% |
| 2025-05-30 | 2025-05-30 | 0.888 | 1.94% | 4.54% |
| 2025-06-30 | 2025-06-30 | 0.869 | 2.19% | 6.10% |
| 2025-07-31 | 2025-07-31 | 0.908 | 2.10% | 7.86% |
| 2025-08-29 | 2025-08-29 | 0.863 | 2.54% | 7.79% |
| 2025-11-28 | 2025-11-28 | 0.732 | 2.08% | 10.22% |
| 2025-12-31 | 2025-12-31 | 0.850 | 2.13% | 16.51% |
| 2026-01-30 | 2026-01-30 | 0.751 | 2.21% | 10.71% |

Average MAD across snapshots:

```text
2.22%
```

Interpretation:
- Most months are reasonably close.
- March/April 2025 remain weak but improved from the original severe March miss.
- Next expected improvement should come from loading official proxies and explicit calibration.

### 2.11 Continued After Handoff

Files changed:
- `pipeline/utils.py`
- `app/views/RAAMStrategy.py`
- `task.md`
- `changelog.md`

Completed:
- Ran `make pipeline && make export` to load official-like RAAM proxy data.
- First pipeline run failed at DuckDB registration with Pandas 3 `StringDtype` columns: `Data type 'str' not recognized`.
- Fixed `insert_df_to_duckdb()` by copying the dataframe and converting pandas string extension columns to plain object columns before `cursor.register(...)`.
- Re-ran `make pipeline && make export`; it completed. Yahoo logged a timeout for `DIS`, but the RAAM proxies were downloaded/exported successfully.
- Confirmed exported RAAM app data now includes all official-like proxies: `LQD`, `USRT`, `SDY`, `NOBL`, `IEV`, `BITO`, `GBTC`, `JNK`, `CTA`, `GSG`.
- Tested no-lookahead monthly smoothing after cap/bucket constraints. Best simple alpha tested was `0.30`.
- Added `ALLOCATION_SMOOTHING_ALPHA = 0.30` and `_smooth_weights(...)` to `app/views/RAAMStrategy.py`.
- `generate_weights(...)` now applies smoothing by default after `_normalize_and_enforce(...)`.
- RAAM tuning controls now expose `Smoothing alpha` and pass it into tuned weight generation.

Verification:
- `source .venv/bin/activate && python -m compileall -q app pipeline` passed.
- Allocation-fit script passed with expected Streamlit bare-mode warnings.

Updated allocation fit after official proxies + smoothing:

| Snapshot | Model Date | Corr | MAD | Max Abs Delta |
|---|---:|---:|---:|---:|
| 2025-01-30 | 2024-12-31 | 0.855 | 1.84% | 5.15% |
| 2025-02-27 | 2025-01-31 | 0.895 | 1.68% | 4.46% |
| 2025-03-31 | 2025-03-31 | 0.906 | 1.58% | 7.14% |
| 2025-04-30 | 2025-04-30 | 0.827 | 1.93% | 5.75% |
| 2025-05-30 | 2025-05-30 | 0.928 | 1.76% | 5.18% |
| 2025-06-30 | 2025-06-30 | 0.904 | 2.06% | 7.42% |
| 2025-07-31 | 2025-07-31 | 0.930 | 2.09% | 8.99% |
| 2025-08-29 | 2025-08-29 | 0.862 | 2.53% | 9.39% |
| 2025-11-28 | 2025-11-28 | 0.859 | 2.05% | 4.83% |
| 2025-12-31 | 2025-12-31 | 0.865 | 2.23% | 14.20% |
| 2026-01-30 | 2026-01-30 | 0.868 | 1.62% | 7.78% |

Average correlation: `0.882`.
Average MAD: `1.94%`.

### 2.12 Further Fit Refinement + UI Descriptions

File changed:
- `app/views/RAAMStrategy.py`

Completed:
- Ran a focused calibration over nearby trend/HRP blend and smoothing alpha values.
- Updated defaults from `TREND_HRP_BLEND = 0.55` and `ALLOCATION_SMOOTHING_ALPHA = 0.30` to:

```python
TREND_HRP_BLEND = 0.65
ALLOCATION_SMOOTHING_ALPHA = 0.20
```

- Added Streamlit descriptions near the RAAM tab header:
  - `st.info(...)` explaining this is a research model, not an exact official replication.
  - `How to read this tab` expander.
  - `Model calibration` expander showing default blend/smoothing and fit objective.

Updated allocation fit after refined defaults:

| Snapshot | Model Date | Corr | MAD | Max Abs Delta |
|---|---:|---:|---:|---:|
| 2025-01-30 | 2024-12-31 | 0.870 | 1.87% | 6.80% |
| 2025-02-27 | 2025-01-31 | 0.894 | 1.77% | 6.08% |
| 2025-03-31 | 2025-03-31 | 0.924 | 1.59% | 4.82% |
| 2025-04-30 | 2025-04-30 | 0.844 | 1.93% | 4.91% |
| 2025-05-30 | 2025-05-30 | 0.954 | 1.45% | 2.93% |
| 2025-06-30 | 2025-06-30 | 0.914 | 1.90% | 6.00% |
| 2025-07-31 | 2025-07-31 | 0.946 | 1.74% | 7.73% |
| 2025-08-29 | 2025-08-29 | 0.875 | 2.33% | 8.48% |
| 2025-11-28 | 2025-11-28 | 0.874 | 1.94% | 4.67% |
| 2025-12-31 | 2025-12-31 | 0.884 | 2.09% | 12.51% |
| 2026-01-30 | 2026-01-30 | 0.893 | 1.55% | 5.81% |

Average correlation: `0.897`.
Average MAD: `1.83%`.
Max absolute snapshot delta: `12.51%`.

### 2.13 Fix Latest Nasdaq Underweight

User flagged latest official Nasdaq weight as `18.38%` while strategy showed roughly `10-12%`.

Root cause found:
- Nasdaq pre-smoothing/latest model signal was close to official, but final live allocation was dragged down by smoothing from prior allocations.
- Loose caps on tiny benchmark equity sleeves let EM ex-China/Japan/Small Cap take too much equity sleeve when ranked highly, crowding out Nasdaq and US Large Cap.

Changes:
- Tightened satellite equity caps:
  - `EM ex-China`: `10%` -> `2%`
  - `Europe`: `10%` -> `3%`
  - `Japan`: `10%` -> `3%`
  - `US Small Cap`: `10%` -> `4%`
- Updated `TREND_HRP_BLEND` to `0.80`.
- Modified `_smooth_weights(...)` so completed business-month-end rebalances still use default smoothing, but the latest incomplete month is unsmoothed.
- Added `_is_business_month_end(...)` helper.

Verification:
- `source .venv/bin/activate && python -m compileall -q app pipeline` passed.
- Latest model date: `2026-05-15`.
- Latest Nasdaq model weight: `18.43%`, close to official `18.38%`.
- Latest US Large Cap model weight: `20.20%`.
- Historical average correlation/MAD improved to `0.915`/`1.71%`.
- Historical max absolute snapshot delta: `10.23%`.

---

## 3. Current State Of Relevant Files / Variables / Decisions

### 3.1 Modified Source/Data Files

Uncommitted source/data changes:
- `app/views/RAAMStrategy.py`
- `resources/instrument_info.csv`
- `HANDOFF.md` itself

Dirty unrelated/local artifacts remain; do not assume clean worktree.

### 3.2 RAAM Constants And Defaults

Current key constants in `app/views/RAAMStrategy.py`:

```python
TREND_WINDOWS = [21, 42, 63, 126, 189, 252]
WINDOW_WEIGHTS = {21: 0.12, 42: 0.16, 63: 0.20, 126: 0.22, 189: 0.16, 252: 0.14}
TREND_BLEND = (0.50, 0.50, 0.00)
TREND_MULT_LOW = 0.05
TREND_MULT_HIGH = 2.50  # unused, kept for reference
TREND_HRP_BLEND = 0.55
RISK_FREE_RATE = cfg.RISK_FREE_RATE
MIN_HISTORY = 252
VOL_LOOKBACK = 126
BENCHMARK_60 = ("SPY", 0.60)
BENCHMARK_40 = ("TLT", 0.40)
BUCKET_BOUNDS = {
    "Alternatives": (0.13, 0.30),
    "Equities": (0.34, 0.60),
    "Fixed Income": (0.20, 0.45),
}
```

### 3.3 Current UI Decisions

- RAAM tab is integrated into `app/PerformanceTable.py` and rendered by `RAAMStrategy.render()`.
- `Run RAAM Strategy` defaults to checked.
- Controls are local to the RAAM tab, not in `st.sidebar`.
- `Enable tuning` still exposes rank-to-weight parameters and trend/HRP blend as a what-if tool.
- Historical allocation check uses fixed dated snapshot labels, no longer a generic year selector.

### 3.4 Current Worktree Status

Last observed `git status --short`:

```text
 M .DS_Store
 M HANDOFF.md
 M app/.DS_Store
 M app/views/RAAMStrategy.py
 M resources/instrument_info.csv
?? .agent/
?? .playwright-mcp/
?? RealAssetAllocation.pdf
?? allocator/.DS_Store
?? allocator_v2/.DS_Store
?? breakout-tab.png
?? export.csv
?? page-initial.txt
?? page-loaded.txt
?? raam-full-final.png
?? raam-full-page.png
?? raam-loading.txt
?? raam-rebuild-final.png
?? raam-rebuild-test.png
?? raam-state.txt
?? raam-trend-breadth.png
?? raam-v2-check.png
?? raam-v2-snap.txt
?? raam-v2.png
?? raam-waterfall.png
?? resources/2025-05.jpg
?? resources/2025-06.jpg
?? resources/2025-09.jpg
?? resources/2026-01.jpg
?? "resources/Historical allocations.jpg"
```

Do not commit `.DS_Store`, `.agent/`, `.playwright-mcp/`, screenshots/page dumps, `export.csv`, or runtime artifacts unless explicitly requested.

Need user decision on whether to commit screenshot files under `resources/`.

---

## 4. What Remains To Be Done

### 4.1 Load Official Proxy Data

Run:

```bash
make pipeline && make export
```

Purpose:
- Download/export official proxy tickers now added to `instrument_info.csv`.
- Let RAAM use `LQD`, `USRT`, `NOBL`/`SDY`, `IEV`, `BITO`/`GBTC`, `JNK`, `CTA` instead of fallbacks.

After this, rerun allocation checks and compare correlation/MAD again.

### 4.2 Verify Streamlit UI After Latest Changes

Recommended smoke test:

```bash
source .venv/bin/activate
cd app
streamlit run PerformanceTable.py
```

Check:
- App initial load with RAAM default-on.
- RAAM tab renders.
- `Historical Allocation Check` shows all snapshots.
- Trend visuals render.
- `Enable tuning` still works.

### 4.3 Add Calibration / Grid Search

Next high-value model work:
- Implement a small offline calibration helper, probably under `app/views/RAAMStrategy.py` first or separate script if it grows.
- Grid search over:
  - `WINDOW_WEIGHTS`
  - `TREND_BLEND`
  - `TREND_HRP_BLEND`
  - rank-to-weight exponents
  - floor fraction
  - bucket bounds
  - optional smoothing strength
- Objective:
  - maximize average correlation to `HISTORICAL_RAA_ALLOCATIONS`
  - minimize average MAD
  - penalize bad snapshots like `2025-03-31`, `2025-04-30`, `2025-12-31`

### 4.4 Add Allocation Smoothing

Likely official behavior is slower than raw monthly signals.

Candidate implementation:

```python
final_weight = alpha * raw_model_weight + (1 - alpha) * prior_month_weight
```

Suggested initial test:
- `alpha` between `0.35` and `0.65`.

Need to preserve no-lookahead:
- Smoothing should use only prior model weights, not future official allocations.

### 4.5 Split Bucket Allocation From Within-Bucket Ranking

Current model ranks all assets cross-sectionally, then applies broad bucket bounds.

Possible better official-like structure:
- Determine bucket weights first.
- Rank assets within each bucket.
- Allocate within each bucket subject to asset max caps.

This may prevent bonds from dominating equities because they are “less bad” during an equity drawdown.

### 4.6 User-Provided Sleeve Constraints

The user said they can provide sleeve-level constraints.

When provided:
- Replace inferred `BUCKET_BOUNDS` with user/official constraints.
- Consider max monthly bucket move constraints as well.

### 4.7 Commit Strategy

No commit has been made after `cf2c8de`.

When ready, consider separate commits:
1. RAAM performance/trend visualization/default-load changes.
2. Official proxies + historical allocation snapshots.
3. Trend/sleeve calibration model changes.

Do not commit local artifacts unless explicitly requested.

---

## 5. Blockers / Open Questions

### Blockers

No hard blockers.

Soft blockers:
- Official proxy tickers are not yet available in exported app data except `GSG`.
- Need `make pipeline && make export` before evaluating proxy-substitution impact.

### Open Questions

- Should screenshot files under `resources/` be committed, or are they local research artifacts only?
- Should RAAM remain default-on despite eager Streamlit tab execution, or should we move to page-level navigation?
- Should `BUCKET_BOUNDS` use official user-provided constraints once supplied?
- Should the model have two modes:
  - “Research model” for unconstrained dynamic allocation.
  - “RAA-fit model” calibrated to official historical allocations.
- Should calibration prioritize correlation, MAD, backtest performance, live ETF fit, or stress-period protection?
- Should Bitcoin use `BITO`, `GBTC`, or `IBIT` as primary for live behavior vs historical matching?
- Should Dividend Payers use `NOBL` or `SDY` as primary? Current primary is `NOBL`.
- Should Managed Futures use `CTA` or `DBMF` as primary? Current primary is `CTA`.

---

## 6. Verification Commands Run Recently

Passed:

```bash
source .venv/bin/activate && python -m compileall -q app
```

Passed, with expected Streamlit cache warnings outside runtime:

```bash
cd app && source ../.venv/bin/activate && python -c "import views.RAAMStrategy; print('import ok')"
```

Expanded allocation check script ran successfully and produced the metrics in section 2.10.

---

## 7. Suggested Next Agent Actions

1. Run `git status --short` before making further changes.
2. Do not revert local artifacts or user files.
3. Run `make pipeline && make export` to load official proxies if the user approves the time/network cost.
4. Re-run expanded allocation checks after export.
5. If fit improves, update the handoff/changelog and consider a commit.
6. If fit still lags, implement calibration/grid search and allocation smoothing next.
