# Tasks

## 2026-05-30 07:30 - Block Alert Emails On Stale Data

- Goal: Prevent live alert emails from being sent unless the alert data date is today, while preserving explicit testing/development escape hatches.
- Scope: alert freshness helper, breakout and strategy send scripts, env example, tests, and tracking updates.
- Assumptions: `--dry-run` should bypass freshness for strategy development; live sends can bypass only with `--allow-stale-data` or `FINTRACKER_ALLOW_STALE_ALERTS=1`.
- Plan: Add a reusable freshness guard, call it before emails/state saves, add CLI/env overrides, and test stale blocking plus dry-run/development bypasses.
- Verify: freshness/alert focused tests; Ruff error checks; Python compile checks; alert runner syntax check; full test suite run documents unrelated existing construction/strategy failures.
- Status: completed locally, uncommitted.

## 2026-05-30 00:00 - Share Alert Scanner Logic With Dashboard Views

- Goal: Remove duplicated alert strategy rules by sharing pure scanner functions between strategy emails and the Streamlit dashboard pages.
- Scope: `app/strategy_scanners.py`, alert signal builders, affected Streamlit views, focused tests, and tracking updates.
- Assumptions: Dashboard UI behavior should stay the same; scanner/filter/scoring logic should become reusable and testable; chart/email rendering stays separate.
- Plan: Extract pure scanners, replace duplicated view/alert logic with calls to those scanners, add regression tests proving alert wrappers use shared scanners, and run focused checks.
- Verify: focused alert/consolidation/sector tests; Ruff error checks; Python compile checks; wrapper syntax check; app-directory import check for affected views; alert script import smoke check. Full test suite run documents unrelated existing construction/strategy failures.
- Status: completed locally, uncommitted.

## 2026-05-29 13:25 - Use Existing Cron For Strategy Alerts

- Goal: Remove launchd strategy alert scheduling and rely on the existing cron schedules.
- Scope: strategy alert schedule files, Makefile install target, tracking updates.
- Assumptions: Cron schedules already exist outside the repo; the repo only needs to provide `scripts/run_strategy_alerts.sh us|eu` and `make strategy-alerts`/`make breakout-alerts` targets.
- Plan: Delete strategy alert `.plist` files, remove the strategy alert install target, and update deploy guidance to point existing cron at the runner script.
- Verify: `zsh -n scripts/run_strategy_alerts.sh`; focused alert tests.
- Status: completed locally, uncommitted.

## 2026-05-29 13:05 - Revert To Local DuckDB File

- Goal: Remove Quack/remote DuckDB server usage and make UI/alerts/pipeline use the local `duckdb.db` file directly.
- Scope: app data connection, alert scripts/wrapper, launchd/Makefile/deploy instructions, Quack server files, tracking updates.
- Assumptions: All runtime processes can access the local `duckdb.db` file; no remote Quack server is needed.
- Plan: Delete Quack launch agent/script, remove `install-duckdb-server`, remove `DUCKDB_REMOTE_HOST` code paths, simplify strategy alert runner, and unload/remove the installed stale launch agent.
- Verify: grep for Quack/remote references, Python compile checks, shell syntax check, focused alert tests, and read-only local DB connect count check.
- Status: completed locally, uncommitted.

## 2026-05-29 11:25 - Expand Large/Mid Cap Stock Universe

- Goal: Add systematic large and mid cap stock coverage so momentum/turnaround alerts scan beyond the existing thematic watchlists.
- Scope: `resources/instrument_info.csv`, tracking updates.
- Assumptions: Add missing S&P 500, S&P MidCap 400, FTSE 100, and Euro Stoxx 50 constituents; keep existing manually curated rows and skip duplicate tickers.
- Plan: Fetch public constituent tables, normalize Yahoo tickers (`.` to `-` for US classes, `.L` for FTSE, native suffixes for Euro Stoxx), append missing rows with `fund_type=stock`, and validate no duplicate tickers.
- Verify: CSV parse/duplicate check; `PYTHONPATH=. uv run python -m pytest tests/test_strategy_alerts.py tests/test_consolidation_setup.py -q`.
- Status: completed locally, uncommitted.

## 2026-05-29 10:30 - Add Momentum Breakout Alert

- Goal: Keep the focused turnaround alert and add a broader watchlist for stocks/ETFs making 4/8/12-week local highs, split into recovery breakouts and tight base breakouts near highs.
- Scope: `app/alerts/signals.py`, `tests/test_strategy_alerts.py`, tracking updates.
- Assumptions: The focused `turnaround` alert should include names within 2% of 4/8/12W local highs while still more than 10% below 52W highs; broader recovery breakouts are local highs still more than 10% below 52W or 104W highs; base breakouts are local highs within 10% of 52W highs after a tight 20D range and non-expanding short-term volatility.
- Plan: Restore and widen `turnaround` as a focused near-local-high recovery alert; add helper metrics for local-high windows, 20D range, and 104W drawdown; classify broader `momentum_breakout` signals as `Recovery breakout` or `Base breakout near highs`; include the metrics in email summaries.
- Verify: `PYTHONPATH=. uv run python -m pytest tests/test_strategy_alerts.py tests/test_consolidation_setup.py -q`; `PYTHONPATH=. uv run python -m py_compile app/alerts/signals.py scripts/send_strategy_alerts.py`; `zsh -n scripts/run_strategy_alerts.sh`; dry-run `scripts/send_strategy_alerts.py --session us --strategy momentum_breakout --dry-run --max-items 5`.
- Status: completed locally, uncommitted.

## 2026-05-27 00:00 - Add Strategy Email Alerts

- Goal: Send separate beautiful pre-market emails for active FinTracker strategies, with both changes and active-signal versions, while keeping RAAM excluded.
- Scope: alert helper modules/scripts, launchd scheduling, Makefile targets, tests, and tracking updates.
- Assumptions: Use the same SMTP recipients as breakout alerts; default strategy presets can be refined after observing live emails; `.L` instruments are EU/UK-session alerts and non-`.L` instruments are US-session alerts.
- Plan: Add pure signal builders, reusable email/state helpers, a strategy alert CLI with session filtering and dry-run support, update scheduling targets, and verify focused tests/checks.
- Test-first approach: Add focused unit tests for session classification/filtering, change detection, and representative strategy signal extraction.
- Verify: `uv run python -m pytest tests/test_strategy_alerts.py tests/test_consolidation_setup.py`; `uv run python -m py_compile app/alerts/session.py app/alerts/state.py app/alerts/email.py app/alerts/signals.py scripts/send_strategy_alerts.py scripts/send_breakout_alerts.py`; `uv run ruff check --select E9,F63,F7,F82 app/alerts scripts/send_strategy_alerts.py scripts/send_breakout_alerts.py tests/test_strategy_alerts.py`; `zsh -n` wrapper checks; plist parse checks. Full `uv run python -m pytest tests/` still has pre-existing allocator/strategy failures unrelated to alerts.
- Status: completed locally, uncommitted.

## 2026-05-27 00:00 - Run Alerts On Mac Mini

- Goal: Ensure the scheduled alert jobs are installed and run from the mac mini deployment checkout.
- Scope: alert launch agents and deploy instructions.
- Assumptions: `make deploy-breakout-alerts` syncs this repo to `~/fin-tracker-ui` on the mac mini, so launch agents should execute from that path.
- Plan: Point alert plists at `$HOME/fin-tracker-ui` and update deploy guidance to install combined strategy alerts.
- Verify: plist parse checks and `make -n deploy-breakout-alerts install-strategy-alerts` syntax checks; deployed via `make deploy-breakout-alerts`; installed launch agents on mac mini; ran EU and US alert batches once; verified `com.fintracker.strategy-alerts.eu/us` are loaded and alert state files were written.
- Status: completed locally, uncommitted.

## 2026-05-27 00:00 - Stop Duplicate Alert Scheduling

- Goal: Stop duplicate/undesired alert emails immediately and prevent old breakout scheduling from coexisting with combined alert scheduling.
- Scope: mac mini launchctl state and `Makefile` install target.
- Assumptions: No alert agents should remain loaded until the email format is corrected.
- Plan: Unload EU/US strategy alert agents and the standalone breakout alert agent; make strategy alert installation unload the old breakout agent defensively.
- Verify: `launchctl list | grep fintracker` should show no alert agents.
- Status: completed locally, uncommitted.

## 2026-05-23 00:00 - Add Bull Consolidation Setup Scanner

- Goal: Add a stock/ETF strategy scanner for bull-regime assets that are not overextended, have consolidated, and are near but not through breakout resistance.
- Scope: `app/views/ConsolidationSetup.py`, `app/PerformanceTable.py`, `tests/test_consolidation_setup.py`, plus tracking updates.
- Assumptions: Use the Reddit strategy notes as source direction: separate regime detection from trade filtering, classify bull regime with 200-day slope plus ADR band, use ADR units for extension and breakout distance, and exclude assets already breaking out.
- Plan: Add pure scanner logic with tests, query OHLC data from exported app prices, add a Streamlit tab with tunable thresholds, and verify focused tests/checks.
- Test-first approach: Add synthetic tests for a valid bull consolidation, an already-broken-out asset, and a non-bull asset.
- Verify: `"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m pytest tests/test_consolidation_setup.py`; `"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m py_compile app/views/ConsolidationSetup.py app/PerformanceTable.py tests/test_consolidation_setup.py`; `"/Users/dbg/code/fin-tracker-ui/.venv/bin/ruff" check --select E9,F63,F7,F82 app/views/ConsolidationSetup.py app/PerformanceTable.py tests/test_consolidation_setup.py`.
- Status: completed locally, uncommitted.

## 2026-05-23 00:00 - Add Breakout Email Alerts

- Goal: Send email when a stock/ETF breaks out above prior consolidation resistance.
- Scope: `app/views/ConsolidationSetup.py`, `scripts/send_breakout_alerts.py`, `Makefile`, `tests/test_consolidation_setup.py`, plus tracking updates.
- Assumptions: Use runtime SMTP environment variables and do not commit secrets; only email when fresh breakout triggers exist.
- Plan: Add a pure breakout trigger scanner, add an email script, expose it through `make breakout-alerts`, and verify focused tests/checks.
- Verify: focused scanner tests, compile checks, and syntax-critical ruff checks.
- Status: completed locally, uncommitted.

## 2026-05-23 00:00 - Schedule Breakout Email Alerts

- Goal: Run pipeline/export and send breakout email alerts automatically after market close on weekdays.
- Scope: `scripts/run_breakout_alerts.sh`, `scripts/com.fintracker.breakout-alerts.plist`, `Makefile`, plus tracking updates.
- Assumptions: macOS `launchd` is acceptable; schedule defaults to Monday-Friday at 22:15 local time.
- Plan: Add a wrapper script, launchd plist, and `make install-breakout-alerts` target.
- Verify: compile/check changed alert script and install launch agent.
- Status: completed locally, uncommitted.

## 2026-05-22 17:00 - Fix Sector Rotation LSE Ticker Matching

- Goal: Make Sector Rotation load `.L` universe constituents from exported app data after pipeline/export refresh.
- Scope: `app/data.py`, `app/views/SectorRotation.py`, `tests/test_sector_rotation.py`, plus tracking updates.
- Assumptions: Keep Yahoo-compatible `.L` symbols in sector universe definitions; app data may expose stripped `ticker` plus full `ticker_full`.
- Plan: Reproduce the missing Global UCITS rows, filter price loads by `ticker` or `ticker_full`, normalize Sector Rotation prices to `ticker_full` where available, and add a regression test.
- Test-first approach: Add a focused test proving `WTEL` plus `WTEL.L` maps into a `.L` price matrix.
- Verify: `make pipeline`; `make export`; `"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m pytest tests/test_sector_rotation.py`; fixed-code exported-data check confirms all 11 Global UCITS tickers have rows; `py_compile`; focused `ruff` syntax checks.
- Status: completed in commit `a59f0f5`.

## 2026-05-22 00:00 - Add Global UCITS Sector Rotation Universe

- Goal: Add a complete global sector rotation universe using UCITS ETFs suitable for UK/LSE execution.
- Scope: `app/views/SectorRotation.py`, `resources/instrument_info.csv`, `tests/test_sector_rotation.py`, plus tracking updates.
- Assumptions: Prefer one provider family for consistency; use SPDR MSCI World sector UCITS ETFs where available and `DPYG.L` as the existing developed-market property proxy for real estate.
- Plan: Use justETF sector ETF coverage as the source direction, validate Yahoo-compatible LSE symbols, add missing metadata, wire the universe into Sector Rotation, and extend tests.
- Test-first approach: Extend universe tests to assert a complete 11-ticker `.L` global UCITS universe.
- Verify: `"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m pytest tests/test_sector_rotation.py`; `"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m py_compile app/views/SectorRotation.py tests/test_sector_rotation.py`; `"/Users/dbg/code/fin-tracker-ui/.venv/bin/ruff" check --select E9,F63,F7,F82 app/views/SectorRotation.py tests/test_sector_rotation.py`.
- Status: completed locally, uncommitted.

## 2026-05-18 10:40 - Expand Sector Rotation Universes

- Goal: Fill sector-rotation gaps by covering all 11 SPDR sectors, adding broader US industry ETFs, and adding a complete LSE-listed Europe sector universe.
- Scope: `app/views/SectorRotation.py`, `resources/instrument_info.csv`, `tests/test_sector_rotation.py`, plus tracking updates.
- Assumptions: The US SPDR universe should remain the canonical 11-sector GICS set; the expanded US universe can include narrower industry ETFs; Europe uses validated LSE `.L` sector ETFs and may require a pipeline/export refresh for new tickers.
- Plan: Add missing Europe sector metadata, expand Europe universe to 11 sectors, add an explicit LSE Europe universe and a US extended universe, then run targeted tests/checks.
- Test-first approach: Extend sector-universe tests to assert complete SPDR and LSE Europe coverage.
- Verify: `uv run pytest tests/test_sector_rotation.py`; `uv run python -m py_compile app/views/SectorRotation.py tests/test_sector_rotation.py`; `uv run ruff check --select E9,F63,F7,F82 app/views/SectorRotation.py tests/test_sector_rotation.py`.
- Status: completed locally, uncommitted.

## 2026-05-18 10:24 - Scrape Official RAA Holdings

- Goal: Download official RAA holdings/allocation from the 3Fourteen RAA holdings page and use it as the live allocation anchor.
- Scope: `app/raa_official.py`, `scripts/download_raa_holdings.py`, `app/views/RAAMStrategy.py`, `Makefile`, `resources/raa_current_allocation.json`, plus tracking updates.
- Assumptions: The page's server-rendered `Asset Allocation` section is the authoritative current allocation source; if live scraping fails, the app can fall back to the last cached scrape.
- Plan: Add a stdlib HTML scraper, normalize official labels to RAAM asset names, cache successful downloads, wire RAAM latest incomplete-month anchoring to scraped data, and add a Make target.
- Test-first approach: Verify scraper output against the official page and then verify latest RAAM weights match scraped allocation.
- Verify: `uv run python scripts/download_raa_holdings.py`; `source .venv/bin/activate && python -m compileall -q app pipeline scripts`; latest RAAM comparison to scraped holdings has MAD `0.0000%`, max delta `0.0000%`, correlation `1.0000`.
- Status: completed locally, uncommitted.

## 2026-05-18 08:35 - Anchor Latest RAAM To Official Holdings

- Goal: Make latest RAAM allocation match the official live holdings supplied by the user.
- Scope: `app/views/RAAMStrategy.py`, plus tracking updates.
- Assumptions: Current official holdings should be treated as the authoritative live allocation when available; historical/backtest behavior should remain model-driven.
- Plan: Add current official holdings as a latest incomplete-month anchor, apply it only to the latest incomplete month, and document the behavior in the tab.
- Test-first approach: Compare latest model weights directly against the supplied official holdings.
- Verify: `source .venv/bin/activate && python -m compileall -q app pipeline`; latest live comparison has MAD `0.0000%`, max delta `0.0000%`, correlation `1.0000`, and sum `100.00%`.
- Status: completed locally, uncommitted.

## 2026-05-18 07:43 - Fix RAAM Live Nasdaq Underweight

- Goal: Bring latest RAAM Nasdaq weight closer to the official source while preserving historical snapshot fit.
- Scope: `app/views/RAAMStrategy.py`, plus tracking updates.
- Assumptions: Official-like behavior should not let tiny satellite equity sleeves crowd out Nasdaq/US large cap, and current incomplete-month live allocations should not be dragged by stale smoothing.
- Plan: Inspect latest weight decomposition, tighten satellite equity caps, skip smoothing for the current incomplete month, and verify live Nasdaq plus historical fit metrics.
- Test-first approach: Use the official Nasdaq `18.38%` example as the failing live check, then rerun allocation-fit verification.
- Verify: `source .venv/bin/activate && python -m compileall -q app pipeline`; latest model Nasdaq is `18.43%`; historical average correlation/MAD is `0.915`/`1.71%`.
- Status: completed locally, uncommitted.

## 2026-05-18 07:38 - Refine RAAM Fit And Explain Tab

- Goal: Further improve RAAM historical allocation fit and make the Streamlit tab easier to interpret.
- Scope: `app/views/RAAMStrategy.py`, plus tracking updates.
- Assumptions: Keep calibration surgical by only changing existing blend/smoothing defaults and adding descriptive UI text.
- Plan: Test focused trend/HRP blend and smoothing alpha combinations, update the best simple defaults, and add model/readme descriptions in the tab.
- Test-first approach: Use allocation-fit script before changing defaults; then compile and rerun the fit check.
- Verify: `source .venv/bin/activate && python -m compileall -q app pipeline`; allocation-fit script confirmed average correlation/MAD improved to `0.897`/`1.83%`.
- Status: completed locally, uncommitted.

## 2026-05-18 00:00 - Add Sector Rotation Strategy

- Goal: Implement a Faber-style sector rotation dashboard using the existing price data.
- Scope: `app/views/SectorRotation.py`, `app/PerformanceTable.py`, tests, plus tracking updates.
- Assumptions: Use monthly closes, average 1/3/6/9/12-month relative strength, equal-weight top N sectors, optional 10-month benchmark SMA cash filter, and existing total-return price series as the data source.
- Plan: Add pure ranking/backtest functions with tests, create a Streamlit tab for current ranks/backtest/rebalances, then compile and run targeted tests.
- Test-first approach: Add unit tests for ranking order and backtest/stat generation before verification.
- Verify: `uv run pytest tests/test_sector_rotation.py`; `uv run python -m py_compile app/PerformanceTable.py app/views/SectorRotation.py tests/test_sector_rotation.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/SectorRotation.py tests/test_sector_rotation.py`.
- Status: completed locally, uncommitted.

## 2026-05-18 00:00 - Add Robotics Stock Tracker

- Goal: Add a dedicated dashboard tab for the requested robotics stock universe with performance-table style data and price visualisation.
- Scope: `app/PerformanceTable.py`, `app/views/RoboticsStocks.py`, `resources/instrument_info.csv`, plus tracking updates.
- Assumptions: Use the existing yfinance/DuckDB/export pipeline; only add clearly public or reasonably mappable symbols; show unclear/private names separately rather than inventing tickers.
- Alignment: User requested a new tab like the Performance table and asked to parallelise with subagents.
- Plan: Add robotics metadata rows, create a focused Streamlit view using existing latest performance and price loaders, wire it into top-level tabs, then compile/check syntax.
- Test-first approach: No Streamlit UI tests cover tabs; use compile and syntax-critical lint checks plus CSV duplicate validation.
- Verify: `uv run python -m py_compile app/PerformanceTable.py app/views/RoboticsStocks.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py app/views/RoboticsStocks.py`; `make pipeline`; `make export`; CSV validation confirmed required mapped robotics tickers are present and no duplicate tickers exist; DuckDB check found `48/48` robotics tracker tickers in `latest_performance`.
- Status: completed locally, uncommitted.

## 2026-05-17 11:27 - Continue RAAM Calibration

- Goal: Load official-like RAAM proxy data and improve historical allocation fit against transcribed official snapshots.
- Scope: `app/views/RAAMStrategy.py`, `pipeline/utils.py`, exported app data, plus tracking updates.
- Assumptions: Monthly smoothing is plausible official-like behavior and must use only prior model weights, not future official allocations.
- Alignment: Continued from `HANDOFF.md`; no repeated questions for decisions already recorded there.
- Plan: Run pipeline/export, fix any pipeline compatibility issue blocking official proxies, rerun allocation checks, then add the smallest calibration change if fit still lags.
- Test-first approach: Use allocation-fit scripts and compile checks; no existing UI automation covers RAAM calibration.
- Verify: `source .venv/bin/activate && python -m compileall -q app pipeline`; RAAM allocation-fit script confirmed all official proxies present and average correlation/MAD improved to `0.882`/`1.94%`.
- Status: completed locally, uncommitted.

## 2026-05-12 00:00 - Improve Trade Simulation Validity

- Goal: Reduce lookahead bias in Today trade simulation and compare simulated trades against the dashboard benchmark.
- Scope: `app/PerformanceTable.py`, plus tracking updates.
- Assumptions: Historical signal rows should remain same-day signals, but simulated execution should use the next available ticker trading row; benchmark-relative return should use `DEFAULT_BENCHMARK` prices over the simulated entry/exit window.
- Alignment: Continued from `HANDOFF.md`; this was the next listed improvement after the investable Today decision columns already present in the working tree.
- Plan: Include benchmark prices in the historical backtest dataset, shift simulated entries to the next available trading day, add benchmark and relative return fields, and surface average relative return plus recent-trade columns.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Status: completed.

## 2026-05-02 00:00 - Add Nuclear And Uranium Theme Universe

- Goal: Add the requested uranium/nuclear ETFs and stocks to the instrument universe with clear nuclear sub-theme labels.
- Scope: `resources/instrument_info.csv`, plus tracking updates.
- Assumptions: Use the existing `sector` column as the theme label; preserve one row per ticker; update existing uranium rows rather than duplicating them.
- Alignment: User provided the ticker list and theme buckets; no extra clarification needed.
- Plan: Add missing tickers, update existing uranium ETF/stock theme labels, validate requested ticker coverage and duplicate tickers.
- Test-first approach: CSV/data change; use a Python CSV validation check rather than UI tests.
- Verify: `uv run python - <<'PY' ...` confirmed all requested tickers are present and no duplicate tickers exist.
- Status: completed.

## 2026-05-02 00:00 - Make Today Decisions Investable

- Goal: Convert Today from vague watchlist labels into an explicit decision queue.
- Scope: `app/PerformanceTable.py`, plus tracking updates.
- Assumptions: Preserve underlying signal names for filtering/backtest compatibility, but add decision-specific entry, invalidation, exit, review, and edge columns for human actionability.
- Alignment: Continued from `HANDOFF.md`; user had pushed back that the Today table was not actionable and asked for investment criteria.
- Plan: Add decision state mapping for each signal type, use confirmation conditions to distinguish buy candidates from wait/avoid/risk states, and update Today metrics/filter wording.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Status: completed.

## 2026-05-02 - Batch yfinance Pipeline Downloads

- Goal: make the price pipeline pull from yfinance faster.
- Scope: `pipeline/executor.py` historical price fetch path only. Preserve existing DuckDB insert, dividend backup, and per-ticker raw file behavior.
- Assumptions: a single batched `yf.download()` over all missing jobs is preferable to one `Ticker.history()` request per ticker plus a 3-second sleep per job.
- Plan: download all missing ticker ranges in one request, split results back by job, reuse existing `execute_job()` handling, compile, and run a no-fetch smoke check if the DuckDB lock allows it.
- Verify: `uv run python -m py_compile pipeline/executor.py` passes. `uv run python -m pipeline.executor --skip_data_fetch` was attempted but DuckDB was locked by another running Python process.
- Status: completed locally, uncommitted.

Track intended work for this project. Add newest entries at the top.

## 2026-05-02 00:00 - Replace Fixed Horizon Signal Backtest

- Goal: Make Today backtesting more investable by replacing fixed-horizon forward returns with trigger-based trade simulation.
- Scope: `app/PerformanceTable.py`, plus tracking updates.
- Assumptions: Simulate long-entry signals first (`Buy Watch`, `Breakout Watch`, `Capitulation Watch`); keep `Trim` and `Short Monitor` as risk/exit signals for now.
- Alignment: User rejected fixed 21-day exits and asked for more complicated/state-based exits.
- Plan: Enter on historical signal days, skip overlapping trades per ticker, exit on failed bounce, MA63 trend stop, profit protection, no-bounce confirmation, hard drawdown stop, or max-hold cap.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Status: completed.

## 2026-05-02 00:00 - Today Signal Backtest

- Goal: Start automating the Today workflow by validating signal rules historically before building portfolio automation.
- Scope: `app/PerformanceTable.py`, plus tracking updates.
- Assumptions: Start with signal backtests rather than capital allocation; use forward price returns over fixed holding periods as first-pass validation.
- Alignment: User chose option 3: both signal backtest and portfolio automation eventually, starting with signal backtest.
- Plan: Factor Today signal construction into reusable candidates, load historical perf rows and prices, compute 5/10/21/63 trading-day forward returns, and render summary stats plus recent historical samples.
- Test-first approach: No UI tests exist; use compile and syntax-critical ruff checks.
- Verify: `uv run python -m py_compile app/PerformanceTable.py`; `uv run ruff check --select E9,F63,F7,F82 app/PerformanceTable.py`.
- Status: completed.

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
