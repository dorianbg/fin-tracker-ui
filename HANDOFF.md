# Handoff — fin-tracker-ui Dashboard Rebuild / Sector Rotation / RAAM

**Date:** 2026-05-22  
**Repo:** `/Users/dbg/code/fin-tracker-ui`  
**Branch:** `main`  
**Current HEAD:** `e520ad7 Add global UCITS sector universe`  
**Current worktree status at handoff:** clean (`git status --short` returned no output)  
**Important workflow rule:** For future non-trivial code changes, create a separate git worktree before editing unless the user explicitly asks to use the current worktree.

---

## 1. Goal / Current Task

We are rebuilding and extending the Streamlit dashboard, with the current focus on making the **Sector Rotation** strategy investable and complete.

Primary current goals:
- Implement Faber-style sector rotation from the supplied Meb Faber PDF / ChartSchool / Quantpedia descriptions.
- Cover all standard 11 GICS sectors for US SPDR rotation.
- Add broader US industry/sector ETF rotation.
- Add Europe/LSE sector rotation alternatives.
- Add a global UCITS sector rotation universe suitable for UK/LSE execution, using justETF as the source direction and Yahoo-compatible tickers for the pipeline.
- Keep RAAM and other dashboard rebuild work committed so a new chat can continue safely.

Secondary active project context:
- RAAM Strategy tab has been heavily rebuilt/calibrated and committed as a checkpoint.
- Robotics tab and universe work are included in the latest checkpoint commit.
- The user explicitly allowed deleting the temporary sector worktree and asked to commit all changes.

---

## 2. Completed So Far

### 2.1 Commit History Relevant To This Handoff

Recent commits on `main`:

```text
e520ad7 Add global UCITS sector universe
896b0ba Checkpoint dashboard and allocation updates
09d3efb Expand sector rotation universes
4b12687 Clarify agent worktree workflow
f65331e Add sector rotation strategy
cf2c8de Add RAAM strategy and allocator tooling
```

### 2.2 Sector Rotation Base Implementation

Commit: `f65331e Add sector rotation strategy`

Files added/changed:
- `app/views/SectorRotation.py`
- `tests/test_sector_rotation.py`
- `app/PerformanceTable.py`
- `task.md`
- `changelog.md`

Implemented:
- New `Sector Rotation` tab in the Streamlit app.
- Faber-style relative strength model:
  - monthly rebalance
  - ranks sectors by average trailing returns over `1, 3, 6, 9, 12` months
  - holds top N equal weight
  - optional 10-month SMA benchmark cash filter
- Current sector ranking table with sparklines.
- Backtest stats table.
- Growth-of-100 log equity chart.
- Recent rebalance history.
- Unit tests for ranking, backtest generation, and stats.

Important implementation details in `app/views/SectorRotation.py`:
- `SECTOR_UNIVERSES` defines all selectable universes.
- `LOOKBACK_MONTHS = [1, 3, 6, 9, 12]`.
- `current_sector_ranks(...)` computes the latest composite rank.
- `build_sector_rotation_backtest(...)` forms month-end signals and applies them from the next trading day.
- `performance_stats(...)` computes CAGR, volatility, Sharpe, and max drawdown.
- Streamlit data imports (`load_prices`, `add_sparkline_column`) are inside `render()` to avoid Streamlit import issues during tests.

### 2.3 Worktree Rule Clarified

Commit: `4b12687 Clarify agent worktree workflow`

Changed:
- `CLAUDE.md` now explicitly says non-trivial code changes require a separate git worktree before editing unless the user explicitly asks to use the current branch/worktree.

Reason:
- Earlier sector work was initially done in the dirty main worktree. User called this out. Rule is now explicit in the project instructions.

### 2.4 Sector Universes Expanded

Commit: `09d3efb Expand sector rotation universes`

Changed:
- Expanded `SECTOR_UNIVERSES` in `app/views/SectorRotation.py`.
- Added validated Europe/LSE sector ETF metadata to `resources/instrument_info.csv`.
- Extended `tests/test_sector_rotation.py`.

US SPDR universe is confirmed complete with all 11 standard sectors:
- Communication Services: `XLC`
- Consumer Discretionary: `XLY`
- Consumer Staples: `XLP`
- Energy: `XLE`
- Financials: `XLF`
- Healthcare: `XLV`
- Industrials: `XLI`
- Materials: `XLB`
- Real Estate: `XLRE`
- Technology: `XLK`
- Utilities: `XLU`

Added `US Extended Sector ETFs` universe:
- Includes the 11 SPDR sector ETFs plus narrower ETFs:
  - `KRE` regional banks
  - `XBI` biotech
  - `XHB` homebuilders
  - `XME` metals/mining
  - `ITA` aerospace/defence
  - `IYT` transport
  - `IGV` software
  - `KIE` insurance
  - `ITB` home construction
  - `XOP` oil/gas exploration
  - `VOX`, `VGT`, `VNQ` Vanguard sector proxies

Europe/LSE sector universe decisions:
- Guessed Europe sector tickers were not committed.
- Yahoo validation found the following usable LSE/Yahoo tickers:
  - Communication Services: `TELE.L`
  - Consumer Discretionary: `ESIC.L`
  - Consumer Staples: `ESIS.L`
  - Energy: `ESIE.L`
  - Financials: `ESIF.L`
  - Healthcare: `ESIH.L`
  - Industrials: `ESIN.L`
  - Technology: `ESIT.L`
  - Materials: `MTRL.L`
  - Real Estate: `IPRP.L`
  - Utilities: `UTIL.L`
- `IPRP.L` is an iShares European Property Yield ETF, used as the Europe real-estate proxy.

### 2.5 Global UCITS Sector Universe Added

Commit: `e520ad7 Add global UCITS sector universe`

Changed:
- Added `Global UCITS Sector ETFs` to `SECTOR_UNIVERSES`.
- Added SPDR MSCI World sector UCITS rows to `resources/instrument_info.csv`.
- Extended tests to assert a complete 11-ticker global UCITS `.L` universe.

Global UCITS universe in `app/views/SectorRotation.py`:

```python
"Global UCITS Sector ETFs": {
    "benchmark": "VWRP.L",
    "cash": "ERNS.L",
    "tickers": [
        "WTEL.L",
        "WCOD.L",
        "WCOS.L",
        "WNRG.L",
        "WFIN.L",
        "WHEA.L",
        "WNDU.L",
        "WTEC.L",
        "WMAT.L",
        "DPYG.L",
        "WUTI.L",
    ],
}
```

Global UCITS sector mapping:
- Communication Services: `WTEL.L` — SPDR MSCI World Communication Services UCITS ETF
- Consumer Discretionary: `WCOD.L` — SPDR MSCI World Consumer Discretionary UCITS ETF
- Consumer Staples: `WCOS.L` — SPDR MSCI World Consumer Staples UCITS ETF
- Energy: `WNRG.L` — SPDR MSCI World Energy UCITS ETF
- Financials: `WFIN.L` — SPDR MSCI World Financials UCITS ETF
- Healthcare: `WHEA.L` — SPDR MSCI World Health Care UCITS ETF
- Industrials: `WNDU.L` — SPDR MSCI World Industrials UCITS ETF
- Technology: `WTEC.L` — SPDR MSCI World Technology UCITS ETF
- Materials: `WMAT.L` — SPDR MSCI World Materials UCITS ETF
- Real Estate: `DPYG.L` — iShares Developed Markets Property Yield UCITS ETF GBP Hedged, existing proxy
- Utilities: `WUTI.L` — SPDR MSCI World Utilities UCITS ETF

Decision:
- justETF was used as the source direction for UCITS global sector coverage.
- Yahoo-compatible LSE symbols were validated using the existing main repo virtualenv at `/Users/dbg/code/fin-tracker-ui/.venv`.
- No new virtualenv should be created in worktrees. Use the main repo venv explicitly for verification commands.
- A SPDR MSCI World Real Estate LSE ticker was not found; `DPYG.L` was retained as the existing global/developed real-estate proxy.

### 2.6 RAAM / Robotics / Dashboard Checkpoint

Commit: `896b0ba Checkpoint dashboard and allocation updates`

This commit intentionally checkpointed all previously dirty work after the user requested committing all changes.

Included notable files/features:
- RAAM updates in `app/views/RAAMStrategy.py`.
- Official RAA holdings scraper:
  - `app/raa_official.py`
  - `scripts/download_raa_holdings.py`
  - `resources/raa_current_allocation.json`
  - `make raa-holdings` target in `Makefile`
- Robotics tab:
  - `app/views/RoboticsStocks.py`
  - `app/PerformanceTable.py` integration
  - robotics universe rows in `resources/instrument_info.csv`
- Tracking updates in `task.md`, `changelog.md`, and `HANDOFF.md`.
- Various screenshots/PDFs/artifacts were included because the user explicitly asked to commit all changes across all files.

Important caveat:
- This was a large checkpoint commit, not a cleanly isolated feature commit.
- It included generated/local artifacts such as `.DS_Store`, `.playwright-mcp/`, screenshots, PDFs, and image files because the user asked to commit all changes.

---

## 3. Current State Of Relevant Files / Variables / Decisions

### 3.1 Repository State

- Current worktree is clean.
- Temporary worktrees:
  - `/Users/dbg/code/fin-tracker-ui-sector-rotation` was removed after explicit approval.
  - `/Users/dbg/code/fin-tracker-ui-global-ucits` was used for `e520ad7`; at the time of this handoff it may still exist unless removed after this file is written. Check `git worktree list` before starting new work.

### 3.2 Sector Rotation Files

Primary file:
- `app/views/SectorRotation.py`

Tests:
- `tests/test_sector_rotation.py`

Instrument metadata:
- `resources/instrument_info.csv`

Top-level app integration:
- `app/PerformanceTable.py` imports and renders `views.SectorRotation` in the `Sector Rotation` tab.

Current sector universes in `SECTOR_UNIVERSES`:
- `US Select Sector SPDRs`
- `US iShares UCITS S&P 500 Sectors`
- `US Extended Sector ETFs`
- `Global UCITS Sector ETFs`
- `Europe iShares MSCI Europe Sectors`
- `LSE Europe Sectors`

Cash proxies:
- Most sector universes use `ERNS.L`.

Benchmarks:
- US universes use `CSP1` / S&P 500 proxy.
- Global UCITS uses `VWRP.L`.
- Europe universes use `IMEA.L`.

### 3.3 `resources/instrument_info.csv` Sector Metadata

Global UCITS rows added:

```csv
WTEL.L,SPDR MSCI World Communication Services UCITS ETF,USD,eq,,Sector - Communication Services
WCOD.L,SPDR MSCI World Consumer Discretionary UCITS ETF,USD,eq,,Sector - Consumer Discretionary
WCOS.L,SPDR MSCI World Consumer Staples UCITS ETF,USD,eq,,Sector - Consumer Staples
WNRG.L,SPDR MSCI World Energy UCITS ETF,USD,eq,,Sector - Energy
WFIN.L,SPDR MSCI World Financials UCITS ETF,USD,eq,,Sector - Financials
WHEA.L,SPDR MSCI World Health Care UCITS ETF,USD,eq,,Sector - Healthcare
WNDU.L,SPDR MSCI World Industrials UCITS ETF,USD,eq,,Sector - Industrials
WTEC.L,SPDR MSCI World Technology UCITS ETF,USD,eq,,Sector - Technology
WMAT.L,SPDR MSCI World Materials UCITS ETF,USD,eq,,Sector - Materials
WUTI.L,SPDR MSCI World Utilities UCITS ETF,USD,eq,,Sector - Utilities
```

Europe/LSE sector rows added earlier:

```csv
ESIS.L,iShares MSCI Europe Consumer Staples Sector,GBP,eq,,Sector - Consumer Staples
MTRL.L,SPDR MSCI Europe Materials UCITS ETF,EUR,eq,,Sector - Materials
UTIL.L,SPDR MSCI Europe Utilities UCITS ETF,EUR,eq,,Sector - Utilities
IPRP.L,iShares European Property Yield UCITS ETF,GBP,eq,,Sector - Real Estate
TELE.L,SPDR MSCI Europe Communication Services UCITS ETF,EUR,eq,,Sector - Communication Services
```

### 3.4 Validation Commands Used

For Sector Rotation and Global UCITS universe, verification used the main repo venv explicitly:

```bash
"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m pytest tests/test_sector_rotation.py
"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m py_compile app/views/SectorRotation.py tests/test_sector_rotation.py
"/Users/dbg/code/fin-tracker-ui/.venv/bin/ruff" check --select E9,F63,F7,F82 app/views/SectorRotation.py tests/test_sector_rotation.py
```

All passed.

Full project validation before `896b0ba`:
- `uv run python -m compileall -q app pipeline scripts allocator allocator_v2 tests` passed.
- `uv run pytest tests` had 5 failures, apparently unrelated to Sector Rotation:
  - `tests/test_construction.py::test_build_portfolio_plan_emits_full_baseline_plan`
  - 4 failures in `tests/test_strategy.py`

Do not assume full test suite is currently green.

### 3.5 Environment Decision

- Do not create a new `.venv` inside git worktrees.
- If working from a worktree, use the main repo venv explicitly:

```bash
/Users/dbg/code/fin-tracker-ui/.venv/bin/python
/Users/dbg/code/fin-tracker-ui/.venv/bin/ruff
```

This was prompted by `uv run` trying to create `/Users/dbg/code/fin-tracker-ui-global-ucits/.venv` in the worktree. That partial env was removed.

---

## 4. What Remains To Be Done

### 4.1 Immediate Sector Rotation Follow-Ups

- Run `make pipeline` and `make export` so newly added ETF metadata is fetched into DuckDB and exported to encrypted Parquet for the app.
- After export, open the Sector Rotation tab and check that `Global UCITS Sector ETFs` shows all expected tickers with price history.
- Confirm whether `DPYG.L` is acceptable as the global/developed real-estate proxy, or whether the user wants a different global real-estate UCITS ETF.
- Consider adding source comments or docs for each sector universe, especially where a proxy is not exact.

### 4.2 Full Test Suite Follow-Up

- Investigate 5 existing failing tests:
  - allocator construction baseline mismatch: expected `150000.0`, observed `157500.0`
  - strategy signal expectations changed from `BUY`/`AVOID`/`WATCH` to `HOLD`/`ACCUMULATE`
- Determine whether tests are stale relative to current allocator/strategy behavior or whether recent commits introduced regressions.

### 4.3 Cleanup / Repo Hygiene

- Consider removing or ignoring committed generated artifacts in a future cleanup PR/commit if the user agrees:
  - `.DS_Store`
  - `.playwright-mcp/`
  - screenshot PNGs/JPGs used for debugging
  - temporary page state text files
- Do not remove them without explicit user approval because they were committed after the user asked to commit all changes.

### 4.4 RAAM Follow-Ups

- Refresh official RAA holdings with `make raa-holdings` or `uv run python scripts/download_raa_holdings.py`.
- Run `make pipeline` and `make export` after instrument universe changes so official proxies and new sector ETFs appear in app data.
- Re-check RAAM latest allocation and historical fit after refreshed data.
- If full tests remain failing, check if RAAM/allocator changes changed assumptions in allocator tests.

---

## 5. Blockers / Open Questions

Open questions:
- Is `DPYG.L` acceptable as the real-estate proxy in `Global UCITS Sector ETFs`, or should a different global UCITS real estate ETF be sourced from justETF?
- Should the Europe sector universes be kept now that a global UCITS universe exists, or should the UI default to Global UCITS for UK users?
- Should `US Extended Sector ETFs` include thematic sectors like semiconductors, cybersecurity, infrastructure, agriculture, timber, airlines, etc., or stay limited to sector/industry ETFs already added?
- Should Sector Rotation backtests include transaction costs/slippage/turnover metrics?
- Should the cash leg use a real ETF return (`ERNS.L`) or zero return when unavailable?

Known blockers / risks:
- Full project pytest suite is not green; targeted Sector Rotation tests pass.
- New ETF rows require pipeline/export refresh before Streamlit can show data.
- justETF pages are not a structured API; ETF source verification used justETF directionally and Yahoo validation for executable ticker symbols.
- Worktree discipline must be maintained for future changes.

---

## 6. Recommended Next Commands

Use from `/Users/dbg/code/fin-tracker-ui` unless working in a separate worktree.

```bash
git status --short
make pipeline
make export
"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m pytest tests/test_sector_rotation.py
```

If starting more non-trivial code work:

```bash
git worktree add /Users/dbg/code/fin-tracker-ui-<task-name> -b <task-branch>
```

Then use the main venv from inside that worktree:

```bash
"/Users/dbg/code/fin-tracker-ui/.venv/bin/python" -m pytest tests/test_sector_rotation.py
```
