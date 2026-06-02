# Handoff: Refine FinTracker Strategy Email Alerts

## 1. Current Goal

Refine the FinTracker strategy email alert system: remove useless low-volatility rotation alerts, and ensure individual stocks appear in all non-breakout alert types.

## 2. Constraints & Decisions

- **Never restart mac mini launch agents** without explicit user request. All are currently stopped.
- **RAAM excluded** from all alert strategies.
- **Email format must be preserved** — exactly like breakout alerts: `What this targets:` explanation box, `ALERT # (Score)`, `Exchange`, `Reason`, `Trigger`, `Performance`, `Volatility`, and two inline charts (1Y signal + 3Y context).
- **Use same SMTP recipients** as breakout alerts (`BREAKOUT_ALERT_TO` / `EMAIL_RECIPIENTS`).
- `.L` tickers = EU/UK session; others = US session.
- **Mac mini path**: `~/fin-tracker-ui`

## 3. Files to Load

### Will be edited
- `scripts/send_strategy_alerts.py` — main alert sender (also reference for format)
- `app/alerts/signals.py` — signal builders and `build_all_signals()`
- `tests/test_strategy_alerts.py` — tests to update

### Reference / dependencies
- `scripts/send_breakout_alerts.py` — breakout format reference (charts, email rendering)
- `app/alerts/session.py` — session classification/filtering
- `app/alerts/state.py` — change detection and state persistence
- `resources/instrument_info.csv` — fund_type values for stocks
- `app/duckdb_importer.py` — column constants
- `Makefile` — strategy-alerts target

## 4. What Has Been Completed

- Strategy alert system built with breakout-style email format
- "What this targets:" explanation box added to every strategy email (text + HTML)
- Breakout chart annotations improved with performance callout box (1W/1M/3M/6M/1Y/3Y)
- EU/US session filtering for `.L` vs non-`.L` tickers
- Deployed to mac mini; verified alert state files written for all strategies
- Launch agents stopped on mac mini
- `/handoff` opencode command repaired (removed broken `opencode-handoff` plugin, replaced with file-based `.handoffs/` system)

## 5. What Remains To Do

1. **Remove `rotation_low_vol`** from `build_all_signals()` and `rotation_signals()` in `app/alerts/signals.py`
2. **Fix stock inclusion** — review `_fund_filter()` calls in all signal builders. The filter uses `"^(?:" + "|".join(prefixes) + ")"` which is anchor-at-start. Ensure `fund_type="stock"` passes through. Check `resources/instrument_info.csv` for actual fund_type values of individual stocks.
3. **Update tests** in `tests/test_strategy_alerts.py` to reflect removal and filter changes
4. **Deploy** changed files to mac mini with rsync
5. **Send examples** with `ssh macmini 'cd ~/fin-tracker-ui && make strategy-alerts SESSION=... ARGS="--active-only ..."'`

## 6. Verification Already Run

- `uv run python -m pytest tests/test_strategy_alerts.py` — 6 passed
- `uv run python -m py_compile` on all changed files — passed
- `uv run ruff check --select E9,F63,F7,F82` on changed files — passed
- Full test suite has unrelated pre-existing failures in allocator/strategy tests

## 7. Deployment State

- Mac mini: `~/fin-tracker-ui` (deployed via `rsync`)
- All alert launch agents stopped (`launchctl list | grep fintracker` shows only `duckdb-server` and `streamlit`)
- Alert state files at `~/fin-tracker-ui/storage/alerts/`
- `.env` has SMTP vars configured
