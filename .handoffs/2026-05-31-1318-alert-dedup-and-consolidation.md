# Alert System — Dedup, Freshness, and Consolidation Plan

## Goal

Fix duplicate/runaway alert emails and work toward a consolidated single-email format where instruments are ranked by how many alert signals they trigger.

## Key Decisions

- **Local DB only** — Quack/remote DuckDB removed everywhere. App and alerts read `duckdb.db` directly.
- **Breakout alerts are standalone** — `send_breakout_alerts.py` has its own email format with ADR/breakout-level charts. `breakout_signals` was removed from `build_all_signals()` to prevent duplication.
- **Cron, not launchd** — Mac mini uses crontab (`5 7 * * 1-5` for EU, `5 20 * * 1-5` for US) calling `scripts/cron_alerts.sh`.
- **Freshness** — 3 calendar-day window (covers weekends/Mondays). Blocks sends with `--allow-stale-data` or `FINTRACKER_ALLOW_STALE_ALERTS=1` escape hatches.
- **Shared scanners** — `app/strategy_scanners.py` holds pure filter/scoring functions used by both alert email senders AND Streamlit dashboard views. No more duplicated logic.
- **State tracking** — Both breakout and strategy senders save alert state after each send. Subsequent runs skip if no changes. Strategy active emails now gated on `not changes.empty`.
- **Expanded universe** — `resources/instrument_info.csv` has 1394 tickers (S&P 500, S&P MidCap 400, FTSE 100, Euro Stoxx 50).

## Files to Load

| Purpose | Files |
|---|---|
| Core alert logic | `app/alerts/signals.py`, `app/alerts/session.py`, `app/alerts/state.py`, `app/alerts/freshness.py` |
| Shared scanners | `app/strategy_scanners.py` |
| Email senders | `scripts/send_breakout_alerts.py`, `scripts/send_strategy_alerts.py` |
| Runner + cron | `scripts/run_strategy_alerts.sh`, `scripts/cron_alerts.sh` (mac mini only) |
| Tests | `tests/test_strategy_alerts.py`, `tests/test_alert_freshness.py`, `tests/test_consolidation_setup.py` |
| Universe data | `resources/instrument_info.csv` |
| Dashboard views (affected) | `app/views/PullbackScanner.py`, `app/views/LaggardBreakout.py`, `app/views/PukeDetector.py`, `app/views/TodaysCrossings.py` |

## Completed

- **Alert architecture** — `app/alerts/signals.py` with 12 strategy signal builders (consolidation, sector_rotation, puke_buy, pullback, laggard_awakening, turnaround, momentum_breakout, todays_crossings, rotation_momentum, rotation_mean_reversion, rotation_vol_adjusted, rotation_puke). Breakout is standalone via `send_breakout_alerts.py`.
- **Dedup at all layers**:
  - Strategy sender: active emails only sent when `not changes.empty`
  - Breakout sender: state-tracked via `load_previous`/`detect_changes`/`save_current`
  - `_finalize()`: drops duplicate `(alert_ticker, signal)` combinations
- **Freshness guard** — 3-day delta check with `--allow-stale-data` and env override.
- **Shared scanners** — Extracted pure functions from Streamlit views into `app/strategy_scanners.py`.
- **Mac mini deployment** — `rsync` deployed, duckdb-server removed, cron installed, tested end-to-end.
- **Chart fixes** — performance annotation at bottom-left, 1D return in z-score summaries.

## Remaining / Next

1. **Consolidated email** — User's idea: single email per session with all instruments ranked by signal count across strategies. Instead of 12 separate strategy emails + 1 breakout, produce one ranked instrument list showing which tickers triggered the most signals.
2. **Investigate root cause of duplicate ticker bug** — AUTO.L appeared 8 times in rotation_puke email. DB had no duplicate rows. `_finalize` dedup is a band-aid defense; root cause unknown.
3. **Commit pending** — Latest fixes (dedup in signals.py, freshness 3-day window, breakout state tracking, strategy active dedup gating) are uncommitted in the `jammy-ferret` worktree. User must run `git commit` + `git push`.
4. **State file noise on first run** — First run per session sends "all New" emails because no previous state exists. Acceptable baseline, but could silence initial `--active-only` to avoid flood.
5. **Mac mini DB completeness** — Historical prices has 1408 tickers, total_return has 1152. Some tickers still missing data from partial pipeline kills earlier. Run `make pipeline` from mac mini on Monday morning to fill gaps.

## Verification

| Check | Result |
|---|---|
| `pytest tests/test_strategy_alerts.py tests/test_alert_freshness.py tests/test_consolidation_setup.py` | 23 passed |
| `ruff check --select E9,F63,F7,F82` on changed files | All passed |
| `py_compile` all changed files | All passed |
| `zsh -n scripts/run_strategy_alerts.sh` | Passed |
| Breakout dedup (run twice on mac mini) | Second run: "No new breakout changes" |
| Full test suite (`pytest tests/`) | 5 pre-existing unrelated failures (test_construction, test_strategy) |

## Deployment State

- **Mac mini**: `~/fin-tracker-ui/` deployed via rsync. `duckdb-server` launch agent removed. Crontab: `5 7,20 * * 1-5 /Users/dbg/fin-tracker-ui/scripts/cron_alerts.sh`. Log: `/Users/dbg/Library/Logs/fintracker-alerts.log`.
- **Worktree**: `jammy-ferret` branch at `/Users/dbg/.local/share/opencode/worktree/.../jammy-ferret`. DB symlinked to `/Users/dbg/code/fin-tracker-ui/duckdb.db`.
- **Main repo**: `/Users/dbg/code/fin-tracker-ui/` — expanded instrument_info.csv was copied here for pipeline runs.
- **Changes uncommitted**: `app/alerts/signals.py`, `app/alerts/freshness.py`, `scripts/send_breakout_alerts.py`, `scripts/send_strategy_alerts.py`, plus earlier refactor files.
