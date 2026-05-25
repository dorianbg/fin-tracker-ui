# fin-tracker — Data Pipeline + Streamlit Dashboard

Unified monorepo: data pipeline (yfinance → DuckDB) + multi-page Streamlit dashboard (via encrypted Parquet).

## Project Structure

| Directory/File | Purpose |
|------|---------
| `pipeline/` | Data pipeline — fetches prices from yfinance, stores in DuckDB |
| `pipeline/executor.py` | Main pipeline entry point. Downloads missing price data, handles dividends/rewrites |
| `pipeline/consts.py` | DB paths, table names, SQL DDL, instrument info CSV path |
| `pipeline/utils.py` | DuckDB connection management, data insertion, missing date range detection |
| `pipeline/process_data.py` | Experimental volatility calculations |
| `resources/instrument_info.csv` | Master instrument list (ticker, description, fund_type, sector) |
| `resources/latest_performance.sql` | SQL view definition for performance metrics |
| `app/PerformanceTable.py` | Main page. Performance table with Sharpe ratios, custom sorting, correlation matrix |
| `app/duckdb_importer.py` | Reads DuckDB, exports to encrypted Parquet. Defines all column name constants |
| `app/data.py` | Query builder + data access. `get_conn()`, `create_query()`, `get_data()` |
| `app/config.py` | Constants: benchmarks, chart sizes, risk-free rate (0.05) |
| `app/utils.py` | Plotting (Altair/Plotly), table styling, correlation matrix, dataframe filtering |

## Running

```bash
source .venv/bin/activate
cd app
streamlit run PerformanceTable.py
```

**Required env var**: None (reads from `duckdb.db` directly). For remote access, set `DUCKDB_REMOTE_HOST` and `QUACK_AUTH_TOKEN`.

## Agent Workflow

For non-trivial code changes, create and work in a separate git worktree before editing. This is required when the current tree is dirty, the task is risky, or concurrent work may be happening. Edit in place only for trivial changes or when the user explicitly asks to use the current branch/worktree.

**Commits:** NEVER commit without asking first. Even if encouraged ("we could commit this"), ask for explicit confirmation before running `git commit`.

## Data Loading

The app reads directly from `duckdb.db` via `data.get_conn()`. SQL views (`total_return`, `latest_performance`, `latest_performance_sharpe`) are created by the pipeline and must exist in the database.

## Column Constants (from `duckdb_importer.py`)

| Constant | Columns |
|----------|---------|
| `perf_desc_cols_start` | `date`, `description` |
| `perf_desc_cols_end` | `ticker`, `fund_type` |
| `perf_vol_cols` | `vol_1mo`, `vol_1y` |
| `perf_mavg_cols` | `ma_21`, `ma_63`, `ma_126`, `ma_252`, `drawdown_52w`, `drawdown_3y` |
| `perf_returns_cols` | `r_1d` through `r_5y` (10 periods) |
| `perf_sharpe_cols` | `r_1d_s` through `r_5y_s` |
| `perf_z_score_cols` | `z_1d`, `z_1w`, `z_2w`, `z_1mo` |
| `perf_rownames_cols` | `rown` (1 = latest day) |

## Benchmarks (from `config.py`)

| Label | Ticker |
|-------|--------|
| S&P 500 | CSP1 |
| MSCI USA | CUSS |
| MSCI Europe | IMEA |
| Emerging Markets | EEM |

## Makefile Targets

| Target | What it does |
|--------|-------------|
| `make pipeline` | Download latest price data from yfinance into DuckDB |
| `make pipeline-rewrite` | Full rewrite of all ticker data |
| `make pipeline-postgres` | Upload latest_performance to Postgres |
| `make export` | Export DuckDB → encrypted Parquet (for the UI) |
| `make ui` | Start the Streamlit dashboard |
| `make cron` | Pipeline + export (for scheduled runs) |
| `make clean` | Remove `__pycache__` directories |

## Environment Variables

| Variable | Required By | Purpose |
|----------|-------------|---------|
| `PARQUET_ENCRYPTION_KEY` | UI (`duckdb_importer.py`) | Encrypt/decrypt Parquet files |
| `POSTGRES_HOST` / `POSTGRES_PORT` / etc. | Pipeline (`make pipeline-postgres`) | Postgres upload (optional) |
| `DUCKDB_REMOTE_HOST` | UI, alerts | Connect to remote DuckDB via Quack |
| `QUACK_AUTH_TOKEN` | UI, alerts | Auth token for Quack server |
