"""External data fetching with DuckDB persistent cache.

Fetches:
  - ETF prices + MA200 via yfinance (all recommended tickers)
  - Macro data (US/UK real yields, EM HY spread, ACWI metrics) via FRED
  - Shiller CAPE data (monthly CSV)

Uses a DuckDB file (allocator_cache.duckdb) as the persistence layer so
reloads are fast and the app works offline using the last fetched values.

All public functions return plain Python dicts/dataclasses so callers
(including Streamlit cached wrappers) stay simple.
"""

import logging
import os
import sys
import importlib.util
from collections import defaultdict
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd

from instruments import get_yfinance_ticker_map, lookup

log = logging.getLogger(__name__)

_CACHE_DB = os.path.join(os.path.dirname(__file__), "allocator_cache.duckdb")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
_REPO_DATA_MODULE = None

# FRED series IDs
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRED_SERIES = {
    "us_real_10y": "DFII10",           # US 10y TIPS yield
    "uk_nominal_10y": "IRLTLT01GBM156N",  # UK 10y nominal gilt
    "us_10y_nominal": "DGS10",          # US 10y nominal
    "em_hy_spread": "BAMLH0A0HYM2EY",   # EM HY OAS spread (% yield)
}

_ETF_YFINANCE = {
    **get_yfinance_ticker_map(),
    "GBPUSD": "GBPUSD=X",   # FX rate for USD→GBP conversion
    "ACWI": "ACWI",         # for global drawdown + PE reference
}

_SHILLER_URL = (
    "https://shillerdata.com/data/ie_data.xls"
)


def _load_repo_data_module():
    """Load the existing app/data.py module directly from disk.

    The allocator should prefer the repo's primary DuckDB/parquet pipeline for
    prices and performance metrics. This loader avoids package-layout issues by
    importing the module from its file path.
    """
    global _REPO_DATA_MODULE
    if _REPO_DATA_MODULE is not None:
        return _REPO_DATA_MODULE

    data_py = os.path.join(_APP_DIR, "data.py")
    if not os.path.exists(data_py):
        return None

    if _APP_DIR not in sys.path:
        sys.path.insert(0, _APP_DIR)

    spec = importlib.util.spec_from_file_location("fin_tracker_app_data", data_py)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _REPO_DATA_MODULE = module
    return module


def _refresh_etf_prices_from_repo(conn: duckdb.DuckDBPyConnection) -> bool:
    """Populate allocator cache from the repo's main price pipeline."""
    app_data = _load_repo_data_module()
    if app_data is None:
        return False

    tickers = tuple(_ETF_YFINANCE.keys())
    try:
        px_df = app_data.load_prices(tickers=tickers)
        perf_df = app_data.load_latest_perf(tickers=tickers, max_rown=1)
    except Exception as e:
        log.warning("Repo price pipeline unavailable; falling back to local fetch: %s", e)
        return False

    if px_df.empty:
        return False

    px_df = px_df.copy()
    px_df = px_df.rename(columns={"price": "close"})
    px_df["date"] = pd.to_datetime(px_df["date"])
    perf_idx = (
        perf_df.sort_values("date").groupby("ticker").tail(1).set_index("ticker")
        if not perf_df.empty
        else pd.DataFrame()
    )

    now = datetime.now()
    inserted = 0
    loaded_tickers: set[str] = set()
    for ticker, grp in px_df.groupby("ticker"):
        grp = grp.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        series = grp["close"].dropna()
        if series.empty:
            continue

        rows = [
            (ticker, str(row.date.date()), float(row.close))
            for row in grp[["date", "close"]].itertuples(index=False)
        ]
        conn.execute("DELETE FROM etf_prices WHERE ticker = ?", [ticker])
        conn.executemany("INSERT INTO etf_prices VALUES (?, ?, ?)", rows)

        trailing_252 = series.tail(252) if len(series) >= 252 else series
        perf_row = perf_idx.loc[ticker] if not perf_idx.empty and ticker in perf_idx.index else None
        ma200 = None
        if perf_row is not None:
            ma200 = perf_row.get("ma_252")
        if pd.isna(ma200) or ma200 is None or float(ma200) <= 0:
            ma200 = float(series.tail(200).mean()) if len(series) >= 200 else float(series.mean())

        last_price = float(perf_row.get("price")) if perf_row is not None and pd.notna(perf_row.get("price")) else float(series.iloc[-1])
        last_date = str(grp["date"].iloc[-1].date())
        low_52w = float(trailing_252.min())
        high_52w = float(trailing_252.max())

        conn.execute(
            """
            INSERT INTO etf_meta
                (ticker, ma200, low_52w, high_52w, last_price, last_date, refreshed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                ma200=excluded.ma200, low_52w=excluded.low_52w,
                high_52w=excluded.high_52w, last_price=excluded.last_price,
                last_date=excluded.last_date, refreshed=excluded.refreshed
            """,
            [ticker, float(ma200), low_52w, high_52w, last_price, last_date, now],
        )
        inserted += 1
        loaded_tickers.add(ticker)

    conn.commit()
    missing_tickers = [ticker for ticker in _ETF_YFINANCE if ticker not in loaded_tickers]
    if missing_tickers:
        log.info(
            "Repo price pipeline loaded %d tickers; %d missing will fall back to yfinance",
            inserted, len(missing_tickers)
        )
    else:
        log.info("Loaded %d tickers into allocator cache from repo price pipeline", inserted)
    return inserted > 0


def _get_cache() -> duckdb.DuckDBPyConnection:
    """Open (and initialise) the persistent DuckDB cache."""
    conn = duckdb.connect(_CACHE_DB)
    for stmt in [
        """CREATE TABLE IF NOT EXISTS etf_prices (
            ticker   TEXT,
            date     DATE,
            close    DOUBLE,
            PRIMARY KEY (ticker, date)
        )""",
        """CREATE TABLE IF NOT EXISTS etf_meta (
            ticker     TEXT PRIMARY KEY,
            ma200      DOUBLE,
            low_52w    DOUBLE,
            high_52w   DOUBLE,
            last_price DOUBLE,
            last_date  DATE,
            refreshed  TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS macro_latest (
            series    TEXT PRIMARY KEY,
            value     DOUBLE,
            as_of     DATE,
            refreshed TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS shiller_cape (
            date           DATE PRIMARY KEY,
            cape           DOUBLE,
            earnings_yield DOUBLE
        )""",
        """CREATE TABLE IF NOT EXISTS etf_pe (
            ticker      TEXT PRIMARY KEY,
            pe_ratio    DOUBLE,
            pb_ratio    DOUBLE,
            source      TEXT,
            as_of       DATE,
            refreshed   TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS factor_data (
            ticker               TEXT,
            date                 DATE,
            trailing_pe          DOUBLE,
            peg_ratio            DOUBLE,
            earnings_growth_5y   DOUBLE,
            five_year_avg_return DOUBLE,
            price_to_book        DOUBLE,
            source               TEXT,
            refreshed            TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )""",
        """CREATE TABLE IF NOT EXISTS ibkr_fundamentals (
            ticker               TEXT PRIMARY KEY,
            trailing_pe          DOUBLE,
            peg_ratio            DOUBLE,
            earnings_growth_5y   DOUBLE,
            five_year_avg_return DOUBLE,
            price_to_book        DOUBLE,
            source_file          TEXT,
            as_of                DATE,
            refreshed            TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS etf_constituents (
            etf_ticker     TEXT,
            holding_symbol TEXT,
            holding_name   TEXT,
            mapped_ticker  TEXT,
            mapped_sleeve  TEXT,
            weight_pct     DOUBLE,
            source         TEXT,
            as_of          DATE,
            refreshed      TIMESTAMP,
            PRIMARY KEY (etf_ticker, holding_symbol, as_of)
        )""",
    ]:
        conn.execute(stmt)

    # Lightweight schema migrations for existing local caches.
    def _ensure_column(table: str, column: str, definition: str) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    _ensure_column("etf_meta", "low_52w", "DOUBLE")
    _ensure_column("etf_meta", "high_52w", "DOUBLE")
    return conn



def _cache_is_fresh(conn: duckdb.DuckDBPyConnection, table: str, max_age_h: int = 12) -> bool:
    """True if the table has a row refreshed within the last max_age_h hours."""
    try:
        row = conn.execute(
            f"SELECT MAX(refreshed) FROM {table}"
        ).fetchone()
        if not row or row[0] is None:
            return False
        last = pd.to_datetime(row[0])
        return (datetime.now() - last.to_pydatetime()) < timedelta(hours=max_age_h)
    except Exception:
        return False


# ── ETF prices + MA200 ────────────────────────────────────────────────
def refresh_etf_prices(force: bool = False) -> None:
    """Download 400 days of prices for all universe tickers, compute MA200."""
    conn = _get_cache()
    if not force and _cache_is_fresh(conn, "etf_meta", max_age_h=8):
        conn.close()
        return

    repo_loaded = _refresh_etf_prices_from_repo(conn)

    try:
        import yfinance as yf
    except ImportError:
        if repo_loaded:
            conn.close()
            return
        log.warning("yfinance not installed; skipping ETF price refresh")
        conn.close()
        return

    existing = {
        row[0]
        for row in conn.execute("SELECT ticker FROM etf_meta").fetchall()
    }
    missing_map = {k: v for k, v in _ETF_YFINANCE.items() if k not in existing}
    if repo_loaded and not missing_map:
        conn.close()
        return

    tickers_yf = list(missing_map.values()) if missing_map else list(_ETF_YFINANCE.values())
    log.info("Downloading ETF prices for %d tickers", len(tickers_yf))
    raw = yf.download(
        tickers_yf, period="400d", interval="1d",
        auto_adjust=True, progress=False, threads=True
    )

    if raw.empty:
        log.warning("yfinance returned empty dataframe")
        conn.close()
        return

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    now = datetime.now()
    active_map = missing_map if missing_map else _ETF_YFINANCE
    for our_ticker, yf_ticker in active_map.items():
        if yf_ticker not in close.columns:
            continue
        series = close[yf_ticker].dropna()
        if series.empty:
            continue

        rows = [(our_ticker, str(d.date()), float(p)) for d, p in series.items()]
        conn.execute("DELETE FROM etf_prices WHERE ticker = ?", [our_ticker])
        conn.executemany("INSERT INTO etf_prices VALUES (?, ?, ?)", rows)

        ma200 = float(series.tail(200).mean()) if len(series) >= 200 else float(series.mean())
        trailing_252 = series.tail(252) if len(series) >= 252 else series
        low_52w = float(trailing_252.min())
        high_52w = float(trailing_252.max())
        last_price = float(series.iloc[-1])
        last_date = str(series.index[-1].date())
        conn.execute("""
            INSERT INTO etf_meta
                (ticker, ma200, low_52w, high_52w, last_price, last_date, refreshed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                ma200=excluded.ma200, low_52w=excluded.low_52w,
                high_52w=excluded.high_52w, last_price=excluded.last_price,
                last_date=excluded.last_date, refreshed=excluded.refreshed
        """, [our_ticker, ma200, low_52w, high_52w, last_price, last_date, now])

    conn.commit()
    conn.close()
    log.info("ETF price refresh complete")


def get_etf_meta() -> pd.DataFrame:
    """Return DataFrame of (ticker, ma200, last_price, last_date) from cache."""
    conn = _get_cache()
    df = conn.execute(
        "SELECT ticker, ma200, low_52w, high_52w, last_price, last_date FROM etf_meta"
    ).df()
    conn.close()
    if df.empty:
        refresh_etf_prices(force=False)
        conn = _get_cache()
        df = conn.execute(
            "SELECT ticker, ma200, low_52w, high_52w, last_price, last_date FROM etf_meta"
        ).df()
        conn.close()
    return df


def get_price_history(tickers: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Return cached price history for all or selected tickers."""
    conn = _get_cache()
    where = ""
    params: list[str] = []
    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        where = f"WHERE ticker IN ({placeholders})"
        params = list(tickers)
    df = conn.execute(
        f"SELECT ticker, date, close FROM etf_prices {where} ORDER BY ticker, date",
        params,
    ).df()
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


_TIMING_EMPTY_COLUMNS = [
    "ticker", "r_1w", "r_1m", "r_3m", "r_6m", "r_1y", "r_3y", "r_5y",
    "vol_1y", "vol_3m", "z_1mo", "z_2w", "z_1w", "z_1d",
    "drawdown_52w", "drawdown_3y", "ma_252", "ma_126", "ma_63", "ma_21",
    "price", "low_52w", "high_52w", "pct_above_ma200",
    "range_52w_pos", "ma200_slope_20d",
]


def _timing_from_repo_perf(tickers: tuple[str, ...] | None) -> pd.DataFrame:
    """Build entry-timing frame from the repo's latest_performance export."""
    app_data = _load_repo_data_module()
    if app_data is None:
        return pd.DataFrame()
    try:
        perf = app_data.load_latest_perf(tickers=tickers, max_rown=1)
    except Exception as e:
        log.debug("load_latest_perf unavailable: %s", e)
        return pd.DataFrame()
    if perf.empty:
        return pd.DataFrame()

    out = perf.rename(
        columns={
            "r_1mo": "r_1m",
            "r_3mo": "r_3m",
            "r_6mo": "r_6m",
            "vol_1mo": "vol_3m",  # pipeline vol_1mo is a 21d rolling vol — our closest proxy to vol_3m
        }
    ).copy()

    out["pct_above_ma200"] = (out["price"] / out["ma_252"] - 1.0) * 100.0

    out["low_52w"] = np.nan
    out["high_52w"] = np.nan
    out["range_52w_pos"] = np.nan
    out["ma200_slope_20d"] = np.nan
    keep = [
        "ticker", "r_1w", "r_1m", "r_3m", "r_6m", "r_1y", "r_3y", "r_5y",
        "vol_1y", "vol_3m", "z_1mo", "z_2w", "z_1w", "z_1d",
        "drawdown_52w", "drawdown_3y", "ma_252", "ma_126", "ma_63", "ma_21",
        "price", "low_52w", "high_52w", "pct_above_ma200",
        "range_52w_pos", "ma200_slope_20d",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].copy()


def _enrich_with_price_tails(df: pd.DataFrame, tickers: tuple[str, ...] | None) -> pd.DataFrame:
    """Backfill low_52w/high_52w/range_52w_pos/ma200_slope_20d from local price history."""
    if df.empty:
        return df

    target_tickers = tuple(df["ticker"].astype(str).unique()) if tickers is None else tickers
    prices = get_price_history(target_tickers)
    if prices.empty:
        return df

    prices["date"] = pd.to_datetime(prices["date"])
    enrich_rows: dict[str, dict] = {}
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if grp.empty:
            continue
        trailing_252 = grp.tail(252) if len(grp) >= 252 else grp
        high_52w = float(trailing_252["close"].max())
        low_52w = float(trailing_252["close"].min())
        px_now = float(grp.iloc[-1]["close"])
        range_52w_pos = float((px_now - low_52w) / (high_52w - low_52w)) if high_52w > low_52w else None

        ma_series = grp["close"].rolling(window=200, min_periods=100).mean()
        ma_latest = float(ma_series.iloc[-1]) if pd.notna(ma_series.iloc[-1]) else None
        ma_20d_ago = float(ma_series.iloc[-21]) if len(ma_series) > 21 and pd.notna(ma_series.iloc[-21]) else None
        ma200_slope_20d = (
            ((ma_latest / ma_20d_ago) - 1.0) * 100.0
            if (ma_latest and ma_20d_ago and ma_20d_ago > 0)
            else None
        )
        enrich_rows[ticker] = {
            "low_52w": low_52w,
            "high_52w": high_52w,
            "range_52w_pos": range_52w_pos,
            "ma200_slope_20d": ma200_slope_20d,
        }

    if not enrich_rows:
        return df

    enrich_df = pd.DataFrame.from_dict(enrich_rows, orient="index").reset_index().rename(columns={"index": "ticker"})
    df = df.drop(columns=["low_52w", "high_52w", "range_52w_pos", "ma200_slope_20d"], errors="ignore")
    return df.merge(enrich_df, on="ticker", how="left")


def get_entry_timing_metrics(tickers: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Entry-timing metrics for the universe, sourced from the repo's cleaned
    ``latest_performance`` export when available, else from the allocator's
    local price cache.

    Repo-derived columns: r_1w, r_1m, r_3m, r_6m, r_1y, r_3y, r_5y,
      vol_1y, vol_3m (≈vol_1mo), z_1d/z_1w/z_2w/z_1mo,
      drawdown_52w, drawdown_3y, ma_252/126/63/21, price.
    Locally enriched: low_52w, high_52w, range_52w_pos, ma200_slope_20d,
      and pct_above_ma200 (price/ma_252 - 1).
    """
    tickers_tuple = tuple(tickers) if tickers else None
    repo_df = _timing_from_repo_perf(tickers_tuple)
    if not repo_df.empty:
        return _enrich_with_price_tails(repo_df, tickers_tuple)

    # Fallback: derive everything locally from etf_prices
    conn = _get_cache()
    where = ""
    params: list[str] = []
    if tickers:
        placeholders = ",".join(["?"] * len(tickers))
        where = f"WHERE ticker IN ({placeholders})"
        params = list(tickers)

    prices = conn.execute(
        f"SELECT ticker, date, close FROM etf_prices {where} ORDER BY ticker, date",
        params,
    ).df()
    if prices.empty:
        return pd.DataFrame(columns=_TIMING_EMPTY_COLUMNS)

    prices["date"] = pd.to_datetime(prices["date"])
    rows: list[dict] = []
    for ticker, grp in prices.groupby("ticker"):
        grp = grp.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if grp.empty:
            continue
        latest = grp.iloc[-1]

        def _price_near(days: int) -> float | None:
            cutoff = pd.Timestamp(latest["date"]) - pd.Timedelta(days=days)
            hist_local = grp.copy()
            hist_local["dist_days"] = (hist_local["date"] - cutoff).abs().dt.days
            row = hist_local.sort_values(["dist_days", "date"]).iloc[0]
            return float(row["close"]) if pd.notna(row["close"]) else None

        cutoff = pd.Timestamp(latest["date"]) - pd.Timedelta(days=365)
        hist = grp.copy()
        hist["dist_days"] = (hist["date"] - cutoff).abs().dt.days
        px_1y_row = hist.sort_values(["dist_days", "date"]).iloc[0]
        px_now = float(latest["close"])
        px_1w = _price_near(7)
        px_1m = _price_near(30)
        px_3m = _price_near(91)
        px_6m = _price_near(182)
        px_1y = float(px_1y_row["close"]) if pd.notna(px_1y_row["close"]) else None
        trailing_252 = grp.tail(252) if len(grp) >= 252 else grp
        trailing_63 = grp.tail(63) if len(grp) >= 63 else grp
        high_52w = float(trailing_252["close"].max()) if not trailing_252.empty else None
        low_52w = float(trailing_252["close"].min()) if not trailing_252.empty else None
        ma_252 = float(trailing_252["close"].mean()) if not trailing_252.empty else None
        ma_126 = float(grp.tail(126)["close"].mean()) if len(grp) >= 63 else None
        ma_63 = float(grp.tail(63)["close"].mean()) if len(grp) >= 30 else None
        ma_21 = float(grp.tail(21)["close"].mean()) if len(grp) >= 10 else None
        daily_returns = trailing_252["close"].pct_change().dropna()
        vol_1y = float(daily_returns.std() * np.sqrt(252) * 100.0) if len(daily_returns) >= 20 else None
        daily_returns_3m = trailing_63["close"].pct_change().dropna()
        vol_3m = float(daily_returns_3m.std() * np.sqrt(252) * 100.0) if len(daily_returns_3m) >= 20 else None

        ma_series = grp["close"].rolling(window=200, min_periods=100).mean()
        ma_latest = float(ma_series.iloc[-1]) if pd.notna(ma_series.iloc[-1]) else None
        ma_20d_ago = float(ma_series.iloc[-21]) if len(ma_series) > 21 and pd.notna(ma_series.iloc[-21]) else None
        pct_above_ma200 = ((px_now / ma_latest) - 1.0) * 100.0 if ma_latest and ma_latest > 0 else None
        ma200_slope_20d = ((ma_latest / ma_20d_ago) - 1.0) * 100.0 if (ma_latest and ma_20d_ago and ma_20d_ago > 0) else None

        range_52w_pos = None
        if high_52w is not None and low_52w is not None and high_52w > low_52w:
            range_52w_pos = float((px_now - low_52w) / (high_52w - low_52w))

        r_1w = ((px_now / px_1w) - 1.0) * 100.0 if px_1w and px_1w > 0 else None
        r_1m = ((px_now / px_1m) - 1.0) * 100.0 if px_1m and px_1m > 0 else None
        r_3m = ((px_now / px_3m) - 1.0) * 100.0 if px_3m and px_3m > 0 else None
        r_6m = ((px_now / px_6m) - 1.0) * 100.0 if px_6m and px_6m > 0 else None
        r_1y = ((px_now / px_1y) - 1.0) * 100.0 if px_1y and px_1y > 0 else None
        drawdown_52w = ((px_now / high_52w) - 1.0) * 100.0 if high_52w and high_52w > 0 else None
        rows.append(
            {
                "ticker": ticker,
                "r_1w": r_1w,
                "r_1m": r_1m,
                "r_3m": r_3m,
                "r_6m": r_6m,
                "r_1y": r_1y,
                "r_3y": None,
                "r_5y": None,
                "vol_1y": vol_1y,
                "vol_3m": vol_3m,
                "z_1mo": None,
                "z_2w": None,
                "z_1w": None,
                "z_1d": None,
                "drawdown_52w": drawdown_52w,
                "drawdown_3y": None,
                "ma_252": ma_252,
                "ma_126": ma_126,
                "ma_63": ma_63,
                "ma_21": ma_21,
                "price": px_now,
                "low_52w": low_52w,
                "high_52w": high_52w,
                "pct_above_ma200": pct_above_ma200,
                "range_52w_pos": range_52w_pos,
                "ma200_slope_20d": ma200_slope_20d,
            }
        )
    return pd.DataFrame(rows)


def get_gbpusd() -> float:
    """Last available GBPUSD rate from cache, or 1.27 as a fallback."""
    conn = _get_cache()
    row = conn.execute(
        "SELECT last_price FROM etf_meta WHERE ticker = 'GBPUSD'"
    ).fetchone()
    conn.close()
    if row and row[0]:
        return float(row[0])
    refresh_etf_prices(force=False)
    conn = _get_cache()
    row = conn.execute(
        "SELECT last_price FROM etf_meta WHERE ticker = 'GBPUSD'"
    ).fetchone()
    conn.close()
    return float(row[0]) if row and row[0] else 1.27


# ── Macro data (FRED) ─────────────────────────────────────────────────
def refresh_macro(force: bool = False) -> None:
    """Fetch latest values for all FRED series."""
    conn = _get_cache()
    if not force and _cache_is_fresh(conn, "macro_latest", max_age_h=12):
        conn.close()
        return

    now = datetime.now()
    for name, series_id in _FRED_SERIES.items():
        try:
            url = f"{_FRED_BASE}?id={series_id}"
            df = pd.read_csv(url, parse_dates=["DATE"], na_values=".")
            df = df.dropna()
            if df.empty:
                continue
            last = df.iloc[-1]
            conn.execute("""
                INSERT INTO macro_latest VALUES (?, ?, ?, ?)
                ON CONFLICT(series) DO UPDATE SET
                    value=excluded.value, as_of=excluded.as_of, refreshed=excluded.refreshed
            """, [name, float(last.iloc[1]) / 100.0, str(last["DATE"].date()), now])
        except Exception as e:
            log.warning("FRED fetch failed for %s: %s", series_id, e)

    conn.commit()
    conn.close()


def get_macro() -> dict[str, float | None]:
    """Return latest macro values from cache. None if not yet fetched."""
    conn = _get_cache()
    rows = conn.execute("SELECT series, value FROM macro_latest").fetchall()
    conn.close()
    out = {name: None for name in _FRED_SERIES}
    out.update({r[0]: r[1] for r in rows})
    # Derive UK real yield ≈ UK nominal 10y - 3% long-run inflation expectation
    uk_nom = out.get("uk_nominal_10y")
    out["uk_real_10y"] = round(uk_nom - 0.03, 4) if uk_nom is not None else None
    # EM HY spread: FRED series is total yield, approximate spread = yield - US 10y
    em_yield = out.get("em_hy_spread")
    us_10y = out.get("us_10y_nominal")
    if em_yield is not None and us_10y is not None:
        out["em_hy_spread_over_ust"] = max(0.0, em_yield - us_10y)
    else:
        out["em_hy_spread_over_ust"] = None
    return out


# ── Shiller CAPE ──────────────────────────────────────────────────────
def refresh_shiller(force: bool = False) -> None:
    """Download Shiller US CAPE data and cache in DuckDB."""
    conn = _get_cache()
    if not force and _cache_is_fresh(conn, "shiller_cape", max_age_h=24 * 30):
        conn.close()
        return
    try:
        df = pd.read_excel(
            _SHILLER_URL,
            sheet_name="Data",
            skiprows=7,
            usecols=[0, 1, 4],
            header=0,
        )
        df.columns = ["date_raw", "price", "cape"]
        df = df.dropna(subset=["cape"])
        # Date column is like "1881.01" — convert to first day of month
        df["date"] = pd.to_datetime(
            df["date_raw"].astype(str).str.replace(r"\.1$", ".10", regex=True),
            format="%Y.%m", errors="coerce"
        )
        df = df.dropna(subset=["date"])
        df["earnings_yield"] = 1.0 / df["cape"]
        rows = [
            (str(r["date"].date()), float(r["cape"]), float(r["earnings_yield"]))
            for _, r in df.iterrows()
        ]
        conn.executemany("""
            INSERT INTO shiller_cape VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET cape=excluded.cape, earnings_yield=excluded.earnings_yield
        """, rows)
        conn.commit()
        log.info("Shiller CAPE refreshed — %d rows", len(rows))
    except Exception as e:
        log.warning("Shiller CAPE fetch failed: %s", e)
    finally:
        conn.close()


def get_latest_cape() -> float | None:
    """Latest Shiller US CAPE from cache."""
    conn = _get_cache()
    row = conn.execute(
        "SELECT cape FROM shiller_cape ORDER BY date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return float(row[0]) if row else None


# ── Analytics: compute current weights and drift (DuckDB) ─────────────
def compute_portfolio_analytics(
    holdings_db_path: str, bucket_sizes_gbp: dict[str, float]
) -> pd.DataFrame:
    """Join holdings with current prices and compute current weight vs target.

    Uses DuckDB to ATTACH the SQLite holdings DB alongside the cache.
    Returns a DataFrame per row: (account_type, ticker, gbp_value, current_pct,
    target_pct, drift_pp, drift_rel).

    bucket_sizes_gbp: {'SIPP': 300000, 'ISA': 150000, 'GIA': 100000}
    """
    conn = duckdb.connect(_CACHE_DB)
    try:
        # Attach the holdings SQLite DB read-only
        conn.execute(f"ATTACH '{holdings_db_path}' AS hdb (TYPE sqlite, READ_ONLY)")
    except Exception:
        pass  # already attached

    try:
        # Pull holdings + last prices in one query
        df = conn.execute("""
            SELECT
                h.account_type,
                h.ticker,
                h.qty,
                h.ccy,
                COALESCE(m.last_price, 0)  AS last_price,
                m.ma200
            FROM hdb.holdings h
            LEFT JOIN etf_meta m ON m.ticker = h.ticker
        """).df()
    except Exception as e:
        log.warning("Holdings analytics query failed: %s", e)
        conn.close()
        return pd.DataFrame()

    gbpusd = get_gbpusd()

    def to_gbp(row):
        price = row["last_price"]
        # LSE tickers are often quoted in pence — heuristic: price > 200 and ccy GBP → pence
        if row["ccy"] == "GBP" and price > 200:
            price = price / 100.0
        if row["ccy"] == "USD":
            price = price / gbpusd
        return row["qty"] * price

    if not df.empty:
        df["gbp_value"] = df.apply(to_gbp, axis=1)

        rows = []
        for account_type, grp in df.groupby("account_type"):
            total_gbp = bucket_sizes_gbp.get(account_type, grp["gbp_value"].sum())
            for _, r in grp.iterrows():
                rows.append({
                    "account_type": account_type,
                    "ticker": r["ticker"],
                    "gbp_value": r["gbp_value"],
                    "current_pct": 100.0 * r["gbp_value"] / total_gbp if total_gbp > 0 else 0,
                    "last_price": r["last_price"],
                    "ma200": r["ma200"],
                })
        conn.close()
        return pd.DataFrame(rows)
    conn.close()
    return pd.DataFrame(columns=["account_type", "ticker", "gbp_value", "current_pct", "last_price", "ma200"])


# ── Regional P/E via iShares + yfinance ──────────────────────────────
# Regional proxies: each ETF's published P/E = weighted avg P/E of holdings,
# so it directly represents the regional market valuation.
_REGION_PE_TICKERS: dict[str, str] = {
    "US":     "CSPX",   # iShares Core S&P 500 → US P/E
    "World":  "IWDA",   # iShares Core MSCI World → DM aggregate
    "EM":     "EIMI",   # iShares Core MSCI EM IMI → EM P/E
    "Japan":  "IJPA",   # iShares MSCI Japan
    "Europe": "IMEA",   # iShares Core MSCI Europe (broader than VEUR for P/E)
    "UK":     "ISF",    # iShares Core FTSE 100 (energy/mining-heavy = commodity proxy)
}

_ISHARES_BASE = "https://www.ishares.com/uk/individual/en/products"
_ISHARES_SUFFIX = "/1478372549651.ajax?tab=keyFacts&fileType=json"
_ISHARES_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*",
    "Referer": "https://www.ishares.com/",
}


def _fetch_ishares_pe(ticker: str, product_id: str, slug: str) -> float | None:
    """Fetch P/E ratio from iShares UK product page JSON.

    Returns None on any failure so the caller can fall back to yfinance.
    The response contains a 'fundCharacteristics' block with 'priceToEarningsRatio'.
    """
    try:
        import requests
        url = f"{_ISHARES_BASE}/{product_id}/{slug}{_ISHARES_SUFFIX}"
        resp = requests.get(url, headers=_ISHARES_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # iShares nests characteristics under different keys across fund types;
        # walk the likely paths in order of preference.
        for path in [
            ("fundCharacteristics", "priceToEarningsRatio", "r"),
            ("fundCharacteristics", "priceToEarningsRatio"),
            ("portfolioCharacteristics", "priceToEarnings", "r"),
            ("portfolioCharacteristics", "priceToEarnings"),
        ]:
            node = data
            for key in path:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    node = None
                    break
            if isinstance(node, (int, float)) and node > 0:
                return float(node)
            if isinstance(node, str):
                try:
                    v = float(node.replace(",", ""))
                    if v > 0:
                        return v
                except ValueError:
                    pass
    except Exception as e:
        log.debug("iShares P/E fetch failed for %s: %s", ticker, e)
    return None


def _fetch_yfinance_pe(yf_ticker: str) -> float | None:
    """Fallback P/E from yfinance .info dict."""
    try:
        import yfinance as yf
        info = yf.Ticker(yf_ticker).info
        for field in ("forwardPE", "trailingPE"):
            v = info.get(field)
            if v and isinstance(v, (int, float)) and 1 < v < 200:
                return float(v)
    except Exception as e:
        log.debug("yfinance P/E fetch failed for %s: %s", yf_ticker, e)
    return None


def refresh_region_pe(force: bool = False) -> None:
    """Fetch P/E for each regional proxy ETF.

    Strategy:
      1. iShares UK product JSON (primary — most accurate for UCITS ETFs)
      2. yfinance .info (fallback)
    Results cached in etf_pe table; stale after 24 h.
    """
    from instruments import ISHARES_PRODUCTS
    conn = _get_cache()
    if not force and _cache_is_fresh(conn, "etf_pe", max_age_h=24):
        conn.close()
        return

    now = datetime.now()
    today = str(now.date())

    for region, our_ticker in _REGION_PE_TICKERS.items():
        yf_ticker = _ETF_YFINANCE.get(our_ticker, our_ticker + ".L")
        pe: float | None = None
        pb: float | None = None
        source = "none"

        # 1. iShares JSON
        if our_ticker in ISHARES_PRODUCTS:
            pid, slug = ISHARES_PRODUCTS[our_ticker]
            pe = _fetch_ishares_pe(our_ticker, pid, slug)
            if pe:
                source = "ishares"

        # 2. yfinance fallback
        if pe is None:
            pe = _fetch_yfinance_pe(yf_ticker)
            if pe:
                source = "yfinance"

        if pe is None:
            log.warning("No P/E found for %s (%s)", region, our_ticker)
            continue

        conn.execute("""
            INSERT INTO etf_pe VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                pe_ratio=excluded.pe_ratio, pb_ratio=excluded.pb_ratio,
                source=excluded.source, as_of=excluded.as_of, refreshed=excluded.refreshed
        """, [our_ticker, pe, pb, source, today, now])
        log.info("P/E %s (%s): %.1f  [%s]", region, our_ticker, pe, source)

    conn.commit()
    conn.close()


def get_region_pe() -> dict[str, dict]:
    """Return {region: {ticker, pe, source}} from cache, or empty dict if not yet fetched."""
    from instruments import ISHARES_PRODUCTS  # noqa: F401
    conn = _get_cache()
    rows = conn.execute(
        "SELECT ticker, pe_ratio, source, as_of FROM etf_pe"
    ).fetchall()
    conn.close()

    # Build reverse map: ticker → region
    ticker_to_region = {v: k for k, v in _REGION_PE_TICKERS.items()}
    result = {}
    for ticker, pe, source, as_of in rows:
        region = ticker_to_region.get(ticker, ticker)
        result[region] = {"ticker": ticker, "pe": pe, "source": source, "as_of": as_of}
    return result


# ── ACWI drawdown ─────────────────────────────────────────────────────
def get_acwi_drawdown_30d() -> float:
    """30-day drawdown of ACWI from cache."""
    conn = _get_cache()
    df = conn.execute(
        "SELECT close FROM etf_prices WHERE ticker = 'ACWI' ORDER BY date DESC LIMIT 30"
    ).df()
    conn.close()
    if df.empty or len(df) < 2:
        return 0.0
    prices = df["close"].values
    peak = float(np.max(prices))
    latest = float(prices[0])
    return round((latest - peak) / peak, 4) if peak > 0 else 0.0


# ── Factor data: PE history + PEG ─────────────────────────────────────
# Data sources, in priority order:
#
#   1. yfinance .info — free, covers US ETFs well.
#      Returns: trailingPE, pegRatio, priceToBook, fiveYearAverageReturn.
#      pegRatio is often missing for ETFs (it's set for individual stocks).
#      fiveYearAverageReturn is used as a growth proxy when pegRatio absent.
#
#   2. Derived PEG = trailingPE / (fiveYearAverageReturn * 100)
#      This is an approximation — price return ≠ earnings growth, but for
#      diversified ETFs it's a serviceable proxy over 5-year windows.
#
#   3. Quarterly timeseries accumulation: each refresh appends one dated row
#      per ticker. After ~8 refreshes (2 years) the PE z-score becomes reliable.
#      Before that, pe_history_signal will show "INSUFFICIENT_HISTORY".
#
#   4. Bootstrap for US broad market: Shiller CAPE (already fetched) provides
#      a deep historical anchor, accessible via get_shiller_pe_zscore().

# Tickers to include in factor screening (all meaningful; skips pure FX + ACWI)
_FACTOR_TICKERS: set[str] = set(_ETF_YFINANCE.keys()) - {"GBPUSD", "ACWI"}
# Suffix map for yfinance (same as _ETF_YFINANCE)
_FACTOR_YF_MAP: dict[str, str] = {k: v for k, v in _ETF_YFINANCE.items()
                                   if k not in ("GBPUSD", "ACWI")}

_IBKR_IMPORT_PATHS = [
    os.path.join(os.path.dirname(__file__), "ibkr_fundamentals.csv"),
    os.path.join(os.path.dirname(__file__), "ibkr_fundamentals.parquet"),
]


def refresh_ibkr_fundamentals(force: bool = False) -> int:
    """Load locally exported IBKR fundamentals if present.

    Supported columns are flexible; we normalise a small required subset.
    This is intentionally local-file only so the allocator can prefer higher
    quality data without requiring live TWS connectivity inside Streamlit.
    """
    conn = _get_cache()
    if not force and _cache_is_fresh(conn, "ibkr_fundamentals", max_age_h=12):
        conn.close()
        return 0

    src = next((path for path in _IBKR_IMPORT_PATHS if os.path.exists(path)), None)
    if src is None:
        conn.close()
        return 0

    if src.endswith(".parquet"):
        df = pd.read_parquet(src)
    else:
        df = pd.read_csv(src)

    if df.empty:
        conn.close()
        return 0

    rename_map = {
        "symbol": "ticker",
        "Symbol": "ticker",
        "Ticker": "ticker",
        "pe": "trailing_pe",
        "PE": "trailing_pe",
        "trailingPE": "trailing_pe",
        "peg": "peg_ratio",
        "PEG": "peg_ratio",
        "pegRatio": "peg_ratio",
        "earningsGrowth": "earnings_growth_5y",
        "eps_growth_5y": "earnings_growth_5y",
        "fiveYearAverageReturn": "five_year_avg_return",
        "priceToBook": "price_to_book",
        "pb": "price_to_book",
    }
    df = df.rename(columns=rename_map)
    required_cols = ["ticker", "trailing_pe"]
    if not set(required_cols).issubset(df.columns):
        conn.close()
        return 0

    for col in [
        "trailing_pe",
        "peg_ratio",
        "earnings_growth_5y",
        "five_year_avg_return",
        "price_to_book",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[[
        "ticker",
        "trailing_pe",
        "peg_ratio",
        "earnings_growth_5y",
        "five_year_avg_return",
        "price_to_book",
    ]].copy()
    for col in ["trailing_pe", "peg_ratio", "earnings_growth_5y", "five_year_avg_return", "price_to_book"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # IBKR/Reuters occasionally uses large negative sentinels (for example -999.9999)
    # to mean "not available". Drop those before the factor model consumes them.
    df.loc[df["earnings_growth_5y"] <= -10, "earnings_growth_5y"] = np.nan
    df.loc[df["peg_ratio"] <= 0, "peg_ratio"] = np.nan
    df.loc[df["trailing_pe"] <= 0, "trailing_pe"] = np.nan
    df.loc[df["price_to_book"] <= 0, "price_to_book"] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["ticker", "trailing_pe"]).drop_duplicates(subset=["ticker"])

    today = str(datetime.now().date())
    now = datetime.now()
    rows = [
        (
            row["ticker"],
            None if pd.isna(row["trailing_pe"]) else float(row["trailing_pe"]),
            None if pd.isna(row["peg_ratio"]) else float(row["peg_ratio"]),
            None if pd.isna(row["earnings_growth_5y"]) else float(row["earnings_growth_5y"]),
            None if pd.isna(row["five_year_avg_return"]) else float(row["five_year_avg_return"]),
            None if pd.isna(row["price_to_book"]) else float(row["price_to_book"]),
            os.path.basename(src),
            today,
            now,
        )
        for _, row in df.iterrows()
    ]
    conn.executemany(
        """
        INSERT INTO ibkr_fundamentals
            (ticker, trailing_pe, peg_ratio, earnings_growth_5y,
             five_year_avg_return, price_to_book, source_file, as_of, refreshed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (ticker) DO UPDATE SET
            trailing_pe=excluded.trailing_pe,
            peg_ratio=excluded.peg_ratio,
            earnings_growth_5y=excluded.earnings_growth_5y,
            five_year_avg_return=excluded.five_year_avg_return,
            price_to_book=excluded.price_to_book,
            source_file=excluded.source_file,
            as_of=excluded.as_of,
            refreshed=excluded.refreshed
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def refresh_factor_data(force: bool = False) -> None:
    """Fetch PE, PEG, and growth data for all universe tickers via yfinance.

    Appends one row per ticker per date to factor_data. Skips if a row for
    today already exists (TTL = 1 calendar day).
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed; skipping factor data refresh")
        return

    conn = _get_cache()
    today = str(datetime.now().date())
    now = datetime.now()

    refresh_ibkr_fundamentals(force=force)
    ibkr_df = conn.execute(
        "SELECT ticker, trailing_pe, peg_ratio, earnings_growth_5y, "
        "five_year_avg_return, price_to_book FROM ibkr_fundamentals"
    ).df()
    ibkr_map = (
        ibkr_df.set_index("ticker").to_dict(orient="index")
        if not ibkr_df.empty
        else {}
    )

    # Check if today's data is already stored
    if not force:
        row = conn.execute(
            "SELECT COUNT(*) FROM factor_data WHERE date = ?", [today]
        ).fetchone()
        if row and row[0] > 0:
            conn.close()
            log.debug("Factor data already fresh for %s — skipping", today)
            return

    log.info("Refreshing factor data for %d tickers", len(_FACTOR_YF_MAP))
    inserted = 0
    for our_ticker, yf_ticker in _FACTOR_YF_MAP.items():
        try:
            info = yf.Ticker(yf_ticker).info
        except Exception as e:
            log.debug("yfinance .info failed for %s: %s", yf_ticker, e)
            info = {}

        ibkr_row = ibkr_map.get(our_ticker, {})

        trailing_pe = ibkr_row.get("trailing_pe")
        if trailing_pe is None or pd.isna(trailing_pe):
            trailing_pe = info.get("trailingPE")

        peg_ratio = ibkr_row.get("peg_ratio")
        if peg_ratio is None or pd.isna(peg_ratio):
            peg_ratio = info.get("pegRatio")

        earnings_growth_5y = ibkr_row.get("earnings_growth_5y")
        if earnings_growth_5y is None or pd.isna(earnings_growth_5y):
            earnings_growth_5y = info.get("earningsGrowth")

        five_year_avg = ibkr_row.get("five_year_avg_return")
        if five_year_avg is None or pd.isna(five_year_avg):
            five_year_avg = info.get("fiveYearAverageReturn")

        price_to_book = ibkr_row.get("price_to_book")
        if price_to_book is None or pd.isna(price_to_book):
            price_to_book = info.get("priceToBook")

        # Skip if no PE data at all — nothing useful to store
        if trailing_pe is None:
            continue

        conn.execute(
            """
            INSERT INTO factor_data
                (ticker, date, trailing_pe, peg_ratio, earnings_growth_5y,
                 five_year_avg_return, price_to_book, source, refreshed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker, date) DO UPDATE SET
                trailing_pe=excluded.trailing_pe,
                peg_ratio=excluded.peg_ratio,
                earnings_growth_5y=excluded.earnings_growth_5y,
                five_year_avg_return=excluded.five_year_avg_return,
                price_to_book=excluded.price_to_book,
                source=excluded.source,
                refreshed=excluded.refreshed
            """,
            [our_ticker, today, trailing_pe, peg_ratio, earnings_growth_5y,
             five_year_avg, price_to_book,
             "ibkr_csv" if our_ticker in ibkr_map else "yfinance", now],
        )
        inserted += 1

    conn.commit()
    conn.close()
    log.info("Factor data refresh complete: %d tickers stored for %s", inserted, today)


def get_factor_data(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return full factor_data timeseries from cache.

    Columns: ticker, date, trailing_pe, peg_ratio, earnings_growth_5y,
             five_year_avg_return, price_to_book, source

    If tickers is None, returns all. Sorted by ticker, date ascending.
    """
    conn = _get_cache()
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        df = conn.execute(
            f"SELECT ticker, date, trailing_pe, peg_ratio, earnings_growth_5y, "
            f"five_year_avg_return, price_to_book, source "
            f"FROM factor_data WHERE ticker IN ({placeholders}) "
            f"ORDER BY ticker, date",
            tickers,
        ).df()
    else:
        df = conn.execute(
            "SELECT ticker, date, trailing_pe, peg_ratio, earnings_growth_5y, "
            "five_year_avg_return, price_to_book, source "
            "FROM factor_data ORDER BY ticker, date"
        ).df()
    conn.close()
    return df


def get_shiller_pe_zscore() -> float | None:
    """Z-score of the latest US Shiller CAPE vs its own full history.

    Uses the Shiller dataset already cached by refresh_shiller().
    Negative z-score = US market cheap vs own long-run history.
    """
    conn = _get_cache()
    df = conn.execute("SELECT date, cape FROM shiller_cape ORDER BY date").df()
    conn.close()
    if df.empty or len(df) < 12:
        return None
    capes = df["cape"].dropna()
    latest = float(capes.iloc[-1])
    mean = float(capes.mean())
    std = float(capes.std())
    if std < 0.1:
        return None
    return round((latest - mean) / std, 3)


def upsert_live_prices_for_tickers(tickers: list[str]) -> int:
    """Fetch missing/current prices for a small ticker list via yfinance and cache them."""
    if not tickers:
        return 0

    try:
        import yfinance as yf
    except ImportError:
        return 0

    tickers = sorted(set(tickers))
    yf_map = get_yfinance_ticker_map()
    active = {t: yf_map.get(t) for t in tickers if yf_map.get(t)}
    if not active:
        return 0

    raw = yf.download(
        list(active.values()),
        period="400d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return 0

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    conn = _get_cache()
    now = datetime.now()
    inserted = 0
    for our_ticker, yf_ticker in active.items():
        if yf_ticker not in close.columns:
            continue
        series = close[yf_ticker].dropna()
        if series.empty:
            continue
        rows = [(our_ticker, str(d.date()), float(p)) for d, p in series.items()]
        conn.execute("DELETE FROM etf_prices WHERE ticker = ?", [our_ticker])
        conn.executemany("INSERT INTO etf_prices VALUES (?, ?, ?)", rows)
        ma200 = float(series.tail(200).mean()) if len(series) >= 200 else float(series.mean())
        trailing_252 = series.tail(252) if len(series) >= 252 else series
        low_52w = float(trailing_252.min())
        high_52w = float(trailing_252.max())
        last_price = float(series.iloc[-1])
        last_date = str(series.index[-1].date())
        conn.execute(
            """
            INSERT INTO etf_meta
                (ticker, ma200, low_52w, high_52w, last_price, last_date, refreshed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                ma200=excluded.ma200,
                low_52w=excluded.low_52w,
                high_52w=excluded.high_52w,
                last_price=excluded.last_price,
                last_date=excluded.last_date,
                refreshed=excluded.refreshed
            """,
            [our_ticker, ma200, low_52w, high_52w, last_price, last_date, now],
        )
        inserted += 1
    conn.close()
    return inserted


def replace_etf_constituents(snapshot_df: pd.DataFrame, as_of: str | None = None, source: str = "yfinance") -> int:
    """Persist an ETF constituent snapshot into DuckDB."""
    if snapshot_df.empty:
        return 0

    as_of = as_of or str(datetime.now().date())
    now = datetime.now()
    conn = _get_cache()
    rows = [
        (
            str(row["ETF"]),
            str(row["Holding symbol"]),
            str(row["Holding name"]),
            None if pd.isna(row.get("Mapped ticker")) else str(row.get("Mapped ticker")),
            None if pd.isna(row.get("Mapped sleeve")) else str(row.get("Mapped sleeve")),
            float(row["Weight %"]),
            source,
            as_of,
            now,
        )
        for _, row in snapshot_df.iterrows()
    ]
    etfs = sorted({r[0] for r in rows})
    for etf in etfs:
        conn.execute("DELETE FROM etf_constituents WHERE etf_ticker = ? AND as_of = ?", [etf, as_of])
    conn.executemany(
        """
        INSERT INTO etf_constituents
            (etf_ticker, holding_symbol, holding_name, mapped_ticker, mapped_sleeve,
             weight_pct, source, as_of, refreshed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.close()
    return len(rows)


def get_etf_constituents(etf_ticker: str, as_of: str | None = None) -> pd.DataFrame:
    """Load ETF constituents from DuckDB, latest snapshot by default."""
    conn = _get_cache()
    if as_of is None:
        row = conn.execute(
            "SELECT MAX(as_of) FROM etf_constituents WHERE etf_ticker = ?",
            [etf_ticker],
        ).fetchone()
        as_of = str(row[0]) if row and row[0] is not None else None
    if as_of is None:
        conn.close()
        return pd.DataFrame()
    df = conn.execute(
        """
        SELECT
            etf_ticker AS ETF,
            holding_symbol AS "Holding symbol",
            holding_name AS "Holding name",
            mapped_ticker AS "Mapped ticker",
            mapped_sleeve AS "Mapped sleeve",
            weight_pct AS "Weight %",
            source,
            as_of
        FROM etf_constituents
        WHERE etf_ticker = ? AND as_of = ?
        ORDER BY weight_pct DESC
        """,
        [etf_ticker, as_of],
    ).df()
    conn.close()
    return df


def compute_true_exposure(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Expand ETF holdings into underlying-stock lookthrough exposure.

    Returns rows at the underlying-security level with both direct and indirect
    exposure legs preserved.
    """
    if portfolio_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, holding in portfolio_df.iterrows():
        source_ticker = str(holding["ticker"])
        source_value = float(holding["gbp_value"])
        account_type = str(holding["account_type"])
        ins = lookup(source_ticker)
        if ins is None:
            continue

        if ins.vehicle_type == "stock":
            rows.append(
                {
                    "account_type": account_type,
                    "source_ticker": source_ticker,
                    "underlying_ticker": source_ticker,
                    "exposure_type": "direct",
                    "gbp_value": source_value,
                }
            )
            continue

        constituents = get_etf_constituents(source_ticker)
        if constituents.empty:
            rows.append(
                {
                    "account_type": account_type,
                    "source_ticker": source_ticker,
                    "underlying_ticker": source_ticker,
                    "exposure_type": "fund_unexpanded",
                    "gbp_value": source_value,
                }
            )
            continue

        expanded_any = False
        for _, row in constituents.iterrows():
            mapped = row.get("Mapped ticker")
            if pd.isna(mapped) or not mapped:
                continue
            expanded_any = True
            rows.append(
                {
                    "account_type": account_type,
                    "source_ticker": source_ticker,
                    "underlying_ticker": str(mapped),
                    "exposure_type": "indirect",
                    "gbp_value": source_value * float(row["Weight %"]) / 100.0,
                }
            )
        if not expanded_any:
            rows.append(
                {
                    "account_type": account_type,
                    "source_ticker": source_ticker,
                    "underlying_ticker": source_ticker,
                    "exposure_type": "fund_unmapped",
                    "gbp_value": source_value,
                }
            )

    return pd.DataFrame(rows)


def summarize_true_exposure(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate direct + indirect lookthrough exposure by underlying ticker."""
    expanded = compute_true_exposure(portfolio_df)
    if expanded.empty:
        return pd.DataFrame()

    agg: dict[tuple[str, str], dict[str, float | str]] = {}
    for _, row in expanded.iterrows():
        key = (row["account_type"], row["underlying_ticker"])
        rec = agg.setdefault(
            key,
            {
                "account_type": row["account_type"],
                "underlying_ticker": row["underlying_ticker"],
                "direct_gbp": 0.0,
                "indirect_gbp": 0.0,
                "unmapped_fund_gbp": 0.0,
            },
        )
        gbp_value = float(row["gbp_value"])
        if row["exposure_type"] == "direct":
            rec["direct_gbp"] += gbp_value
        elif row["exposure_type"] == "indirect":
            rec["indirect_gbp"] += gbp_value
        else:
            rec["unmapped_fund_gbp"] += gbp_value

    out = pd.DataFrame(agg.values())
    out["total_gbp"] = out["direct_gbp"] + out["indirect_gbp"] + out["unmapped_fund_gbp"]
    out["duplicate_overlap"] = (out["direct_gbp"] > 0) & (out["indirect_gbp"] > 0)
    return out.sort_values(["account_type", "total_gbp"], ascending=[True, False]).reset_index(drop=True)
