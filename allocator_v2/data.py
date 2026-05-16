"""Data access for the all-weather allocator.

Reads directly from the repo's existing pipeline:

- ``app.data.load_prices(tickers=...)`` → long price history (used by the 90d
  covariance matrix and the sizers' return inputs).
- ``app.data.load_latest_perf(tickers=..., max_rown=N)`` → the cleaned
  ``latest_performance`` table with pre-computed returns, vol, z-scores.

We do NOT maintain a separate DuckDB cache here. The repo's pipeline already
updates parquet files; this module just loads them. The only local state is
Streamlit's ``@st.cache_data`` decorator, which keeps re-queries cheap during
a session.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP_DIR = os.path.join(_REPO_ROOT, "app")
_REPO_DATA_MODULE = None


def _load_app_data():
    """Import the repo's ``app/data.py`` by path so we don't depend on PYTHONPATH."""
    global _REPO_DATA_MODULE
    if _REPO_DATA_MODULE is not None:
        return _REPO_DATA_MODULE

    if _APP_DIR not in sys.path:
        sys.path.insert(0, _APP_DIR)

    data_py = os.path.join(_APP_DIR, "data.py")
    spec = importlib.util.spec_from_file_location("fin_tracker_app_data", data_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import app/data.py from {data_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _REPO_DATA_MODULE = module
    return module


def load_prices(tickers: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Long-format price history (ticker, date, price) from the repo parquet."""
    app_data = _load_app_data()
    df = app_data.load_prices(tickers=tuple(tickers))
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_latest_perf(tickers: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Latest-row performance metrics (vol, z-scores, returns, drawdown)."""
    app_data = _load_app_data()
    df = app_data.load_latest_perf(tickers=tuple(tickers), max_rown=1)
    if df.empty:
        return df
    return df.copy()


def returns_wide(prices_long: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Wide daily-return matrix (date × ticker). Columns align to ``tickers``."""
    if prices_long.empty:
        return pd.DataFrame(columns=tickers)
    wide = prices_long.pivot_table(index="date", columns="ticker", values="price")
    wide = wide.sort_index().ffill(limit=3)
    rets = wide.pct_change(fill_method=None).dropna(how="all")
    for t in tickers:
        if t not in rets.columns:
            rets[t] = np.nan
    return rets[tickers]
