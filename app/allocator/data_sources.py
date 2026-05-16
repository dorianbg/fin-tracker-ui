"""External data fetchers — FRED with local CSV fallback.

All fetchers are designed to be called from Streamlit with @st.cache_data.
They return DataFrames or dicts that the valuation engine consumes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "macro_cache"

FRED_SERIES = {
    "us_real_yield_10y": "DFII10",
    "uk_real_yield_10y": "FII10",
    "em_hy_spread": "BAMLEMHBHYCRPIUSOAS",
    "us_10y_nominal": "DGS10",
}


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_fred_series(series_id: str, start: str = "2020-01-01") -> pd.Series:
    """Fetch a FRED series via pandas-datareader. Falls back to local CSV."""
    cache_file = CACHE_DIR / f"{series_id}.csv"
    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_id, "fred", start=start)
        _ensure_cache_dir()
        df.to_csv(cache_file)
        return df.iloc[:, 0].dropna()
    except Exception as e:
        logging.warning(f"FRED fetch failed for {series_id}: {e}; trying local cache")
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df.iloc[:, 0].dropna()
        return pd.Series(dtype=float)


def fetch_all_fred() -> dict[str, float]:
    """Fetch latest values for all tracked FRED series."""
    out: dict[str, float] = {}
    for label, sid in FRED_SERIES.items():
        s = fetch_fred_series(sid)
        if not s.empty:
            out[label] = float(s.iloc[-1]) / 100.0
    return out


def build_macro_data():
    """Build a MacroData instance from live FRED data."""
    from allocator.valuation import MacroData
    fred = fetch_all_fred()
    return MacroData(
        uk_real_yield_10y=fred.get("uk_real_yield_10y", 0.0),
        em_hy_spread=fred.get("em_hy_spread", 0.0),
        us_10y_nominal=fred.get("us_10y_nominal", 0.0),
    )
