"""ETF lookthrough utilities.

Maps ETF top holdings onto the direct-stock universe where possible so ETFs can
be analysed as baskets of individual equities rather than opaque tickers.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from instruments import (
    STOCK_LOOKTHROUGH_ALIASES,
    get_yfinance_ticker_map,
    get_yfinance_to_internal_map,
    lookup,
)

log = logging.getLogger(__name__)


def fetch_etf_top_holdings(etf_ticker: str, limit: int = 10) -> pd.DataFrame:
    """Return top holdings for an ETF using yfinance funds_data.

    Columns:
    - ETF
    - Holding symbol
    - Holding name
    - Weight %
    """
    etf = lookup(etf_ticker)
    if etf is None:
        return pd.DataFrame()

    yf_symbol = get_yfinance_ticker_map().get(etf_ticker, etf_ticker)
    try:
        top = yf.Ticker(yf_symbol).funds_data.top_holdings
    except Exception as exc:
        log.warning("ETF lookthrough fetch failed for %s: %s", etf_ticker, exc)
        return pd.DataFrame()

    if top is None or top.empty:
        return pd.DataFrame()

    top = top.reset_index().rename(
        columns={"Symbol": "Holding symbol", "Name": "Holding name", "Holding Percent": "Weight %"}
    )
    top["ETF"] = etf_ticker
    top["Weight %"] = top["Weight %"].astype(float) * 100.0
    return top[["ETF", "Holding symbol", "Holding name", "Weight %"]].head(limit)


def map_holdings_to_universe(holdings_df: pd.DataFrame, score_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map ETF holdings symbols to internal stock instruments and existing factor data."""
    if holdings_df.empty:
        return pd.DataFrame()

    reverse_map = get_yfinance_to_internal_map()
    score_map = score_df.set_index("Ticker").to_dict(orient="index") if score_df is not None and not score_df.empty else {}

    rows = []
    for _, row in holdings_df.iterrows():
        raw_symbol = str(row["Holding symbol"]).strip().upper()
        internal = STOCK_LOOKTHROUGH_ALIASES.get(raw_symbol)
        if internal is None:
            internal = reverse_map.get(raw_symbol)
        # Some non-US holdings arrive already normalized without suffix in our stock universe.
        if internal is None:
            internal = reverse_map.get(raw_symbol.replace("-", "."))
        ins = lookup(internal) if internal else None
        score = score_map.get(internal, {})
        rows.append(
            {
                "ETF": row["ETF"],
                "Holding symbol": raw_symbol,
                "Holding name": row["Holding name"],
                "Weight %": row["Weight %"],
                "Mapped ticker": internal,
                "Mapped sleeve": ins.sleeve if ins else None,
                "Trailing PE": score.get("Trailing PE"),
                "PE percentile": score.get("PE percentile"),
                "PEG": score.get("PEG"),
                "Price/MA200": score.get("Price/MA200"),
                "Composite": score.get("Composite"),
            }
        )

    return pd.DataFrame(rows)


def build_constituent_snapshot(etf_tickers: list[str], score_df: pd.DataFrame | None = None, limit: int = 10) -> pd.DataFrame:
    """Build a combined ETF constituent snapshot for persistence."""
    frames = []
    for etf_ticker in etf_tickers:
        holdings_df = fetch_etf_top_holdings(etf_ticker, limit=limit)
        if holdings_df.empty:
            continue
        mapped_df = map_holdings_to_universe(holdings_df, score_df=score_df)
        if mapped_df.empty:
            continue
        frames.append(mapped_df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["as_of"] = str(datetime.now().date())
    return out
