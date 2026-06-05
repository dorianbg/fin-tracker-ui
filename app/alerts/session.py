from __future__ import annotations

import pandas as pd


ASIA_SUFFIXES = (".T", ".KS", ".KQ", ".HK")
EU_SUFFIXES = (
    ".L",
    ".PA",
    ".DE",
    ".AS",
    ".SW",
    ".CO",
    ".ST",
    ".MI",
    ".MC",
    ".BR",
    ".HE",
)
VALID_SESSIONS = ("all", "asia", "eu", "us")


def ticker_for_session(row: pd.Series) -> str:
    full = row.get("ticker_full")
    ticker = row.get("ticker")
    if pd.notna(full) and str(full):
        return str(full)
    return str(ticker)


def market_session_for_ticker(ticker: str) -> str:
    ticker = str(ticker)
    if ticker.endswith(ASIA_SUFFIXES):
        return "asia"
    if ticker.endswith(EU_SUFFIXES):
        return "eu"
    return "us"


def add_alert_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "ticker_full" in out.columns:
        out["alert_ticker"] = out.apply(ticker_for_session, axis=1)
    else:
        out["alert_ticker"] = out["ticker"].astype(str)
    return out


def filter_by_session(df: pd.DataFrame, session: str) -> pd.DataFrame:
    if session not in VALID_SESSIONS:
        raise ValueError(f"Unknown session: {session}")
    out = add_alert_ticker(df)
    if session == "all" or out.empty:
        return out
    mask = out["alert_ticker"].map(market_session_for_ticker) == session
    return out[mask].copy()


def session_label(session: str) -> str:
    return {"all": "All markets", "asia": "Asia", "eu": "EU/UK", "us": "US"}[session]
