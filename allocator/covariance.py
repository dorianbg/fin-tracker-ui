"""Pairwise-correlation helpers used by the covariance-warning panel.

We track two signals derived from daily closes:

- A current 90-day realised correlation matrix — fed to the ERC sizer and
  used to flag holdings whose combined weight sits on top of a correlated
  pair. Above 0.75 the pair is treated as a single sleeve for risk
  purposes, so the warning nudges the user to trim one side.
- A "corr drift" read: current 90d correlation minus 1y correlation per
  pair. Positive drift means the pair has been correlating more tightly
  recently — relevant when a supposed diversifier (e.g. linkers vs
  equity) starts moving with stocks in a rates-led selloff.

Both are expensive enough that Streamlit callers should cache them with
``@st.cache_data(ttl=3600)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_sources as ds


def _returns_wide(prices_long: pd.DataFrame) -> pd.DataFrame:
    wide = prices_long.pivot_table(index="date", columns="ticker", values="close")
    wide = wide.sort_index().ffill(limit=3)
    rets = wide.pct_change(fill_method=None).dropna(how="all")
    return rets


def correlation_matrix(tickers: list[str] | tuple[str, ...], window_days: int = 90) -> pd.DataFrame:
    """Realised correlation over the trailing ``window_days`` trading days.

    Returns an empty DataFrame if price history is missing for any ticker
    (we prefer explicit failure to silently-padded correlations).
    """
    if not tickers:
        return pd.DataFrame()
    prices = ds.get_price_history(tuple(tickers))
    if prices.empty:
        return pd.DataFrame()
    rets = _returns_wide(prices)
    if rets.empty:
        return pd.DataFrame()
    tail = rets.tail(window_days)
    # Drop tickers without enough observations in the window so we don't
    # return correlations driven by 10 overlapping days.
    min_obs = max(30, window_days // 3)
    cols = [c for c in tail.columns if tail[c].count() >= min_obs]
    if not cols:
        return pd.DataFrame()
    return tail[cols].corr()


def correlation_drift(
    tickers: list[str] | tuple[str, ...],
    short_window: int = 90,
    long_window: int = 252,
) -> pd.DataFrame:
    """Per-pair (short_corr - long_corr). Positive = tightening recently."""
    if not tickers:
        return pd.DataFrame()
    prices = ds.get_price_history(tuple(tickers))
    if prices.empty:
        return pd.DataFrame()
    rets = _returns_wide(prices)
    if rets.empty:
        return pd.DataFrame()
    short = rets.tail(short_window).corr()
    long = rets.tail(long_window).corr()
    common = [c for c in short.columns if c in long.columns]
    if not common:
        return pd.DataFrame()
    return short.loc[common, common] - long.loc[common, common]


def concentrated_pairs(
    sized_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    drift_df: pd.DataFrame | None = None,
    corr_threshold: float = 0.75,
    weight_col: str = "risk_scaled_weight",
    min_combined_weight: float = 0.10,
) -> pd.DataFrame:
    """Flag pairs of holdings with high correlation AND non-trivial combined weight.

    Parameters
    ----------
    sized_df : DataFrame
        Per-bucket sized plan (output of ``sizing.inverse_vol_weights`` or
        ``sizing.erc_weights``).
    corr_df : DataFrame
        Pairwise correlation matrix (output of ``correlation_matrix``).
    drift_df : DataFrame, optional
        Pairwise correlation drift; if present, the drift is annotated
        onto each flagged pair so the UI can distinguish "structurally
        correlated" from "tightening in the current regime".
    corr_threshold : float
        Minimum correlation to flag. 0.75 is the default — anything above
        that means the pair moves essentially together day-to-day.
    min_combined_weight : float
        Ignore pairs whose combined weight is under this. A 2% + 2%
        overlap isn't worth alerting on even when correlation is 0.95.
    """
    if sized_df.empty or corr_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for account_type, grp in sized_df.groupby("account_type", sort=False):
        active = grp[grp["action"] != "NOT_ACTIVE"]
        if len(active) < 2:
            continue
        active_tickers = [t for t in active["ticker"].astype(str).tolist() if t in corr_df.columns]
        weight_lookup = dict(zip(active["ticker"].astype(str), active[weight_col]))
        sub = corr_df.loc[active_tickers, active_tickers]
        for i, a in enumerate(active_tickers):
            for b in active_tickers[i + 1 :]:
                corr = float(sub.loc[a, b])
                if not np.isfinite(corr) or corr < corr_threshold:
                    continue
                wa = float(weight_lookup.get(a, 0.0))
                wb = float(weight_lookup.get(b, 0.0))
                if wa + wb < min_combined_weight:
                    continue
                drift_val = None
                if drift_df is not None and not drift_df.empty and a in drift_df.columns and b in drift_df.index:
                    d = drift_df.loc[a, b]
                    if np.isfinite(d):
                        drift_val = float(d)
                rows.append(
                    {
                        "account_type": account_type,
                        "ticker_a": a,
                        "ticker_b": b,
                        "correlation_90d": corr,
                        "correlation_drift_90_vs_252": drift_val,
                        "weight_a": wa,
                        "weight_b": wb,
                        "combined_weight": wa + wb,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["account_type", "combined_weight"], ascending=[True, False]).reset_index(drop=True)
    return out
