"""Factor computation: PE-vs-history and PEG-growth factors.

Two factors for screening instruments and tilting allocations:

Factor 1 — PE z-score vs own history
--------------------------------------
How cheap is this instrument vs its own trailing PE distribution?

  z = (current_PE - mean_PE_history) / std_PE_history

  z < -1  → "CHEAP" (bottom ~16th percentile of own history)
  z ∈ [-1, +1] → "FAIR"
  z > +1  → "DEAR" (top ~16th percentile)

Data source: `factor_data` table in allocator_cache.duckdb, populated by
`data_sources.refresh_factor_data()`. Each quarterly refresh adds one row per
ticker. After ~8 refreshes (2 years of quarterly data), the z-score is
meaningful. Before that, it is marked 'INSUFFICIENT_HISTORY'.

Bootstrap for US market: Shiller CAPE data (already in cache) provides a
deep historical anchor for the US broad market. Sector-level history
self-builds from quarterly snapshots.

Factor 2 — PEG ratio (growth-adjusted value)
----------------------------------------------
PEG = forward (or trailing) PE / EPS growth rate

  PEG < 1.0  → paying < 1 year's worth of growth per unit of earnings → "CHEAP"
  PEG 1–2    → fair value
  PEG > 2    → expensive vs growth rate

Data sources available (in priority order):
  1. yfinance `.info["pegRatio"]` — directly available for many US ETFs/stocks
  2. Computed from trailing_PE / trailing_EPS_3yr_growth (own quarterly history)
  3. Computed from trailing_PE / fiveYearAverageReturn as growth proxy (approximate)

Note: ETFs report weighted average PE/PEG of their holdings — this is what we want.
yfinance returns these for US-listed ETFs. For GBP UCITS ETFs, we use the iShares
product JSON which gives PE; PEG is computed from PE + price return history as proxy.

Composite signal
-----------------
A combined signal merges both factors. Instruments in the top quartile on both
(cheap vs history AND low PEG) are flagged as "HIGH CONVICTION BUY SIGNAL".
The MA200 filter from valuation.py is applied on top: if price < 0.85×MA200,
the signal is suppressed ("falling knife — wait for stabilisation").
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class FactorScore:
    ticker: str
    current_pe: float | None
    pe_zscore: float | None          # negative = cheap vs own history
    pe_percentile: float | None      # low percentile = cheap vs own history
    pe_range_position: float | None  # 0 = bottom of observed range, 1 = top
    pe_history_signal: str           # "CHEAP" | "FAIR" | "DEAR" | "INSUFFICIENT_HISTORY"
    peg_ratio: float | None          # < 1 = growth is cheap
    peg_source: str                  # "direct" | "computed" | "unavailable"
    peg_signal: str                  # "CHEAP" | "FAIR" | "DEAR" | "N/A"
    earnings_growth_pct: float | None  # annualised EPS growth used for PEG compute
    price_ma200_ratio: float | None   # from etf_meta cache
    price_range_52w: float | None     # 0 = at 52w low, 1 = at 52w high
    distance_to_52w_low_pct: float | None
    ma200_signal: str                # "OK" | "FALLING_KNIFE" | "EXTENDED" | "N/A"
    composite_signal: str            # "HIGH_CONVICTION" | "CHEAP" | "FAIR" | "DEAR" | "AVOID"


def _safe_percentile_rank(series: pd.Series, value: float) -> float | None:
    clean = series.dropna()
    if len(clean) < 4:
        return None
    return float((clean <= value).mean())


def _safe_range_position(series: pd.Series, value: float) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    floor = float(clean.min())
    ceiling = float(clean.max())
    spread = ceiling - floor
    if spread <= 0:
        return None
    return float((value - floor) / spread)


def _pe_history_label(z: float | None, n_points: int) -> str:
    if z is None or n_points < 4:
        return "INSUFFICIENT_HISTORY"
    if z < -1.0:
        return "CHEAP"
    if z > 1.0:
        return "DEAR"
    return "FAIR"


def _peg_label(peg: float | None) -> str:
    if peg is None:
        return "N/A"
    if peg < 1.0:
        return "CHEAP"
    if peg < 2.0:
        return "FAIR"
    return "DEAR"


def _ma200_label(ratio: float | None) -> str:
    if ratio is None:
        return "N/A"
    if ratio < 0.85:
        return "FALLING_KNIFE"
    if ratio > 1.30:
        return "EXTENDED"
    return "OK"


def _composite(pe_signal: str, peg_signal: str, ma200_signal: str) -> str:
    """Combine the three signals into one actionable label."""
    if ma200_signal == "FALLING_KNIFE":
        return "AVOID"  # never buy a crashing knife regardless of valuation
    if ma200_signal == "EXTENDED":
        return "AVOID"  # don't chase parabolic moves
    cheap_signals = sum(s == "CHEAP" for s in (pe_signal, peg_signal))
    dear_signals = sum(s == "DEAR" for s in (pe_signal, peg_signal))
    if cheap_signals == 2:
        return "HIGH_CONVICTION"
    if cheap_signals == 1 and dear_signals == 0:
        return "CHEAP"
    if dear_signals >= 1:
        return "DEAR"
    return "FAIR"


# ── Factor computation ────────────────────────────────────────────────

def compute_factor_scores(
    factor_df: pd.DataFrame,  # rows from factor_data table (ticker, date, trailing_pe, ...)
    meta_df: pd.DataFrame,    # rows from etf_meta table (ticker, last_price, ma200)
) -> list[FactorScore]:
    """
    Compute both factors for each ticker that appears in factor_df.

    factor_df columns: ticker, date, trailing_pe, peg_ratio, earnings_growth_5y,
                       five_year_avg_return
    meta_df columns:   ticker, ma200, low_52w, high_52w, last_price

    Returns one FactorScore per ticker (latest data point + history-based z-score).
    """
    scores = []
    meta_idx = meta_df.set_index("ticker") if not meta_df.empty else pd.DataFrame()

    for ticker, grp in factor_df.groupby("ticker"):
        grp = grp.sort_values("date")
        latest = grp.iloc[-1]

        # ── Factor 1: PE z-score ─────────────────────────────────────
        pe_series = grp["trailing_pe"].dropna()
        current_pe = float(latest["trailing_pe"]) if pd.notna(latest["trailing_pe"]) else None
        pe_zscore = None
        pe_percentile = None
        pe_range_position = None
        if current_pe is not None and len(pe_series) >= 4:
            mean_pe = float(pe_series.mean())
            std_pe = float(pe_series.std())
            if std_pe > 0.5:  # avoid division by near-zero std
                pe_zscore = (current_pe - mean_pe) / std_pe
            pe_percentile = _safe_percentile_rank(pe_series, current_pe)
            pe_range_position = _safe_range_position(pe_series, current_pe)
        n_points = int(len(pe_series))

        # ── Factor 2: PEG ratio ──────────────────────────────────────
        # Priority: (a) direct pegRatio field, (b) computed from PE + growth, (c) N/A
        peg = None
        peg_source = "unavailable"
        earnings_growth = None

        direct_peg = latest.get("peg_ratio") if "peg_ratio" in latest.index else None
        if pd.notna(direct_peg) and isinstance(direct_peg, (int, float)) and direct_peg > 0:
            peg = float(direct_peg)
            peg_source = "direct"

        if peg is None:
            # Try computing from PE / earnings growth (5yr annualised return as proxy)
            growth = latest.get("earnings_growth_5y") if "earnings_growth_5y" in latest.index else None
            if pd.isna(growth) or growth is None:
                growth = latest.get("five_year_avg_return") if "five_year_avg_return" in latest.index else None
            if pd.notna(growth) and isinstance(growth, (int, float)) and growth > 0.01:
                earnings_growth = float(growth) * 100  # as percentage
                if current_pe is not None and earnings_growth > 0:
                    peg = current_pe / earnings_growth
                    peg_source = "computed"

        # ── MA200 ratio ───────────────────────────────────────────────
        ma200_ratio = None
        price_range_52w = None
        distance_to_52w_low_pct = None
        if not meta_idx.empty and ticker in meta_idx.index:
            row = meta_idx.loc[ticker]
            ma200 = row.get("ma200")
            last_price = row.get("last_price")
            low_52w = row.get("low_52w")
            high_52w = row.get("high_52w")
            if pd.notna(ma200) and pd.notna(last_price) and float(ma200) > 0:
                ma200_ratio = float(last_price) / float(ma200)
            if (
                pd.notna(low_52w)
                and pd.notna(high_52w)
                and pd.notna(last_price)
                and float(high_52w) > float(low_52w) > 0
            ):
                price_range_52w = (float(last_price) - float(low_52w)) / (
                    float(high_52w) - float(low_52w)
                )
                distance_to_52w_low_pct = (float(last_price) / float(low_52w)) - 1.0

        # ── Signals ───────────────────────────────────────────────────
        pe_signal = _pe_history_label(pe_zscore, n_points)
        peg_signal = _peg_label(peg)
        ma200_signal = _ma200_label(ma200_ratio)
        composite = _composite(pe_signal, peg_signal, ma200_signal)

        scores.append(FactorScore(
            ticker=str(ticker),
            current_pe=current_pe,
            pe_zscore=round(pe_zscore, 3) if pe_zscore is not None else None,
            pe_percentile=round(pe_percentile, 3) if pe_percentile is not None else None,
            pe_range_position=round(pe_range_position, 3) if pe_range_position is not None else None,
            pe_history_signal=pe_signal,
            peg_ratio=round(peg, 3) if peg is not None else None,
            peg_source=peg_source,
            peg_signal=peg_signal,
            earnings_growth_pct=round(earnings_growth, 2) if earnings_growth is not None else None,
            price_ma200_ratio=round(ma200_ratio, 3) if ma200_ratio is not None else None,
            price_range_52w=round(price_range_52w, 3) if price_range_52w is not None else None,
            distance_to_52w_low_pct=round(distance_to_52w_low_pct, 3) if distance_to_52w_low_pct is not None else None,
            ma200_signal=ma200_signal,
            composite_signal=composite,
        ))

    return scores


def as_dataframe(scores: list[FactorScore]) -> pd.DataFrame:
    """Return factor scores as a styled-ready DataFrame."""
    return pd.DataFrame([
        {
            "Ticker": s.ticker,
            "Trailing PE": s.current_pe,
            "PE z-score": s.pe_zscore,
            "PE percentile": s.pe_percentile,
            "PE range pos": s.pe_range_position,
            "PE signal": s.pe_history_signal,
            "PEG": s.peg_ratio,
            "PEG source": s.peg_source,
            "PEG signal": s.peg_signal,
            "EPS growth %": s.earnings_growth_pct,
            "Price/MA200": s.price_ma200_ratio,
            "52W range pos": s.price_range_52w,
            "Dist. from 52W low": s.distance_to_52w_low_pct,
            "MA200 signal": s.ma200_signal,
            "Composite": s.composite_signal,
        }
        for s in scores
    ])
