"""Price/momentum buy-signal rules.

Signals are derived purely from the timing metrics computed in
``data_sources.get_entry_timing_metrics`` plus the per-instrument regime from
``themes.classify_instrument_regimes``. Valuation fields are explicitly NOT
used — at a long horizon and an ETF granularity, reported P/Es are noisy,
sparse (no PE for bonds/gold/commodities) and shorter-horizon than our
investment clock. Price/trend/vol and a few sleeve-specific macro triggers
deliver a more consistent signal.

Signal vocabulary (unchanged from the old PE-driven engine so downstream
consumers don't break):

  BUY        — oversold repair with a visible turn: washed_out/basing/repairing
               regime AND short-term return positive.
  ACCUMULATE — healthy but constructive: strong regime, pullback inside band.
  HOLD       — dead_money or unclear regime.
  WATCH      — strong_but_stretched: valid sleeve, wrong entry window.
  WAIT       — same as WATCH today; retained for backwards-compat in call sites.
  AVOID      — falling_knife regime.
"""

from dataclasses import dataclass

import pandas as pd

from instruments import INSTRUMENTS, THEMATIC_EXTRAS, lookup, wrapper_rules_summary
from themes import classify_instrument_regimes


@dataclass(frozen=True)
class StrategyThresholds:
    """Price/momentum thresholds. Tuned for weekly/monthly rebalancing cadence."""

    # Dormancy / basing band
    dormant_abs_r_1y_pct: float = 8.0    # |r_1y| < this → dormant
    basing_min_r_3m_pct: float = 0.0     # r_3m > this → short-term turn up

    # Oversold repair
    oversold_pct_below_ma200: float = -8.0   # pct_above_ma200 < this
    oversold_min_r_3m_pct: float = -3.0      # require some stabilisation
    oversold_dd_threshold_pct: float = -15.0

    # Stretched / extended
    stretched_pct_above_ma200: float = 15.0
    stretched_range_52w_pos: float = 0.85
    stretched_r_1y_pct: float = 40.0

    # Falling knife
    knife_pct_below_ma200: float = -15.0
    knife_range_52w_pos: float = 0.10
    knife_r_3m_pct: float = -5.0


def _as_float(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


_REGIME_TO_SIGNAL = {
    "falling_knife": "AVOID",
    "strong_but_stretched": "WATCH",
    "washed_out": "ACCUMULATE",  # only promotes to BUY with a short-term turn
    "basing": "BUY",
    "repairing": "BUY",
    "strong": "ACCUMULATE",
    "dead_money": "HOLD",
}


def classify_buy_signal(
    row: pd.Series,
    thresholds: StrategyThresholds = StrategyThresholds(),
) -> tuple[str, str]:
    """Return ``(signal, rationale)`` from price/momentum fields + regime.

    Accepts either:
      - a row from the old factor/score dataframe (contains Price/MA200 etc., no regime)
      - a row from the new timing+regime dataframe (has ``regime`` column)

    When the ``regime`` column is present it is the primary driver; timing
    fields feed the rationale string and one promotion rule (washed_out →
    BUY when r_1m > 0).
    """
    regime = str(row.get("regime") or "").lower()
    r_1m = _as_float(row.get("r_1m"))
    r_3m = _as_float(row.get("r_3m"))
    r_1y = _as_float(row.get("r_1y"))
    pct_ma = _as_float(row.get("pct_above_ma200"))
    dd = _as_float(row.get("drawdown_52w"))
    rng = _as_float(row.get("range_52w_pos"))

    # Fallback: the score_df path (no regime column). Derive an approximate
    # signal from Price/MA200 + 52W range pos only. No PE inputs.
    if regime not in _REGIME_TO_SIGNAL:
        ma_ratio = _as_float(row.get("Price/MA200"))
        px_range = _as_float(row.get("52W range pos"))
        if ma_ratio is not None:
            pct_ma = (ma_ratio - 1.0) * 100.0
        if px_range is not None and rng is None:
            rng = px_range
        if pct_ma is not None and pct_ma < thresholds.knife_pct_below_ma200:
            if rng is not None and rng < thresholds.knife_range_52w_pos:
                return "AVOID", "falling knife: price well below MA200 and pinned to 52w low"
        if pct_ma is not None and pct_ma > thresholds.stretched_pct_above_ma200:
            return "WATCH", "stretched: price well above MA200"
        if rng is not None and rng >= thresholds.stretched_range_52w_pos:
            return "WATCH", "near 52w high — wait for pullback"
        if pct_ma is not None and pct_ma < thresholds.oversold_pct_below_ma200:
            return "ACCUMULATE", "oversold vs MA200"
        return "HOLD", "neutral"

    signal = _REGIME_TO_SIGNAL[regime]
    bits = []
    if pct_ma is not None:
        bits.append(f"{pct_ma:+.1f}% vs MA200")
    if rng is not None:
        bits.append(f"52w pos {rng*100:.0f}%")
    if r_3m is not None:
        bits.append(f"r3m {r_3m:+.0f}%")
    if r_1y is not None:
        bits.append(f"r1y {r_1y:+.0f}%")
    if dd is not None:
        bits.append(f"dd {dd:.0f}%")

    # Promotion: washed_out with a visible short-term turn → BUY.
    if regime == "washed_out" and r_1m is not None and r_1m > 0 and r_3m is not None and r_3m >= thresholds.oversold_min_r_3m_pct:
        signal = "BUY"
        return signal, f"washed-out with turn up ({', '.join(bits)})"

    # Demotion: basing but r_3m barely positive → keep ACCUMULATE (don't lump-sum yet).
    if regime == "basing" and r_3m is not None and r_3m < 2.0:
        signal = "ACCUMULATE"

    rationale = f"{regime}: {', '.join(bits)}" if bits else regime
    return signal, rationale


def build_buy_candidates(
    input_df: pd.DataFrame,
    thresholds: StrategyThresholds = StrategyThresholds(),
) -> pd.DataFrame:
    """Build a per-ticker signal table.

    ``input_df`` can be either:
      - a factor ``score_df`` (legacy path, no regime column)
      - a timing dataframe from ``get_entry_timing_metrics``, which will be
        enriched with a ``regime`` column via ``classify_instrument_regimes``.

    Output always has a "Ticker" column and a "Strategy signal" column so that
    existing consumers keep working.
    """
    if input_df.empty:
        return pd.DataFrame()

    df = input_df.copy()
    has_regime = "regime" in df.columns

    # Timing path: run the regime classifier so we have the label
    if not has_regime and "ticker" in df.columns and "pct_above_ma200" in df.columns:
        df = classify_instrument_regimes(df)
        has_regime = True

    # Normalise column name so downstream code can always use "Ticker"
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "Ticker"})

    rows = []
    for _, row in df.iterrows():
        ticker = str(row["Ticker"])
        ins = lookup(ticker)
        signal, rationale = classify_buy_signal(row, thresholds=thresholds)
        rows.append(
            {
                **row.to_dict(),
                "Ticker": ticker,
                "Theme": (ins.sleeve if ins else "unknown").replace("_", " ").title(),
                "Strategy signal": signal,
                "Rationale": rationale,
            }
        )

    out = pd.DataFrame(rows)
    order = pd.CategoricalDtype(
        ["BUY", "ACCUMULATE", "WATCH", "HOLD", "WAIT", "AVOID"], ordered=True
    )
    out["Strategy signal"] = out["Strategy signal"].astype(order)
    sort_cols = ["Strategy signal"]
    for candidate in ("pct_above_ma200", "52W range pos", "range_52w_pos"):
        if candidate in out.columns:
            sort_cols.append(candidate)
            break
    return out.sort_values(by=sort_cols, ascending=[True] + [True] * (len(sort_cols) - 1), na_position="last").reset_index(drop=True)


def build_theme_watchlist(input_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the best candidate per preferred theme (THEMATIC_EXTRAS)."""
    if input_df.empty:
        return pd.DataFrame()

    candidates = build_buy_candidates(input_df)
    rows = []
    for sleeve, meta in THEMATIC_EXTRAS.items():
        ticker = meta["preferred_ticker"]
        match = candidates[candidates["Ticker"] == ticker]
        signal = match.iloc[0] if not match.empty else None
        rows.append(
            {
                "Theme": sleeve.replace("_", " ").title(),
                "Ticker": ticker,
                "Signal": signal["Strategy signal"] if signal is not None else "NO_DATA",
                "Regime": signal.get("regime") if signal is not None else None,
                "% vs MA200": signal.get("pct_above_ma200") if signal is not None else None,
                "52w pos": signal.get("range_52w_pos") if signal is not None else None,
                "Max GIA weight": meta["max_weight_gia"],
                "Activation": meta["activation"],
            }
        )
    return pd.DataFrame(rows)


def _candidate_score(row: pd.Series, thresholds: StrategyThresholds) -> float:
    """Higher is better. Pure price/momentum scoring.

    Ranking goal: for a given sleeve, pick the instrument that is most clearly
    in a good regime (basing / repairing / washed-out-with-turn). Penalise
    instruments that are stretched or breaking down.
    """
    score = 0.0
    signal = str(row.get("Strategy signal") or "")
    regime = str(row.get("regime") or "").lower()

    signal_bonus = {"BUY": 4.0, "ACCUMULATE": 2.5, "WATCH": -1.0, "HOLD": 0.0, "WAIT": -1.0, "AVOID": -4.0}
    score += signal_bonus.get(signal, 0.0)

    regime_bonus = {
        "basing": 2.0,
        "repairing": 1.5,
        "washed_out": 1.0,
        "strong": 0.5,
        "dead_money": -0.5,
        "strong_but_stretched": -2.0,
        "falling_knife": -4.0,
    }
    score += regime_bonus.get(regime, 0.0)

    pct_ma = _as_float(row.get("pct_above_ma200"))
    rng = _as_float(row.get("range_52w_pos"))
    r_3m = _as_float(row.get("r_3m"))
    slope = _as_float(row.get("ma200_slope_20d"))

    if pct_ma is not None:
        if -8.0 <= pct_ma <= 8.0:
            score += 1.0  # sweet spot
        elif pct_ma > thresholds.stretched_pct_above_ma200:
            score -= 2.0
        elif pct_ma < thresholds.knife_pct_below_ma200:
            score -= 3.0
    if rng is not None:
        if rng <= 0.40:
            score += 1.0
        elif rng >= thresholds.stretched_range_52w_pos:
            score -= 1.5
    if r_3m is not None and r_3m > 0:
        score += 0.5
    if slope is not None and slope > 0:
        score += 0.3

    return round(score, 3)


def build_wrapper_candidate_table(
    input_df: pd.DataFrame,
    thresholds: StrategyThresholds = StrategyThresholds(),
) -> pd.DataFrame:
    """Best investable instruments by wrapper and sleeve (price/momentum only)."""
    if input_df.empty:
        return pd.DataFrame()

    buy_df = build_buy_candidates(input_df, thresholds=thresholds).copy()
    if buy_df.empty:
        return pd.DataFrame()

    buy_idx = buy_df.set_index("Ticker")

    rows = []
    for ticker, instrument in INSTRUMENTS.items():
        if ticker not in buy_idx.index:
            continue
        base = buy_idx.loc[ticker]
        for wrapper, eligible in instrument.wrapper_eligible.items():
            if not eligible:
                continue
            row = {
                "Wrapper": wrapper,
                "Sleeve": instrument.sleeve.replace("_", " ").title(),
                "Ticker": ticker,
                "Name": instrument.name,
                "Vehicle": instrument.vehicle_type,
                "Reporting": instrument.is_reporting_fund,
                "Strategy signal": base["Strategy signal"],
                "Regime": base.get("regime"),
                "Candidate score": _candidate_score(base, thresholds),
                "% vs MA200": base.get("pct_above_ma200"),
                "52W range pos": base.get("range_52w_pos"),
                "r_3m": base.get("r_3m"),
                "r_1y": base.get("r_1y"),
                "Rationale": base.get("Rationale"),
                "Wrapper rule": wrapper_rules_summary(wrapper),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(
        by=["Wrapper", "Sleeve", "Candidate score"],
        ascending=[True, True, False],
        na_position="last",
    )
    out["Rank"] = out.groupby(["Wrapper", "Sleeve"]).cumcount() + 1
    return out.reset_index(drop=True)
