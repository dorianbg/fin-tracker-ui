"""Theme-level analytics built from instrument sleeves and cached price history.

The snapshot builder produces per-theme diagnostics: median 1m/3m/6m/1y return,
drawdown, price/MA200, range position, share of constituents above MA200, etc.

Those diagnostics are then collapsed into a single ``regime`` label per theme,
which is what actually feeds the constructor. Regimes intentionally use a small
vocabulary so they can be reasoned about quickly:

  - ``washed_out``         — deeply drawn down and still below trend; contrarian
                              buy zone if and only if it is not still falling
  - ``repairing``          — bottomed, starting to climb back toward MA200
  - ``strong``             — healthy trend, reasonable proximity to highs
  - ``strong_but_stretched`` — trend intact but extension past MA200 / near 52w
                              high makes fresh entry a bad idea now
  - ``dead_money``         — flat, range-bound, no edge
  - ``falling_knife``      — still in freefall; do not buy

These labels are consumed by ``construction.build_portfolio_plan`` to gate the
``APPROVED`` action. Only ``repairing`` and ``strong`` produce buy-now entries.
``washed_out`` is escalated only when combined with a valuation signal (handled
in the constructor, not here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from instruments import INSTRUMENTS, lookup


def sleeve_label(sleeve: str) -> str:
    return str(sleeve).replace("_", " ").title()


def instrument_theme_rows(include_stocks: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker, ins in INSTRUMENTS.items():
        if not include_stocks and ins.vehicle_type == "stock":
            continue
        rows.append(
            {
                "ticker": ticker,
                "theme": ins.sleeve,
                "theme_label": sleeve_label(ins.sleeve),
                "vehicle_type": ins.vehicle_type,
                "name": ins.name,
            }
        )
    return pd.DataFrame(rows)


def build_theme_snapshot(timing_df: pd.DataFrame, include_stocks: bool = True) -> pd.DataFrame:
    """Aggregate price/performance diagnostics by theme/sleeve."""
    if timing_df.empty:
        return pd.DataFrame()

    theme_df = instrument_theme_rows(include_stocks=include_stocks)
    merged = theme_df.merge(timing_df, on="ticker", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["price_ma200"] = merged["price"] / merged["ma_252"].replace(0, np.nan)
    merged["range_52w_pos"] = (
        (merged["price"] - merged["low_52w"]) / (merged["high_52w"] - merged["low_52w"]).replace(0, np.nan)
    )
    merged["above_ma200"] = merged["price_ma200"] > 1.0
    merged["near_high"] = merged["drawdown_52w"] >= -7.0

    out = (
        merged.groupby(["theme", "theme_label"], as_index=False)
        .agg(
            instruments=("ticker", "nunique"),
            stocks=("vehicle_type", lambda s: int((s == "stock").sum())),
            funds=("vehicle_type", lambda s: int((s != "stock").sum())),
            r_1m=("r_1m", "median"),
            r_3m=("r_3m", "median"),
            r_6m=("r_6m", "median"),
            r_1y=("r_1y", "median"),
            vol_1y=("vol_1y", "median"),
            drawdown_52w=("drawdown_52w", "median"),
            price_ma200=("price_ma200", "median"),
            range_52w_pos=("range_52w_pos", "median"),
            pct_above_ma200=("above_ma200", "mean"),
            pct_near_high=("near_high", "mean"),
        )
        .sort_values(["r_3m", "r_1y"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    return out


REGIME_LABELS = (
    "washed_out",
    "basing",
    "repairing",
    "strong",
    "strong_but_stretched",
    "dead_money",
    "falling_knife",
)

REGIME_BUY_NOW = {"repairing", "strong", "basing"}
REGIME_WATCH = {"washed_out", "strong_but_stretched"}
REGIME_REJECT = {"falling_knife"}
REGIME_NEUTRAL = {"dead_money"}


def _classify_row(row: pd.Series) -> str:
    """Classify a single theme-snapshot row into a regime label.

    The rules are intentionally simple and explainable. All thresholds are
    collected here so they can be tuned from one place.
    """
    ma = row.get("price_ma200")
    dd = row.get("drawdown_52w")
    r_3m = row.get("r_3m")
    r_1m = row.get("r_1m")
    rng_pos = row.get("range_52w_pos")
    pct_above = row.get("pct_above_ma200")

    ma_ok = ma is not None and not pd.isna(ma)
    dd_ok = dd is not None and not pd.isna(dd)

    # Falling knife: still below trend AND still giving back ground recently.
    if ma_ok and ma < 0.85:
        if r_1m is not None and not pd.isna(r_1m) and r_1m < 0:
            return "falling_knife"
        return "washed_out"

    # Washed out: deep drawdown, sub-trend, but not freshly falling.
    if dd_ok and dd <= -20.0 and ma_ok and ma < 0.95:
        return "washed_out"

    # Strong but stretched: above trend AND near 52w highs / hot trailing return.
    if ma_ok and ma > 1.20:
        return "strong_but_stretched"
    if rng_pos is not None and not pd.isna(rng_pos) and rng_pos >= 0.85 and ma_ok and ma >= 1.05:
        return "strong_but_stretched"

    # Strong: healthy trend, breadth above MA200, not yet overextended.
    if ma_ok and ma >= 1.02 and (pct_above is None or pd.isna(pct_above) or pct_above >= 0.55):
        return "strong"

    # Repairing: below-ish trend, but medium-term return positive — bottoming process.
    if ma_ok and 0.90 <= ma < 1.05 and r_3m is not None and not pd.isna(r_3m) and r_3m >= 0:
        return "repairing"

    # Everything else: dead money / no edge.
    return "dead_money"


def _regime_rationale(regime: str, row: pd.Series) -> str:
    ma = row.get("price_ma200")
    dd = row.get("drawdown_52w")
    rng_pos = row.get("range_52w_pos")
    r_3m = row.get("r_3m")

    def _fmt(val, spec):
        if val is None or pd.isna(val):
            return "—"
        return format(val, spec)

    fragments = []
    if ma is not None and not pd.isna(ma):
        fragments.append(f"MA200 {_fmt(ma, '.2f')}")
    if dd is not None and not pd.isna(dd):
        fragments.append(f"dd {_fmt(dd, '.0f')}%")
    if rng_pos is not None and not pd.isna(rng_pos):
        fragments.append(f"52w pos {_fmt(rng_pos * 100, '.0f')}%")
    if r_3m is not None and not pd.isna(r_3m):
        fragments.append(f"r3m {_fmt(r_3m, '+.0f')}%")
    body = ", ".join(fragments) if fragments else "no signals"
    return f"{regime} — {body}"


def classify_theme_regimes(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``regime`` and ``regime_rationale`` columns to a theme snapshot."""
    if snapshot_df.empty:
        return snapshot_df
    out = snapshot_df.copy()
    out["regime"] = out.apply(_classify_row, axis=1)
    out["regime_rationale"] = out.apply(lambda r: _regime_rationale(r["regime"], r), axis=1)
    return out


def get_theme_regimes(snapshot_df: pd.DataFrame) -> dict[str, str]:
    """Return {theme: regime} for downstream consumers (constructor etc)."""
    if snapshot_df.empty or "theme" not in snapshot_df.columns:
        return {}
    labelled = classify_theme_regimes(snapshot_df)
    return dict(zip(labelled["theme"], labelled["regime"]))


def _classify_instrument(row: pd.Series, universe_vol_median: float | None) -> str:
    """Classify a single instrument into a regime using price / momentum only.

    Inputs come from ``data_sources.get_entry_timing_metrics``:
      pct_above_ma200, drawdown_52w, range_52w_pos, r_1m, r_3m, r_1y,
      vol_3m, vol_1y, ma200_slope_20d.

    The rule order matters — earlier rules win so falling-knife always
    pre-empts basing, and stretched always pre-empts strong.
    """
    pct_ma = row.get("pct_above_ma200")
    dd = row.get("drawdown_52w")
    rng = row.get("range_52w_pos")
    r_1m = row.get("r_1m")
    r_3m = row.get("r_3m")
    r_1y = row.get("r_1y")
    vol_3m = row.get("vol_3m")
    vol_1y = row.get("vol_1y")
    slope = row.get("ma200_slope_20d")

    def _ok(v):
        return v is not None and not pd.isna(v)

    # 1. Falling knife: deeply sub-trend AND still giving back ground AND pinned to 52w lows.
    if _ok(pct_ma) and pct_ma < -15.0:
        if _ok(r_3m) and r_3m < -5.0 and _ok(rng) and rng < 0.10:
            return "falling_knife"

    # 2. Strong but stretched: clearly above trend or near 52w high or heavy 1y move.
    if _ok(pct_ma) and pct_ma > 15.0:
        return "strong_but_stretched"
    if _ok(rng) and rng >= 0.85 and _ok(pct_ma) and pct_ma >= 5.0:
        return "strong_but_stretched"
    if _ok(r_1y) and r_1y > 40.0 and _ok(pct_ma) and pct_ma > 5.0:
        return "strong_but_stretched"

    # 3. Washed out: deep drawdown, sub-trend, not yet turning.
    if _ok(dd) and dd <= -20.0 and _ok(pct_ma) and pct_ma < -5.0:
        if not (_ok(r_3m) and r_3m > 0):
            return "washed_out"

    # 4. Basing: dormant price (flat-ish r_1y AND low recent vol) with
    #    short-term turn up and MA200 flattening or curling up.
    if _ok(r_1y) and abs(r_1y) < 8.0 and _ok(r_3m) and r_3m > 0:
        low_vol_ok = (
            _ok(vol_3m)
            and universe_vol_median is not None
            and vol_3m < universe_vol_median
        )
        slope_ok = _ok(slope) and slope > -0.2
        if low_vol_ok and slope_ok:
            return "basing"

    # 5. Repairing: below to just-above trend, 3m return positive, and NOT
    #    pinned to 52w highs (which would be "strong" or stretched instead).
    if _ok(pct_ma) and -8.0 <= pct_ma <= 5.0 and _ok(r_3m) and r_3m > 0:
        too_deep = _ok(dd) and dd < -25.0
        already_at_highs = _ok(rng) and rng >= 0.90
        if not too_deep and not already_at_highs:
            return "repairing"

    # 6. Strong: trending up, not yet stretched.
    if _ok(pct_ma) and 5.0 <= pct_ma <= 15.0:
        if _ok(rng) and 0.50 <= rng <= 0.85:
            return "strong"
        if not _ok(rng):
            return "strong"

    # 7. Dead money / default.
    return "dead_money"


def _instrument_rationale(regime: str, row: pd.Series) -> str:
    pct_ma = row.get("pct_above_ma200")
    dd = row.get("drawdown_52w")
    rng = row.get("range_52w_pos")
    r_3m = row.get("r_3m")
    r_1y = row.get("r_1y")
    slope = row.get("ma200_slope_20d")

    def _fmt(v, spec):
        if v is None or pd.isna(v):
            return "—"
        return format(v, spec)

    parts = []
    if pct_ma is not None and not pd.isna(pct_ma):
        parts.append(f"{_fmt(pct_ma, '+.1f')}% vs MA200")
    if dd is not None and not pd.isna(dd):
        parts.append(f"dd {_fmt(dd, '.0f')}%")
    if rng is not None and not pd.isna(rng):
        parts.append(f"52w pos {_fmt(rng * 100, '.0f')}%")
    if r_3m is not None and not pd.isna(r_3m):
        parts.append(f"r3m {_fmt(r_3m, '+.0f')}%")
    if r_1y is not None and not pd.isna(r_1y):
        parts.append(f"r1y {_fmt(r_1y, '+.0f')}%")
    if slope is not None and not pd.isna(slope):
        parts.append(f"MA slope {_fmt(slope, '+.2f')}%")
    return f"{regime} — {', '.join(parts) if parts else 'no signals'}"


def classify_instrument_regimes(timing_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker regime classifier using the price/momentum fields.

    Returns the input frame with added ``regime`` and ``regime_rationale`` columns.
    The universe median 3-month volatility is used as the reference for the
    "dormant" (low-vol) test inside ``_classify_instrument``.
    """
    if timing_df.empty:
        return timing_df.assign(regime=pd.Series(dtype=str), regime_rationale=pd.Series(dtype=str))
    out = timing_df.copy()
    universe_vol_median = (
        float(out["vol_3m"].dropna().median())
        if "vol_3m" in out.columns and out["vol_3m"].notna().any()
        else None
    )
    out["regime"] = out.apply(lambda r: _classify_instrument(r, universe_vol_median), axis=1)
    out["regime_rationale"] = out.apply(lambda r: _instrument_rationale(r["regime"], r), axis=1)
    return out


def get_instrument_regimes(timing_df: pd.DataFrame) -> dict[str, str]:
    """Return ``{ticker: regime}`` for the constructor's per-row gating."""
    if timing_df.empty or "ticker" not in timing_df.columns:
        return {}
    labelled = classify_instrument_regimes(timing_df)
    return dict(zip(labelled["ticker"], labelled["regime"]))


def build_theme_return_panel(price_df: pd.DataFrame, include_stocks: bool = True) -> pd.DataFrame:
    """Equal-weight daily return panel per theme."""
    if price_df.empty:
        return pd.DataFrame()

    theme_df = instrument_theme_rows(include_stocks=include_stocks)
    merged = price_df.merge(theme_df, on="ticker", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(["ticker", "date"])
    merged["ret"] = merged.groupby("ticker")["close"].pct_change()
    merged = merged.dropna(subset=["ret"])
    if merged.empty:
        return pd.DataFrame()

    panel = (
        merged.groupby(["date", "theme_label"])["ret"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="theme_label", values="ret")
        .sort_index()
    )
    return panel


def build_theme_correlation(price_df: pd.DataFrame, include_stocks: bool = True, lookback_days: int = 252) -> pd.DataFrame:
    panel = build_theme_return_panel(price_df, include_stocks=include_stocks)
    if panel.empty:
        return pd.DataFrame()
    panel = panel.tail(lookback_days)
    panel = panel.dropna(axis=1, thresh=max(40, lookback_days // 4))
    if panel.shape[1] < 2:
        return pd.DataFrame()
    return panel.corr()


def build_portfolio_weighted_theme_correlation(
    price_df: pd.DataFrame,
    ticker_weights: dict[str, float],
    lookback_days: int = 252,
) -> pd.DataFrame:
    """Theme correlation where each ticker is weighted by its strategic allocation.

    ``ticker_weights`` maps ticker → GBP (or normalised) weight. Each theme's
    daily return is the weight-weighted mean of the returns of its *held*
    tickers. Tickers with zero or missing weight are excluded.

    This collapses a theme with one £40k ETF and one £200 stock into a near
    ETF-only series — which is what we actually care about for portfolio-level
    correlation decisions. It also naturally excludes themes that are not in
    the strategic portfolio.
    """
    if price_df.empty or not ticker_weights:
        return pd.DataFrame()

    theme_df = instrument_theme_rows(include_stocks=True)
    merged = price_df.merge(theme_df, on="ticker", how="inner")
    if merged.empty:
        return pd.DataFrame()

    weights = {str(k): float(v) for k, v in ticker_weights.items() if v and v > 0}
    merged = merged[merged["ticker"].astype(str).isin(weights.keys())]
    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values(["ticker", "date"])
    merged["ret"] = merged.groupby("ticker")["close"].pct_change()
    merged = merged.dropna(subset=["ret"])
    if merged.empty:
        return pd.DataFrame()

    merged["weight"] = merged["ticker"].astype(str).map(weights).fillna(0.0)

    def _weighted_mean(grp: pd.DataFrame) -> float:
        w = grp["weight"].to_numpy()
        r = grp["ret"].to_numpy()
        total = w.sum()
        if total <= 0:
            return float("nan")
        return float(np.average(r, weights=w))

    panel = (
        merged.groupby(["date", "theme_label"], group_keys=False)
        .apply(lambda g: pd.Series({"ret": _weighted_mean(g)}))
        .reset_index()
        .pivot(index="date", columns="theme_label", values="ret")
        .sort_index()
    )
    panel = panel.tail(lookback_days).dropna(axis=1, thresh=max(40, lookback_days // 4))
    if panel.shape[1] < 2:
        return pd.DataFrame()
    return panel.corr()


def build_theme_lookthrough_concentration(
    expanded_df: pd.DataFrame,
    total_portfolio_gbp: float | None = None,
) -> pd.DataFrame:
    """Aggregate true-exposure rows (from ``compute_true_exposure``) up to the theme level.

    The input is the row-per-exposure-leg frame: each row is either a ``direct``
    stock holding, an ``indirect`` ETF underlying, or an unexpanded fund bucket.
    The underlying ticker's sleeve is the concentration key; unexpanded funds
    fall back to the source ETF's sleeve so nothing is dropped.

    Returns one row per (account_type, theme) with direct/indirect/unmapped
    breakdown + % of portfolio. This is what the user needs to see "my SIPP is
    47% US large-cap tech after lookthrough".
    """
    if expanded_df.empty:
        return pd.DataFrame()

    def _theme_for(row: pd.Series) -> str:
        ticker = str(row.get("underlying_ticker") or row.get("source_ticker") or "")
        ins = lookup(ticker)
        if ins is not None and ins.sleeve:
            return sleeve_label(ins.sleeve)
        # unmapped lookthrough legs: attribute to the source ETF's sleeve so the
        # fund's own theme is preserved rather than dropped into "Unknown".
        src = lookup(str(row.get("source_ticker") or ""))
        if src is not None and src.sleeve:
            return sleeve_label(src.sleeve)
        return "Unknown"

    working = expanded_df.copy()
    working["theme_label"] = working.apply(_theme_for, axis=1)
    working["gbp_value"] = working["gbp_value"].astype(float)

    agg = (
        working.groupby(["account_type", "theme_label", "exposure_type"], as_index=False)["gbp_value"]
        .sum()
        .pivot_table(
            index=["account_type", "theme_label"],
            columns="exposure_type",
            values="gbp_value",
            fill_value=0.0,
        )
        .reset_index()
    )
    for col in ("direct", "indirect", "fund_unexpanded", "fund_unmapped"):
        if col not in agg.columns:
            agg[col] = 0.0
    agg["total_gbp"] = agg["direct"] + agg["indirect"] + agg["fund_unexpanded"] + agg["fund_unmapped"]
    agg = agg.rename(
        columns={
            "direct": "direct_gbp",
            "indirect": "indirect_gbp",
            "fund_unexpanded": "unexpanded_gbp",
            "fund_unmapped": "unmapped_gbp",
        }
    )
    if total_portfolio_gbp and total_portfolio_gbp > 0:
        agg["pct_of_portfolio"] = agg["total_gbp"] / float(total_portfolio_gbp) * 100.0
    else:
        agg["pct_of_portfolio"] = None
    return agg.sort_values(["account_type", "total_gbp"], ascending=[True, False]).reset_index(drop=True)


def build_sleeve_ticker_weights(bucket_sizes: dict[str, float]) -> dict[str, float]:
    """Return {ticker: gbp_weight} from the strategic baseline targets.

    Pure function over ``buckets.ALL_BUCKETS`` — does not look at actual
    holdings. Used to drive the portfolio-weighted correlation view so it
    reflects the *plan*, not whatever happens to be in the account today.
    """
    from buckets import ALL_BUCKETS  # local import to avoid circularity

    out: dict[str, float] = {}
    for account, bucket in ALL_BUCKETS.items():
        size = float(bucket_sizes.get(account, 0.0))
        if size <= 0:
            continue
        for target in bucket.targets:
            if target.is_tactical:
                continue
            out[target.primary_ticker] = out.get(target.primary_ticker, 0.0) + size * float(target.weight)
    return out


def build_theme_stock_screen(score_df: pd.DataFrame, timing_df: pd.DataFrame) -> pd.DataFrame:
    """Direct-stock screen grouped by theme instead of geography."""
    if score_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    timing_idx = timing_df.set_index("ticker") if not timing_df.empty else pd.DataFrame()
    for _, row in score_df.iterrows():
        ticker = str(row["Ticker"])
        ins = lookup(ticker)
        if ins is None or ins.vehicle_type != "stock":
            continue
        trow = timing_idx.loc[ticker] if not timing_idx.empty and ticker in timing_idx.index else None
        rows.append(
            {
                "Theme": sleeve_label(ins.sleeve),
                "Ticker": ticker,
                "Name": ins.name,
                "PE percentile": row.get("PE percentile"),
                "PE range pos": row.get("PE range pos"),
                "52W range pos": row.get("52W range pos"),
                "Dist. from 52W low": row.get("Dist. from 52W low"),
                "Price/MA200": row.get("Price/MA200"),
                "Composite": row.get("Composite"),
                "r_1m": None if trow is None or pd.isna(trow.get("r_1m")) else float(trow["r_1m"]),
                "r_3m": None if trow is None or pd.isna(trow.get("r_3m")) else float(trow["r_3m"]),
                "r_6m": None if trow is None or pd.isna(trow.get("r_6m")) else float(trow["r_6m"]),
                "r_1y": None if trow is None or pd.isna(trow.get("r_1y")) else float(trow["r_1y"]),
                "drawdown_52w": None if trow is None or pd.isna(trow.get("drawdown_52w")) else float(trow["drawdown_52w"]),
                "vol_1y": None if trow is None or pd.isna(trow.get("vol_1y")) else float(trow["vol_1y"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["Theme", "52W range pos", "Price/MA200", "PE percentile"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    return out.reset_index(drop=True)

