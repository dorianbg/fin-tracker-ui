from __future__ import annotations

import numpy as np
import pandas as pd


MA_CROSSOVER_COLS: dict[str, str] = {
    "21D MA": "ma_21",
    "63D MA": "ma_63",
    "126D MA": "ma_126",
    "252D MA": "ma_252",
}

Z_SCORE_COLS: dict[str, str] = {
    "1D Z": "z_1d",
    "1W Z": "z_1w",
    "2W Z": "z_2w",
    "1M Z": "z_1mo",
}


def scan_pullbacks(
    df: pd.DataFrame,
    *,
    pullback_ma_col: str = "ma_21",
    pullback_depth: float = 0,
    min_uptrend_strength: float = 0,
    require_intermediate_ok: bool = True,
    best_only: bool = True,
    max_quality_drawdown: float = -20,
    require_bounce: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    pullbacks = df[df["ma_252"] >= min_uptrend_strength].copy()
    if require_intermediate_ok:
        pullbacks = pullbacks[pullbacks["ma_126"] >= 0]
    pullbacks = pullbacks[pullbacks[pullback_ma_col] <= pullback_depth]
    if pullbacks.empty:
        return pullbacks

    pullbacks["bounce_signal"] = pullbacks["r_1w"] > 0
    pullbacks["bounce_or_stabilizing"] = (pullbacks["r_1d"] > 0) | (
        pullbacks["r_1w"] > 0
    )
    pullbacks["trend_score"] = pullbacks["ma_252"].clip(lower=0) + pullbacks[
        "ma_126"
    ].clip(lower=0)
    pullbacks["pullback_depth_score"] = (-pullbacks[pullback_ma_col]).clip(
        lower=0, upper=15
    )
    pullbacks["bounce_score"] = pullbacks["bounce_or_stabilizing"].astype(int) * 10
    pullbacks["drawdown_penalty"] = (-pullbacks["drawdown_52w"] - 20).clip(lower=0)
    pullbacks["breakdown_penalty"] = (-pullbacks["ma_63"] - 5).clip(lower=0) + (
        -pullbacks["ma_126"]
    ).clip(lower=0)
    pullbacks["pullback_score"] = pullbacks["ma_252"] * (-pullbacks[pullback_ma_col])
    pullbacks["quality_score"] = (
        pullbacks["trend_score"]
        + pullbacks["pullback_depth_score"]
        + pullbacks["bounce_score"]
        - pullbacks["drawdown_penalty"]
        - pullbacks["breakdown_penalty"]
    )

    if best_only:
        pullbacks = pullbacks[
            (pullbacks["ma_126"] > 0)
            & (pullbacks["drawdown_52w"] >= max_quality_drawdown)
        ].copy()
        if require_bounce:
            pullbacks = pullbacks[pullbacks["bounce_or_stabilizing"]].copy()
        pullbacks = pullbacks.sort_values("quality_score", ascending=False)
    else:
        pullbacks = pullbacks.sort_values(pullback_ma_col, ascending=True)

    if limit is not None:
        pullbacks = pullbacks.head(limit)
    return pullbacks.reset_index(drop=True)


def scan_laggard_awakening(
    df: pd.DataFrame,
    *,
    benchmark_ticker: str,
    laggard_period: str = "1Y",
    awakening_period: str = "1W",
    underperf_threshold: float = 10,
    max_abs_return: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_cols = {"1Y": "r_1y", "2Y": "r_2y", "3Y": "r_3y", "5Y": "r_5y"}
    rs_period_cols = {"1D": "r_1d", "1W": "r_1w", "1M": "r_1mo"}
    laggard_col = period_cols[laggard_period]
    rs_col = f"rs_{awakening_period}"
    deficit_col = f"rs_{laggard_period}"

    if df.empty or deficit_col not in df.columns or rs_col not in df.columns:
        empty = df.iloc[0:0].copy()
        return empty, empty

    mask = (df[deficit_col] <= -underperf_threshold) & (
        df["ticker"] != benchmark_ticker
    )
    if max_abs_return is not None and laggard_col in df.columns:
        mask = mask & (df[laggard_col].abs() <= max_abs_return)

    laggards = df[mask].copy()
    awakening = laggards[laggards[rs_col] > 0].copy()
    sleeping = laggards[laggards[rs_col] <= 0].copy()
    if not awakening.empty:
        awakening["awakening_score"] = awakening[rs_col] * (-awakening[deficit_col])
        awakening = awakening.sort_values("awakening_score", ascending=False)
    return awakening, sleeping


def scan_laggard_breakout_confirmations(
    laggards: pd.DataFrame, *, awakening_period: str = "1W"
) -> pd.DataFrame:
    rs_col = f"rs_{awakening_period}"
    if laggards.empty or rs_col not in laggards.columns:
        return laggards.iloc[0:0].copy()
    confirmed = laggards[(laggards["ma_21"] > 0) & (laggards[rs_col] > 0)].copy()
    if confirmed.empty:
        return confirmed
    confirmed["above_63d"] = confirmed["ma_63"] > 0
    confirmed["above_252d"] = confirmed["ma_252"] > 0
    confirmed["ma_cross_count"] = (
        (confirmed["ma_21"] > 0).astype(int)
        + (confirmed["ma_63"] > 0).astype(int)
        + (confirmed["ma_126"] > 0).astype(int)
        + (confirmed["ma_252"] > 0).astype(int)
    )
    return confirmed.sort_values("ma_cross_count", ascending=False)


def scan_puke_capitulation(
    df: pd.DataFrame,
    *,
    vol_ratio_threshold: float,
    drawdown_threshold: float,
    drawdown_col: str = "drawdown_52w",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = df.copy()
        return empty, empty

    work = df.copy()
    if "vol_ratio" not in work.columns:
        work["vol_ratio"] = work["vol_1mo"] / work["vol_1y"].replace(0, np.nan)

    dd_source = drawdown_col if drawdown_col in work.columns else "ma_252"
    mask_strict = (work["vol_ratio"] >= vol_ratio_threshold) & (
        work[dd_source] <= drawdown_threshold
    )
    vol_relaxed = max(0.8, vol_ratio_threshold * 0.8)
    dd_relaxed = min(-5.0, drawdown_threshold * 0.75)
    mask_relaxed = (
        (work["vol_ratio"] >= vol_relaxed)
        & (~mask_strict)
        & (work[dd_source] <= dd_relaxed)
    )

    def _score(candidates: pd.DataFrame) -> pd.DataFrame:
        out = candidates.copy()
        if out.empty:
            return out
        dd_val = out[dd_source].fillna(0)
        out["severity"] = (-dd_val) * out["vol_ratio"].fillna(1.0)
        out["bounce_starting"] = out["r_1w"] > 0
        return out.sort_values("severity", ascending=False, ignore_index=True)

    return _score(work[mask_strict]), _score(work[mask_relaxed])


def scan_puke_buy_candidates(
    active_cap: pd.DataFrame, cap_watchlist: pd.DataFrame | None = None
) -> pd.DataFrame:
    if active_cap.empty:
        return active_cap.copy()
    buys = active_cap[active_cap.get("bounce_starting", False)].copy()
    if buys.empty and cap_watchlist is not None and not cap_watchlist.empty:
        buys = cap_watchlist[cap_watchlist.get("bounce_starting", False)].copy()
    if buys.empty:
        return buys
    return buys.sort_values("severity", ascending=False).reset_index(drop=True)


def split_latest_two_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    today = raw[raw["rown"] == 1].drop_duplicates("ticker").set_index("ticker")
    yesterday = raw[raw["rown"] == 2].drop_duplicates("ticker").set_index("ticker")
    common = today.index.intersection(yesterday.index)
    return today.loc[common], yesterday.loc[common]


def scan_new_52w_highs(today: pd.DataFrame, yesterday: pd.DataFrame) -> pd.DataFrame:
    new_highs = today[
        (today["drawdown_52w"] >= -0.5) & (yesterday["drawdown_52w"] < -0.5)
    ].copy()
    if not new_highs.empty:
        new_highs["prev_drawdown"] = yesterday.loc[new_highs.index, "drawdown_52w"]
    return new_highs


def scan_ma_crossovers(
    today: pd.DataFrame, yesterday: pd.DataFrame, *, include_bearish: bool = True
) -> pd.DataFrame:
    common = today.index.intersection(yesterday.index)
    rows = []
    for label, col in MA_CROSSOVER_COLS.items():
        for ticker in common:
            today_val = today.loc[ticker, col]
            yest_val = yesterday.loc[ticker, col]
            if pd.isna(today_val) or pd.isna(yest_val):
                continue
            base = {
                "Ticker": ticker,
                "Instrument": today.loc[ticker, "description"],
                "MA": label.replace("D MA", "d MA"),
                "Today": today_val,
                "Yesterday": yest_val,
                "1D Return": today.loc[ticker, "r_1d"],
            }
            if today_val > 0 >= yest_val:
                rows.append({**base, "Direction": "🟢 Crossed ABOVE"})
            elif include_bearish and today_val < 0 <= yest_val:
                rows.append({**base, "Direction": "🔴 Crossed BELOW"})
    return pd.DataFrame(rows)


def scan_z_score_spikes(today: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    rows = []
    for ticker, row in today.iterrows():
        for label, col in Z_SCORE_COLS.items():
            z_val = row.get(col)
            if pd.notna(z_val) and z_val >= z_threshold:
                rows.append(
                    {
                        "Ticker": ticker,
                        "Instrument": row.get("description"),
                        "Metric": label,
                        "Z-Score": z_val,
                        "1D Return": row.get("r_1d"),
                        "1W Return": row.get("r_1w"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Z-Score", ascending=False)


def scan_todays_alert_crossings(
    raw: pd.DataFrame, *, z_threshold: float = 2.0
) -> pd.DataFrame:
    today, yesterday = split_latest_two_rows(raw)
    rows = []

    for ticker, row in scan_new_52w_highs(today, yesterday).iterrows():
        rows.append(
            {
                **row.to_dict(),
                "ticker": ticker,
                "signal": "New 52W high",
                "score": row.get("r_1d", 0),
                "summary": "Fresh 52-week high or within 0.5% after not being there yesterday.",
            }
        )

    bullish_crosses = scan_ma_crossovers(today, yesterday, include_bearish=False)
    for _, row in bullish_crosses.iterrows():
        ticker = row["Ticker"]
        source = today.loc[ticker]
        ma_label = str(row["MA"]).replace("d", "D")
        rows.append(
            {
                **source.to_dict(),
                "ticker": ticker,
                "signal": f"Bullish {ma_label} crossover",
                "score": row["Today"],
                "summary": f"Crossed above {ma_label}: {float(row['Yesterday']):+.2f}% → {float(row['Today']):+.2f}%.",
            }
        )

    z_spikes = scan_z_score_spikes(today, z_threshold)
    for _, row in z_spikes.iterrows():
        ticker = row["Ticker"]
        source = today.loc[ticker]
        rows.append(
            {
                **source.to_dict(),
                "ticker": ticker,
                "signal": "Z-score spike",
                "score": row["Z-Score"],
                "summary": f"Unusual move: {row['Metric']} at {float(row['Z-Score']):.2f}σ, 1D return {float(row['1D Return']):+.2f}%.",
            }
        )

    active = pd.DataFrame(rows)
    if not active.empty:
        active = active.sort_values("score", ascending=False)
    return active
