from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.alerts.session import add_alert_ticker, filter_by_session
from app.views.ConsolidationSetup import (
    scan_breakout_triggers,
    scan_consolidation_setups,
)
from app.views.SectorRotation import (
    SECTOR_UNIVERSES,
    current_sector_ranks,
    normalise_prices,
)
from app.strategy_scanners import (
    scan_elite_relative_strength,
    scan_laggard_awakening,
    scan_leaders_weakening,
    scan_pullbacks,
    scan_todays_alert_crossings,
)


@dataclass(frozen=True)
class StrategySignals:
    strategy_id: str
    title: str
    commentary: str
    active: pd.DataFrame


def _finalize(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_alert_ticker(df)
    # deduplicate by display ticker + signal — guards against transient data duplicates
    dedup_cols = [c for c in ("alert_ticker", "ticker") if c in out.columns]
    if "signal" in out.columns:
        dedup_cols.append("signal")
    out = out.drop_duplicates(subset=dedup_cols).copy()
    if limit > 0:
        out = out.head(limit)
    out = out.reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    if "signal" not in out.columns:
        out["signal"] = out.get("strategy", "signal")
    return out


def _bounded_limit(limit: int, default_n: int) -> int:
    return min(limit, default_n) if limit > 0 else 0


def _fund_filter(
    df: pd.DataFrame, prefixes: tuple[str, ...] = ("eq", "stock")
) -> pd.DataFrame:
    if df.empty or "fund_type" not in df.columns:
        return df.copy()
    pattern = "^(?:" + "|".join(prefixes) + ")"
    return df[df["fund_type"].fillna("").str.contains(pattern, regex=True)].copy()


def _score_summary(row: pd.Series, text: str) -> str:
    return text.format(**{k: row.get(k, np.nan) for k in row.index})


def _finite_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _attach_ticker_full(df: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker_full" in df.columns or "ticker_full" not in source.columns:
        return df
    meta_cols = ["ticker", "ticker_full"]
    meta = source.sort_values("date").drop_duplicates("ticker", keep="last")
    return df.merge(meta[meta_cols], on="ticker", how="left")


def breakout_signals(prices: pd.DataFrame, session: str, limit: int) -> StrategySignals:
    scoped = filter_by_session(prices, session)
    alerts = scan_breakout_triggers(scoped)
    alerts = _attach_ticker_full(alerts, scoped)
    if not alerts.empty:
        alerts = alerts.rename(columns={"breakout_score": "score"})
        alerts["signal"] = "Fresh resistance breakout"
        alerts["summary"] = alerts.apply(
            lambda r: _score_summary(
                r,
                "Close crossed prior 30-day resistance; extension {breakout_extension_adr:.2f} ADR above breakout and {extension_adr:.1f} ADR from 200D MA.",
            ),
            axis=1,
        )
    return StrategySignals(
        "breakout",
        "FinTracker breakout alerts",
        "Fresh breakouts above prior consolidation resistance. These are execution/watchlist alerts, not long-term allocation advice.",
        _finalize(alerts, limit),
    )


def consolidation_signals(
    prices: pd.DataFrame, session: str, limit: int
) -> StrategySignals:
    scoped = _fund_filter(filter_by_session(prices, session))
    setups = scan_consolidation_setups(scoped)
    setups = _attach_ticker_full(setups, scoped)
    if not setups.empty:
        setups = setups.rename(columns={"setup_score": "score"})
        setups["signal"] = "Bull consolidation near resistance"
        setups["summary"] = setups.apply(
            lambda r: _score_summary(
                r,
                "Bull regime, tight range, {breakout_gap_adr:.2f} ADR below breakout, {extension_adr:.1f} ADR from 200D MA.",
            ),
            axis=1,
        )
    return StrategySignals(
        "consolidation",
        "FinTracker consolidation setup alerts",
        "Bull-market assets coiling near resistance before a possible breakout.",
        _finalize(setups, limit),
    )


def rotation_signals(
    latest: pd.DataFrame, session: str, limit: int
) -> list[StrategySignals]:
    df = _fund_filter(
        filter_by_session(latest, session),
        ("eq", "eq-reit", "commod", "bonds", "stock"),
    )
    results = []
    sharpe = df.copy()
    if not sharpe.empty:
        if "r_3mo_s" in sharpe.columns:
            sharpe["sharpe_score"] = sharpe["r_3mo_s"]
        else:
            sharpe["sharpe_score"] = sharpe["r_3mo"] / sharpe["vol_1y"].replace(
                0, np.nan
            )
        for strategy_id, title, ascending, default_n, commentary in [
            (
                "rotation_vol_adjusted",
                "Rotation Vol-Adjusted",
                False,
                5,
                "Best 3M return per unit of 1Y volatility.",
            ),
            (
                "rotation_puke",
                "Rotation Puke Contrarian",
                True,
                10,
                "Worst risk-adjusted names for contrarian distress monitoring.",
            ),
        ]:
            active = (
                sharpe.dropna(subset=["sharpe_score"])
                .sort_values("sharpe_score", ascending=ascending)
                .copy()
            )
            active["score"] = active["sharpe_score"]
            active["signal"] = title
            active["summary"] = active.apply(
                lambda r: f"3M/1Y-vol score {float(r['sharpe_score']):+.2f}.", axis=1
            )
            results.append(
                StrategySignals(
                    strategy_id,
                    f"FinTracker {title} alerts",
                    commentary,
                    _finalize(active, _bounded_limit(limit, default_n)),
                )
            )
    return results


def sector_rotation_signals(
    prices: pd.DataFrame, session: str, limit: int
) -> StrategySignals:
    if session == "asia":
        return StrategySignals(
            "sector_rotation",
            "FinTracker sector rotation alerts",
            "Sector rotation alerts are currently configured for US and EU/UK sector ETF universes only.",
            pd.DataFrame(),
        )
    universe_name = (
        "Global UCITS Sector ETFs" if session == "eu" else "US Select Sector SPDRs"
    )
    universe = SECTOR_UNIVERSES[universe_name]
    scoped = normalise_prices(prices)
    try:
        ranks = current_sector_ranks(scoped, list(universe["tickers"]))
    except (IndexError, KeyError, ValueError):
        ranks = pd.DataFrame()
    if not ranks.empty:
        meta = scoped.sort_values("date").drop_duplicates("ticker", keep="last")
        ranks = ranks.merge(
            meta[["ticker", "description", "fund_type"]], on="ticker", how="left"
        )
        ranks["score"] = ranks["rs_score"]
        ranks["signal"] = f"{universe_name} top sector rank"
        ranks["summary"] = ranks.apply(
            lambda r: (
                f"Composite 1/3/6/9/12M relative-strength score {float(r['rs_score']):+.2f}."
            ),
            axis=1,
        )
    return StrategySignals(
        "sector_rotation",
        "FinTracker sector rotation alerts",
        f"Current Faber-style sector rotation leaders for {universe_name}; default top sectors are equal-weight candidates.",
        _finalize(ranks, min(limit, 5)),
    )


def puke_buy_signals(latest: pd.DataFrame, session: str, limit: int) -> StrategySignals:
    df = _fund_filter(filter_by_session(latest, session))
    z_cols = [c for c in ["z_1d", "z_1w", "z_2w", "z_1mo"] if c in df.columns]
    if df.empty or not z_cols:
        active = pd.DataFrame()
    else:
        active = df.copy()
        active["max_z"] = active[z_cols].max(axis=1)
        active["vol_ratio"] = active["vol_1mo"] / active["vol_1y"].replace(0, np.nan)
        active = active[(active["max_z"] >= 2.0) & (active["r_1w"] > 0)].copy()
        active["score"] = active["max_z"] * active["r_1w"].clip(lower=0)
        active = active.sort_values("score", ascending=False)
        active["signal"] = "Puke bounce buy signal"
        active["summary"] = active.apply(
            lambda r: (
                f"Stress z-score {float(r['max_z']):.2f} with positive 1W bounce {float(r['r_1w']):+.2f}%."
            ),
            axis=1,
        )
    return StrategySignals(
        "puke_buy",
        "FinTracker puke buy alerts",
        "Capitulation or stress names that have started to bounce.",
        _finalize(active, limit),
    )


def pullback_signals(latest: pd.DataFrame, session: str, limit: int) -> StrategySignals:
    df = _fund_filter(filter_by_session(latest, session))
    active = scan_pullbacks(
        df,
        pullback_ma_col="ma_21",
        pullback_depth=0,
        min_uptrend_strength=0,
        require_intermediate_ok=True,
        best_only=True,
        max_quality_drawdown=-20,
        require_bounce=True,
    )
    if not active.empty:
        active["score"] = active["quality_score"]
        active["signal"] = "Uptrend pullback stabilising"
        active["summary"] = active.apply(
            lambda r: (
                f"Above 252D/126D trend, below 21D by {float(r['ma_21']):+.2f}%, with early bounce evidence."
            ),
            axis=1,
        )
    return StrategySignals(
        "pullback",
        "FinTracker pullback alerts",
        "Uptrending assets pulling back to short-term support and beginning to stabilize.",
        _finalize(active, limit),
    )


def laggard_signals(
    latest: pd.DataFrame, session: str, limit: int, benchmark: str = "VWRP"
) -> StrategySignals:
    df = _fund_filter(filter_by_session(latest, session))
    bm = latest[latest["ticker"] == benchmark]
    if df.empty or bm.empty:
        active = pd.DataFrame()
    else:
        bm_1y = float(bm["r_1y"].iloc[0])
        bm_1w = float(bm["r_1w"].iloc[0])
        rs_df = df.copy()
        rs_df["rs_1Y"] = rs_df["r_1y"] - bm_1y
        rs_df["rs_1W"] = rs_df["r_1w"] - bm_1w
        active, _ = scan_laggard_awakening(
            rs_df,
            benchmark_ticker=benchmark,
            laggard_period="1Y",
            awakening_period="1W",
            underperf_threshold=10,
        )
        active = active.rename(columns={"awakening_score": "score", "rs_1W": "rs_1w"})
        active["signal"] = "Laggard awakening"
        active["summary"] = active.apply(
            lambda r: (
                f"1Y laggard now outperforming {benchmark} by {float(r['rs_1w']):+.2f}% over 1W."
            ),
            axis=1,
        )
    return StrategySignals(
        "laggard_awakening",
        "FinTracker laggard awakening alerts",
        "Long-term underperformers starting to show short-term relative strength.",
        _finalize(active, limit),
    )


def _first_local_high_window(ticker_prices: pd.DataFrame) -> int | None:
    if ticker_prices.empty:
        return None
    closes = ticker_prices.sort_values("date")["price"]
    latest_close = float(closes.iloc[-1])
    for window in (60, 40, 20):
        if (
            len(closes) >= window
            and latest_close >= float(closes.tail(window).max()) * 0.999
        ):
            return window
    return None


def _first_near_local_high_window(
    ticker_prices: pd.DataFrame, tolerance: float
) -> tuple[int, float] | None:
    if ticker_prices.empty:
        return None
    closes = ticker_prices.sort_values("date")["price"]
    latest_close = float(closes.iloc[-1])
    for window in (60, 40, 20):
        if len(closes) < window:
            continue
        local_high = float(closes.tail(window).max())
        if local_high <= 0:
            continue
        distance = (latest_close / local_high - 1) * 100
        if latest_close >= local_high * tolerance:
            return window, distance
    return None


def _range_pct(ticker_prices: pd.DataFrame, window: int) -> float:
    closes = ticker_prices.sort_values("date")["price"].tail(window)
    if closes.empty:
        return np.nan
    low = float(closes.min())
    high = float(closes.max())
    if low <= 0:
        return np.nan
    return (high / low - 1) * 100


def _drawdown_from_high(ticker_prices: pd.DataFrame, window: int) -> float:
    closes = ticker_prices.sort_values("date")["price"].tail(window)
    if closes.empty:
        return np.nan
    high = float(closes.max())
    if high <= 0:
        return np.nan
    return (float(closes.iloc[-1]) / high - 1) * 100


def momentum_breakout_signals(
    latest: pd.DataFrame, prices: pd.DataFrame, session: str, limit: int
) -> StrategySignals:
    df = _fund_filter(filter_by_session(latest, session))
    if df.empty or prices.empty:
        active = pd.DataFrame()
    else:
        rows = []
        for _, row in df.iterrows():
            ticker = row.get("ticker")
            ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")
            local_high_window = _first_local_high_window(ticker_prices)
            if local_high_window is None:
                continue
            drawdown_52w = row.get("drawdown_52w", np.nan)
            drawdown_104w = _drawdown_from_high(ticker_prices, 504)
            range_20d = _range_pct(ticker_prices, 20)
            range_60d = _range_pct(ticker_prices, 60)
            r_1w = row.get("r_1w", 0)
            r_1mo = row.get("r_1mo", 0)
            positive_momentum = (
                max(
                    float(r_1w) if pd.notna(r_1w) else 0,
                    float(r_1mo) if pd.notna(r_1mo) else 0,
                )
                > 0
            )
            if not positive_momentum:
                continue

            setup_type = None
            if (pd.notna(drawdown_52w) and drawdown_52w <= -10) or (
                pd.notna(drawdown_104w) and drawdown_104w <= -10
            ):
                setup_type = "Recovery breakout"
            elif (
                pd.notna(drawdown_52w)
                and -10 < drawdown_52w <= 0.5
                and pd.notna(range_20d)
                and range_20d <= 10
                and (
                    pd.isna(row.get("vol_1mo"))
                    or pd.isna(row.get("vol_1y"))
                    or row.get("vol_1mo") <= row.get("vol_1y")
                )
            ):
                setup_type = "Base breakout near highs"
            if setup_type is None:
                continue

            out = row.to_dict()
            out["local_high_window"] = local_high_window
            out["drawdown_104w"] = drawdown_104w
            out["range_20d"] = range_20d
            out["range_60d"] = range_60d
            out["setup_type"] = setup_type
            rows.append(out)

        active = pd.DataFrame(rows)
        if not active.empty:
            active["score"] = active.apply(
                lambda r: (
                    _finite_float(r.get("local_high_window")) / 20 * 10
                    + max(_finite_float(r.get("r_1w")), 0)
                    + max(_finite_float(r.get("r_1mo")), 0)
                    + (
                        max(
                            -_finite_float(r.get("drawdown_52w")),
                            -_finite_float(r.get("drawdown_104w")),
                        )
                        / 2
                        if r["setup_type"] == "Recovery breakout"
                        else max(10 - _finite_float(r.get("range_20d"), 10), 0)
                    )
                ),
                axis=1,
            )
            active = active.sort_values("score", ascending=False)
            active["signal"] = active["setup_type"]
            active["summary"] = active.apply(
                lambda r: (
                    f"{r['setup_type']}: {int(r['local_high_window'] / 5)}W local high; "
                    f"52W drawdown {float(r['drawdown_52w']):+.1f}%, "
                    f"104W drawdown {float(r['drawdown_104w']):+.1f}%, "
                    f"20D range {float(r['range_20d']):.1f}%, "
                    f"1W {float(r.get('r_1w', 0)):+.2f}%, 1M {float(r.get('r_1mo', 0)):+.2f}%."
                ),
                axis=1,
            )
    return StrategySignals(
        "momentum_breakout",
        "FinTracker momentum breakout alerts",
        "Stocks and ETFs making 4–12 week local highs, split into recovery breakouts below prior peaks and tight base breakouts near 52-week highs.",
        _finalize(active, limit),
    )


def turnaround_signals(
    latest: pd.DataFrame, prices: pd.DataFrame, session: str, limit: int
) -> StrategySignals:
    df = _fund_filter(filter_by_session(latest, session))
    if df.empty or prices.empty:
        active = pd.DataFrame()
    else:
        candidates = df[df["drawdown_52w"] < -10].copy()
        rows = []
        for ticker in candidates["ticker"].unique():
            ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")
            near_high = _first_near_local_high_window(ticker_prices, tolerance=0.98)
            if near_high is None:
                continue
            local_high_window, distance_to_local_high = near_high
            row = candidates[candidates["ticker"] == ticker].iloc[0].to_dict()
            row["local_high_window"] = local_high_window
            row["distance_to_local_high"] = distance_to_local_high
            rows.append(row)
        active = pd.DataFrame(rows)
        if not active.empty:
            active["score"] = -active["drawdown_52w"] * active["r_1w"].clip(lower=0)
            active = active.sort_values("score", ascending=False)
            active["signal"] = "Turnaround: near 4-12W high but still below 52W peak"
            active["summary"] = active.apply(
                lambda r: (
                    f"Within {float(r['distance_to_local_high']):+.1f}% of {int(r['local_high_window'] / 5)}W high, "
                    f"still {float(r['drawdown_52w']):+.1f}% below 52W high, "
                    f"1W return {float(r['r_1w']):+.2f}%."
                ),
                axis=1,
            )
    return StrategySignals(
        "turnaround",
        "FinTracker turnaround alerts",
        "Equities within 2% of 4–12 week highs while still down more than 10% from their 52-week peak — focused recovery watchlist.",
        _finalize(active, limit),
    )


def todays_crossings_signals(
    raw: pd.DataFrame, session: str, limit: int, z_threshold: float = 2.0
) -> StrategySignals:
    scoped = filter_by_session(raw, session)
    active = scan_todays_alert_crossings(scoped, z_threshold=z_threshold)
    return StrategySignals(
        "todays_crossings",
        "FinTracker today's crossings alerts",
        "Fresh threshold crossings: new highs, bullish MA crosses, and unusual z-score moves.",
        _finalize(active, limit),
    )


def leaders_weakening_signals(
    latest: pd.DataFrame, session: str, limit: int, benchmark: str = "VWRP"
) -> StrategySignals:
    df = _fund_filter(filter_by_session(latest, session))
    active = scan_leaders_weakening(df, benchmark_ticker=benchmark)
    if not active.empty:
        active["score"] = active["weakening_score"]
        active["signal"] = "Leader weakening"
        active["summary"] = active.apply(
            lambda r: (
                f"6M {float(r['r_6mo']):+.1f}% → 1W {float(r['r_1w']):+.2f}%, "
                f"below 21D MA by {float(r['ma_21']):+.2f}%, 52W dd {float(r['drawdown_52w']):+.1f}%."
            ),
            axis=1,
        )
    return StrategySignals(
        "leaders_weakening",
        "FinTracker leaders weakening alerts",
        "Former momentum leaders showing early breakdown signals: strong 6M history but recent 1W weakness and price below 21D MA.",
        _finalize(active, limit),
    )


def _relative_strength_benchmark(latest: pd.DataFrame, session: str) -> str:
    candidates = ("SPY", "CSP1", "VWRP") if session == "us" else ("VWRP", "CSP1", "SPY")
    tickers = set(latest.get("ticker", pd.Series(dtype=str)).dropna().astype(str))
    for ticker in candidates:
        if ticker in tickers:
            return ticker
    return "VWRP"


def elite_relative_strength_signals(
    latest: pd.DataFrame, prices: pd.DataFrame, session: str, limit: int
) -> StrategySignals:
    scoped_latest = _fund_filter(filter_by_session(latest, session), ("eq", "stock"))
    scoped_prices = filter_by_session(prices, session)
    benchmark = _relative_strength_benchmark(latest, session)
    benchmark_latest = latest[latest["ticker"] == benchmark]
    benchmark_prices = prices[prices["ticker"] == benchmark]
    scan_latest = pd.concat([scoped_latest, benchmark_latest], ignore_index=True)
    scan_prices = pd.concat([scoped_prices, benchmark_prices], ignore_index=True)
    scan_latest = scan_latest.drop_duplicates(subset=["ticker"], keep="last")
    price_dedup_cols = [
        col for col in ("ticker_full", "ticker", "date") if col in scan_prices.columns
    ]
    scan_prices = scan_prices.drop_duplicates(subset=price_dedup_cols, keep="last")
    active = scan_elite_relative_strength(
        scan_latest,
        scan_prices,
        benchmark_ticker=benchmark,
    )
    if not active.empty:
        active["score"] = active["elite_rs_score"]

        def _signal(row: pd.Series) -> str:
            if row["rs_new_high_before_price"]:
                return "RS new high before price"
            if row["resilient_during_index_pullback"]:
                return "Resilient during index pullback"
            return "RS line 52W high"

        active["signal"] = active.apply(_signal, axis=1)
        active["summary"] = active.apply(
            lambda r: (
                f"RS {float(r['rs_1y_percentile']):.0f}th percentile vs {benchmark}; "
                f"1Y relative {float(r['rs_1y']):+.1f}%, 3M relative {float(r['rs_3mo']):+.1f}%, "
                f"52W drawdown {float(r['drawdown_52w']):+.1f}%."
            ),
            axis=1,
        )
    return StrategySignals(
        "elite_relative_strength",
        "FinTracker elite relative strength alerts",
        "Top relative-strength stocks and equity ETFs showing RS-line breakouts or resilience while the benchmark is weak.",
        _finalize(active, limit),
    )


def build_all_signals(
    latest: pd.DataFrame,
    raw_two_rows: pd.DataFrame,
    prices: pd.DataFrame,
    session: str,
    limit: int,
    selected: set[str] | None = None,
) -> list[StrategySignals]:
    # breakout_signals is handled by the standalone send_breakout_alerts.py
    # to avoid duplicate emails from the same scan_breakout_triggers scanner.
    builders = [
        ("consolidation", lambda: consolidation_signals(prices, session, limit)),
        ("sector_rotation", lambda: sector_rotation_signals(prices, session, limit)),
        (
            "elite_relative_strength",
            lambda: elite_relative_strength_signals(latest, prices, session, limit),
        ),
        ("puke_buy", lambda: puke_buy_signals(latest, session, limit)),
        ("pullback", lambda: pullback_signals(latest, session, limit)),
        ("laggard_awakening", lambda: laggard_signals(latest, session, limit)),
        ("turnaround", lambda: turnaround_signals(latest, prices, session, limit)),
        (
            "momentum_breakout",
            lambda: momentum_breakout_signals(latest, prices, session, limit),
        ),
        (
            "leaders_weakening",
            lambda: leaders_weakening_signals(latest, session, limit),
        ),
    ]
    signals = [
        builder()
        for strategy_id, builder in builders
        if selected is None or strategy_id in selected
    ]
    rotation_ids = {
        "rotation_vol_adjusted",
        "rotation_puke",
    }
    if selected is None or selected.intersection(rotation_ids):
        signals.extend(
            strategy
            for strategy in rotation_signals(latest, session, limit)
            if selected is None or strategy.strategy_id in selected
        )
    return signals
