"""
RAAM Strategy — Real Asset Allocation Model
Based on 3Fourteen Research: "Real Asset Allocation: The World Has Changed" (May 2023).

Core mechanics:
  - 20-asset expanded universe (Alternatives + Equities + Fixed Income)
  - Regression-based trend breadth, strength, and mean-reversion scoring
  - Cross-sectional trend ranking → benchmark-anchored weight multipliers
  - Two-level hierarchical risk parity (bucket-level + within-bucket)
  - 55% trend / 45% HRP final weight blend
  - Monthly rebalance, no lookahead
  - Backtest vs 60/40 (60% SPY / 40% TLT) and static RAAM benchmark
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import altair as alt
import config as cfg
from data import load_prices


# ─────────────────────────────────────────────────────────────────
# SECTION 1: Asset Configuration
# ─────────────────────────────────────────────────────────────────

ASSETS = {
    # ── Alternatives ──
    "Bitcoin":         {"proxy": "IBIT",  "bucket": "Alternatives", "benchmark": 0.02, "max": 0.03},
    "Commodities":     {"proxy": "PDBC",  "bucket": "Alternatives", "benchmark": 0.04, "max": 0.16},
    "Energy":          {"proxy": "XLE",   "bucket": "Alternatives", "benchmark": 0.02, "max": 0.10},
    "Gold":            {"proxy": "GLD",   "bucket": "Alternatives", "benchmark": 0.02, "max": 0.10},
    "Managed Futures": {"proxy": "DBMF",  "bucket": "Alternatives", "benchmark": 0.06, "max": 0.16},
    "Miners":          {"proxy": "PICK",  "bucket": "Alternatives", "benchmark": 0.02, "max": 0.10},
    "Real Estate":     {"proxy": "VNQ",   "bucket": "Alternatives", "benchmark": 0.02, "max": 0.16},
    # ── Equities ──
    "Dividend Payers": {"proxy": "SCHD",  "bucket": "Equities",     "benchmark": 0.05, "max": 0.10},
    "EM ex-China":     {"proxy": "EMXC",  "bucket": "Equities",     "benchmark": 0.01, "max": 0.10},
    "Europe":          {"proxy": "VGK",   "bucket": "Equities",     "benchmark": 0.01, "max": 0.10},
    "Japan":           {"proxy": "EWJ",   "bucket": "Equities",     "benchmark": 0.01, "max": 0.10},
    "Nasdaq":          {"proxy": "QQQ",   "bucket": "Equities",     "benchmark": 0.20, "max": 0.40},
    "US Large Cap":    {"proxy": "SPY",   "bucket": "Equities",     "benchmark": 0.20, "max": 0.40},
    "US Small Cap":    {"proxy": "IWM",   "bucket": "Equities",     "benchmark": 0.02, "max": 0.10},
    # ── Fixed Income ──
    "Corporate Bonds":     {"proxy": "VCIT",  "bucket": "Fixed Income", "benchmark": 0.10, "max": 0.30},
    "EM Bonds":            {"proxy": "EMB",   "bucket": "Fixed Income", "benchmark": 0.02, "max": 0.10},
    "High Yield":          {"proxy": "HYG",   "bucket": "Fixed Income", "benchmark": 0.06, "max": 0.20},
    "Long-Term Treasuries":{"proxy": "TLT",   "bucket": "Fixed Income", "benchmark": 0.10, "max": 0.40},
    "T-Bills":             {"proxy": "BIL",   "bucket": "Fixed Income", "benchmark": 0.00, "max": 0.20},
    "TIPS":                {"proxy": "TIP",   "bucket": "Fixed Income", "benchmark": 0.02, "max": 0.40},
}

BUCKETS = ["Alternatives", "Equities", "Fixed Income"]

TREND_WINDOWS = [21, 42, 63, 126, 189, 252]
WINDOW_WEIGHTS = {21: 0.30, 42: 0.25, 63: 0.20, 126: 0.12, 189: 0.08, 252: 0.05}
TREND_BLEND = (0.50, 0.50, 0.00)       # breadth, strength, mean_reversion (MR disabled)
TREND_MULT_LOW  = 0.05                   # floor fraction when rank=0 (5% of benchmark)
TREND_MULT_HIGH = 2.50                   # unused (kept for reference)
TREND_HRP_BLEND = 0.55                   # 55% trend, 45% HRP

RISK_FREE_RATE  = cfg.RISK_FREE_RATE     # 0.05
MIN_HISTORY     = 252                    # days required before asset is eligible
VOL_LOOKBACK    = 126                    # days for trailing volatility estimate

BENCHMARK_60   = ("SPY", 0.60)
BENCHMARK_40   = ("TLT", 0.40)


# ─────────────────────────────────────────────────────────────────
# SECTION 2: Data Loading
# ─────────────────────────────────────────────────────────────────

def _all_proxies():
    """Return list of every proxy ticker used by the model + benchmarks."""
    raam = [cfg["proxy"] for cfg in ASSETS.values()]
    bench = [BENCHMARK_60[0], BENCHMARK_40[0]]
    return list(dict.fromkeys(raam + bench))  # unique, order-preserving


@st.cache_data(ttl=300)
def _load_raw_prices():
    """Load raw price history from DuckDB for all needed tickers."""
    tickers = tuple(_all_proxies())
    df = load_prices(tickers)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _prepare_prices(raw_df):
    """Pivot raw prices to wide DataFrame: rows=dates, cols=tickers."""
    prices_wide = raw_df.pivot_table(
        index="date", columns="ticker", values="price", aggfunc="last"
    )
    prices_wide = prices_wide.sort_index()
    prices_wide = prices_wide.ffill()
    return prices_wide


@st.cache_data(ttl=600)
def load_raam_data():
    """
    Load and prepare all data needed by the strategy.
    Returns (prices_wide, returns_wide, rebalance_dates).
    """
    raw = _load_raw_prices()
    prices = _prepare_prices(raw)
    returns = prices.pct_change().dropna(how="all")
    rebalance_dates = _month_end_dates(prices.index)
    return prices, returns, rebalance_dates


def _month_end_dates(date_index):
    """Find the last business day of each month in the index."""
    s = pd.Series(date_index, index=date_index)
    month_end_mask = s.groupby(s.dt.to_period("M")).transform("max") == s
    return s[month_end_mask].tolist()


# ─────────────────────────────────────────────────────────────────
# SECTION 3: Trend Engine
# ─────────────────────────────────────────────────────────────────

def _regression_metrics(log_prices, window):
    """
    Run OLS regression of log(price) ~ time for the last `window` observations.
    Returns (slope, r_squared, predicted_last, residual_zscore).
    Returns NaN tuple if insufficient data.
    """
    y = log_prices.tail(window).values
    n = len(y)
    if n < max(10, window // 4):
        return (np.nan, np.nan, np.nan, np.nan)
    x = np.arange(n, dtype=float)
    result = sp_stats.linregress(x, y)
    slope = result.slope
    r2 = result.rvalue ** 2
    intercept = result.intercept
    predicted_last = intercept + slope * (n - 1)
    residuals = y - (intercept + slope * x)
    residual_std = np.std(residuals, ddof=1) if len(residuals) > 1 else np.nan
    residual_z = (y[-1] - predicted_last) / residual_std if residual_std and residual_std > 0 else 0.0
    return (slope, r2, predicted_last, residual_z)


def _trend_features_for_date(prices_wide, eval_date, log_prices_all):
    """
    Compute trend breadth, strength, and mean-reversion for every asset
    at a single evaluation date, using only data through eval_date.
    """
    rows = []
    for asset_name, cfg in ASSETS.items():
        proxy = cfg["proxy"]
        if proxy not in prices_wide.columns:
            continue
        series = prices_wide[proxy].loc[:eval_date].dropna()
        if len(series) < MIN_HISTORY:
            continue
        log_px = np.log(series)

        breadth_count = 0
        breadth_total = 0
        strength_components = []

        for w in TREND_WINDOWS:
            slope, r2, _, _ = _regression_metrics(log_px, w)
            if np.isnan(slope):
                continue
            wgt = WINDOW_WEIGHTS.get(w, 1.0)
            breadth_total += wgt
            if slope > 0:
                breadth_count += wgt
            if w in (63, 126, 252):
                strength_components.append((slope * max(r2, 0), WINDOW_WEIGHTS.get(w, 1.0)))

        breadth = breadth_count / breadth_total if breadth_total > 0 else np.nan
        strength = (sum(s * w for s, w in strength_components) /
                    sum(w for _, w in strength_components)
                    if strength_components else np.nan)
        _, _, _, residual_z = _regression_metrics(log_px, 63)
        mr_score = -residual_z if not np.isnan(residual_z) else np.nan

        rows.append({
            "asset":     asset_name,
            "proxy":     proxy,
            "bucket":    cfg["bucket"],
            "date":      eval_date,
            "breadth":   breadth,
            "strength":  strength,
            "mr_score":  mr_score,
            "price":     series.iloc[-1],
            "benchmark": cfg["benchmark"],
            "max":       cfg["max"],
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def compute_all_trends(prices_wide, rebalance_dates):
    """Compute trend features for every asset at every rebalance date."""
    log_prices_all = {c: np.log(prices_wide[c].dropna()) for c in prices_wide.columns}
    frames = []
    for d in rebalance_dates:
        f = _trend_features_for_date(prices_wide, d, log_prices_all)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = _add_trend_ranks(df)
    return df


def _add_trend_ranks(trend_df):
    """Add cross-sectional z-scores, final score, and percentile rank per date.
    Ranks are then centered so the benchmark-weighted average is ~0.50,
    eliminating systematic over/under-weight bias that normalization would crush."""
    result = []
    for d, grp in trend_df.groupby("date"):
        grp = grp.copy()
        for col in ["breadth", "strength", "mr_score"]:
            vals = grp[col].dropna()
            if len(vals) < 2:
                grp[f"z_{col}"] = 0.0
            else:
                mu, sd = vals.mean(), vals.std(ddof=0)
                grp[f"z_{col}"] = ((grp[col] - mu) / sd).fillna(0.0) if sd > 0 else 0.0

        b, s, m = TREND_BLEND
        grp["final_score"] = (b * grp["z_breadth"]
                              + s * grp["z_strength"]
                              + m * grp["z_mr_score"])
        scored = grp["final_score"].dropna()
        if len(scored) < 2:
            grp["rank_pct"] = 0.5
        else:
            grp["rank_pct"] = scored.rank(pct=True).reindex(grp.index).fillna(0.5)
        result.append(grp)

    result = pd.concat(result, ignore_index=True)
    result = _center_ranks(result)
    return result


def _center_ranks(trend_df):
    """Shift percentile ranks so benchmark-weighted average = 0.50.
    This prevents normalisation from compressing active tilts."""
    centered = []
    for d, grp in trend_df.groupby("date"):
        grp = grp.copy()
        bench = grp["benchmark"].values
        ranks = grp["rank_pct"].values
        bw_avg = np.average(ranks, weights=bench) if bench.sum() > 0 else 0.5
        shift = bw_avg - 0.5
        grp["rank_pct"] = (ranks - shift).clip(0.0, 1.0)
        centered.append(grp)
    return pd.concat(centered, ignore_index=True)


# ─────────────────────────────────────────────────────────────────
# SECTION 4: HRP Engine (Two-Level)
# ─────────────────────────────────────────────────────────────────

def _hrp_weights_for_date(returns_wide, eval_date):
    """
    Two-level hierarchical risk parity at eval_date.
    Level 1: equal risk budget across 3 buckets (1/3 each).
    Level 2: clipped inverse-vol within each bucket (prevents low-vol
             concentration that pure inverse-vol would cause).
    """
    trailing = returns_wide.loc[:eval_date].tail(VOL_LOOKBACK)
    if len(trailing) < 20:
        return {}

    # ── Level 1: Equal risk budget per bucket (1/3 each) ──
    bucket_wt = {b: 1.0 / 3.0 for b in BUCKETS}

    # ── Level 2: Inverse-vol within each bucket ──
    weights = {}
    for b in BUCKETS:
        bw = bucket_wt[b]
        assets_in_b = [(a, cfg["proxy"]) for a, cfg in ASSETS.items()
                       if cfg["bucket"] == b]
        available = [(a, p) for a, p in assets_in_b if p in trailing.columns]
        if not available:
            continue
        cols = [p for _, p in available]
        vols = trailing[cols].std()
        vols = vols.replace(0, np.nan)
        if vols.isna().all():
            raw = np.ones(len(available)) / len(available)
        else:
            vols = vols.fillna(vols.max())
            raw = (1.0 / vols).values
            raw = np.clip(raw, 0.25 * raw.mean(), 4.0 * raw.mean())
            raw = raw / raw.sum()
        for (a, _), w in zip(available, raw):
            weights[a] = bw * w

    return weights


@st.cache_data(ttl=600)
def compute_all_hrp(returns_wide, rebalance_dates):
    """Compute HRP weights for every rebalance date."""
    records = []
    for d in rebalance_dates:
        wt = _hrp_weights_for_date(returns_wide, d)
        for asset_name, w in wt.items():
            records.append({"date": d, "asset": asset_name, "hrp_weight": w})
    if not records:
        return pd.DataFrame(columns=["date", "asset", "hrp_weight"])
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# SECTION 5: Weight Generation
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def generate_weights(trend_df, hrp_df, rebalance_dates,
                     trend_hrp_blend=None, rank_params=None):
    if trend_hrp_blend is None:
        trend_hrp_blend = TREND_HRP_BLEND
    if rank_params is None:
        rank_params = {}
    concave_exp = rank_params.get("concave_exp", 0.6)
    quad_exp = rank_params.get("quad_exp", 2.0)
    floor_pct = rank_params.get("floor_pct", 0.05)

    if trend_df.empty:
        return pd.DataFrame(columns=["date", "asset", "weight", "trend_weight",
                                      "hrp_weight", "benchmark", "max", "pre_norm_blend"])

    hrp_map = {}
    if not hrp_df.empty:
        for _, row in hrp_df.iterrows():
            hrp_map[(row["date"], row["asset"])] = row["hrp_weight"]

    records = []
    for d in rebalance_dates:
        day = trend_df[trend_df["date"] == d].copy()
        if day.empty:
            continue
        if day["max"].sum() < 1.0:
            continue

        day["trend_w"] = day.apply(
            lambda r: _rank_to_weight(r["benchmark"], r["max"], r["rank_pct"],
                                      concave_exp=concave_exp,
                                      quad_exp=quad_exp,
                                      floor_pct=floor_pct),
            axis=1,
        )

        for idx, row in day.iterrows():
            a = row["asset"]
            tw = row["trend_w"]
            hw = hrp_map.get((d, a), row["benchmark"] / len(day))
            if hw is None or np.isnan(hw):
                hw = row["benchmark"] / len(day)
            bw = row["benchmark"]
            raw = trend_hrp_blend * tw + (1 - trend_hrp_blend) * hw
            cap = row["max"]
            final = min(raw, cap)
            records.append({
                "date": d, "asset": a, "weight": final,
                "trend_weight": tw, "hrp_weight": hw,
                "benchmark": bw, "max": cap,
                "pre_norm_blend": final,
            })

    weights_df = pd.DataFrame(records)
    if weights_df.empty:
        return weights_df

    weights_df = _normalize_and_enforce(weights_df)
    return weights_df


def _rank_to_weight(benchmark, max_wt, rank_pct, concave_exp=0.6, quad_exp=2.0,
                    floor_pct=0.05):
    floor = benchmark * floor_pct
    if rank_pct >= 0.5:
        t = ((rank_pct - 0.5) / 0.5) ** concave_exp
        return benchmark + (max_wt - benchmark) * t
    else:
        t = (rank_pct / 0.5) ** quad_exp
        return floor + (benchmark - floor) * t


def _normalize_and_enforce(weights_df):
    """Normalize weights to 1.0 per date and enforce max caps."""
    result = []
    for d, grp in weights_df.groupby("date"):
        grp = grp.copy()
        grp["weight"] = _normalize_capped_weights(grp["weight"], grp["max"])
        result.append(grp)
    return pd.concat(result, ignore_index=True)


def _normalize_capped_weights(weights, max_weights):
    weights = weights.clip(lower=0, upper=max_weights).astype(float)
    max_weights = max_weights.astype(float)
    if weights.sum() <= 0:
        return weights
    if max_weights.sum() < 1.0:
        return weights

    fixed = pd.Series(False, index=weights.index)
    out = weights.copy()
    for _ in range(len(out) + 1):
        remaining = ~fixed
        remaining_total = out[remaining].sum()
        fixed_total = out[fixed].sum()
        target_remaining = 1.0 - fixed_total
        if remaining_total <= 0 or target_remaining <= 0:
            break
        out.loc[remaining] = out.loc[remaining] * target_remaining / remaining_total
        breached = remaining & (out > max_weights)
        if not breached.any():
            break
        out.loc[breached] = max_weights.loc[breached]
        fixed.loc[breached] = True
    return out


# ─────────────────────────────────────────────────────────────────
# SECTION 6: Backtest
# ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def run_backtest(returns_wide, weights_df, rebalance_dates, label="RAAM Dynamic"):
    """
    Simulate portfolio returns by applying weights at each rebalance date
    to the subsequent period's daily returns. No lookahead.
    """
    if weights_df.empty or not rebalance_dates:
        return pd.Series(dtype=float, name=label)

    all_returns = []
    rebalance_dates_sorted = sorted(rebalance_dates)

    for i, d in enumerate(rebalance_dates_sorted[:-1]):
        d_next = rebalance_dates_sorted[i + 1]
        w_day = weights_df[weights_df["date"] == d]
        if w_day.empty:
            continue

        period_rets = returns_wide.loc[d:].loc[:d_next].iloc[1:]  # exclude d itself
        if period_rets.empty:
            continue

        port_ret = pd.Series(0.0, index=period_rets.index, name=label)
        for _, row in w_day.iterrows():
            asset = row["asset"]
            proxy = ASSETS.get(asset, {}).get("proxy")
            if proxy and proxy in period_rets.columns:
                port_ret += row["weight"] * period_rets[proxy].fillna(0.0)

        all_returns.append(port_ret)

    if not all_returns:
        return pd.Series(dtype=float, name=label)

    result = pd.concat(all_returns).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result.name = label
    return result


@st.cache_data(ttl=600)
def backtest_6040(returns_wide, rebalance_dates):
    """Backtest 60% SPY / 40% TLT, rebalanced monthly."""
    spy, tlt = BENCHMARK_60[0], BENCHMARK_40[0]
    all_ret = []
    for i, d in enumerate(rebalance_dates[:-1]):
        d_next = rebalance_dates[i + 1]
        period = returns_wide.loc[d:].loc[:d_next].iloc[1:]
        if period.empty:
            continue
        pr = pd.Series(0.0, index=period.index)
        if spy in period.columns:
            pr += 0.60 * period[spy].fillna(0.0)
        if tlt in period.columns:
            pr += 0.40 * period[tlt].fillna(0.0)
        all_ret.append(pr)
    if not all_ret:
        return pd.Series(dtype=float)
    r = pd.concat(all_ret).sort_index()
    r = r[~r.index.duplicated(keep="first")]
    r.name = "60/40"
    return r


@st.cache_data(ttl=600)
def backtest_static_raam(rebalance_dates, returns_wide):
    """Backtest static RAAM benchmark weights, monthly rebalanced."""
    proxy_map = {a: cfg["proxy"] for a, cfg in ASSETS.items()}
    w_map = {a: cfg["benchmark"] for a, cfg in ASSETS.items()}
    all_ret = []
    for i, d in enumerate(rebalance_dates[:-1]):
        d_next = rebalance_dates[i + 1]
        period = returns_wide.loc[d:].loc[:d_next].iloc[1:]
        if period.empty:
            continue
        pr = pd.Series(0.0, index=period.index)
        for asset, bw in w_map.items():
            p = proxy_map.get(asset)
            if p and p in period.columns:
                pr += bw * period[p].fillna(0.0)
        all_ret.append(pr)
    if not all_ret:
        return pd.Series(dtype=float)
    r = pd.concat(all_ret).sort_index()
    r = r[~r.index.duplicated(keep="first")]
    r.name = "RAAM Static"
    return r


def _compute_metrics(ret_series):
    """Compute standard portfolio metrics from a daily return series."""
    if ret_series.empty or len(ret_series) < 20:
        return {}
    r = ret_series.dropna()
    years = len(r) / 252
    ann_ret = (1 + r).prod() ** (1 / years) - 1 if years > 0 else 0
    ann_vol = r.std() * np.sqrt(252) if r.std() > 0 else 0
    sharpe = ((ann_ret - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else 0
    downside = r[r < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino = ((ann_ret - RISK_FREE_RATE) / down_vol) if down_vol and down_vol > 0 else 0
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum / running_max - 1)
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd and max_dd != 0 else 0

    yearly = r.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    best_yr = yearly.max() if not yearly.empty else None
    worst_yr = yearly.min() if not yearly.empty else None

    return {
        "CAGR": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Best Year": best_yr,
        "Worst Year": worst_yr,
        "Cumulative Return": cum.iloc[-1] - 1,
    }


# ─────────────────────────────────────────────────────────────────
# SECTION 7: UI Rendering
# ─────────────────────────────────────────────────────────────────

def render():
    st.title("📊 RAAM Strategy — Real Asset Allocation Model")

    st.markdown(
        "Based on 3Fourteen Research's *Real Asset Allocation: The World Has Changed* "
        "(May 2023). This implementation uses **20 US-listed ETFs**, regression-based "
        "trend scoring across 6 timeframes, two-level hierarchical risk parity, and "
        "monthly rebalancing with no lookahead."
    )

    if not st.checkbox("Load RAAM Strategy", value=False, key="raam_load_strategy"):
        st.info("Enable this when you want to run the RAAM model. Streamlit tabs execute eagerly, so this prevents the full backtest running while you are using other tabs.")
        return

    # ── Load data ──
    try:
        prices_wide, returns_wide, rebalance_dates = load_raam_data()
    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        st.info("Run `make pipeline` to fetch missing ticker data, then `make export`.")
        return

    if not rebalance_dates:
        st.warning("Not enough data to determine rebalance dates.")
        return

    with st.spinner("Computing trend features …"):
        trend_df = compute_all_trends(prices_wide, rebalance_dates)
    with st.spinner("Computing HRP weights …"):
        hrp_df = compute_all_hrp(returns_wide, rebalance_dates)
    with st.spinner("Generating portfolio weights …"):
        weights_df = generate_weights(trend_df, hrp_df, rebalance_dates)
    with st.spinner("Running backtest …"):
        raam_dynamic = run_backtest(returns_wide, weights_df, rebalance_dates, "RAAM Dynamic")
        raam_static  = backtest_static_raam(rebalance_dates, returns_wide)
        bench_6040   = backtest_6040(returns_wide, rebalance_dates)
        if not raam_dynamic.empty:
            start, end = raam_dynamic.index.min(), raam_dynamic.index.max()
            raam_static = raam_static.loc[start:end]
            bench_6040 = bench_6040.loc[start:end]

    if trend_df.empty or weights_df.empty:
        st.warning("Not enough data to compute trends — need at least 1 year of price history per asset.")
        return

    latest_date = weights_df["date"].max()
    latest_weights = weights_df[weights_df["date"] == latest_date]

    # ── Local Controls ──
    with st.expander("RAAM Controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rebalance Dates", len(weights_df["date"].unique()))
        c2.metric("Proxy Tickers", len(_all_proxies()))
        c3.metric("Latest Rebalance", latest_date.date().isoformat())

        st.markdown("#### Parameter Tuning")
        tuned = st.checkbox("Enable tuning", value=False,
                            help="Override default parameters and compare backtest results")

        tune_params = {}
        if tuned:
            st.caption("Adjust the rank-to-weight mapping and blend ratio. This is an interactive what-if tool, not a walk-forward optimiser.")
            p1, p2 = st.columns(2)
            tune_concave_exp = p1.slider(
                "Concave exponent", 0.3, 3.0, 0.6, 0.1,
                help="Higher → more conviction for top-ranked assets")
            tune_quad_exp = p2.slider(
                "Convex exponent", 0.5, 4.0, 2.0, 0.1,
                help="Higher → faster decay for low-ranked assets")
            tune_floor_pct = p1.slider(
                "Floor fraction", 0.01, 0.30, 0.05, 0.01,
                help="Minimum weight as fraction of benchmark when rank=0")
            tune_blend = p2.slider(
                "Trend-HRP blend", 0.0, 1.0, TREND_HRP_BLEND, 0.05,
                help="0 = pure risk-parity sleeve, 1 = pure trend following")

            tune_params = {
                "concave_exp": tune_concave_exp,
                "quad_exp": tune_quad_exp,
                "floor_pct": tune_floor_pct,
            }

            with st.spinner("Generating tuned weights …"):
                tuned_weights_df = generate_weights(trend_df, hrp_df, rebalance_dates,
                                                    trend_hrp_blend=tune_blend,
                                                    rank_params=tune_params)
            with st.spinner("Running tuned backtest …"):
                tuned_raam = run_backtest(returns_wide, tuned_weights_df, rebalance_dates,
                                           "RAAM Tuned")
                if not raam_dynamic.empty:
                    tuned_raam = tuned_raam.loc[raam_dynamic.index.min():raam_dynamic.index.max()]

            st.markdown("#### Tuned Backtest")
            if not tuned_raam.empty:
                tm = _compute_metrics(tuned_raam)
                dm = _compute_metrics(raam_dynamic)
                if tm and dm:
                    mcols = st.columns(4)
                    for col, (key, label) in zip(mcols, [("CAGR", "CAGR"), ("Sharpe", "Sharpe"),
                                                          ("Sortino", "Sortino"),
                                                          ("Max Drawdown", "MaxDD")]):
                        delta = tm[key] - dm[key]
                        fmt = "{:.2f}" if key in ("Sharpe", "Sortino") else "{:.1%}"
                        col.metric(label, fmt.format(tm[key]),
                                   delta=f"{delta:+.2f}" if key in ("Sharpe", "Sortino") else f"{delta:+.1%}")

            tuned_latest_date = tuned_weights_df["date"].max()
            tuned_latest_weights = tuned_weights_df[tuned_weights_df["date"] == tuned_latest_date]

    # ── MAIN SECTION 1: Current Allocation ──
    st.header("🎯 Default Current Model Allocation")
    _render_allocation_table(latest_weights, trend_df, latest_date)

    # ── MAIN SECTION 2: Rank → Weight Attribution ──
    st.header("🔗 Trend Rank → Weight Attribution")
    _render_pipeline_attribution(trend_df, hrp_df, latest_date)

    # ── MAIN SECTION 3: Backtest ──
    st.header("📈 Backtest Performance")
    _render_backtest(raam_dynamic, raam_static, bench_6040, returns_wide, rebalance_dates)

    # ── Tuned vs Default Comparison ──
    if tuned:
        st.header("🔧 Tuned vs Default Comparison")

        st.subheader("Tuned Allocation vs Default")
        wt_default = latest_weights.set_index("asset")[["weight"]].rename(
            columns={"weight": "Default"})
        wt_tuned = tuned_latest_weights.set_index("asset")[["weight"]].rename(
            columns={"weight": "Tuned"})
        wt_cmp = wt_default.join(wt_tuned, how="outer").fillna(0)
        wt_cmp["Delta"] = wt_cmp["Tuned"] - wt_cmp["Default"]
        wt_cmp["Bucket"] = [ASSETS.get(a, {}).get("bucket", "") for a in wt_cmp.index]

        def _color_delta(val):
            if abs(val) > 0.01:
                return "font-weight: bold; " + ("color: #2e7d32" if val > 0 else "color: #c62828")
            return "color: #888"

        st.dataframe(
            wt_cmp.style
            .format({"Default": "{:.1%}", "Tuned": "{:.1%}", "Delta": "{:+.1%}"})
            .map(_color_delta, subset=["Delta"]),
            height=720,
        )

        st.subheader("Backtest Metrics Comparison")
        dm = _compute_metrics(raam_dynamic)
        tm = _compute_metrics(tuned_raam)
        if dm and tm:
            cmp_rows = []
            for key, label in [
                ("CAGR", "CAGR"), ("Annualized Vol", "Ann. Vol"),
                ("Sharpe", "Sharpe"), ("Sortino", "Sortino"),
                ("Max Drawdown", "Max DD"), ("Calmar", "Calmar"),
                ("Best Year", "Best Year"), ("Worst Year", "Worst Year"),
                ("Cumulative Return", "Cum. Return"),
            ]:
                cmp_rows.append({
                    "Metric": label,
                    "Default": dm[key],
                    "Tuned": tm[key],
                    "Δ": tm[key] - dm[key],
                })
            cmp_df = pd.DataFrame(cmp_rows).set_index("Metric")
            ratio_metrics = {"Sharpe", "Sortino", "Calmar"}
            for metric_name in cmp_df.index:
                fmt_str = "{:.2f}" if metric_name in ratio_metrics else "{:.2%}"
                for col in ["Default", "Tuned"]:
                    val = cmp_df.loc[metric_name, col]
                    if isinstance(val, (int, float)):
                        cmp_df.loc[metric_name, col] = fmt_str.format(val)
                delta_val = cmp_df.loc[metric_name, "Δ"]
                if isinstance(delta_val, (int, float)):
                    cmp_df.loc[metric_name, "Δ"] = f"{delta_val:+.2%}"
            st.dataframe(
                cmp_df.style.map(
                    lambda v: "color: #2e7d32; font-weight: bold"
                    if isinstance(v, str) and v.startswith("+")
                    else ("color: #c62828; font-weight: bold"
                          if isinstance(v, str) and v.startswith("-") else ""),
                    subset=["Δ"],
                ),
                height=380,
            )

    # ── MAIN SECTION 4: Trend Signals ──
    st.header("🔍 Trend Signals")
    _render_trend_table(trend_df, latest_date)

    # ── MAIN SECTION 5: Trend Diagnostics ──
    st.header("🔬 Trend Diagnostics")
    _render_trend_breadth_grid(prices_wide, latest_date)

    # ── MAIN SECTION 6: Weight Waterfall ──
    st.header("🌊 Weight Attribution Waterfall")
    _render_weight_waterfall(trend_df, hrp_df, latest_date, latest_weights)

    # ── MAIN SECTION 7: Details ──
    st.header("📋 Details")
    with st.expander("Allocation History (weight heatmap)", expanded=False):
        _render_allocation_history(weights_df)
    with st.expander("Trend Score History", expanded=False):
        _render_trend_history(trend_df)
    with st.expander("Bucket Allocation Over Time", expanded=False):
        _render_bucket_history(weights_df)


def _render_trend_breadth_grid(prices_wide, latest_date):
    """Show per-asset, per-window regression trend direction as a colour grid."""
    st.subheader("Trend Breadth — Per-Window Regression Direction")

    grid_rows = []
    for asset_name in sorted(ASSETS.keys(), key=lambda a: ASSETS[a]["bucket"]):
        cfg = ASSETS[asset_name]
        proxy = cfg["proxy"]
        if proxy not in prices_wide.columns:
            continue
        series = prices_wide[proxy].loc[:latest_date].dropna()
        if len(series) < 50:
            continue
        log_px = np.log(series)

        row = {"Asset": asset_name, "Bucket": cfg["bucket"]}
        for w in TREND_WINDOWS:
            slope, r2, _, _ = _regression_metrics(log_px, w)
            if np.isnan(slope):
                row[f"{w}d"] = "—"
            elif slope > 0:
                row[f"{w}d"] = f"▲ {slope * 252:.1%}"  # annualized
            else:
                row[f"{w}d"] = f"▼ {slope * 252:.1%}"
        grid_rows.append(row)

    df = pd.DataFrame(grid_rows).set_index("Asset")

    def _color_trend(val):
        if isinstance(val, str) and val.startswith("▲"):
            return "background-color: #e8f5e9; color: #1b5e20"
        if isinstance(val, str) and val.startswith("▼"):
            return "background-color: #ffebee; color: #b71c1c"
        return "color: #ccc"
    
    st.dataframe(
        df.style.map(_color_trend),
        height=680,
    )
    st.caption(
        "▲ = positive regression slope (uptrend).  ▼ = negative (downtrend).  "
        "Value = annualised regression slope.  "
        "Trend Breadth = count(▲) / 6."
    )


def _render_weight_waterfall(trend_df, hrp_df, latest_date, latest_weights):
    """Show step-by-step weight derivation for a selected asset."""
    assets_list = sorted(ASSETS.keys())
    selected = st.selectbox("Select asset", assets_list, index=assets_list.index("Miners"),
                            key="waterfall_asset")

    cfg = ASSETS[selected]
    lt = trend_df[trend_df["date"] == latest_date]
    lt_a = lt[lt["asset"] == selected]
    rk = lt_a["rank_pct"].values[0] if len(lt_a) else 0.5

    lw = latest_weights[latest_weights["asset"] == selected]
    final_w = lw["weight"].values[0] if len(lw) else 0
    trend_w_raw = lw["trend_weight"].values[0] if len(lw) else 0

    h_map = {}
    if not hrp_df.empty:
        lh = hrp_df[hrp_df["date"] == latest_date]
        h_map = lh.set_index("asset")["hrp_weight"].to_dict()
    hrp_w = h_map.get(selected, cfg["benchmark"])

    blend = TREND_HRP_BLEND * trend_w_raw + (1 - TREND_HRP_BLEND) * hrp_w
    capped = min(blend, cfg["max"])

    steps = [
        ("1. Benchmark weight", cfg["benchmark"]),
        ("2. Trend rank percentile", rk),
        ("3. Rank × tilt → raw trend weight", trend_w_raw),
        ("4. HRP weight (bucket risk parity)", hrp_w),
        ("5. Blended (%.0f%% trend / %.0f%% HRP)" % (TREND_HRP_BLEND * 100, (1 - TREND_HRP_BLEND) * 100), blend),
        ("6. Capped at max (%.0f%%)" % (cfg["max"] * 100), capped),
        ("7. Final (after normalisation)", final_w),
    ]

    waterfall_df = pd.DataFrame(steps, columns=["Step", "Value"])
    waterfall_df["Label"] = waterfall_df.apply(
        lambda r: f"{r['Value']:.1%}" if r["Step"] != "2. Trend rank percentile"
        else f"{r['Value']:.0%}",
        axis=1,
    )

    import plotly.express as px
    fig = px.bar(
        waterfall_df,
        x="Step",
        y="Value",
        text="Label",
        title=f"Weight Attribution — {selected} ({cfg['proxy']})",
        color_discrete_sequence=["#1565c0"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(tickformat=".0%", title="Weight"),
        xaxis=dict(title=""),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig)

    st.caption(
        f"Benchmark: {cfg['benchmark']:.1%}  |  Max: {cfg['max']:.1%}  |  "
        f"Trend rank: {rk:.0%}  |  Final weight: {final_w:.1%}"
    )


def _render_pipeline_attribution(trend_df, hrp_df, latest_date):
    """Show the full pipeline: trend rank → trend weight → HRP blend → final weight."""
    latest_t = trend_df[trend_df["date"] == latest_date].set_index("asset")
    hrp_map = {}
    if not hrp_df.empty:
        lh = hrp_df[hrp_df["date"] == latest_date]
        hrp_map = lh.set_index("asset")["hrp_weight"].to_dict()

    if latest_t.empty:
        st.info("No trend data for attribution.")
        return

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Rank → Weight Pipeline")

        rows = []
        total_pre = 0.0
        for a in sorted(ASSETS.keys()):
            cfg = ASSETS[a]
            rk = latest_t.loc[a, "rank_pct"] if a in latest_t.index else 0.5
            tw = _rank_to_weight(cfg["benchmark"], cfg["max"], rk)
            hw = hrp_map.get(a, cfg["benchmark"])
            if hw is None or np.isnan(hw):
                hw = cfg["benchmark"]
            blend = TREND_HRP_BLEND * tw + (1 - TREND_HRP_BLEND) * hw
            capped = min(blend, cfg["max"])
            total_pre += capped
            rows.append({
                "Asset": a,
                "Rank %": f"{rk:.0%}",
                "Bench": f"{cfg['benchmark']:.1%}",
                "TrendW": tw,
                "HRP_w": hw,
                "Blend": capped,
                "Max": f"{cfg['max']:.1%}",
                "MaxVal": cfg["max"],
            })

        df = pd.DataFrame(rows)
        df["Final"] = _normalize_capped_weights(df["Blend"], df["MaxVal"])

        def _color_blend(val):
            if val > 0.15:
                return "color: #1b5e20; font-weight: bold"
            if val < 0.02:
                return "color: #c62828"
            return ""

        st.dataframe(
            df.drop(columns=["MaxVal"]).set_index("Asset").style
            .format({"TrendW": "{:.1%}", "HRP_w": "{:.1%}", "Blend": "{:.1%}", "Final": "{:.1%}"})
            .map(_color_blend, subset=["Final"])
            .background_gradient(subset=["TrendW", "Blend", "Final"], cmap="YlOrRd"),
            height=700,
        )

        st.caption(
            f"Pre-normalisation sum: **{total_pre:.1%}**.  "
            f"Normalisation target: 100% after cap enforcement.  "
            f"Benchmark-weighted avg rank: "
            f"**{np.average(latest_t['rank_pct'], weights=latest_t['benchmark']):.3f}**  "
            f"(target 0.500)"
        )

    with col2:
        st.subheader("Rank vs Final Weight")
        df["rank_val"] = df["Rank %"].str.rstrip("%").astype(float) / 100
        df["final_val"] = df["Final"]
        df["bench_val"] = [ASSETS[a]["benchmark"] for a in sorted(ASSETS.keys())]
        df["max_val"] = [ASSETS[a]["max"] for a in sorted(ASSETS.keys())]

        chart_data = df[["Asset", "rank_val", "final_val", "bench_val", "max_val"]].copy()
        chart_data["label"] = chart_data.apply(
            lambda r: f"{r['Asset']} ({r['bench_val']:.0%}→{r['final_val']:.1%})", axis=1
        )

        points = alt.Chart(chart_data).mark_circle(size=120).encode(
            x=alt.X("rank_val:Q", title="Trend Rank Percentile", axis=alt.Axis(format=".0%")),
            y=alt.Y("final_val:Q", title="Final Model Weight", axis=alt.Axis(format=".0%")),
            color=alt.Color("final_val:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=["Asset:N", "rank_val:Q", "final_val:Q", "bench_val:Q"],
        )

        bench_line = alt.Chart(chart_data).mark_line(
            strokeDash=[4, 4], color="gray", opacity=0.5
        ).encode(
            x="rank_val:Q",
            y=alt.Y("bench_val:Q"),
        )

        st.altair_chart(
            (points + bench_line).properties(height=420).interactive(),
        )
        st.caption("Dashed line = benchmark weight at each rank. Points above = overweight.")

    # Live RAA comparison
    st.divider()
    st.subheader("Comparison: Our Model vs Live RAA ETF")
    _render_live_comparison(df)

    # Normalisation diagnostic
    st.divider()
    _render_norm_diagnostic(latest_t, hrp_map)


def _render_live_comparison(attr_df):
    """Compare our model allocation to the live RAA ETF."""
    live_raa = {
        "Nasdaq": 0.1838, "US Large Cap": 0.1403, "US Small Cap": 0.0393,
        "Dividend Payers": 0.0387, "EM ex-China": 0.0203, "Japan": 0.0202,
        "Europe": 0.0195,
        "Long-Term Treasuries": 0.0673, "Corporate Bonds": 0.0582,
        "High Yield": 0.0581, "EM Bonds": 0.0389, "TIPS": 0.0287, "T-Bills": 0.0117,
        "Miners": 0.0746, "Managed Futures": 0.0591, "Gold": 0.0496,
        "Commodities": 0.0397, "Bitcoin": 0.0225, "Energy": 0.0198,
        "Real Estate": 0.0097,
    }

    cmp_rows = []
    for _, row in attr_df.iterrows():
        a = row["Asset"]
        lr = live_raa.get(a, 0)
        ow = row["Final"]
        d = ow - lr
        cmp_rows.append({
            "Asset": a,
            "Live RAA": f"{lr:.1%}",
            "Our Model": f"{ow:.1%}",
            "Delta": d,
        })

    cmp_df = pd.DataFrame(cmp_rows).set_index("Asset")
    cmp_df = cmp_df.reindex(
        cmp_df.index[np.argsort(cmp_df["Delta"].abs())[::-1]]
    )

    def _color_delta(val):
        if abs(val) > 0.025:
            return "font-weight: bold; " + ("color: #2e7d32" if val > 0 else "color: #c62828")
        return "color: #888"

    st.dataframe(
        cmp_df.style
        .format({"Delta": "{:+.1%}"})
        .map(_color_delta, subset=["Delta"]),
    )

    # Bucket summary
    b_rows = []
    for b_name in ["Alternatives", "Equities", "Fixed Income"]:
        b_assets = [a for a, c in ASSETS.items() if c["bucket"] == b_name]
        lr = sum(live_raa.get(a, 0) for a in b_assets)
        ow = sum(attr_df[attr_df["Asset"].isin(b_assets)]["Final"])
        bw = sum(ASSETS[a]["benchmark"] for a in b_assets)
        b_rows.append(f"**{b_name}**: {ow:.1%}  (live: {lr:.1%}, benchmark: {bw:.1%})")
    st.markdown("  |  ".join(b_rows))

    keys = sorted(ASSETS.keys())
    corr = np.corrcoef(
        [live_raa.get(a, 0) for a in keys],
        [attr_df[attr_df["Asset"] == a]["Final"].values[0] if a in attr_df["Asset"].values else 0 for a in keys],
    )[0, 1]
    mad = np.mean([abs(live_raa.get(a, 0) - (attr_df[attr_df["Asset"] == a]["Final"].values[0] if a in attr_df["Asset"].values else 0)) for a in keys])
    st.caption(
        f"Weight correlation with live RAA ETF: **{corr:.3f}**  |  "
        f"Mean absolute deviation: **{mad:.1%}**  |  "
        f"[View live RAA holdings](https://3fourteensmi.com/raa#holdings)"
    )


def _render_norm_diagnostic(latest_t, hrp_map):
    """Show diagnostic on how normalisation affects each weight step."""
    st.subheader("Normalisation Compression Diagnostic")
    st.markdown(
        "Trend weights naturally sum above or below 100% depending on whether "
        "most assets are trending above or below their benchmarks. The HRP blend "
        "pulls toward 100% (HRP always sums to exactly 100%), then final "
        "cap-aware normalisation targets a 100%-invested portfolio."
    )

    total_t = 0
    total_h = 0
    for a in sorted(ASSETS.keys()):
        cfg = ASSETS[a]
        rk = latest_t.loc[a, "rank_pct"] if a in latest_t.index else 0.5
        tw = _rank_to_weight(cfg["benchmark"], cfg["max"], rk)
        hw = hrp_map.get(a, cfg["benchmark"])
        if hw is None or np.isnan(hw):
            hw = cfg["benchmark"]
        total_t += tw
        total_h += hw

    total_blend = TREND_HRP_BLEND * total_t + (1 - TREND_HRP_BLEND) * total_h

    col1, col2, col3 = st.columns(3)
    col1.metric("Trend sum", f"{total_t:.1%}",
                delta=f"{total_t - 1.0:+.1%}" if total_t != 1.0 else None)
    col2.metric("HRP sum", f"{total_h:.1%}",
                delta=f"{total_h - 1.0:+.1%}" if abs(total_h - 1.0) > 0.001 else "= 100%")
    col3.metric("Blend (pre-norm)", f"{total_blend:.1%}",
                delta=f"{total_blend - 1.0:+.1%}")
    st.caption(
        "When pre-normalisation sum differs from 100%, weights are rescaled while "
        "respecting asset caps. Rank centering (benchmark-weighted avg rank = 0.500) "
        "minimises this normalisation effect."
    )


def _render_allocation_table(latest_weights, trend_df, latest_date):
    """Render the current allocation table with benchmark comparison."""
    if latest_weights.empty:
        st.info("No weight data yet.")
        return

    latest_trend = trend_df[trend_df["date"] == latest_date]
    trend_map = {}
    if not latest_trend.empty:
        trend_map = latest_trend.set_index("asset")[["breadth", "strength",
                                                       "mr_score", "rank_pct"]].to_dict("index")

    rows = []
    for _, row in latest_weights.iterrows():
        a = row["asset"]
        cfg = ASSETS.get(a, {})
        proxy = cfg.get("proxy", "")
        bucket = cfg.get("bucket", "")
        tinfo = trend_map.get(a, {})
        pre = row.get("pre_norm_blend", 0)
        fw = row["weight"]
        compress = fw / pre if pre > 0.0001 else float("nan")
        rows.append({
            "Asset": a,
            "Proxy": proxy,
            "Bucket": bucket,
            "Benchmark": f"{row['benchmark']:.1%}",
            "Pre-Norm": pre,
            "Weight": fw,
            "Active": fw - row["benchmark"],
            "Compress": compress,
            "Max": f"{row['max']:.1%}",
            "Breadth": f"{tinfo.get('breadth', 0):.2f}",
            "Strength": f"{tinfo.get('strength', 0):.4f}",
            "MR Score": f"{tinfo.get('mr_score', 0):.2f}",
            "Rank %": f"{tinfo.get('rank_pct', 0):.1%}",
        })

    table = pd.DataFrame(rows).set_index("Asset")

    def _color_active(val):
        if val > 0.005:
            return "color: #2e7d32; font-weight: bold"
        if val < -0.005:
            return "color: #c62828; font-weight: bold"
        return "color: #888"

    def _color_compress(val):
        if isinstance(val, float) and not np.isnan(val):
            if val < 0.98:
                return "color: #c62828; font-weight: bold"
            if val > 1.02:
                return "color: #2e7d32; font-weight: bold"
        return ""

    st.dataframe(
        table.style
        .format({"Pre-Norm": "{:.1%}", "Weight": "{:.1%}", "Active": "{:.1%}",
                  "Compress": "{:.3f}"})
        .map(_color_active, subset=["Active"])
        .map(_color_compress, subset=["Compress"]),
        height=760,
    )

    total_pre = table["Pre-Norm"].sum()
    st.caption(
        f"Pre-norm weight sum: **{total_pre:.1%}**  |  "
        f"Final weights are cap-normalised to 100%  |  "
        f"Compress > 1 = weight gained in normalisation; < 1 = lost."
    )

    st.subheader("Pre-Normalisation vs Final Weights")
    chart_df = table.reset_index()
    melted = pd.DataFrame({
        "Asset": list(chart_df["Asset"]) * 2,
        "Bucket": list(chart_df["Bucket"]) * 2,
        "Stage": ["Pre-Normalisation"] * len(chart_df) + ["Final"] * len(chart_df),
        "Weight": list(chart_df["Pre-Norm"]) + list(chart_df["Weight"]),
    })

    bar = alt.Chart(melted).mark_bar().encode(
        x=alt.X("Asset:N", sort=None, title=""),
        y=alt.Y("Weight:Q", title="", axis=alt.Axis(format=".0%")),
        color=alt.Color("Stage:N", scale=alt.Scale(
            domain=["Pre-Normalisation", "Final"],
            range=["#90a4ae", "#1565c0"])),
        column=alt.Column("Bucket:N", title=None,
                          sort=["Alternatives", "Equities", "Fixed Income"]),
        tooltip=["Asset", "Stage", alt.Tooltip("Weight:Q", format=".1%")],
    ).properties(height=280)

    st.altair_chart(bar)

    st.subheader("Bucket Summary")
    bucket_summary = []
    for b in BUCKETS:
        bucket_mask = latest_weights["asset"].isin(
            [a for a, c in ASSETS.items() if c["bucket"] == b]
        )
        w = latest_weights[bucket_mask]["weight"].sum()
        bw = latest_weights[bucket_mask]["benchmark"].sum()
        pw = latest_weights[bucket_mask].get("pre_norm_blend", pd.Series(dtype=float))
        pw_sum = pw.sum() if not pw.empty else 0
        bucket_summary.append(
            f"**{b}**: {w:.1%}  (benchmark: {bw:.1%}, active: {w-bw:+.1%}, "
            f"pre-norm: {pw_sum:.1%})"
        )
    st.markdown("  |  ".join(bucket_summary))


def _render_backtest(raam_dynamic, raam_static, bench_6040, returns_wide, rebalance_dates):
    """Render backtest charts and metrics table."""
    if raam_dynamic.empty and raam_static.empty and bench_6040.empty:
        st.info("Insufficient data for backtest.")
        return

    col1, col2 = st.columns([1, 1])

    # Drop NaN gaps before computing cumulative returns (consistent with _compute_metrics)
    rd_clean = raam_dynamic.dropna()
    rs_clean = raam_static.dropna()
    b60_clean = bench_6040.dropna()

    with col1:
        st.subheader("Equity Curve")
        combined = pd.DataFrame({
            "RAAM Dynamic": (1 + rd_clean).cumprod(),
            "RAAM Static": (1 + rs_clean).cumprod(),
            "60/40": (1 + b60_clean).cumprod(),
        }).dropna(how="all")

        if not combined.empty:
            chart_data = combined.reset_index().melt(
                id_vars="date", var_name="Portfolio", value_name="Growth of $1"
            )
            chart = (
                alt.Chart(chart_data)
                .mark_line()
                .encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("Growth of $1:Q", title="Growth of $1"),
                    color=alt.Color(
                        "Portfolio:N",
                        scale=alt.Scale(
                            domain=["RAAM Dynamic", "RAAM Static", "60/40"],
                            range=["#1b5e20", "#4caf50", "#90a4ae"],
                        ),
                    ),
                    tooltip=["date:T", "Portfolio:N", "Growth of $1:Q"],
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(chart)

    with col2:
        st.subheader("Drawdowns")
        dd_data = pd.DataFrame({
            "RAAM Dynamic": (1 + rd_clean).cumprod().pipe(
                lambda c: c / c.cummax() - 1
            ),
            "60/40": (1 + b60_clean).cumprod().pipe(
                lambda c: c / c.cummax() - 1
            ),
        }).dropna(how="all")

        if not dd_data.empty:
            dd_chart_data = dd_data.reset_index().melt(
                id_vars="date", var_name="Portfolio", value_name="Drawdown"
            )
            dd_chart = (
                alt.Chart(dd_chart_data)
                .mark_area(opacity=0.7)
                .encode(
                    x=alt.X("date:T", title=""),
                    y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
                    color=alt.Color("Portfolio:N"),
                    tooltip=["date:T", "Portfolio:N", "Drawdown:Q"],
                )
                .properties(height=350)
                .interactive()
            )
            st.altair_chart(dd_chart)

    # Metrics table
    st.subheader("Performance Metrics")
    metrics = {}
    for name, series in [
        ("RAAM Dynamic", raam_dynamic),
        ("RAAM Static", raam_static),
        ("60/40", bench_6040),
    ]:
        if not series.empty:
            m = _compute_metrics(series)
            if m:
                metrics[name] = m

    if metrics:
        met_df = pd.DataFrame(metrics).T
        fmt_cols = ["CAGR", "Annualized Vol", "Max Drawdown", "Best Year", "Worst Year", "Cumulative Return"]
        st.dataframe(
            met_df.style.format(
                {c: "{:.2%}" for c in fmt_cols if c in met_df.columns}
                | {"Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Calmar": "{:.2f}"}
            ),
        )


def _render_trend_table(trend_df, latest_date):
    """Render the latest trend signals for all assets."""
    latest = trend_df[trend_df["date"] == latest_date].copy()
    if latest.empty:
        st.info("No trend data for the latest date.")
        return

    latest = latest.sort_values("rank_pct", ascending=False)
    display = latest[["asset", "bucket", "breadth", "strength",
                       "mr_score", "final_score", "rank_pct"]].copy()
    display.columns = ["Asset", "Bucket", "Breadth", "Strength",
                        "MR Score", "Final Score", "Rank %"]

    st.dataframe(
        display.set_index("Asset").style
        .format({
            "Breadth": "{:.2f}",
            "Strength": "{:.5f}",
            "MR Score": "{:.2f}",
            "Final Score": "{:.2f}",
            "Rank %": "{:.1%}",
        })
        .background_gradient(subset=["Breadth", "Strength", "Final Score", "Rank %"],
                             cmap="RdYlGn"),
    )


def _render_allocation_history(weights_df):
    """Heatmap of asset weights over time."""
    if weights_df.empty:
        return
    pivot = weights_df.pivot_table(
        index="date", columns="asset", values="weight", aggfunc="sum"
    ).fillna(0)
    if pivot.empty or pivot.shape[1] < 2:
        return

    import plotly.express as px
    fig = px.imshow(
        pivot.T,
        labels={"x": "Date", "y": "Asset", "color": "Weight"},
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig)


def _render_trend_history(trend_df):
    """Score over time per asset."""
    if trend_df.empty:
        return
    assets = sorted(trend_df["asset"].unique())
    selected = st.multiselect("Assets", assets, default=assets[:6], key="trend_hist_sel")
    if not selected:
        return
    subset = trend_df[trend_df["asset"].isin(selected)]
    chart = (
        alt.Chart(subset)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("final_score:Q", title="Final Trend Score"),
            color=alt.Color("asset:N", legend=alt.Legend(title=None)),
            tooltip=["date:T", "asset:N", "final_score:Q", "rank_pct:Q"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart)


def _render_bucket_history(weights_df):
    """Stacked area chart of bucket allocation over time."""
    if weights_df.empty:
        return
    bucket_map = {}
    for a, c in ASSETS.items():
        bucket_map[a] = c["bucket"]

    weights_df = weights_df.copy()
    weights_df["bucket"] = weights_df["asset"].map(bucket_map)
    bucket_w = weights_df.groupby(["date", "bucket"])["weight"].sum().reset_index()
    pivot = bucket_w.pivot(index="date", columns="bucket", values="weight").fillna(0)

    if pivot.empty:
        return

    chart_data = pivot.reset_index().melt(
        id_vars="date", var_name="Bucket", value_name="Weight"
    )
    chart = (
        alt.Chart(chart_data)
        .mark_area()
        .encode(
            x=alt.X("date:T", title=""),
            y=alt.Y("Weight:Q", title="Allocation", axis=alt.Axis(format=".0%")),
            color=alt.Color("Bucket:N"),
            tooltip=["date:T", "Bucket:N", "Weight:Q"],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(chart)
