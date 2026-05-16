"""Walk-forward backtest for the risk-parity allocator.

At each rebalance date (monthly, last business day), uses *only* prices up
to that date to (a) build the 90d covariance matrix, (b) compute price/MA252
ratios, (c) call ``sleeves.bucket_weights`` with the caller's policy. Holds
those weights until the next rebalance, compounding daily returns.

Benchmarks for comparison:
  - CSP1 (S&P 500) — pure US equity
  - VWRP (FTSE All-World) — global equity / ACWI proxy
  - 60/40 — 60% CSP1 + 40% TLT, rebalanced monthly

No look-ahead: at rebalance date ``t`` we slice the return matrix to ``.loc[:t]``
before cov / MA computation.

No transaction costs modelled. With monthly rebalancing on a ~15-asset book
the realised drag is 10–30 bps/yr — material for level but not for ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from allocator_v2 import covariance as cov_mod
from allocator_v2 import ensemble
from allocator_v2 import sleeves as sl


BENCHMARK_TICKERS = ("CSP1", "IWDA", "TLT")
_YF_BENCHMARK = {"CSP1": "CSP1.L", "IWDA": "IWDA.L", "TLT": "TLT"}


@dataclass
class BacktestResult:
    equity: pd.Series                 # portfolio equity curve (starts at 1.0)
    daily_returns: pd.Series          # daily arithmetic returns
    weights_over_time: pd.DataFrame   # rebalance_date × ticker
    rebalance_dates: pd.DatetimeIndex


def _month_end_rebalances(index: pd.DatetimeIndex, min_lookback: int = 252) -> pd.DatetimeIndex:
    """Last trading day of each month present in ``index``, starting after ``min_lookback`` days."""
    if len(index) <= min_lookback:
        return pd.DatetimeIndex([])
    eligible = index[min_lookback:]
    ser = pd.Series(eligible, index=eligible)
    # Group by (year, month), keep the last date in each group.
    key = eligible.to_period("M")
    last_per_month = ser.groupby(key).last().values
    return pd.DatetimeIndex(sorted(last_per_month))


def _ma_ratios_at(prices_wide: pd.DataFrame, asof: pd.Timestamp, window: int = 252) -> dict[str, float]:
    """Price / MA(window) as of ``asof``, using only data up to asof."""
    hist = prices_wide.loc[:asof]
    if hist.empty:
        return {}
    tail = hist.tail(window)
    if len(tail) < window // 2:
        return {}
    ma = tail.mean(axis=0)
    last = hist.iloc[-1]
    out: dict[str, float] = {}
    for t in last.index:
        m = ma.get(t)
        p = last.get(t)
        if m is None or p is None or not np.isfinite(m) or not np.isfinite(p) or m <= 0:
            continue
        out[t] = float(p / m)
    return out


def _weights_at(
    rets_wide: pd.DataFrame,
    prices_wide: pd.DataFrame,
    asof: pd.Timestamp,
    policy: sl.SleevePolicy,
    cov_window: int,
    method: str,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    """Compute target weights as of ``asof`` — no look-ahead."""
    hist_rets = rets_wide.loc[:asof]
    cov = cov_mod.cov_matrix(hist_rets, window=cov_window)
    corr = cov_mod.corr_matrix(hist_rets, window=cov_window)
    if cov.empty or len(cov) < 3:
        return pd.Series(dtype=float)

    ma_ratios = _ma_ratios_at(prices_wide, asof)
    pol = replace(policy, ma_ratios=ma_ratios)

    if method == "mesh":
        sizer_df = ensemble.build_all_sizers(cov, corr, policy=pol)
        w = ensemble.mesh(sizer_df)
    else:
        w = sl.bucket_weights(cov, corr, policy=pol, method=method)
    if min_weight > 0 and not w.empty:
        w = sl.apply_min_weight_floor(w, min_weight, sl.ASSET_SLEEVE)
    if max_weight > 0 and not w.empty:
        w = sl.apply_max_weight_cap(w, max_weight, sl.ASSET_SLEEVE)
    return w


def run_backtest(
    rets_wide: pd.DataFrame,
    prices_wide: pd.DataFrame,
    policy: sl.SleevePolicy,
    cov_window: int = 90,
    method: str = "erc",
    min_weight: float = 0.03,
    max_weight: float = 0.10,
    min_lookback: int = 252,
    start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Walk forward, rebalancing monthly.

    ``rets_wide`` and ``prices_wide`` share a DatetimeIndex. ``prices_wide`` is
    used for MA computation (levels), ``rets_wide`` for cov + PnL.
    """
    if rets_wide.empty or prices_wide.empty:
        return BacktestResult(
            equity=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            weights_over_time=pd.DataFrame(),
            rebalance_dates=pd.DatetimeIndex([]),
        )

    rebal_dates = _month_end_rebalances(rets_wide.index, min_lookback=min_lookback)
    if start is not None:
        rebal_dates = rebal_dates[rebal_dates >= start]
    if len(rebal_dates) == 0:
        return BacktestResult(
            equity=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            weights_over_time=pd.DataFrame(),
            rebalance_dates=rebal_dates,
        )

    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    daily_ret_parts: list[pd.Series] = []

    for i, rd in enumerate(rebal_dates):
        w = _weights_at(
            rets_wide, prices_wide, rd, policy,
            cov_window=cov_window, method=method,
            min_weight=min_weight, max_weight=max_weight,
        )
        if w.empty:
            continue
        weights_rows[rd] = w

        # Holding period: day after rebalance up to (and including) next rebalance.
        next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rets_wide.index[-1]
        mask = (rets_wide.index > rd) & (rets_wide.index <= next_rd)
        period = rets_wide.loc[mask, w.index].fillna(0.0)
        if period.empty:
            continue
        # Daily portfolio return = w · r  (start-of-day weights, ignoring intra-period drift).
        pr = period.values @ w.values
        daily_ret_parts.append(pd.Series(pr, index=period.index))

    if not daily_ret_parts:
        return BacktestResult(
            equity=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            weights_over_time=pd.DataFrame(),
            rebalance_dates=rebal_dates,
        )

    daily_ret = pd.concat(daily_ret_parts).sort_index()
    equity = (1.0 + daily_ret).cumprod()

    wdf = pd.DataFrame(weights_rows).T.sort_index().fillna(0.0)
    wdf.index.name = "rebalance_date"

    return BacktestResult(
        equity=equity,
        daily_returns=daily_ret,
        weights_over_time=wdf,
        rebalance_dates=pd.DatetimeIndex(list(weights_rows.keys())),
    )


def benchmark_equity(
    bench_rets: pd.DataFrame,
    weights: dict[str, float],
    start: pd.Timestamp | None = None,
) -> pd.Series:
    """Equity curve for a fixed-weight benchmark, rebalanced monthly on month-end."""
    if bench_rets.empty:
        return pd.Series(dtype=float)
    cols = [t for t in weights if t in bench_rets.columns]
    if not cols:
        return pd.Series(dtype=float)
    w_arr = np.array([weights[t] for t in cols], dtype=float)
    w_arr = w_arr / w_arr.sum()

    rets = bench_rets[cols].copy()
    if start is not None:
        rets = rets.loc[rets.index >= start]
    rets = rets.dropna(how="all").fillna(0.0)
    if rets.empty:
        return pd.Series(dtype=float)

    # Reset weights at each month-end; compound within the month using drifted weights.
    out = []
    cur_w = w_arr.copy()
    last_month = rets.index[0].to_period("M")
    for dt, row in rets.iterrows():
        m = dt.to_period("M")
        if m != last_month:
            cur_w = w_arr.copy()
            last_month = m
        # Portfolio return for the day.
        r = float(cur_w @ row.values)
        out.append((dt, r))
        # Drift weights by today's growth.
        grow = 1.0 + row.values
        cur_w = cur_w * grow
        s = cur_w.sum()
        if s > 0:
            cur_w = cur_w / s

    s = pd.Series({d: r for d, r in out}).sort_index()
    return (1.0 + s).cumprod()


def summary_stats(equity: pd.Series) -> dict[str, float]:
    """CAGR, annualised vol, Sharpe, max DD, total return."""
    if equity.empty or len(equity) < 2:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan, "total_return": np.nan}
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(rets.mean() * 252 / vol) if vol > 0 else np.nan
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min())
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
    }
