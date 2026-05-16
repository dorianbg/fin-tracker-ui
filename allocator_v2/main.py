"""Risk-Parity Allocator — v2 dashboard.

Runs two sizers (bucket-ERC / bucket-HRP) on the same 90-day covariance
matrix, shows them side-by-side, and meshes them via a 50/50 default blend
(user-adjustable). Sleeve caps are hard-enforced; factor tilts (e.g.
momentum on IWMO) nudge equity weights inside the sleeve.

Layout:

  - Sidebar: universe selector, sleeve caps + bond cap, factor-tilt
    strength, covariance window, mesh weights.
  - Target weights (side-by-side ERC / HRP / Mesh).
  - Per-sizer risk contributions.
  - Correlation heatmap + diagnostics.

Run from the repo root:

    streamlit run allocator_v2/main.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure sibling packages import cleanly whether launched via the run script or
# directly with ``streamlit run allocator_v2/main.py``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from allocator_v2 import backtest as bt_mod
from allocator_v2 import covariance as cov_mod
from allocator_v2 import data as data_mod
from allocator_v2 import ensemble
from allocator_v2 import sleeves as sl
from allocator_v2 import universe as uni
from allocator_v2.sizers.erc import risk_contributions


st.set_page_config(page_title="Risk-Parity Allocator", layout="wide")


# ── Data loading ─────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def _load_returns(tickers: tuple[str, ...]) -> pd.DataFrame:
    prices = data_mod.load_prices(tickers)
    return data_mod.returns_wide(prices, list(tickers))


@st.cache_data(ttl=600)
def _load_perf(tickers: tuple[str, ...]) -> pd.DataFrame:
    return data_mod.load_latest_perf(tickers)


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Risk-Parity Allocator")

all_tickers = uni.tickers()
selected = st.sidebar.multiselect(
    "Universe",
    options=all_tickers,
    default=all_tickers,
    help="Assets included in the covariance + sizing. Removing one drops it from all three models.",
)
if len(selected) < 3:
    st.error("Select at least 3 assets to size a portfolio.")
    st.stop()

window = st.sidebar.slider(
    "Covariance window (trading days)",
    min_value=30, max_value=252, value=90, step=10,
)

st.sidebar.subheader("Sleeve caps (policy)")
st.sidebar.caption(
    "Hard sleeve budgets. ERC/HRP equalise risk *within* each sleeve. "
    "Bond cap is enforced — overflow redistributes to equity + real."
)
sleeve_eq = st.sidebar.slider("Equity", 0.0, 0.9, sl.DEFAULT_SLEEVES["equity"], 0.05)
sleeve_real = st.sidebar.slider("Real assets", 0.0, 0.5, sl.DEFAULT_SLEEVES["real"], 0.05)
sleeve_bonds = st.sidebar.slider("Bonds", 0.0, 0.5, sl.DEFAULT_SLEEVES["bonds"], 0.05)
sleeve_cash = st.sidebar.slider("Cash (reserve)", 0.0, 0.5, sl.DEFAULT_SLEEVES["cash"], 0.01)
bond_cap = st.sidebar.slider(
    "Bond hard cap", 0.0, 0.4, 0.15, 0.01,
    help="Upper bound on total bond-sleeve weight. Applies to ERC, HRP and mesh.",
)
real_defensive_share = st.sidebar.slider(
    "Real sleeve — defensive share", 0.0, 1.0, 0.40, 0.05,
    help="Within the real-asset sleeve: share to defensive names (gold, REITs, infrastructure). "
         "Remainder goes to cyclical names (commodities, miners, energy).",
)
factor_tilt_strength = st.sidebar.slider(
    "Factor-tilt strength", 0.0, 1.5, 1.0, 0.1,
    help="Scales the momentum/quality tilts defined in universe.py (IWMO gets the biggest nudge). "
         "0 = off, 1 = defaults, 1.5 = aggressive.",
)

st.sidebar.subheader("MA-relative tilt (mean reversion)")
ma_strength = st.sidebar.slider(
    "MA tilt strength", 0.0, 1.5, 1.0, 0.1,
    help="Overweight below-MA, underweight above-MA. Applied inside equity + real sleeves. "
         "0 = off, 1 = default, 1.5 = aggressive.",
)
ma_band_lo = st.sidebar.slider(
    "Neutral band lower (price/MA)", 0.80, 1.00, 0.90, 0.01,
)
ma_band_hi = st.sidebar.slider(
    "Neutral band upper (price/MA)", 1.00, 1.30, 1.10, 0.01,
)
ma_barbell = st.sidebar.checkbox(
    "Barbell mode (mean-rev + trend)",
    value=False,
    help="U-shaped tilt: overweight oversold (<0.95·MA) AND strong uptrend (1.05–1.35·MA); "
         "underweight the mushy middle and parabolic extremes. Neutral band is ignored.",
)

st.sidebar.subheader("Position floor + cap")
min_weight = st.sidebar.slider(
    "Minimum per-asset weight", 0.0, 0.10, 0.03, 0.005,
    help="Drop any asset below this weight; redistribute pro-rata to survivors in the same sleeve.",
)
max_weight = st.sidebar.slider(
    "Maximum per-asset weight", 0.05, 0.30, 0.10, 0.01,
    help="Cap any asset above this weight; redistribute excess pro-rata to uncapped names in the same sleeve.",
)

# Pipeline's `ma_252` column is the percent deviation from the 252d MA
# (i.e. (price/MA − 1) × 100), not the MA level. Convert back to a ratio.
@st.cache_data(ttl=600)
def _load_ma_ratios(tickers: tuple[str, ...]) -> dict[str, float]:
    perf = data_mod.load_latest_perf(tickers)
    if perf.empty or "ma_252" not in perf.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in perf.iterrows():
        pct = row.get("ma_252")
        if pct is None:
            continue
        try:
            pct_f = float(pct)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pct_f):
            continue
        out[row["ticker"]] = 1.0 + pct_f / 100.0
    return out

ma_ratios = _load_ma_ratios(tuple(selected))

sleeve_policy = sl.SleevePolicy(
    weights={
        "equity": sleeve_eq,
        "real": sleeve_real,
        "bonds": sleeve_bonds,
        "cash": sleeve_cash,
    },
    bond_cap=bond_cap,
    real_split={
        "defensive": real_defensive_share,
        "cyclical": 1.0 - real_defensive_share,
    },
    factor_tilt_strength=factor_tilt_strength,
    ma_ratios=ma_ratios,
    ma_tilt_strength=ma_strength,
    ma_neutral_band=(ma_band_lo, ma_band_hi),
    ma_sleeves=("equity", "real"),
    ma_barbell=ma_barbell,
    min_weight=min_weight,
    max_weight=max_weight,
)

st.sidebar.subheader("Mesh weights")
w_erc = st.sidebar.slider("ERC", 0.0, 1.0, 0.5, 0.05)
w_hrp = st.sidebar.slider("HRP", 0.0, 1.0, 0.5, 0.05)
model_weights = {"erc": w_erc, "hrp": w_hrp}

total_nav = st.sidebar.number_input(
    "Portfolio NAV (£)",
    min_value=0, value=550_000, step=10_000,
    help="Used to convert target weights into £ amounts.",
)


# ── Return matrix + covariance ───────────────────────────────────────
rets = _load_returns(tuple(selected))
if rets.empty:
    st.error("No price history available for the selected universe. Run `make export` first.")
    st.stop()

cov = cov_mod.cov_matrix(rets, window=window)
corr = cov_mod.corr_matrix(rets, window=window)
vols = cov_mod.annualised_vol(rets, window=window)
if cov.empty or len(cov) < 3:
    st.error(
        f"Covariance matrix too small ({len(cov)} assets with enough history). "
        "Lengthen the window or select assets with more data."
    )
    st.stop()

active = list(cov.index)
cond = cov_mod.condition_number(cov)


# ── Sizers side-by-side + mesh ───────────────────────────────────────
st.title("Target weights")

sizer_df = ensemble.build_all_sizers(cov, corr, policy=sleeve_policy)
mesh_weights = ensemble.mesh(sizer_df, model_weights)
if min_weight > 0:
    mesh_weights = sl.apply_min_weight_floor(mesh_weights, min_weight, sl.ASSET_SLEEVE)
if max_weight > 0:
    mesh_weights = sl.apply_max_weight_cap(mesh_weights, max_weight, sl.ASSET_SLEEVE)
sizer_df["mesh"] = mesh_weights

# ── Top-line portfolio stats (mesh) ──────────────────────────────────
_mesh_arr = mesh_weights.values
_mesh_var = float(_mesh_arr @ cov.values @ _mesh_arr)
_mesh_vol = float(np.sqrt(_mesh_var))
_mesh_daily_vol = _mesh_vol / np.sqrt(252)

# Effective number of bets = 1 / HHI on risk contributions.
_rc_mesh_vec = risk_contributions(mesh_weights, cov)
_hhi = float((_rc_mesh_vec ** 2).sum()) if not _rc_mesh_vec.empty else 0.0
_eff_bets = 1.0 / _hhi if _hhi > 0 else float("nan")

# Equity sleeve variance share.
_eq_tickers = [t for t in mesh_weights.index if sl.ASSET_SLEEVE.get(t) == "equity"]
_eq_rc_share = float(_rc_mesh_vec.loc[_eq_tickers].sum()) if _eq_tickers else 0.0

_est_dd_low = _mesh_vol * 2.0
_est_dd_high = _mesh_vol * 2.5

stat1, stat2, stat3, stat4, stat5 = st.columns(5)
stat1.metric("Portfolio vol (ann.)", f"{_mesh_vol:.1%}")
stat2.metric("Daily vol", f"{_mesh_daily_vol:.2%}")
stat3.metric("Equity risk share", f"{_eq_rc_share:.0%}")
stat4.metric("Effective # bets", f"{_eff_bets:.1f}")
stat5.metric(
    "Est. max DD band", f"-{_est_dd_low:.0%} to -{_est_dd_high:.0%}",
    help="Rule-of-thumb: annual max drawdown ≈ 2× to 2.5× annualised vol.",
)

# Per-sleeve totals for sanity-check display.
sleeve_totals = pd.DataFrame(
    {
        col: [
            float(sizer_df.loc[[t for t in sizer_df.index if sl.ASSET_SLEEVE.get(t) == s], col].sum())
            for s in sl.DEFAULT_SLEEVES
            if s != "cash"
        ]
        for col in ("erc", "hrp", "mesh")
    },
    index=[s for s in sl.DEFAULT_SLEEVES if s != "cash"],
)
sleeve_totals.index.name = "Sleeve"

st.subheader("Sleeve totals (policy enforcement)")
st.dataframe(
    sleeve_totals.style.format("{:.1%}"),
    use_container_width=True,
    height=180,
)

weights_table = sizer_df.copy()
weights_table.index.name = "Ticker"
weights_table["Description"] = [uni.UNIVERSE[t].description for t in weights_table.index]
weights_table["Ann. vol"] = [vols.get(t, np.nan) for t in weights_table.index]

display = weights_table[["Description", "erc", "hrp", "mesh", "Ann. vol"]].copy()
display["£ (mesh)"] = display["mesh"] * float(total_nav)

st.dataframe(
    display.style.format(
        {
            "erc": "{:.1%}",
            "hrp": "{:.1%}",
            "mesh": "{:.1%}",
            "Ann. vol": "{:.1%}",
            "£ (mesh)": "£{:,.0f}",
        }
    ),
    use_container_width=True,
    height=560,
)

# Weights bar chart
bar_df = (
    sizer_df[["erc", "hrp", "mesh"]]
    .rename_axis("Ticker")
    .reset_index()
    .melt(id_vars="Ticker", var_name="Sizer", value_name="Weight")
)
fig = px.bar(
    bar_df,
    x="Ticker", y="Weight", color="Sizer",
    barmode="group",
    title="ERC vs HRP vs Mesh — per-asset weight",
)
fig.update_layout(yaxis_tickformat=".0%", height=440)
st.plotly_chart(fig, use_container_width=True)


# ── Risk contributions ───────────────────────────────────────────────
st.title("Risk contribution per sizer")
rc_df = pd.DataFrame(
    {
        "erc":  risk_contributions(sizer_df["erc"], cov),
        "hrp":  risk_contributions(sizer_df["hrp"], cov),
        "mesh": risk_contributions(sizer_df["mesh"], cov),
    }
).fillna(0.0)
rc_df.index.name = "Ticker"
st.dataframe(
    rc_df.style.format("{:.1%}"),
    use_container_width=True,
    height=420,
)


# ── Correlation heatmap ──────────────────────────────────────────────
st.title("Correlation matrix (90d)")
heat = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
    )
)
heat.update_layout(height=520)
st.plotly_chart(heat, use_container_width=True)


# ── Diagnostics ──────────────────────────────────────────────────────
with st.expander("Diagnostics"):
    st.write(f"Active assets: **{len(active)}** / {len(all_tickers)}")
    st.write(f"Covariance window: **{window}** trading days")
    st.write(f"Condition number: **{cond:,.1f}**  (>1e4 ≈ ill-conditioned)")
    st.write(
        "Portfolio vol (mesh): "
        f"**{float(np.sqrt(mesh_weights.values @ cov.values @ mesh_weights.values)):.1%}**"
    )

    perf = _load_perf(tuple(active))
    if not perf.empty:
        st.subheader("Latest performance")
        cols_show = [c for c in ("ticker", "price", "r_1mo", "r_3mo", "r_1y", "drawdown_52w", "vol_1y")
                     if c in perf.columns]
        st.dataframe(
            perf[cols_show].rename(
                columns={"r_1mo": "r_1m", "r_3mo": "r_3m"}
            ).style.format(
                {
                    "price": "{:.2f}",
                    "r_1m": "{:+.1f}%",
                    "r_3m": "{:+.1f}%",
                    "r_1y": "{:+.1f}%",
                    "drawdown_52w": "{:.1f}%",
                    "vol_1y": "{:.1f}%",
                },
                na_rep="—",
            ),
            hide_index=True,
            use_container_width=True,
        )


# ── Backtest ─────────────────────────────────────────────────────────
st.title("Backtest")
st.caption(
    "Walk-forward, monthly rebalance, no look-ahead. "
    "At each rebalance date we use prices up to that date to compute cov + MA, "
    "then hold the resulting weights for the month. Benchmarks: S&P 500 (CSP1), "
    "MSCI World (IWDA), 60/40 (CSP1/TLT)."
)

bt_col1, bt_col2, bt_col3 = st.columns(3)
bt_method = bt_col1.selectbox("Sizer", ["mesh", "erc", "hrp"], index=0, key="bt_method")
bt_start = bt_col2.selectbox(
    "Start",
    ["2012-01-01", "2015-01-01", "2017-01-01", "2018-01-01", "2020-01-01", "2022-01-01"],
    index=2, key="bt_start",
    help="Earliest usable dates depend on asset inceptions: MVOL/SGLD from 2016-02, "
         "TIP/GDX from 2015-12. Starts before these still work — the harness just holds fewer assets early on.",
)
run_bt = bt_col3.button("Run backtest", type="primary")

if run_bt:
    start_ts = pd.Timestamp(bt_start)
    all_bench = list(bt_mod.BENCHMARK_TICKERS)
    # Union of current universe + benchmark tickers for a single price pull.
    all_needed = tuple(sorted(set(selected) | set(all_bench)))
    prices_long = data_mod.load_prices(all_needed)
    if prices_long.empty:
        st.error("No price data available. Run `make export` first.")
        st.stop()

    prices_wide = (
        prices_long.pivot_table(index="date", columns="ticker", values="price")
        .sort_index().ffill(limit=3)
    )
    rets_full = prices_wide.pct_change(fill_method=None).dropna(how="all")

    # Slice to the portfolio universe for the strategy engine.
    port_cols = [c for c in selected if c in rets_full.columns]
    port_rets = rets_full[port_cols]
    port_prices = prices_wide[port_cols]

    with st.spinner("Running walk-forward backtest..."):
        result = bt_mod.run_backtest(
            port_rets, port_prices, sleeve_policy,
            cov_window=window, method=bt_method,
            min_weight=min_weight, max_weight=max_weight,
            start=start_ts,
        )

    if result.equity.empty:
        st.warning("No backtest output — not enough history for the selected universe + start date.")
    else:
        # Align all curves to start at 1.0 on the first in-period date.
        port_curve = result.equity
        port_curve = port_curve / port_curve.iloc[0]

        # Benchmarks on the same index.
        bench_cols = [c for c in all_bench if c in rets_full.columns]
        bench_rets = rets_full[bench_cols].loc[port_curve.index.min():]

        bench_curves: dict[str, pd.Series] = {}
        if "CSP1" in bench_cols:
            c = bt_mod.benchmark_equity(bench_rets, {"CSP1": 1.0}, start=port_curve.index.min())
            if not c.empty:
                bench_curves["S&P 500 (CSP1)"] = c / c.iloc[0]
        if "IWDA" in bench_cols:
            c = bt_mod.benchmark_equity(bench_rets, {"IWDA": 1.0}, start=port_curve.index.min())
            if not c.empty:
                bench_curves["MSCI World (IWDA)"] = c / c.iloc[0]
        if "CSP1" in bench_cols and "TLT" in bench_cols:
            c = bt_mod.benchmark_equity(bench_rets, {"CSP1": 0.6, "TLT": 0.4}, start=port_curve.index.min())
            if not c.empty:
                bench_curves["60/40 (CSP1/TLT)"] = c / c.iloc[0]

        # Stats table.
        stat_rows = {f"Allocator ({bt_method.upper()})": bt_mod.summary_stats(port_curve)}
        for name, curve in bench_curves.items():
            stat_rows[name] = bt_mod.summary_stats(curve)
        stats_df = pd.DataFrame(stat_rows).T
        stats_df = stats_df.rename(columns={
            "cagr": "CAGR", "vol": "Vol", "sharpe": "Sharpe",
            "max_dd": "Max DD", "total_return": "Total Return",
        })
        st.dataframe(
            stats_df.style.format({
                "CAGR": "{:.1%}",
                "Vol": "{:.1%}",
                "Sharpe": "{:.2f}",
                "Max DD": "{:.1%}",
                "Total Return": "{:.1%}",
            }),
            use_container_width=True,
        )

        # Equity curves.
        chart_df = pd.DataFrame({f"Allocator ({bt_method.upper()})": port_curve})
        for name, curve in bench_curves.items():
            chart_df[name] = curve.reindex(chart_df.index).ffill()
        chart_df = chart_df.dropna(how="all").reset_index().rename(columns={"index": "date"})
        if "date" not in chart_df.columns:
            chart_df = chart_df.rename(columns={chart_df.columns[0]: "date"})
        long = chart_df.melt(id_vars="date", var_name="Series", value_name="Equity")
        line = px.line(long, x="date", y="Equity", color="Series",
                       title=f"Equity curves (start = {bt_start}, method = {bt_method.upper()})")
        line.update_layout(height=460)
        st.plotly_chart(line, use_container_width=True)

        # Drawdown curves.
        dd_df = pd.DataFrame()
        for name, curve in {f"Allocator ({bt_method.upper()})": port_curve, **bench_curves}.items():
            dd_df[name] = curve / curve.cummax() - 1.0
        dd_df = dd_df.reset_index().rename(columns={"index": "date"})
        if "date" not in dd_df.columns:
            dd_df = dd_df.rename(columns={dd_df.columns[0]: "date"})
        dd_long = dd_df.melt(id_vars="date", var_name="Series", value_name="Drawdown")
        dd_fig = px.line(dd_long, x="date", y="Drawdown", color="Series", title="Drawdown")
        dd_fig.update_layout(yaxis_tickformat=".0%", height=360)
        st.plotly_chart(dd_fig, use_container_width=True)

        with st.expander("Weights over time"):
            wt = result.weights_over_time
            st.dataframe(wt.style.format("{:.1%}"), use_container_width=True, height=400)
