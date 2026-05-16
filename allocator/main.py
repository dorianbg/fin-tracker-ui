"""Antifragile Two-Bucket Portfolio Allocator.

Run from repo root:
    streamlit run allocator/main.py

Tabs:
  1. Allocation   — current vs target stacked bars per bucket
  2. Valuation    — regional tilts + macro triggers table
  3. Rebalance    — drift alerts (current % vs target %)
  4. Deploy       — 12-month deployment cockpit + tranche history
  5. Holdings     — CRUD for positions per bucket
  6. Warnings     — wrapper eligibility + reporting-fund flags
  7. Factors      — PE vs history (z-score) + PEG ratio screens
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure the allocator package is importable from wherever Streamlit was launched
sys.path.insert(0, os.path.dirname(__file__))

import holdings as h
import data_sources as ds
from buckets import SIPP, ISA, GIA, ALL_BUCKETS, SleeveTarget
from instruments import INSTRUMENTS, THEMATIC_EXTRAS, lookup
from valuation import (
    MacroData, RegionData, DeploymentState,
    compute_region_tilts, apply_tilts_to_weights,
    compute_bond_triggers, compute_deployment_pace,
)
from factors import compute_factor_scores, as_dataframe as factors_df
from strategy import (
    build_buy_candidates,
    build_theme_watchlist,
    build_wrapper_candidate_table,
)
from construction import build_portfolio_plan, summarize_portfolio_plan, build_satellite_candidates
from lookthrough import fetch_etf_top_holdings, map_holdings_to_universe
from themes import (
    build_theme_snapshot,
    build_theme_correlation,
    build_theme_stock_screen,
    build_portfolio_weighted_theme_correlation,
    build_sleeve_ticker_weights,
    build_theme_lookthrough_concentration,
    classify_instrument_regimes,
    classify_theme_regimes,
    get_theme_regimes,
)
from sizing import inverse_vol_weights, risk_contribution_report
from covariance import correlation_matrix, correlation_drift, concentrated_pairs
from rebalance import build_rebalance_signals


def _stock_region(ins) -> str:
    if ins is None or ins.vehicle_type != "stock":
        return "Fund"
    country = str(ins.country or ins.domicile or "").upper()
    if country in {"UK", "GB"}:
        return "UK"
    if country in {"FR", "DE", "IT", "ES", "NL", "BE", "CH", "EU", "GERMANY", "FRANCE", "ITALY", "SPAIN", "NETHERLANDS", "BELGIUM", "SWITZERLAND"}:
        return "Europe"
    if country in {"US", "USA"}:
        return "US"
    if country in {"CN", "CHINA", "HK", "HONG KONG", "TW", "TAIWAN", "KR", "KOREA", "IN", "INDIA"}:
        return "Asia / EM"
    return "Other"


def _is_stock_ticker(ticker: str) -> bool:
    ins = lookup(str(ticker))
    return bool(ins and ins.vehicle_type == "stock")

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Allocator",
    page_icon="🏛",
    layout="wide",
)

# ── Initialise DB ─────────────────────────────────────────────────────
h.init_db()

# ── Sidebar: bucket sizes ─────────────────────────────────────────────
st.sidebar.title("Bucket Sizes (£)")
sipp_size = st.sidebar.number_input("SIPP total (£)", min_value=0, value=300_000, step=5_000)
isa_size = st.sidebar.number_input("ISA total (£)", min_value=0, value=150_000, step=5_000)
gia_size = st.sidebar.number_input("GIA total (£)", min_value=0, value=100_000, step=5_000)
total_size = sipp_size + isa_size + gia_size

BUCKET_SIZES = {"SIPP": sipp_size, "ISA": isa_size, "GIA": gia_size}

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh market data", help="Re-fetches prices, FRED yields, Shiller CAPE"):
    with st.spinner("Fetching prices…"):
        ds.refresh_etf_prices(force=True)
    with st.spinner("Fetching macro data…"):
        ds.refresh_macro(force=True)
    with st.spinner("Fetching Shiller CAPE…"):
        ds.refresh_shiller(force=True)
    with st.spinner("Fetching factor data (PE/PEG)…"):
        ds.refresh_factor_data(force=True)
    ds.refresh_ibkr_fundamentals(force=True)
    st.sidebar.success("Done")

# Trigger background refresh on first load (no-op if cache is fresh)
ds.refresh_etf_prices()
ds.refresh_macro()
ds.refresh_region_pe()
ds.refresh_factor_data()
ds.refresh_ibkr_fundamentals()

# ── Cached analytics ──────────────────────────────────────────────────
@st.cache_data(ttl=300)
def cached_portfolio(sipp: int, isa: int, gia: int) -> pd.DataFrame:
    sizes = {"SIPP": sipp, "ISA": isa, "GIA": gia}
    return ds.compute_portfolio_analytics(h._DB_PATH, sizes)


@st.cache_data(ttl=3600)
def cached_etf_meta() -> pd.DataFrame:
    return ds.get_etf_meta()


@st.cache_data(ttl=3600)
def cached_macro() -> dict:
    return ds.get_macro()


@st.cache_data(ttl=3600 * 6)
def cached_region_pe() -> dict:
    return ds.get_region_pe()


portfolio_df = cached_portfolio(sipp_size, isa_size, gia_size)
etf_meta = cached_etf_meta()
macro = cached_macro()
gbpusd = ds.get_gbpusd()

# Build a MacroData object (use stub values where FRED data unavailable)
macro_data = MacroData(
    uk_real_yield_10y=macro.get("uk_real_10y") or 0.012,
    us_real_yield_10y=macro.get("us_real_10y") or 0.02,
    us_10y_nominal=macro.get("us_10y_nominal") or 0.042,
    em_hy_spread=macro.get("em_hy_spread_over_ust") or 0.04,
    acwi_drawdown_30d=ds.get_acwi_drawdown_30d(),
    acwi_forward_pe_pct=0.85,  # TODO: compute from history when we have more data
)
bond_triggers = compute_bond_triggers(macro_data)


# ── Title ─────────────────────────────────────────────────────────────
st.title("🏛 Antifragile Portfolio Allocator")
col1, col2, col3 = st.columns(3)
col1.metric("SIPP", f"£{sipp_size:,.0f}", f"{100*sipp_size/total_size:.0f}% of total" if total_size else "")
col2.metric("ISA",  f"£{isa_size:,.0f}",  f"{100*isa_size/total_size:.0f}% of total" if total_size else "")
col3.metric("GIA",  f"£{gia_size:,.0f}",  f"{100*gia_size/total_size:.0f}% of total" if total_size else "")

tabs = st.tabs(["🧭 Draft", "🎯 Allocation", "📊 Valuation", "⚖️ Rebalance", "🚀 Deploy", "📋 Holdings", "⚠️ Warnings", "🔬 Factors"])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — DRAFT
# ═══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("First-Draft Portfolio")
    st.caption(
        "Core draft stays on the primary strategic tickers. "
        "Signals are used to classify what to buy now, what to stage, and what still lacks enough data."
    )

    factor_raw = ds.get_factor_data()
    etf_meta_for_factors = ds.get_etf_meta()
    if factor_raw.empty or etf_meta_for_factors.empty:
        st.info("Factor data not ready yet. Refresh market and factor data first.")
    else:
        score_df = factors_df(compute_factor_scores(factor_raw, etf_meta_for_factors))
        # Full-universe timing so the regime/vol layer covers every plan ticker,
        # not only those with factor (PE) coverage. The regime classifier needs
        # all sleeves visible to compute a universe-relative vol median.
        timing_df = classify_instrument_regimes(ds.get_entry_timing_metrics(()))
        theme_snap_for_regime = build_theme_snapshot(timing_df)
        theme_regimes = get_theme_regimes(theme_snap_for_regime)
        draft_df = build_portfolio_plan(
            score_df, BUCKET_SIZES,
            selection_mode="primary_first",
            timing_df=timing_df,
            theme_regimes=theme_regimes,
            bond_triggers=bond_triggers,
        )
        summary_df = summarize_portfolio_plan(draft_df)
        alt_df = build_satellite_candidates(score_df)

        action_colours = {
            "APPROVED": "background-color: #1a7a4a; color: white",
            "BUILD_CORE": "background-color: #d4edda",
            "WATCHLIST": "background-color: #fff3cd",
            "NO_DATA": "background-color: #e2e3e5",
            "NOT_ACTIVE": "background-color: #d6d8db; color: #383d41",
            "REJECT": "background-color: #721c24; color: white",
        }

        if not summary_df.empty:
            st.dataframe(
                summary_df.style.format({"target_gbp": "£{:,.0f}"}),
                hide_index=True,
                use_container_width=True,
            )

        buy_now_df = draft_df[draft_df["action"].isin(["APPROVED", "BUILD_CORE"])].copy()
        stage_df = draft_df[draft_df["action"].isin(["WATCHLIST", "NO_DATA"])].copy()
        dormant_df = draft_df[draft_df["action"] == "NOT_ACTIVE"].copy()

        hide_non_actionable = st.checkbox(
            "Hide NO_DATA / NOT_ACTIVE rows",
            value=False,
            help="When on, the Core Draft table and Stage Later list show only rows with real factor coverage and active tactical triggers.",
        )
        if hide_non_actionable:
            stage_df = stage_df[stage_df["action"] != "NO_DATA"]

        st.subheader("Buy Now / Build Core")
        st.dataframe(
            buy_now_df[[
                "account_type", "ticker", "target_gbp",
                "r_1m", "r_3m", "r_6m", "r_1y",
                "drawdown_52w", "pct_above_ma200", "range_52w_pos",
                "vol_3m", "vol_1y", "z_1mo",
                "strategy_signal", "theme_regime", "action", "rationale"
            ]].style
            .applymap(lambda v: action_colours.get(str(v), ""), subset=["action"])
            .format({
                "target_gbp": lambda x: f"£{x:,.0f}" if pd.notna(x) else "—",
                "r_1m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_3m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_6m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_1y": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "drawdown_52w": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "pct_above_ma200": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "range_52w_pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "vol_3m": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "vol_1y": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "z_1mo": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
            }),
            hide_index=True,
            use_container_width=True,
            height=320,
        )

        st.subheader("Stage Later / Missing Data")
        st.dataframe(
            stage_df[[
                "account_type", "ticker", "target_gbp",
                "r_1m", "r_3m", "r_6m", "r_1y",
                "drawdown_52w", "pct_above_ma200", "range_52w_pos",
                "vol_3m", "vol_1y", "z_1mo",
                "strategy_signal", "theme_regime", "action", "rationale"
            ]].style
            .applymap(lambda v: action_colours.get(str(v), ""), subset=["action"])
            .format({
                "target_gbp": lambda x: f"£{x:,.0f}" if pd.notna(x) else "—",
                "r_1m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_3m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_6m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_1y": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "drawdown_52w": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "pct_above_ma200": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "range_52w_pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "vol_3m": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "vol_1y": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "z_1mo": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
            }),
            hide_index=True,
            use_container_width=True,
            height=320,
        )

        st.subheader("Core Draft by Sleeve")
        core_draft_display = draft_df.copy()
        if hide_non_actionable:
            core_draft_display = core_draft_display[~core_draft_display["action"].isin(["NO_DATA", "NOT_ACTIVE"])]
        st.dataframe(
            core_draft_display[[
                "account_type", "sleeve", "ticker", "name", "target_weight",
                "target_gbp", "r_1y", "drawdown_52w", "range_52w_pos",
                "price_ma200", "strategy_signal", "theme_regime", "action", "candidate_score"
            ]].style
            .applymap(lambda v: action_colours.get(str(v), ""), subset=["action"])
            .format({
                "target_weight": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "target_gbp": lambda x: f"£{x:,.0f}" if pd.notna(x) else "—",
                "r_1y": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "drawdown_52w": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "range_52w_pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "price_ma200": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                "candidate_score": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            }),
            hide_index=True,
            use_container_width=True,
            height=420,
        )

        if not dormant_df.empty:
            st.subheader("Dormant Tactical Sleeves")
            st.caption("Tactical sleeves whose macro trigger is not firing. Zero weight today — kept visible so you can see the dormant option set.")
            st.dataframe(
                dormant_df[[
                    "account_type", "sleeve", "ticker", "name", "rationale"
                ]],
                hide_index=True,
                use_container_width=True,
            )

        if not alt_df.empty:
            st.subheader("Optional Alternatives / Satellite Ideas")
            st.caption("Non-core ideas. Keep these separate from the base draft.")
            st.dataframe(
                alt_df[[
                    "Wrapper", "Sleeve", "Ticker", "Name", "Vehicle", "Strategy signal",
                    "Candidate score", "Rationale"
                ]].style.format({"Candidate score": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
                height=320,
            )


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — ALLOCATION
# ═══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Target Allocation vs Current")

    drawdown_override = st.slider(
        "ISA/GIA drawdown tolerance (%)",
        min_value=-25, max_value=-5, value=-10, step=1,
        help="Soften the -10% target to accept a higher return at the cost of larger potential drawdowns."
    )

    def build_target_df(bucket, label: str) -> pd.DataFrame:
        rows = []
        for t in bucket.targets:
            ticker_label = f"{t.sleeve.replace('_', ' ').title()} ({t.primary_ticker})"
            rows.append({
                "Sleeve": ticker_label,
                "Target %": t.weight * 100,
                "Bucket": label,
                "Tactical": t.is_tactical,
            })
        return pd.DataFrame(rows)

    def build_current_df(account_type: str, bucket_size: float) -> pd.DataFrame:
        sub = portfolio_df[portfolio_df["account_type"] == account_type] if not portfolio_df.empty else pd.DataFrame()
        total_invested = float(sub["gbp_value"].sum()) if not sub.empty else 0.0
        cash = max(0.0, bucket_size - total_invested)
        rows = []
        if not sub.empty:
            for _, row in sub.iterrows():
                ins = lookup(row["ticker"])
                sleeve = ins.sleeve if ins else "unknown"
                rows.append({
                    "Sleeve": f"{sleeve.replace('_', ' ').title()} ({row['ticker']})",
                    "Current %": 100.0 * row["gbp_value"] / bucket_size if bucket_size else 0,
                    "GBP": row["gbp_value"],
                })
        if cash > 0:
            rows.append({
                "Sleeve": "Uninvested Cash",
                "Current %": 100.0 * cash / bucket_size if bucket_size else 0,
                "GBP": cash,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Sleeve", "Current %", "GBP"])

    for bucket_obj, bucket_name, bucket_size in [
        (SIPP, "SIPP", sipp_size),
        (ISA, "ISA", isa_size),
        (GIA, "GIA", gia_size),
    ]:
        st.subheader(f"{bucket_name}  (£{bucket_size:,.0f}  |  target drawdown: {bucket_obj.drawdown_tolerance:.0%})")
        target_df = build_target_df(bucket_obj, bucket_name)
        current_df = build_current_df(bucket_name, bucket_size)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("Target")
            if not target_df.empty:
                fig = px.bar(
                    target_df[~target_df["Tactical"]],
                    x="Target %", y="Bucket", color="Sleeve",
                    orientation="h", height=160,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    target_df[["Sleeve", "Target %"]].rename(columns={"Target %": "Wt %"}),
                    hide_index=True, use_container_width=True,
                )
        with c2:
            st.caption("Current")
            if not current_df.empty:
                fig2 = px.bar(
                    current_df, x="Current %", y=[bucket_name] * len(current_df),
                    color="Sleeve", orientation="h", height=160,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
                fmt = {"GBP": "£{:,.0f}", "Current %": "{:.1f}%"}
                st.dataframe(
                    current_df.style.format(fmt),
                    hide_index=True, use_container_width=True,
                )
            else:
                total_invested = portfolio_df[portfolio_df["account_type"] == bucket_name]["gbp_value"].sum() if not portfolio_df.empty else 0
                cash_pct = 100.0 * (1 - total_invested / bucket_size) if bucket_size else 100
                st.info(f"No holdings entered yet — effectively 100% cash ({cash_pct:.0f}%)")

        # Projected real return banner for this bucket
        eq_w = sum(t.weight for t in bucket_obj.targets if t.sleeve.startswith("equity"))
        ra_w = sum(t.weight for t in bucket_obj.targets if t.sleeve.startswith("real"))
        bd_w = sum(t.weight for t in bucket_obj.targets if t.sleeve.startswith("bonds"))
        ca_w = sum(t.weight for t in bucket_obj.targets if t.sleeve.startswith("cash"))
        proj_real = eq_w * 5.0 + ra_w * 3.0 + bd_w * 2.0 + ca_w * 1.0
        st.caption(
            f"Projected real return (rough): **{proj_real:.1f}%**  |  "
            f"Equity {eq_w:.0%} / Real assets {ra_w:.0%} / Bonds {bd_w:.0%} / Cash {ca_w:.0%}"
        )
        st.divider()

    # Blended portfolio real return
    if total_size > 0:
        weighted_real = (
            sipp_size * sum(t.weight * {"equity": 5, "real": 3, "bonds": 2, "cash": 1}.get(t.sleeve.split("_")[0], 0) for t in SIPP.targets)
            + isa_size * sum(t.weight * {"equity": 5, "real": 3, "bonds": 2, "cash": 1}.get(t.sleeve.split("_")[0], 0) for t in ISA.targets)
            + gia_size * sum(t.weight * {"equity": 5, "real": 3, "bonds": 2, "cash": 1}.get(t.sleeve.split("_")[0], 0) for t in GIA.targets)
        ) / total_size
        st.success(
            f"**Blended projected real return: {weighted_real:.1f}%**  "
            f"(assumes equity 5%, real assets 3%, bonds 2%, cash 1%).  "
            f"The -10% ISA/GIA drawdown constraint is the main return drag — "
            f"soften the slider above to see the tradeoff."
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — VALUATION
# ═══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Valuation Signals & Regional Tilts")

    st.subheader("Macro indicators")
    m1, m2, m3, m4 = st.columns(4)
    def _fmt(v, mul=100, suffix="%", na="N/A"):
        return f"{v * mul:.2f}{suffix}" if v is not None else na

    m1.metric("UK 10y real yield", _fmt(macro.get("uk_real_10y")),
              help="≈ UK nominal 10y gilt - 3% LT inflation. Linker trigger: >1.5%")
    m2.metric("US 10y real yield", _fmt(macro.get("us_real_10y")),
              help="FRED DFII10 TIPS yield")
    m3.metric("US 10y nominal", _fmt(macro.get("us_10y_nominal")),
              help="Long-duration trigger: >5%")
    m4.metric("EM HY spread over UST", _fmt(macro.get("em_hy_spread_over_ust")),
              help="EM HY yield minus US 10y. EM bond trigger: >6%")

    # Bond triggers
    st.subheader("Active bond triggers")
    trig_rows = [
        {"Trigger": "UK linkers (INXG)", "Active": "✅ YES" if bond_triggers["linkers_extra"] > 0 else "❌ NO",
         "Condition": "UK 10y real > 1.5%", "Added weight": f"{bond_triggers['linkers_extra']:.0%}"},
        {"Trigger": "EM USD bonds (EMB)", "Active": "✅ YES" if bond_triggers["em_usd"] > 0 else "❌ NO",
         "Condition": "EM spread > 6%", "Added weight": f"{bond_triggers['em_usd']:.0%}"},
        {"Trigger": "Long duration (TLT)", "Active": "✅ YES" if bond_triggers["long_dur"] > 0 else "❌ NO",
         "Condition": "US 10y nominal > 5%", "Added weight": f"{bond_triggers['long_dur']:.0%}"},
    ]
    st.dataframe(pd.DataFrame(trig_rows), hide_index=True, use_container_width=True)

    # Regional tilt engine
    st.subheader("Regional equity tilts (valuation + momentum filter)")
    st.caption(
        "Adjust each region's forward P/E below. Tilt = f(earnings-yield z-score vs peers) × "
        "momentum dampener (0.7× if price < 0.85·MA200 or > 1.30·MA200)."
    )

    # Allow manual PE + price/MA200 input per region
    _live_pe = cached_region_pe()
    _pe_fallbacks = {"US": 21.5, "Europe": 14.0, "Japan": 14.5, "EM": 12.5, "UK": 11.5}
    region_defaults = {
        "US":     {"fwd_pe": _live_pe.get("US",     {}).get("pe") or _pe_fallbacks["US"],     "acwi_w": 0.64, "_src": _live_pe.get("US",     {}).get("source", "manual")},
        "Europe": {"fwd_pe": _live_pe.get("Europe", {}).get("pe") or _pe_fallbacks["Europe"], "acwi_w": 0.13, "_src": _live_pe.get("Europe", {}).get("source", "manual")},
        "Japan":  {"fwd_pe": _live_pe.get("Japan",  {}).get("pe") or _pe_fallbacks["Japan"],  "acwi_w": 0.06, "_src": _live_pe.get("Japan",  {}).get("source", "manual")},
        "EM":     {"fwd_pe": _live_pe.get("EM",     {}).get("pe") or _pe_fallbacks["EM"],     "acwi_w": 0.11, "_src": _live_pe.get("EM",     {}).get("source", "manual")},
        "UK":     {"fwd_pe": _live_pe.get("UK",     {}).get("pe") or _pe_fallbacks["UK"],     "acwi_w": 0.04, "_src": _live_pe.get("UK",     {}).get("source", "manual")},
    }

    with st.expander("Edit regional inputs", expanded=False):
        region_inputs = {}
        cols = st.columns(len(region_defaults))
        for i, (rname, defaults) in enumerate(region_defaults.items()):
            with cols[i]:
                src_badge = f"<sub style='color:grey'>via {defaults['_src']}</sub>" if defaults["_src"] != "manual" else "<sub style='color:orange'>manual fallback</sub>"
                st.markdown(f"**{rname}** {src_badge}", unsafe_allow_html=True)
                # Get MA200 ratio from cache if we have an ETF proxy
                proxy = {"US": "IWQU", "Europe": "VEUR", "Japan": "IJPA", "EM": "EIMI", "UK": "ISF"}.get(rname)
                meta_row = etf_meta[etf_meta["ticker"] == proxy] if proxy and not etf_meta.empty else pd.DataFrame()
                ma200_ratio_default = float(
                    meta_row["last_price"].iloc[0] / meta_row["ma200"].iloc[0]
                ) if not meta_row.empty and meta_row["ma200"].iloc[0] > 0 else 1.05
                fwd_pe = st.number_input(f"Fwd P/E", value=float(defaults["fwd_pe"]), step=0.5, key=f"pe_{rname}")
                ma_ratio = st.number_input(f"Price/MA200", value=ma200_ratio_default, step=0.01, format="%.3f", key=f"ma_{rname}")
                region_inputs[rname] = {"fwd_pe": fwd_pe, "acwi_w": defaults["acwi_w"], "ma_ratio": ma_ratio}

    regions = [
        RegionData(
            name=rname,
            forward_pe=v["fwd_pe"],
            price=v["ma_ratio"] * 100,
            ma200=100.0,
            acwi_base_weight=v["acwi_w"],
        )
        for rname, v in region_inputs.items()
    ]
    tilts = compute_region_tilts(regions)
    final_weights = apply_tilts_to_weights(regions, tilts)

    tilt_rows = []
    for r in regions:
        ey = 100.0 / r.forward_pe
        tilt_rows.append({
            "Region": r.name,
            "Fwd P/E": r.forward_pe,
            "Earnings yield %": round(ey, 2),
            "Price/MA200": round(region_inputs[r.name]["ma_ratio"], 3),
            "ACWI base %": round(r.acwi_base_weight * 100, 1),
            "Tilt mult": round(tilts.get(r.name, 1.0), 3),
            "Final weight %": round(final_weights.get(r.name, 0) * 100, 1),
        })
    tilt_df = pd.DataFrame(tilt_rows)

    def color_tilt(v):
        if v > 1.1: return "background-color: #d4edda"
        if v < 0.9: return "background-color: #f8d7da"
        return ""

    st.dataframe(
        tilt_df.style.applymap(color_tilt, subset=["Tilt mult"]).format({
            "Fwd P/E": "{:.1f}",
            "Earnings yield %": "{:.2f}%",
            "Price/MA200": "{:.3f}",
            "ACWI base %": "{:.1f}%",
            "Tilt mult": "{:.3f}",
            "Final weight %": "{:.1f}%",
        }),
        hide_index=True, use_container_width=True,
    )

    fig_tilt = go.Figure()
    fig_tilt.add_bar(name="ACWI base", x=tilt_df["Region"], y=tilt_df["ACWI base %"], marker_color="#aec6cf")
    fig_tilt.add_bar(name="After tilt", x=tilt_df["Region"], y=tilt_df["Final weight %"], marker_color="#4c8cbf")
    fig_tilt.update_layout(barmode="group", height=280, margin=dict(l=0, r=0, t=20, b=0), title="Regional weights: ACWI base vs valuation-tilted")
    st.plotly_chart(fig_tilt, use_container_width=True)

    # ACWI drawdown indicator
    dd = macro_data.acwi_drawdown_30d
    st.metric("ACWI 30-day drawdown", f"{dd:.1%}", help="Drives deployment accelerator")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — REBALANCE
# ═══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Rebalance Alerts")
    st.caption("Alerts when a sleeve's actual weight drifts > 200bps or > 25% relative from target.")

    # ── Per-ticker signals driven by regime + drift ──
    st.subheader("Per-ticker signals (regime-aware)")
    st.caption(
        "Combines drift with the instrument's current regime. Stretched names are kept at HOLD even when "
        "slightly underweight; falling_knife and dead_money regimes escalate to SELL / ROTATE."
    )
    if not portfolio_df.empty:
        _reb_timing = classify_instrument_regimes(ds.get_entry_timing_metrics(()))
        reb_df = build_rebalance_signals(portfolio_df, BUCKET_SIZES, ALL_BUCKETS, _reb_timing)
        if not reb_df.empty:
            reb_colours = {
                "SELL":   "background-color: #721c24; color: white",
                "TRIM":   "background-color: #fff3cd",
                "ROTATE": "background-color: #ffe5b4",
                "TOP_UP": "background-color: #1a7a4a; color: white",
                "HOLD":   "background-color: #e2e3e5",
                "KEEP":   "",
            }
            show_cols = [
                "account_type", "ticker", "sleeve", "regime", "strategy_signal",
                "rebalance_action", "current_weight", "target_weight", "drift_pp", "reason",
            ]
            st.dataframe(
                reb_df[show_cols].style
                .applymap(lambda v: reb_colours.get(str(v), ""), subset=["rebalance_action"])
                .format({
                    "current_weight": lambda x: f"{x:.2%}" if pd.notna(x) else "—",
                    "target_weight": lambda x: f"{x:.2%}" if pd.notna(x) else "—",
                    "drift_pp": lambda x: f"{x:+.2f}pp" if pd.notna(x) else "—",
                }),
                hide_index=True, use_container_width=True, height=480,
            )
    else:
        st.info("No holdings recorded yet — add positions in the Holdings tab.")

    st.divider()
    st.subheader("Sleeve-level drift")

    for bucket_obj, bucket_name, bucket_size in [
        (SIPP, "SIPP", sipp_size),
        (ISA, "ISA", isa_size),
        (GIA, "GIA", gia_size),
    ]:
        st.subheader(f"{bucket_name}")
        if bucket_size == 0:
            st.info("Bucket size is 0 — set in sidebar.")
            continue

        # Build a lookup: sleeve → current %
        sub = portfolio_df[portfolio_df["account_type"] == bucket_name] if not portfolio_df.empty else pd.DataFrame()
        sleeve_current: dict[str, float] = {}
        if not sub.empty:
            for _, row in sub.iterrows():
                ins = lookup(row["ticker"])
                sleeve = ins.sleeve if ins else "unknown"
                sleeve_current[sleeve] = sleeve_current.get(sleeve, 0) + float(row["gbp_value"]) / bucket_size * 100

        drift_rows = []
        for t in bucket_obj.targets:
            target_pct = t.weight * 100
            current_pct = sleeve_current.get(t.sleeve, 0.0)
            drift_pp = current_pct - target_pct
            drift_rel = drift_pp / target_pct * 100 if target_pct > 0 else 0
            alert = (abs(drift_pp) > 2.0 or abs(drift_rel) > 25) and not t.is_tactical
            drift_rows.append({
                "Sleeve": t.sleeve.replace("_", " ").title(),
                "Ticker": t.primary_ticker,
                "Target %": round(target_pct, 1),
                "Current %": round(current_pct, 1),
                "Drift pp": round(drift_pp, 2),
                "Drift %": round(drift_rel, 1),
                "Action": ("🔴 BUY" if drift_pp < -2 else "🔵 SELL" if drift_pp > 2 else "✅ OK") if not t.is_tactical else "—",
                "Alert": alert,
            })
        drift_df = pd.DataFrame(drift_rows)
        alert_mask = drift_df["Alert"].to_numpy()
        display_df = drift_df.drop(columns=["Alert"])

        def highlight_alerts(row, mask=alert_mask):
            return ["background-color: #fff3cd"] * len(row) if mask[row.name] else [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_alerts, axis=1).format({
                "Target %": "{:.1f}%", "Current %": "{:.1f}%",
                "Drift pp": "{:+.2f}pp", "Drift %": "{:+.1f}%",
            }),
            hide_index=True, use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — DEPLOY
# ═══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Deployment Cockpit")
    st.caption(
        "Rule: deploy 30% on day 1 into defensive sleeves only. "
        "Then 5%/month over 12 months, accelerated on ACWI drawdowns, "
        "decelerated when global valuations are >90th percentile."
    )

    for bucket_name, bucket_size in BUCKET_SIZES.items():
        if bucket_size == 0:
            continue
        st.subheader(f"{bucket_name}  (£{bucket_size:,.0f})")
        c1, c2 = st.columns([1, 1])

        with c1:
            total_initial = st.number_input(
                f"Initial total to deploy (£)", value=bucket_size,
                key=f"deploy_init_{bucket_name}", step=5_000
            )
            months_elapsed = st.slider(
                "Months since deployment started",
                0, 24, 0, key=f"elapsed_{bucket_name}"
            )
            months_remaining = max(1, 12 - months_elapsed)
            invested = float(portfolio_df[portfolio_df["account_type"] == bucket_name]["gbp_value"].sum()) if not portfolio_df.empty else 0.0
            cash_remaining = max(0.0, total_initial - invested)

            state = DeploymentState(
                total_initial=float(total_initial),
                cash_remaining=cash_remaining,
                months_remaining=months_remaining,
            )
            pace, reason = compute_deployment_pace(state, macro_data)
            tranche_gbp = pace * total_initial

            st.metric("Invested so far", f"£{invested:,.0f}", f"{100*invested/total_initial:.1f}% of total" if total_initial else "")
            st.metric("Cash remaining", f"£{cash_remaining:,.0f}")
            st.metric(
                "This month's tranche",
                f"£{tranche_gbp:,.0f}",
                reason.replace("_", " "),
                delta_color="normal" if "drawdown" in reason else "off",
            )

            trigger_label = {
                "drawdown>20": "🚨 Accelerate: ACWI -20%+",
                "drawdown>10": "⚡ Accelerate: ACWI -10%",
                "drawdown>5":  "📈 Mild accelerate: ACWI -5%",
                "valuation>90pct": "⏸ Slow: valuation >90th pct",
                "default": "📅 Default pace",
                "deployment_complete": "✅ Fully deployed",
            }.get(reason, reason)
            st.info(trigger_label)

        with c2:
            # Deployment projection chart
            sims = []
            rem = cash_remaining
            for mo in range(1, months_remaining + 1):
                sim_state = DeploymentState(
                    total_initial=float(total_initial),
                    cash_remaining=rem,
                    months_remaining=months_remaining - mo + 1,
                )
                p, _ = compute_deployment_pace(sim_state, macro_data)
                deploy = p * float(total_initial)
                deployed = invested + (float(total_initial) - cash_remaining + deploy * (mo))
                sims.append({"Month": months_elapsed + mo, "Deployed (£)": min(deployed, total_initial)})
                rem = max(0, rem - deploy)
            sim_df = pd.DataFrame(sims)
            if not sim_df.empty:
                fig_dep = px.area(
                    sim_df, x="Month", y="Deployed (£)",
                    title="Projected deployment curve (default pace)",
                    height=220,
                    color_discrete_sequence=["#4c8cbf"],
                )
                fig_dep.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_dep, use_container_width=True)

        # Deployment log
        log_rows = h.fetch_deployment_log(account_type=bucket_name)
        if log_rows:
            st.dataframe(
                pd.DataFrame(log_rows, columns=["Date", "Bucket", "Ticker", "Amount (£)", "Trigger"]),
                hide_index=True, use_container_width=True,
            )
            if st.button(f"🗑 Clear log ({bucket_name})", key=f"clrlog_{bucket_name}"):
                with h.get_db() as conn:
                    conn.execute("DELETE FROM deployment_log WHERE account_type = ?", [bucket_name])
                    conn.commit()
                st.rerun()
        else:
            st.caption("No deployment entries yet.")

        if st.button(f"📝 Record manual tranche ({bucket_name})", key=f"log_{bucket_name}"):
            with st.form(f"tranche_form_{bucket_name}", clear_on_submit=True):
                t_ticker = st.text_input("Ticker")
                t_amount = st.number_input("Amount (£)", min_value=0.0, step=500.0)
                t_reason = st.text_input("Trigger / reason", value="manual")
                if st.form_submit_button("Record"):
                    h.record_tranche(bucket_name, t_ticker.upper(), t_amount, t_reason)
                    st.success(f"Recorded £{t_amount:,.0f} → {t_ticker}")
                    st.rerun()
        st.divider()


# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — HOLDINGS
# ═══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Holdings")
    st.caption("Enter actual positions. Qty × live price drives the allocation and drift tabs.")

    # Add holding form
    with st.expander("➕ Add / update holding", expanded=True):
        with st.form("add_holding_form", clear_on_submit=True):
            ah_col1, ah_col2, ah_col3, ah_col4 = st.columns(4)
            ah_account = ah_col1.selectbox("Account", ["SIPP", "ISA", "GIA"])
            ah_ticker = ah_col2.text_input("Ticker", help="e.g. IWQU, GDX, TLT")
            ah_qty = ah_col3.number_input("Quantity (units)", min_value=0.0, step=1.0, format="%.4f")
            ah_ccy = ah_col4.selectbox("Currency", ["GBP", "USD"])
            ah_cost = st.number_input("Cost basis (£, optional)", min_value=0.0, step=100.0)
            if st.form_submit_button("Save"):
                if ah_ticker and ah_qty > 0:
                    h.upsert_holding(
                        ah_account, ah_ticker.upper(), ah_qty,
                        cost_basis_gbp=ah_cost or None,
                        ccy=ah_ccy,
                    )
                    st.success(f"Saved {ah_ticker.upper()} × {ah_qty:.4f} in {ah_account}")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Ticker and quantity required.")

    # Current holdings table per bucket
    all_holdings = h.fetch_holdings()
    if all_holdings:
        hdf = pd.DataFrame([
            {"ID": hh.id, "Account": hh.account_type, "Ticker": hh.ticker,
             "Qty": hh.qty, "Cost (£)": hh.cost_basis_gbp, "CCY": hh.ccy}
            for hh in all_holdings
        ])

        # Join live prices
        if not etf_meta.empty:
            meta_sub = etf_meta[["ticker", "last_price", "last_date"]].copy()
            hdf = hdf.merge(meta_sub, left_on="Ticker", right_on="ticker", how="left").drop(columns=["ticker"])
            hdf["GBP value"] = hdf.apply(
                lambda r: r["Qty"] * (r["last_price"] / 100 if r["CCY"] == "GBP" and r["last_price"] > 200 else
                                       r["last_price"] / gbpusd if r["CCY"] == "USD" else r["last_price"]),
                axis=1
            )
        for acct, grp in hdf.groupby("Account"):
            st.subheader(f"{acct}")
            st.dataframe(grp.drop(columns=["Account"]).style.format({
                "Qty": "{:.4f}",
                "last_price": "{:.2f}",
                "GBP value": "£{:,.0f}",
                "Cost (£)": lambda x: f"£{x:,.0f}" if pd.notna(x) else "",
            }), hide_index=True, use_container_width=True)

        # Delete
        del_id = st.number_input("Delete holding by ID", min_value=0, step=1, value=0)
        if st.button("🗑 Delete") and del_id > 0:
            h.delete_holding(int(del_id))
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("No holdings recorded yet. Add some above.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 6 — WARNINGS
# ═══════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Wrapper & Compliance Warnings")

    warn_rows = []
    if all_holdings:
        for hh in all_holdings:
            ins = INSTRUMENTS.get(hh.ticker.upper())
            if ins is None:
                warn_rows.append({
                    "Account": hh.account_type, "Ticker": hh.ticker,
                    "Issue": "⚠️ Ticker not in instrument universe — verify manually",
                    "Severity": "Medium",
                })
                continue
            if not ins.wrapper_eligible.get(hh.account_type, False):
                warn_rows.append({
                    "Account": hh.account_type, "Ticker": hh.ticker,
                    "Issue": f"🚫 {hh.ticker} is not eligible for {hh.account_type} "
                             f"(eligible: {[w for w, ok in ins.wrapper_eligible.items() if ok]})",
                    "Severity": "High",
                })
            if not ins.is_reporting_fund:
                warn_rows.append({
                    "Account": hh.account_type, "Ticker": hh.ticker,
                    "Issue": "🔴 NOT a UK Reporting Fund — gains taxed at income rates in GIA",
                    "Severity": "High",
                })
            if hh.account_type in ("SIPP", "ISA") and ins.listing in ("NYSE", "NASDAQ"):
                warn_rows.append({
                    "Account": hh.account_type, "Ticker": hh.ticker,
                    "Issue": f"🚫 US-listed ETF in {hh.account_type} — platform will reject this",
                    "Severity": "High",
                })

    if warn_rows:
        w_df = pd.DataFrame(warn_rows)
        high = w_df[w_df["Severity"] == "High"]
        med = w_df[w_df["Severity"] == "Medium"]
        if not high.empty:
            st.error(f"{len(high)} high-severity issue(s)")
            st.dataframe(high.drop(columns=["Severity"]), hide_index=True, use_container_width=True)
        if not med.empty:
            st.warning(f"{len(med)} medium-severity notice(s)")
            st.dataframe(med.drop(columns=["Severity"]), hide_index=True, use_container_width=True)
    else:
        st.success("No warnings — all holdings pass wrapper eligibility and reporting-fund checks.")

    st.divider()
    st.subheader("Correlated-pair risk concentration")
    st.caption(
        "Pairs of holdings whose trailing 90-day correlation is ≥ 0.75 and whose combined bucket weight "
        "is ≥ 10%. Drift column shows how much tighter the pair has become vs the trailing 1y baseline "
        "(positive = tightening now, e.g. linkers moving with equity in a rates-led selloff)."
    )
    if not portfolio_df.empty:
        _cov_timing = classify_instrument_regimes(ds.get_entry_timing_metrics(()))
        _plan = build_portfolio_plan(
            factors_df(compute_factor_scores(ds.get_factor_data(), ds.get_etf_meta())),
            BUCKET_SIZES,
            selection_mode="primary_first",
            timing_df=_cov_timing,
        )
        _sized = inverse_vol_weights(_plan, _cov_timing, blend=0.5)
        _active_tickers = _sized[_sized["action"] != "NOT_ACTIVE"]["ticker"].astype(str).unique().tolist()
        _corr = correlation_matrix(_active_tickers)
        _drift = correlation_drift(_active_tickers)
        pairs = concentrated_pairs(_sized, _corr, _drift, corr_threshold=0.75)
        if pairs.empty:
            st.success("No concentrated correlated pairs above the 0.75 threshold.")
        else:
            st.dataframe(
                pairs.style.format({
                    "correlation_90d": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                    "correlation_drift_90_vs_252": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                    "weight_a": lambda x: f"{x:.1%}" if pd.notna(x) else "—",
                    "weight_b": lambda x: f"{x:.1%}" if pd.notna(x) else "—",
                    "combined_weight": lambda x: f"{x:.1%}" if pd.notna(x) else "—",
                }),
                hide_index=True, use_container_width=True, height=360,
            )
    else:
        st.info("Add holdings to see correlated-pair concentration warnings.")

    st.divider()
    st.subheader("Instrument universe reference")
    uni_rows = []
    for ticker, ins in INSTRUMENTS.items():
        uni_rows.append({
            "Ticker": ticker,
            "Name": ins.name[:50],
            "Sleeve": ins.sleeve.replace("_", " ").title(),
            "AUM $bn": ins.aum_bn_usd,
            "Exchange": ins.listing,
            "CCY": ins.ccy,
            "Acc.": "✅" if ins.accumulating else "—",
            "Rep. Fund": "✅" if ins.is_reporting_fund else "🔴",
            "SIPP": "✅" if ins.wrapper_eligible["SIPP"] else "—",
            "ISA": "✅" if ins.wrapper_eligible["ISA"] else "—",
            "GIA": "✅" if ins.wrapper_eligible["GIA"] else "—",
        })
    st.dataframe(
        pd.DataFrame(uni_rows).style.format({"AUM $bn": "{:.1f}"}),
        hide_index=True, use_container_width=True, height=600,
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 7 — FACTOR SCREENS
# ═══════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Factor Screens")
    st.caption(
        "Two quantitative factors applied to every instrument in the universe. "
        "Both combine with the existing MA200 momentum filter from the Valuation tab."
    )

    col_help1, col_help2 = st.columns(2)
    with col_help1:
        st.info(
            "**Factor 1 — PE vs own history (z-score)**\n\n"
            "z < −1 → instrument is in the bottom 16th percentile of its own PE history → CHEAP\n\n"
            "Requires ≥4 quarterly data points to be meaningful. "
            "History self-builds from quarterly refreshes — after 2 years it is fully reliable. "
            "US broad market bootstrapped from Shiller CAPE (140 years of data)."
        )
    with col_help2:
        st.info(
            "**Factor 2 — PEG ratio**\n\n"
            "PEG = Trailing PE ÷ 5-year annualised earnings growth\n\n"
            "PEG < 1 → paying less than 1×growth for all future earnings → CHEAP\n\n"
            "Source: yfinance `pegRatio` field (direct). Where absent, computed from "
            "`trailingPE ÷ (fiveYearAverageReturn × 100)` — labelled 'computed' in table."
        )

    # ── Shiller CAPE banner ──
    shiller_z = ds.get_shiller_pe_zscore()
    if shiller_z is not None:
        cape = ds.get_latest_cape()
        cape_str = f"US Shiller CAPE: {cape:.1f}x" if cape else "US Shiller CAPE"
        if shiller_z > 1.0:
            st.error(f"⚠️ {cape_str} — z-score {shiller_z:+.2f} vs full history (DEAR — top 16th pct)")
        elif shiller_z < -1.0:
            st.success(f"✅ {cape_str} — z-score {shiller_z:+.2f} vs full history (CHEAP — bottom 16th pct)")
        else:
            st.warning(f"📊 {cape_str} — z-score {shiller_z:+.2f} vs full history (FAIR)")
    else:
        st.caption("Shiller CAPE z-score: not yet available — run 'Refresh market data' to fetch.")

    st.divider()

    # ── Factor scores table ──
    @st.cache_data(ttl=3600)
    def cached_factor_data():
        return ds.get_factor_data()

    factor_raw = cached_factor_data()
    etf_meta_for_factors = cached_etf_meta()

    if factor_raw.empty:
        st.info(
            "No factor data yet. Click '🔄 Refresh market data' in the sidebar to fetch "
            "PE and PEG data for all instruments. Subsequent quarterly refreshes build the "
            "PE history needed for the z-score factor."
        )
    else:
        scores = compute_factor_scores(factor_raw, etf_meta_for_factors)
        score_df = factors_df(scores)
        timing_df = ds.get_entry_timing_metrics(tuple(score_df["Ticker"].astype(str).tolist()))
        price_history_df = ds.get_price_history(tuple(score_df["Ticker"].astype(str).tolist()))

        # ── Signal colour styling ──
        _signal_colours = {
            "HIGH_CONVICTION": "background-color: #1a7a4a; color: white",
            "CHEAP":           "background-color: #d4edda",
            "FAIR":            "",
            "DEAR":            "background-color: #f8d7da",
            "AVOID":           "background-color: #721c24; color: white",
            "INSUFFICIENT_HISTORY": "color: #888",
            "N/A":             "color: #888",
            "FALLING_KNIFE":   "background-color: #721c24; color: white",
            "EXTENDED":        "background-color: #856404; color: white",
            "OK":              "",
        }

        def _colour_cell(val):
            return _signal_colours.get(str(val), "")

        signal_cols = ["PE signal", "PEG signal", "MA200 signal", "Composite"]

        st.subheader("All instruments — factor scores")
        st.dataframe(
            score_df.style
            .applymap(_colour_cell, subset=signal_cols)
            .format({
                "Trailing PE": lambda x: f"{x:.1f}" if pd.notna(x) else "—",
                "PE z-score":  lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "PE percentile": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "PE range pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "PEG":         lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "EPS growth %": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "Price/MA200":  lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                "52W range pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "Dist. from 52W low": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
            }),
            hide_index=True, use_container_width=True, height=500,
        )

        st.subheader("Strategy buy list")
        st.caption(
            "Encodes your rules directly: buy PE near the bottom of its own history, stay near the 52-week low, "
            "avoid falling knives, and prefer repaired momentum inside favoured themes."
        )
        buy_df = build_buy_candidates(timing_df)
        strategy_signal_colours = {
            "BUY": "background-color: #1a7a4a; color: white",
            "ACCUMULATE": "background-color: #d4edda",
            "WATCH": "background-color: #fff3cd",
            "HOLD": "",
            "WAIT": "background-color: #ffe5b4",
            "AVOID": "background-color: #721c24; color: white",
        }
        buy_cols = [
            "Ticker", "Theme", "regime", "Strategy signal", "Rationale",
            "pct_above_ma200", "range_52w_pos", "r_3m", "r_1y", "drawdown_52w", "vol_3m",
        ]
        buy_cols = [c for c in buy_cols if c in buy_df.columns]
        st.dataframe(
            buy_df[buy_cols].style
            .applymap(lambda v: strategy_signal_colours.get(str(v), ""), subset=["Strategy signal"])
            .format({
                "pct_above_ma200": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "range_52w_pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "r_3m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "r_1y": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "drawdown_52w": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                "vol_3m": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
            }),
            hide_index=True,
            use_container_width=True,
            height=360,
        )

        st.subheader("Theme performance snapshot")
        st.caption(
            "Technical view by theme: median momentum, drawdown, volatility, and extension across the mapped instruments. "
            "``regime`` summarises each theme into one of {washed_out, repairing, strong, strong_but_stretched, dead_money, falling_knife}."
        )
        theme_snapshot_df = classify_theme_regimes(build_theme_snapshot(timing_df))
        tab_regimes = get_theme_regimes(theme_snapshot_df)
        if not theme_snapshot_df.empty:
            regime_colours = {
                "strong": "background-color: #1a7a4a; color: white",
                "repairing": "background-color: #d4edda",
                "washed_out": "background-color: #e2e3e5",
                "strong_but_stretched": "background-color: #fff3cd",
                "dead_money": "background-color: #f0f0f0",
                "falling_knife": "background-color: #721c24; color: white",
            }
            st.dataframe(
                theme_snapshot_df[[
                    "theme_label", "regime", "regime_rationale",
                    "instruments", "stocks", "funds",
                    "r_1m", "r_3m", "r_6m", "r_1y", "vol_1y",
                    "drawdown_52w", "price_ma200", "range_52w_pos",
                    "pct_above_ma200", "pct_near_high",
                ]].style
                .applymap(lambda v: regime_colours.get(str(v), ""), subset=["regime"])
                .format({
                    "r_1m": "{:.1f}%",
                    "r_3m": "{:.1f}%",
                    "r_6m": "{:.1f}%",
                    "r_1y": "{:.1f}%",
                    "vol_1y": "{:.1f}%",
                    "drawdown_52w": "{:.1f}%",
                    "price_ma200": "{:.3f}",
                    "range_52w_pos": "{:.0%}",
                    "pct_above_ma200": "{:.0%}",
                    "pct_near_high": "{:.0%}",
                }),
                hide_index=True,
                use_container_width=True,
                height=360,
            )

        st.subheader("Theme correlation")
        corr_mode = st.radio(
            "Correlation weighting",
            options=["Portfolio-weighted (strategic targets)", "Equal-weight (all universe)"],
            index=0,
            horizontal=True,
            help=(
                "Portfolio-weighted uses each theme's strategic GBP target weight, so the "
                "correlation reflects what the allocator will actually hold. Equal-weight "
                "treats every instrument in the universe the same — useful for a diagnostic "
                "view but can be dominated by noisy small-AUM names."
            ),
        )
        if corr_mode.startswith("Portfolio"):
            ticker_weights = build_sleeve_ticker_weights(BUCKET_SIZES)
            corr_df = build_portfolio_weighted_theme_correlation(price_history_df, ticker_weights)
            caption = "Daily-return correlation weighted by strategic target GBP per ticker. Only themes that appear in the targets show up here."
        else:
            corr_df = build_theme_correlation(price_history_df)
            caption = "Equal-weight daily-return correlation across all universe themes over the last 252 trading days."
        st.caption(caption)
        if not corr_df.empty:
            fig_corr = px.imshow(
                corr_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
            )
            fig_corr.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough theme history to compute a correlation matrix yet.")

        st.subheader("Portfolio construction")
        st.caption(
            "Strategic sleeves converted into executable positions by wrapper. "
            "This is the actual constructor: one selected instrument per sleeve, with a clear action label."
        )
        plan_df = build_portfolio_plan(
            score_df,
            BUCKET_SIZES,
            timing_df=timing_df,
            theme_regimes=tab_regimes,
            bond_triggers=bond_triggers,
        )
        plan_summary_df = summarize_portfolio_plan(plan_df)
        if not plan_summary_df.empty:
            st.dataframe(
                plan_summary_df.style.format({"target_gbp": "£{:,.0f}"}),
                hide_index=True,
                use_container_width=True,
            )
        if not plan_df.empty:
            action_colours = {
                "APPROVED": "background-color: #1a7a4a; color: white",
                "BUILD_CORE": "background-color: #d4edda",
                "WATCHLIST": "background-color: #fff3cd",
                "NO_DATA": "background-color: #e2e3e5",
                "NOT_ACTIVE": "background-color: #d6d8db; color: #383d41",
                "REJECT": "background-color: #721c24; color: white",
            }
            st.dataframe(
                plan_df[[
                    "account_type", "sleeve", "ticker", "name", "vehicle", "target_weight",
                    "target_gbp", "r_1y", "drawdown_52w", "range_52w_pos", "price_ma200",
                    "strategy_signal", "theme_regime", "action", "candidate_score",
                    "is_primary_ticker", "rationale"
                ]].style
                .applymap(lambda v: action_colours.get(str(v), ""), subset=["action"])
                .format({
                    "target_weight": "{:.0%}",
                    "target_gbp": "£{:,.0f}",
                    "r_1y": "{:.1f}%",
                    "drawdown_52w": "{:.1f}%",
                    "range_52w_pos": "{:.0%}",
                    "price_ma200": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                    "candidate_score": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                    "is_primary_ticker": lambda x: "Yes" if x else "",
                }),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

        stock_theme_df = build_theme_stock_screen(score_df, timing_df)
        if not stock_theme_df.empty:
            st.subheader("Theme-aware stock screen")
            st.caption("Direct stocks grouped by theme, with price/performance fields alongside valuation.")
            theme_choice = st.selectbox(
                "Stock theme",
                options=["All"] + sorted(stock_theme_df["Theme"].dropna().unique().tolist()),
                index=0,
            )
            display_stock_df = stock_theme_df if theme_choice == "All" else stock_theme_df[stock_theme_df["Theme"] == theme_choice]
            st.dataframe(
                display_stock_df[[
                    "Theme", "Ticker", "Name", "PE percentile", "PE range pos", "52W range pos",
                    "Dist. from 52W low", "Price/MA200", "r_1m", "r_3m", "r_6m", "r_1y",
                    "drawdown_52w", "vol_1y", "Composite"
                ]].style
                .applymap(_colour_cell, subset=["Composite"])
                .format({
                    "PE percentile": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                    "PE range pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                    "52W range pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                    "Dist. from 52W low": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                    "Price/MA200": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                    "r_1m": "{:.1f}%",
                    "r_3m": "{:.1f}%",
                    "r_6m": "{:.1f}%",
                    "r_1y": "{:.1f}%",
                    "drawdown_52w": "{:.1f}%",
                    "vol_1y": "{:.1f}%",
                }),
                hide_index=True,
                use_container_width=True,
                height=360,
            )

        st.subheader("Best entry points by wrapper and sleeve")
        st.caption(
            "This compares the actual investable instruments inside each sleeve and wrapper, then down-ranks names "
            "that are too close to peaks. The goal is to keep the theme but avoid the hottest entry point."
        )
        wrapper_df = build_wrapper_candidate_table(timing_df)
        if not wrapper_df.empty:
            top_wrapper_df = wrapper_df[wrapper_df["Rank"] <= 2].copy()
            wrap_cols = [
                "Wrapper", "Sleeve", "Rank", "Ticker", "Name", "Strategy signal", "Regime",
                "Vehicle", "Reporting", "Candidate score", "% vs MA200",
                "52W range pos", "r_3m", "r_1y", "Rationale",
            ]
            wrap_cols = [c for c in wrap_cols if c in top_wrapper_df.columns]
            st.dataframe(
                top_wrapper_df[wrap_cols].style
                .applymap(lambda v: strategy_signal_colours.get(str(v), ""), subset=["Strategy signal"])
                .format({
                    "Candidate score": "{:.2f}",
                    "Reporting": lambda x: "Yes" if x else "No",
                    "% vs MA200": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "52W range pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                    "r_3m": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "r_1y": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                }),
                hide_index=True,
                use_container_width=True,
                height=420,
            )
            st.caption(
                "SIPP candidates can include direct stocks. ISA stays fund-only. "
                "GIA candidates can include HMRC-reporting funds and direct stocks."
            )

        # ── High conviction shortlist ──
        hc = score_df[score_df["Composite"].isin(["HIGH_CONVICTION", "CHEAP"])].copy()
        if not hc.empty:
            st.subheader("🟢 Buy candidates (PE cheap vs history AND PEG < 2, MA200 passes)")
            st.dataframe(
                hc[["Ticker", "Trailing PE", "PE z-score", "PE signal", "PEG", "PEG signal", "Composite"]]
                .style.applymap(_colour_cell, subset=["PE signal", "PEG signal", "Composite"]),
                hide_index=True, use_container_width=True,
            )

        avoid = score_df[score_df["Composite"] == "AVOID"]
        if not avoid.empty:
            st.subheader("🔴 Avoid (MA200 filter rejected — falling knife or extended)")
            st.dataframe(
                avoid[["Ticker", "Trailing PE", "MA200 signal", "Price/MA200", "Composite"]]
                .style.applymap(_colour_cell, subset=["MA200 signal", "Composite"]),
                hide_index=True, use_container_width=True,
            )

        st.subheader("ETF lookthrough to stocks")
        etf_choices = sorted(
            t for t, ins in INSTRUMENTS.items() if ins.vehicle_type in {"ucits_etf", "us_etf", "etc"}
        )
        selected_etf = st.selectbox("ETF for lookthrough", options=etf_choices, index=0)
        stored_df = ds.get_etf_constituents(selected_etf)
        if stored_df.empty:
            holdings_df = fetch_etf_top_holdings(selected_etf, limit=10)
            mapped_df = map_holdings_to_universe(holdings_df, score_df)
        else:
            mapped_df = stored_df.copy()
            score_map = score_df.set_index("Ticker").to_dict(orient="index") if not score_df.empty else {}
            mapped_df["Trailing PE"] = mapped_df["Mapped ticker"].map(lambda t: score_map.get(t, {}).get("Trailing PE"))
            mapped_df["PE percentile"] = mapped_df["Mapped ticker"].map(lambda t: score_map.get(t, {}).get("PE percentile"))
            mapped_df["PEG"] = mapped_df["Mapped ticker"].map(lambda t: score_map.get(t, {}).get("PEG"))
            mapped_df["Price/MA200"] = mapped_df["Mapped ticker"].map(lambda t: score_map.get(t, {}).get("Price/MA200"))
            mapped_df["Composite"] = mapped_df["Mapped ticker"].map(lambda t: score_map.get(t, {}).get("Composite"))
        if not mapped_df.empty:
            st.dataframe(
                mapped_df.style.format({
                    "Weight %": "{:.2f}%",
                    "Trailing PE": lambda x: f"{x:.1f}" if pd.notna(x) else "—",
                    "PE percentile": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                    "PEG": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                    "Price/MA200": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                }).applymap(_colour_cell, subset=["Composite"]),
                hide_index=True,
                use_container_width=True,
                height=360,
            )
            mapped_count = int(mapped_df["Mapped ticker"].notna().sum())
            st.caption(
                f"Mapped {mapped_count}/{len(mapped_df)} top holdings onto the direct-stock universe. "
                "Unmapped rows are still useful as raw ETF constituents, but they are not yet first-class stock candidates."
            )
            if "as_of" in mapped_df.columns:
                st.caption(f"Constituent snapshot as of {mapped_df['as_of'].iloc[0]}.")
        else:
            st.info("No lookthrough holdings available for this ETF.")

        st.subheader("True lookthrough exposure")
        exposure_df = ds.summarize_true_exposure(portfolio_df)
        if not exposure_df.empty:
            exposure_df["Region"] = exposure_df["underlying_ticker"].map(lambda t: _stock_region(lookup(str(t))))
            exposure_df["Total % of portfolio"] = exposure_df["total_gbp"] / total_size * 100 if total_size else 0.0
            overlap_df = exposure_df[exposure_df["duplicate_overlap"]].copy()
            st.dataframe(
                exposure_df[[
                    "account_type", "Region", "underlying_ticker", "direct_gbp", "indirect_gbp",
                    "unmapped_fund_gbp", "total_gbp", "Total % of portfolio", "duplicate_overlap"
                ]].style.format({
                    "direct_gbp": "£{:,.0f}",
                    "indirect_gbp": "£{:,.0f}",
                    "unmapped_fund_gbp": "£{:,.0f}",
                    "total_gbp": "£{:,.0f}",
                    "Total % of portfolio": "{:.2f}%",
                    "duplicate_overlap": lambda x: "Yes" if x else "",
                }),
                hide_index=True,
                use_container_width=True,
                height=360,
            )
            if not overlap_df.empty:
                st.caption("Duplicate overlap flags names held directly and indirectly via ETF lookthrough in the same account.")

            st.subheader("Concentration by theme (after ETF lookthrough)")
            st.caption(
                "Aggregates the true-exposure rows by the underlying instrument's sleeve. "
                "This is what actually answers 'how much of my portfolio is US tech / gold / linkers "
                "once I expand every ETF into its constituents?'"
            )
            expanded_rows = ds.compute_true_exposure(portfolio_df)
            concentration_df = build_theme_lookthrough_concentration(expanded_rows, total_size)
            if not concentration_df.empty:
                st.dataframe(
                    concentration_df[[
                        "account_type", "theme_label", "direct_gbp", "indirect_gbp",
                        "unexpanded_gbp", "unmapped_gbp", "total_gbp", "pct_of_portfolio",
                    ]].style.format({
                        "direct_gbp": "£{:,.0f}",
                        "indirect_gbp": "£{:,.0f}",
                        "unexpanded_gbp": "£{:,.0f}",
                        "unmapped_gbp": "£{:,.0f}",
                        "total_gbp": "£{:,.0f}",
                        "pct_of_portfolio": lambda x: f"{x:.2f}%" if pd.notna(x) else "—",
                    }),
                    hide_index=True,
                    use_container_width=True,
                    height=360,
                )
                portfolio_total = concentration_df.groupby("theme_label", as_index=False)["total_gbp"].sum()
                portfolio_total["pct_of_portfolio"] = (
                    portfolio_total["total_gbp"] / total_size * 100.0 if total_size else 0.0
                )
                portfolio_total = portfolio_total.sort_values("total_gbp", ascending=False).head(15)
                fig_concentration = px.bar(
                    portfolio_total,
                    x="theme_label",
                    y="pct_of_portfolio",
                    title="Top 15 themes by portfolio-level concentration (after lookthrough)",
                )
                fig_concentration.update_layout(
                    height=360,
                    xaxis_title="",
                    yaxis_title="% of portfolio",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_concentration, use_container_width=True)
            else:
                st.info("No theme-level concentration data yet. Add holdings first.")
        else:
            st.info("No lookthrough exposure yet. Add holdings and refresh ETF constituents.")

    st.divider()

    # ── Thematic opportunity table ──
    st.subheader("Thematic satellite opportunities (GIA only)")
    st.caption(
        "Optional overweights funded by trimming baseline GIA sleeves. Activation requires MA200 filter to pass. "
        "Max per-theme caps shown; typical live exposure 8–12% of GIA when 2–3 themes activate."
    )
    thematic_rows = []
    theme_df = build_theme_watchlist(timing_df if not factor_raw.empty else pd.DataFrame())
    if not theme_df.empty:
        st.dataframe(
            theme_df.style.format({
                "% vs MA200": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "52w pos": lambda x: f"{x:.0%}" if pd.notna(x) else "—",
                "Max GIA weight": "{:.0%}",
            }),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.dataframe(pd.DataFrame(thematic_rows), hide_index=True, use_container_width=True)

    n_history = int(factor_raw["date"].nunique()) if not factor_raw.empty else 0
    st.caption(
        f"PE history depth: {n_history} quarterly snapshot(s). "
        f"z-score is reliable after 8 snapshots (≈2 years of quarterly refreshes). "
        f"Data source priority: DuckDB `ibkr_fundamentals` / local import fallback → yfinance .info; "
        f"regional PE from iShares product JSON where available."
    )
