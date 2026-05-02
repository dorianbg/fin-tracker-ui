"""00_PortfolioAllocator.py — Two-Bucket Antifragile Portfolio Allocator.

Top-of-sidebar page. Sections:
  1. Bucket dashboard — current vs target weights for SIPP and ISA+GIA
  2. Valuation table — regions with forward P/E, CAPE, MA200 ratio, tilts
  3. Rebalance alerts — drift > 200bps flagged with suggested action
  4. Deployment cockpit — cash deployment pace, tranche history
  5. Tax/wrapper warnings — non-reporting funds, wrapper violations
  6. Honest-return banner — projected real return per bucket
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from allocator import buckets, holdings, instruments, valuation
from allocator.data_sources import fetch_all_fred
from data import add_sparkline_column


def render():
    st.title("Portfolio Allocator")

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Allocator Settings")

        sipp_nav = st.number_input(
            "SIPP NAV (£)", min_value=0, value=300_000, step=10_000
        )
        isa_gia_nav = st.number_input(
            "ISA + GIA NAV (£)", min_value=0, value=250_000, step=10_000
        )
        total = sipp_nav + isa_gia_nav

        dd_isa = st.slider(
            "ISA+GIA drawdown tolerance",
            -0.25,
            -0.05,
            -0.10,
            0.01,
            help="Binding constraint on the liquid bucket. More negative = more equity allowed.",
        )

        drift_threshold = st.slider(
            "Drift alert threshold (bps)",
            50,
            500,
            200,
            25,
            help="Flag positions that drift more than this from target.",
        )

        st.subheader("Deployment")
        months_remaining = st.number_input(
            "Months remaining in deploy",
            min_value=0,
            max_value=24,
            value=12,
        )
        initial_cash = st.number_input(
            "Initial cash to deploy (£)",
            min_value=0,
            value=total,
            step=10_000,
        )

    with content_col:
        # ── Section 1: Bucket Dashboard ─────────────────────────────────────
        st.header("1. Bucket Dashboard")

        col1, col2 = st.columns(2)

        for col, bkt_key, nav in [
            (col1, "SIPP", sipp_nav),
            (col2, "ISA_GIA", isa_gia_nav),
        ]:
            bkt = buckets.ALL_BUCKETS[bkt_key]
            with col:
                st.subheader(f"{bkt.name}")
                st.metric("NAV", f"£{nav:,.0f}")
                st.metric("Drawdown target", f"{bkt.drawdown_target:.0%}")
                st.metric("Vol target", f"~{bkt.vol_target:.0%}")

                targets = bkt.sleeve_targets
                current = holdings.current_weights(
                    account_type=bkt.wrappers[0] if len(bkt.wrappers) == 1 else None
                )

                target_df = pd.DataFrame(
                    {
                        "Sleeve": list(targets.keys()),
                        "Target %": [v * 100 for v in targets.values()],
                    }
                )

                if not current.empty:
                    # Map current holdings to sleeves via instruments
                    sleeve_current: dict[str, float] = {}
                    for ticker, wt in current.items():
                        inst = instruments.ALL_INSTRUMENTS.get(ticker)
                        sleeve = inst.sleeve if inst else "unknown"
                        sleeve_current[sleeve] = sleeve_current.get(sleeve, 0.0) + wt
                    target_df["Current %"] = [
                        sleeve_current.get(s, 0.0) * 100 for s in targets.keys()
                    ]
                    target_df["Gap %"] = target_df["Current %"] - target_df["Target %"]
                else:
                    target_df["Current %"] = 0.0
                    target_df["Gap %"] = -target_df["Target %"]

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        name="Target",
                        x=target_df["Sleeve"],
                        y=target_df["Target %"],
                        marker_color="steelblue",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        name="Current",
                        x=target_df["Sleeve"],
                        y=target_df["Current %"],
                        marker_color="coral",
                    )
                )
                fig.update_layout(
                    barmode="group",
                    height=300,
                    yaxis_title="%",
                    margin=dict(t=30, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown budget meter
                actual_dd_budget_used = 0.0
                remaining_budget = abs(bkt.drawdown_target) - actual_dd_budget_used
                st.progress(
                    min(1.0, actual_dd_budget_used / abs(bkt.drawdown_target)),
                    text=f"DD budget used: {actual_dd_budget_used:.0%} / {abs(bkt.drawdown_target):.0%}",
                )

        # ── Section 2: Price Performance & Tilts ────────────────────────────
        st.header("2. Price Performance & Tilts")
        st.caption(
            "Regional equity tilts based on price/MA200 ratio. "
            "Below MA → overweight (mean-reversion). Above MA → underweight (extended). "
            "Bond triggers from FRED macro data."
        )

        REGION_ETFS = {
            "US": "SPY",
            "Europe": "VGK",
            "Japan": "EWJ",
            "EM": "EEM",
            "UK": "ISF",
        }

        @st.cache_data(ttl=600)
        def _load_region_data() -> dict[str, valuation.RegionData]:
            """Build region data from pipeline's performance table (ma_252 column)."""
            import duckdb_importer as di
            from data import get_conn

            tickers = tuple(REGION_ETFS.values())
            tickers_str = "','".join(tickers)
            query = f"""
                SELECT ticker, price, ma_252
                FROM {di.perf_tbl}
                WHERE rown = 1 AND ticker IN ('{tickers_str}')
            """
            try:
                df = get_conn().execute(query).df()
            except Exception:
                df = pd.DataFrame()
            regions: dict[str, valuation.RegionData] = {}
            etf_to_region = {v: k for k, v in REGION_ETFS.items()}
            for _, row in df.iterrows():
                region = etf_to_region.get(row["ticker"])
                if region is None:
                    continue
                price = float(row["price"]) if pd.notna(row["price"]) else 100.0
                ma_pct = float(row["ma_252"]) if pd.notna(row["ma_252"]) else 0.0
                ratio = 1.0 + ma_pct / 100.0
                ma200 = price / ratio if ratio > 0 else price
                regions[region] = valuation.RegionData(
                    name=region, price=price, ma200=ma200
                )
            for r in REGION_ETFS:
                if r not in regions:
                    regions[r] = valuation.RegionData(name=r, price=100.0, ma200=100.0)
            return regions

        REGIONS = _load_region_data()
        tilts = valuation.compute_region_tilts(REGIONS)

        val_data = []
        for r, rd in REGIONS.items():
            ratio = rd.price / rd.ma200 if rd.ma200 > 0 else 1.0
            val_data.append(
                {
                    "Region": r,
                    "ETF": REGION_ETFS.get(r, ""),
                    "Price": rd.price,
                    "MA200": rd.ma200,
                    "Price/MA200": ratio,
                    "Tilt": tilts.get(r, 1.0),
                }
            )
        val_df = pd.DataFrame(val_data)
        val_df["ticker"] = val_df["ETF"]
        add_sparkline_column(val_df)
        add_sparkline_column(val_df, col_name="Price (1y)", days=365)
        st.dataframe(
            val_df[
                [
                    "Region",
                    "ETF",
                    "Price (90d)",
                    "Price (1y)",
                    "Price",
                    "MA200",
                    "Price/MA200",
                    "Tilt",
                ]
            ].style.format(
                {
                    "Price": "{:.2f}",
                    "MA200": "{:.2f}",
                    "Price/MA200": "{:.2f}",
                    "Tilt": "{:.2f}",
                }
            ).background_gradient(subset=["Tilt"], cmap="RdYlGn", vmin=0.5, vmax=1.5),
            column_config={
                "Price (90d)": st.column_config.LineChartColumn(
                    "Price (90d)", width="small"
                ),
                "Price (1y)": st.column_config.LineChartColumn(
                    "Price (1y)", width="small"
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        # Bond triggers
        st.subheader("Bond Triggers")
        with st.expander("Fetch live FRED data", expanded=False):
            if st.button("Refresh FRED data"):
                with st.spinner("Fetching from FRED..."):
                    fred = fetch_all_fred()
                    st.json(fred)
                    st.success("Data refreshed.")
            else:
                st.info("Click to fetch live macro data from FRED.")

        macro = valuation.MacroData()
        triggers = valuation.compute_bond_triggers(macro)
        trigger_df = pd.DataFrame(
            [
                {
                    "Trigger": k.replace("_", " ").title(),
                    "Target weight": v,
                    "Active": v > 0,
                }
                for k, v in triggers.items()
            ]
        )
        st.dataframe(
            trigger_df.style.format({"Target weight": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )

        # ── Section 3: Rebalance Alerts ─────────────────────────────────────
        st.header("3. Rebalance Alerts")

        all_holdings = holdings.load_holdings()
        if all_holdings.empty:
            st.info(
                "No holdings recorded. Add holdings via the Portfolio Manager page, "
                "or seed test data in portfolio.db."
            )
        else:
            for bkt_key in ("SIPP", "ISA_GIA"):
                bkt = buckets.ALL_BUCKETS[bkt_key]
                acct_filter = bkt.wrappers[0] if len(bkt.wrappers) == 1 else None
                cur = holdings.current_weights(account_type=acct_filter)
                if cur.empty:
                    continue

                # Build target series from sleeve_targets × instruments
                # (simplified: use sleeve_targets as-is for now)
                target_series = pd.Series(bkt.sleeve_targets)
                drift_df = holdings.drift(
                    cur, target_series, threshold_bps=drift_threshold
                )
                if not drift_df.empty:
                    st.subheader(f"{bkt.name} — Drift Alerts")
                    st.dataframe(
                        drift_df.style.format(
                            {
                                "current_%": "{:.1%}",
                                "target_%": "{:.1%}",
                                "drift_%": "{:+.1%}",
                                "drift_bps": "{:+.0f}",
                            }
                        ),
                        use_container_width=True,
                    )
                else:
                    st.success(f"{bkt.name}: all positions within tolerance.")

        # ── Section 4: Deployment Cockpit ───────────────────────────────────
        st.header("4. Deployment Cockpit")

        cash_remaining = initial_cash * 0.70  # assume 30% deployed day 1
        state = valuation.DeploymentState(
            months_remaining=months_remaining,
            cash_remaining=cash_remaining,
            total_initial=initial_cash,
        )
        pace = valuation.compute_deployment_pace(state, macro)

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Cash remaining", f"£{cash_remaining:,.0f}")
        dcol2.metric("Months left", f"{months_remaining}")
        dcol3.metric("This month's tranche", f"£{cash_remaining * pace:,.0f}")
        dcol4.metric("Deploy rate", f"{pace:.1%} of remaining")

        # Deployment history
        log = holdings.load_deployment_log()
        if not log.empty:
            st.subheader("Deployment History")
            st.dataframe(log, use_container_width=True, hide_index=True)
        else:
            st.caption("No deployment tranches recorded yet.")

        # Deployment pace simulation
        with st.expander("Deployment pace simulation (12-month Monte Carlo)"):
            st.caption(
                "Simulates 500 paths of 12-month ACWI drawdowns (normal, "
                "mu=0, sigma=15%). Shows how cash deploys faster on drawdowns."
            )
            np.random.seed(42)
            n_paths = 500
            months = 12
            results = []
            for _ in range(n_paths):
                cash = float(initial_cash) * 0.70
                total_init = cash
                for m in range(1, months + 1):
                    dd = np.random.normal(0, 0.05)
                    cumulative_dd = max(-0.30, min(0.0, dd))
                    s = valuation.DeploymentState(months - m + 1, cash, total_init)
                    md = valuation.MacroData(acwi_drawdown_30d=cumulative_dd)
                    p = valuation.compute_deployment_pace(s, md)
                    deploy = cash * p
                    cash -= deploy
                    results.append(
                        {
                            "month": m,
                            "cash_pct": cash / total_init if total_init > 0 else 0,
                        }
                    )

            sim_df = pd.DataFrame(results)
            agg = sim_df.groupby("month")["cash_pct"].agg(
                [
                    "mean",
                    "median",
                    lambda x: x.quantile(0.10),
                    lambda x: x.quantile(0.90),
                ]
            )
            agg.columns = ["Mean", "Median", "P10", "P90"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=agg.index, y=agg["Mean"], name="Mean", line=dict(width=2))
            )
            fig.add_trace(
                go.Scatter(
                    x=agg.index, y=agg["Median"], name="Median", line=dict(dash="dash")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(agg.index) + list(agg.index[::-1]),
                    y=list(agg["P90"]) + list(agg["P10"][::-1]),
                    fill="toself",
                    fillcolor="rgba(100,149,237,0.15)",
                    line=dict(width=0),
                    name="P10–P90",
                )
            )
            fig.update_layout(
                yaxis_title="Cash remaining (%)",
                xaxis_title="Month",
                yaxis_tickformat=".0%",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Section 5: Tax / Wrapper Warnings ───────────────────────────────
        st.header("5. Tax & Wrapper Warnings")

        if not all_holdings.empty:
            violations = instruments.wrapper_violations(all_holdings)
            if violations:
                st.warning(f"{len(violations)} wrapper violation(s) detected:")
                st.dataframe(
                    pd.DataFrame(violations), use_container_width=True, hide_index=True
                )
            else:
                st.success("No wrapper violations.")

            non_reporting = instruments.non_reporting_fund_tickers()
            held_non_reporting = [
                t for t in all_holdings["asset"].unique() if t in non_reporting
            ]
            if held_non_reporting:
                st.warning(
                    f"Non-reporting fund holdings: {', '.join(held_non_reporting)}. "
                    "Check HMRC Reporting Funds list before purchase."
                )
            else:
                st.success("All held instruments have reporting fund status.")
        else:
            st.info("Add holdings to see wrapper/tax warnings.")

        # ── Section 6: Honest Return Banner ─────────────────────────────────
        st.header("6. Expected Real Returns")
        st.caption(
            "Approximate expected real return given current sleeve targets. "
            "Equity 5%, real 3%, bonds 2%, cash 1%."
        )

        ret_sipp = buckets.expected_real_return(buckets.SIPP)
        ret_isa = buckets.expected_real_return(buckets.ISA_GIA)
        blended = (
            (ret_sipp * sipp_nav + ret_isa * isa_gia_nav) / total if total > 0 else 0
        )

        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("SIPP", f"{ret_sipp:.1%} real")
        rcol2.metric("ISA + GIA", f"{ret_isa:.1%} real")
        rcol3.metric("Blended", f"{blended:.1%} real")

        # 20-year projection
        years = 20
        proj_sipp = sipp_nav * (1 + ret_sipp) ** years
        proj_isa = isa_gia_nav * (1 + ret_isa) ** years
        proj_total = proj_sipp + proj_isa
        st.caption(
            f"At current allocations, projected real value in {years} years (no contributions): "
            f"SIPP £{proj_sipp:,.0f} + ISA/GIA £{proj_isa:,.0f} = **£{proj_total:,.0f}**"
        )

        cost_note = abs(0.05 - blended)
        st.caption(
            f"The -10% ISA constraint costs ~{cost_note:.1%} vs an unconstrained equity portfolio. "
            f"Softening to -15% would allow ~10pp more equity and ~50bps higher return."
        )
