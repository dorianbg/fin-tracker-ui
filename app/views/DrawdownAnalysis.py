"""
Drawdown Analysis — dedicated page for analysing max drawdowns, recovery times,
and ranking instruments by their current drawdown severity.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data import (
    load_latest_perf,
    load_prices,
    fund_type_sidebar,
    filter_by_fund_type,
    add_sparkline_column,
)


def render():
    st.title("📉 Drawdown Analysis")
    st.markdown(
        "Analyse drawdowns across all instruments. Identify which are in deep drawdowns, "
        "which are recovering, and historical drawdown patterns."
    )

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Settings")
        fund_type_filter = fund_type_sidebar(key="drawdown_fund_types")
        severity_threshold = st.slider(
            "Severe drawdown threshold (%)",
            -50,
            -5,
            -15,
            key="drawdown_severity_threshold",
        )

    with content_col:
        _latest = load_latest_perf()
        _prices = load_prices()

        if _latest.empty:
            st.warning("No data loaded.")
            st.stop()

        df = filter_by_fund_type(_latest, fund_type_filter)
        price_df = filter_by_fund_type(_prices, fund_type_filter)

        # ═══════════════════════════════════════════
        # Section 1: Current Drawdown Rankings
        # ═══════════════════════════════════════════
        st.header("🏔️ Current Drawdowns from 52-Week High")

        ranked = df.sort_values("drawdown_52w", ascending=True).reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = "Rank"

        # Colour-code by severity
        fig_dd = px.bar(
            ranked,
            x="drawdown_52w",
            y="description",
            orientation="h",
            color="drawdown_52w",
            color_continuous_scale="RdYlGn",
            labels={"drawdown_52w": "Drawdown from 52W High (%)", "description": ""},
            title="All Instruments — Drawdown from 52-Week High",
        )
        fig_dd.update_layout(
            yaxis=dict(autorange="reversed"), height=max(500, len(ranked) * 22)
        )
        fig_dd.add_vline(
            x=severity_threshold,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="Severe",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("At 52W High", f"{(df['drawdown_52w'] >= -0.5).sum()}")
        with col2:
            st.metric(
                "Within 5%",
                f"{((df['drawdown_52w'] >= -5) & (df['drawdown_52w'] < -0.5)).sum()}",
            )
        with col3:
            st.metric(
                "5-15% Drawdown",
                f"{((df['drawdown_52w'] < -5) & (df['drawdown_52w'] >= -15)).sum()}",
            )
        with col4:
            st.metric(
                f"Severe (<{severity_threshold}%)",
                f"{(df['drawdown_52w'] < severity_threshold).sum()}",
            )

        # ═══════════════════════════════════════════
        # Section 2: Drawdown Scatter: 52W vs 3Y
        # ═══════════════════════════════════════════
        st.header("📊 52-Week vs 3-Year Drawdown")
        st.markdown(
            "Instruments in the **bottom-left** are in deep drawdowns on both timeframes — potentially distressed. "
            "Bottom-right (52W deep but 3Y shallow) = recent decline in a long-term uptrend."
        )

        fig_scatter = px.scatter(
            df,
            x="drawdown_3y",
            y="drawdown_52w",
            color="r_1mo",
            color_continuous_scale="RdYlGn",
            text="ticker",
            hover_data=["description", "r_1w", "r_3mo"],
            labels={
                "drawdown_3y": "Drawdown from 3Y High (%)",
                "drawdown_52w": "Drawdown from 52W High (%)",
                "r_1mo": "1M Return",
                "description": "Instrument",
            },
            title="Drawdown: 52-Week vs 3-Year",
        )
        fig_scatter.update_traces(textposition="top center", marker=dict(size=9))
        fig_scatter.add_hline(
            y=severity_threshold, line_dash="dash", line_color="red", opacity=0.3
        )
        fig_scatter.add_vline(
            x=severity_threshold, line_dash="dash", line_color="red", opacity=0.3
        )
        fig_scatter.update_layout(height=550)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ═══════════════════════════════════════════
        # Section 3: Severe Drawdowns Table
        # ═══════════════════════════════════════════
        st.header(f"🔴 Instruments in Severe Drawdown (<{severity_threshold}%)")

        severe = (
            df[df["drawdown_52w"] < severity_threshold]
            .sort_values("drawdown_52w")
            .copy()
        )

        if severe.empty:
            st.success("No instruments in severe drawdown territory.")
        else:
            display_cols = [
                "description",
                "ticker",
                "fund_type",
                "Price (90d)",
                "drawdown_52w",
                "drawdown_3y",
                "r_1d",
                "r_1w",
                "r_1mo",
                "r_3mo",
                "ma_21",
                "ma_252",
                "vol_1y",
            ]
            available = [
                c for c in display_cols if c in severe.columns or c == "Price (90d)"
            ]

            add_sparkline_column(severe)

            st.dataframe(
                severe[available]
                .style.format(
                    subset=[
                        c
                        for c in [
                            "drawdown_52w",
                            "drawdown_3y",
                            "r_1d",
                            "r_1w",
                            "r_1mo",
                            "r_3mo",
                            "ma_21",
                            "ma_252",
                        ]
                        if c in available
                    ],
                    formatter="{:+.2f}%",
                )
                .format(
                    subset=[c for c in ["vol_1y"] if c in available],
                    formatter="{:.2f}%",
                ),
                column_config={
                    "Price (90d)": st.column_config.LineChartColumn(
                        "Price (90d)", width="small"
                    ),
                    "description": st.column_config.TextColumn(
                        "description", width="medium"
                    ),
                    "ticker": st.column_config.TextColumn("ticker", width="small"),
                    "fund_type": st.column_config.TextColumn(
                        "fund_type", width="small"
                    ),
                    "drawdown_52w": st.column_config.NumberColumn(
                        "DD 52W", format="%.2f%%", width="small"
                    ),
                    "drawdown_3y": st.column_config.NumberColumn(
                        "DD 3Y", format="%.2f%%", width="small"
                    ),
                    "r_1d": st.column_config.NumberColumn(
                        "r_1d", format="%.2f%%", width="small"
                    ),
                    "r_1w": st.column_config.NumberColumn(
                        "r_1w", format="%.2f%%", width="small"
                    ),
                    "r_1mo": st.column_config.NumberColumn(
                        "r_1mo", format="%.2f%%", width="small"
                    ),
                    "r_3mo": st.column_config.NumberColumn(
                        "r_3mo", format="%.2f%%", width="small"
                    ),
                    "ma_21": st.column_config.NumberColumn(
                        "ma_21", format="%.2f%%", width="small"
                    ),
                    "ma_252": st.column_config.NumberColumn(
                        "ma_252", format="%.2f%%", width="small"
                    ),
                    "vol_1y": st.column_config.NumberColumn(
                        "vol_1y", format="%.2f%%", width="small"
                    ),
                },
                height=min(400, 50 + len(severe) * 35),
            )

        # ═══════════════════════════════════════════
        # Section 4: Historical Drawdown Timelines
        # ═══════════════════════════════════════════
        st.header("📈 Drawdown Timelines")
        st.markdown("Select instruments to see their historical drawdown paths.")

        available_instruments = sorted(df["description"].unique())
        selected = st.multiselect(
            "Select instruments",
            options=available_instruments,
            default=available_instruments[:3]
            if len(available_instruments) >= 3
            else available_instruments,
            max_selections=8,
        )

        if selected:
            sel_tickers = df[df["description"].isin(selected)]["ticker"].tolist()
            hist = price_df[price_df["ticker"].isin(sel_tickers)].copy()
            hist["date"] = pd.to_datetime(hist["date"])

            # calculate rolling drawdown from expanding max
            dd_lines = []
            for ticker in sel_tickers:
                t_df = hist[hist["ticker"] == ticker].sort_values("date").copy()
                t_df["peak"] = t_df["price"].expanding().max()
                t_df["drawdown"] = (t_df["price"] / t_df["peak"] - 1) * 100
                dd_lines.append(t_df[["date", "ticker", "description", "drawdown"]])

            if dd_lines:
                dd_all = pd.concat(dd_lines, ignore_index=True)
                fig_dd_hist = px.line(
                    dd_all,
                    x="date",
                    y="drawdown",
                    color="description",
                    labels={
                        "drawdown": "Drawdown from Peak (%)",
                        "date": "",
                        "description": "",
                    },
                    title="Historical Drawdown from All-Time High",
                )
                fig_dd_hist.add_hline(
                    y=severity_threshold,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.3,
                )
                fig_dd_hist.update_layout(height=450)
                st.plotly_chart(fig_dd_hist, use_container_width=True)

                # Max drawdown stats
                st.subheader("📋 Max Drawdown Statistics")
                stats = []
                for ticker in sel_tickers:
                    t_df = hist[hist["ticker"] == ticker].sort_values("date").copy()
                    t_df["peak"] = t_df["price"].expanding().max()
                    t_df["drawdown"] = (t_df["price"] / t_df["peak"] - 1) * 100
                    max_dd = t_df["drawdown"].min()
                    max_dd_date = t_df.loc[t_df["drawdown"].idxmin(), "date"]
                    desc = t_df["description"].iloc[0]

                    # find recovery: first date after max_dd_date where price >= peak at that point
                    peak_at_max_dd = t_df.loc[t_df["drawdown"].idxmin(), "peak"]
                    recovery_df = t_df[
                        (t_df["date"] > max_dd_date) & (t_df["price"] >= peak_at_max_dd)
                    ]
                    if not recovery_df.empty:
                        recovery_date = recovery_df["date"].iloc[0]
                        recovery_days = (recovery_date - max_dd_date).days
                        recovery_str = f"{recovery_days} days"
                    else:
                        recovery_str = "Not recovered"

                    stats.append(
                        {
                            "Instrument": desc,
                            "Max Drawdown": f"{max_dd:.1f}%",
                            "Max DD Date": max_dd_date.strftime("%Y-%m-%d"),
                            "Recovery": recovery_str,
                            "Current DD": f"{df[df['ticker'] == ticker]['drawdown_52w'].iloc[0]:+.1f}%"
                            if ticker in df["ticker"].values
                            else "N/A",
                        }
                    )

                st.dataframe(pd.DataFrame(stats), hide_index=True)
