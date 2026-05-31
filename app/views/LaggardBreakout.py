"""
Laggard Breakout Scanner — find long-term underperformers that are starting to move.

Strategy: identify instruments that were "dead capital" (poor 1-2Y returns, lagging
the benchmark) but are now showing short-term relative strength. These are potential
trend-change / breakout opportunities — similar to Japan's equity breakout in 2023-24
after years of underperformance.

Sections:
  1) Laggard Awakening: long-term laggards with improving short-term RS
  2) Relative Strength Table: 1D / 1W / 1M excess returns vs benchmark
  3) Breakout Confirmation: laggards now crossing above key moving averages
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import config
from data import (
    load_latest_perf,
    fund_type_sidebar,
    filter_by_fund_type,
    add_sparkline_column,
)

try:
    from strategy_scanners import (
        scan_laggard_awakening,
        scan_laggard_breakout_confirmations,
    )
except ModuleNotFoundError:
    from app.strategy_scanners import (
        scan_laggard_awakening,
        scan_laggard_breakout_confirmations,
    )


def render():
    st.title("🔄 Laggard Breakout Scanner")
    st.markdown(
        "Find **long-term underperformers** that are just starting to move. "
        "Identify dead capital waking up — buy the breakout, not the bottom."
    )

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Scanner Settings")

        benchmark_label = st.selectbox(
            "Benchmark",
            options=list(config.BENCHMARKS.keys()),
            index=0,
            key="laggard_benchmark",
        )
        benchmark_ticker = config.BENCHMARKS[benchmark_label]

        laggard_period = st.selectbox(
            "Laggard lookback",
            options=["1Y", "2Y", "3Y"],
            index=1,
            help="Period over which the instrument must have underperformed",
            key="laggard_lookback",
        )
        PERIOD_COL = {"1Y": "r_1y", "2Y": "r_2y", "3Y": "r_3y", "5Y": "r_5y"}
        laggard_col = PERIOD_COL[laggard_period]

        underperf_threshold = st.slider(
            f"Min {laggard_period} underperformance vs benchmark (%)",
            min_value=0,
            max_value=60,
            value=10,
            step=5,
            help="How much worse than benchmark over the laggard period to qualify",
            key="laggard_underperf_threshold",
        )

        max_abs_return = st.slider(
            f"Max absolute {laggard_period} return (%)",
            min_value=-30,
            max_value=50,
            value=20,
            step=5,
            help="Filter for 'dead capital' — instruments that barely moved. Set high to include all.",
            key="laggard_max_abs_return",
        )

        rs_awakening = st.selectbox(
            "Awakening detection period",
            options=["1D", "1W", "1M"],
            index=1,
            help="Short-term period to detect the breakout",
            key="laggard_awakening_period",
        )
        RS_COL = {"1D": "r_1d", "1W": "r_1w", "1M": "r_1mo"}
        awakening_col = RS_COL[rs_awakening]

        fund_type_filter = fund_type_sidebar(key="laggard_fund_types")

    with content_col:
        # ── load data ──
        _all_data = load_latest_perf()

        if _all_data.empty:
            st.warning("No data loaded.")
            st.stop()

        df = filter_by_fund_type(_all_data, fund_type_filter)

        # get benchmark row
        benchmark_row = _all_data[_all_data["ticker"] == benchmark_ticker]
        if benchmark_row.empty:
            st.error(f"Benchmark ticker '{benchmark_ticker}' not found in data.")
            st.stop()

        benchmark_desc = benchmark_row["description"].iloc[0]

        # ── compute relative strength columns ──
        RS_PERIODS = {
            "1D": "r_1d",
            "1W": "r_1w",
            "2W": "r_2w",
            "1M": "r_1mo",
            "3M": "r_3mo",
            "6M": "r_6mo",
            "1Y": "r_1y",
            "2Y": "r_2y",
            "3Y": "r_3y",
        }

        for label, col in RS_PERIODS.items():
            bm_val = (
                float(benchmark_row[col].iloc[0])
                if col in benchmark_row.columns
                else 0.0
            )
            df[f"rs_{label}"] = df[col] - bm_val

        # laggard excess return
        bm_laggard_return = float(benchmark_row[laggard_col].iloc[0])
        bm_awakening_return = float(benchmark_row[awakening_col].iloc[0])

        st.caption(
            f"Benchmark: **{benchmark_desc}** ({benchmark_ticker}) — "
            f"{laggard_period} return: {bm_laggard_return:+.2f}% · "
            f"{rs_awakening} return: {bm_awakening_return:+.2f}%"
        )

        # ═══════════════════════════════════════════
        # Section 1: Relative Strength Overview
        # ═══════════════════════════════════════════
        st.header("📊 Relative Strength Overview")
        st.markdown(
            f"Excess return vs **{benchmark_desc}** at every horizon. "
            "Green = outperforming, Red = underperforming."
        )

        # Build RS heatmap data
        rs_display_cols = ["rs_1D", "rs_1W", "rs_1M", "rs_3M", "rs_6M", "rs_1Y"]
        rs_display_labels = ["1D", "1W", "1M", "3M", "6M", "1Y"]

        # Add 2Y and 3Y if they exist in the data
        for extra in ["rs_2Y", "rs_3Y"]:
            if extra in df.columns and df[extra].notna().any():
                rs_display_cols.append(extra)
                rs_display_labels.append(extra.replace("rs_", ""))

        rs_overview = df.set_index("description")[rs_display_cols].copy()
        rs_overview.columns = rs_display_labels
        rs_overview = rs_overview.dropna(subset=["1Y"])

        # Sort by 1W relative strength (most interesting ordering)
        rs_overview = rs_overview.sort_values("1W", ascending=False)

        # Limit to top/bottom for readability
        if len(rs_overview) > 50:
            st.caption(
                f"Showing top 25 and bottom 25 by 1W relative strength (of {len(rs_overview)} total)"
            )
            rs_top = rs_overview.head(25)
            rs_bottom = rs_overview.tail(25)
            rs_show = pd.concat([rs_top, rs_bottom])
        else:
            rs_show = rs_overview

        for col in rs_show.columns:
            rs_show[col] = pd.to_numeric(rs_show[col], errors="coerce")

        vals = rs_show.values.flatten()
        finite = vals[np.isfinite(vals)]
        rng = (
            min(max(float(pd.Series(np.abs(finite)).quantile(0.75)), 10.0), 40.0)
            if len(finite) > 0
            else 20.0
        )

        fig_rs = px.imshow(
            rs_show.values,
            x=list(rs_show.columns),
            y=list(rs_show.index),
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            range_color=[-rng, rng],
            text_auto=".1f",
            aspect="auto",
            labels=dict(color="Excess Return %"),
        )
        fig_rs.update_layout(height=max(400, len(rs_show) * 25))
        st.plotly_chart(fig_rs, width="stretch")

        # ═══════════════════════════════════════════
        # Section 2: Laggard Awakening
        # ═══════════════════════════════════════════
        st.header("🔄 Laggard Awakening")
        st.markdown(
            f"**Long-term laggards** (underperformed benchmark by ≥{underperf_threshold}% over "
            f"{laggard_period}, with absolute return ≤{max_abs_return}%) that now show "
            f"**positive {rs_awakening} relative strength** vs the benchmark."
        )

        awakening, sleeping = scan_laggard_awakening(
            df,
            benchmark_ticker=benchmark_ticker,
            laggard_period=laggard_period,
            awakening_period=rs_awakening,
            underperf_threshold=underperf_threshold,
            max_abs_return=max_abs_return,
        )
        laggards = pd.concat([awakening, sleeping], ignore_index=True)

        if awakening.empty and laggards.empty:
            st.info(
                f"No instruments underperformed the benchmark by ≥{underperf_threshold}% over "
                f"{laggard_period}. Try lowering the threshold."
            )
        else:
            if not awakening.empty:
                st.subheader(f"🟢 Awakening ({len(awakening)} instruments)")
                st.markdown(
                    "These laggards are now **outperforming** the benchmark short-term. "
                    "Ranked by awakening score = (short-term RS) × (long-term deficit)."
                )

                display_cols = [
                    "description",
                    "ticker",
                    "fund_type",
                    laggard_col,
                    f"rs_{laggard_period}",
                    "r_1d",
                    "r_1w",
                    "r_1mo",
                    "rs_1D",
                    "rs_1W",
                    "rs_1M",
                    "ma_21",
                    "ma_63",
                    "ma_252",
                    "drawdown_52w",
                    "awakening_score",
                    "Price (90d)",
                ]
                available = [c for c in display_cols if c in awakening.columns]

                fmt_cols = [
                    c
                    for c in [
                        laggard_col,
                        f"rs_{laggard_period}",
                        "r_1d",
                        "r_1w",
                        "r_1mo",
                        "rs_1D",
                        "rs_1W",
                        "rs_1M",
                        "ma_21",
                        "ma_63",
                        "ma_252",
                        "drawdown_52w",
                    ]
                    if c in available
                ]

                available = [
                    c
                    for c in display_cols
                    if c in awakening.columns or c == "Price (90d)"
                ]

                add_sparkline_column(awakening)

                st.dataframe(
                    awakening[available]
                    .style.format(subset=fmt_cols, formatter="{:+.2f}%")
                    .format(
                        subset=[c for c in ["awakening_score"] if c in available],
                        formatter="{:.1f}",
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
                        laggard_col: st.column_config.NumberColumn(
                            laggard_col, format="%.2f%%", width="small"
                        ),
                        f"rs_{laggard_period}": st.column_config.NumberColumn(
                            f"rs_{laggard_period}", format="%.2f%%", width="small"
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
                        "rs_1D": st.column_config.NumberColumn("rs_1D", width="small"),
                        "rs_1W": st.column_config.NumberColumn("rs_1W", width="small"),
                        "rs_1M": st.column_config.NumberColumn("rs_1M", width="small"),
                        "ma_21": st.column_config.NumberColumn(
                            "ma_21", format="%.2f%%", width="small"
                        ),
                        "ma_63": st.column_config.NumberColumn(
                            "ma_63", format="%.2f%%", width="small"
                        ),
                        "ma_252": st.column_config.NumberColumn(
                            "ma_252", format="%.2f%%", width="small"
                        ),
                        "drawdown_52w": st.column_config.NumberColumn(
                            "DD 52W", format="%.2f%%", width="small"
                        ),
                        "awakening_score": st.column_config.NumberColumn(
                            "awakening_score", format="%.1f", width="small"
                        ),
                    },
                    hide_index=True,
                    height=min(500, 50 + len(awakening) * 35),
                )

                # Top awakening bar chart
                top_awk = awakening.head(15)
                fig_awk = px.bar(
                    top_awk,
                    x="awakening_score",
                    y="description",
                    orientation="h",
                    color=f"rs_{rs_awakening}",
                    color_continuous_scale="YlGn",
                    labels={
                        "awakening_score": "Awakening Score",
                        "description": "",
                        f"rs_{rs_awakening}": f"{rs_awakening} RS %",
                    },
                    title=f"Top {len(top_awk)} — Laggard Awakening Score",
                )
                fig_awk.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=max(300, len(top_awk) * 35),
                    showlegend=False,
                )
                st.plotly_chart(fig_awk, width="stretch")
            else:
                st.info(
                    f"No laggards are currently showing positive {rs_awakening} relative strength. "
                    "They're all still sleeping."
                )

            # Still sleeping
            if not sleeping.empty:
                with st.expander(
                    f"💤 Still Sleeping ({len(sleeping)} laggards)", expanded=False
                ):
                    st.markdown(
                        "These are still underperforming short-term — no breakout yet."
                    )
                    sleep_cols = [
                        "description",
                        "ticker",
                        laggard_col,
                        f"rs_{laggard_period}",
                        awakening_col,
                        f"rs_{rs_awakening}",
                        "ma_252",
                    ]
                    sleep_available = [c for c in sleep_cols if c in sleeping.columns]
                    sleep_fmt = [
                        c
                        for c in [
                            laggard_col,
                            f"rs_{laggard_period}",
                            awakening_col,
                            f"rs_{rs_awakening}",
                            "ma_252",
                        ]
                        if c in sleep_available
                    ]
                    st.dataframe(
                        sleeping.sort_values(f"rs_{rs_awakening}", ascending=False)[
                            sleep_available
                        ].style.format(subset=sleep_fmt, formatter="{:+.2f}%"),
                        hide_index=True,
                        height=400,
                    )

        # ═══════════════════════════════════════════
        # Section 3: Breakout Confirmation
        # ═══════════════════════════════════════════
        st.header("✅ Breakout Confirmation")
        st.markdown(
            "Laggards that have **also crossed above key moving averages** — "
            "combining relative strength improvement with technical breakout signals."
        )

        if not laggards.empty:
            confirmed = scan_laggard_breakout_confirmations(
                laggards, awakening_period=rs_awakening
            )

            if not confirmed.empty:
                st.markdown(
                    f"**{len(confirmed)}** laggards with positive RS and above 21d MA:"
                )

                conf_cols = [
                    "description",
                    "ticker",
                    "fund_type",
                    laggard_col,
                    f"rs_{laggard_period}",
                    f"rs_{rs_awakening}",
                    "ma_21",
                    "ma_63",
                    "ma_126",
                    "ma_252",
                    "above_63d",
                    "above_252d",
                    "ma_cross_count",
                ]
                conf_available = [c for c in conf_cols if c in confirmed.columns]
                conf_fmt = [
                    c
                    for c in [
                        laggard_col,
                        f"rs_{laggard_period}",
                        f"rs_{rs_awakening}",
                        "ma_21",
                        "ma_63",
                        "ma_126",
                        "ma_252",
                    ]
                    if c in conf_available
                ]

                st.dataframe(
                    confirmed[conf_available].style.format(
                        subset=conf_fmt, formatter="{:+.2f}%"
                    ),
                    hide_index=True,
                    height=min(450, 50 + len(confirmed) * 35),
                )
            else:
                st.info(
                    "No laggards are currently both above their 21d MA and outperforming "
                    "the benchmark short-term. The breakout hasn't happened yet."
                )
        else:
            st.info("No laggards detected with current settings.")

        # ═══════════════════════════════════════════
        # Section 4: Full Relative Strength Table
        # ═══════════════════════════════════════════
        st.header("📋 Full Relative Strength Table")
        st.markdown(
            f"Excess returns vs **{benchmark_desc}** ({benchmark_ticker}) at every horizon. "
            "Positive = outperforming, negative = underperforming."
        )

        rs_table_cols = [
            "description",
            "ticker",
            "fund_type",
            "rs_1D",
            "rs_1W",
            "rs_1M",
            "rs_3M",
            "rs_6M",
            "rs_1Y",
        ]
        # Add longer periods if available
        for extra in ["rs_2Y", "rs_3Y"]:
            if extra in df.columns:
                rs_table_cols.append(extra)

        rs_available = [c for c in rs_table_cols if c in df.columns]
        rs_fmt = [c for c in rs_available if c.startswith("rs_")]

        sort_by = st.selectbox(
            "Sort by",
            options=[c for c in rs_available if c.startswith("rs_")],
            index=1,  # default rs_1W
            format_func=lambda x: x.replace("rs_", "") + " Relative Strength",
            key="laggard_rs_table_sort",
        )

        rs_table = df[df["ticker"] != benchmark_ticker][rs_available].copy()
        rs_table = rs_table.sort_values(sort_by, ascending=False).reset_index(drop=True)
        rs_table.index += 1
        rs_table.index.name = "Rank"

        st.dataframe(
            rs_table.style.format(subset=rs_fmt, formatter="{:+.2f}%"),
            height=600,
        )
