"""
Daily Summary — morning briefing page aggregating the most important signals
from across all scanners into one consolidated view.

Sections:
  1) Market Pulse: headline metrics for the day
  2) Biggest Movers: top gainers/losers by fund type
  3) Opportunity Radar: cross-scanner summary (puke, breakout, laggard, pullback)
  4) Z-Score Alerts: unusual statistical moves
  5) MA Crossover Summary: bullish/bearish crossovers today
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import config
from data import (
    load_latest_perf,
    fund_type_sidebar,
    filter_by_fund_type,
    add_sparkline_column,
)


def _add_sparkline_columns(df: pd.DataFrame) -> pd.DataFrame:
    add_sparkline_column(df)
    add_sparkline_column(df, col_name="Price (1y)", days=365)
    return df


def _sparkline_config() -> dict:
    return {
        "Price (90d)": st.column_config.LineChartColumn("Price (90d)", width="small"),
        "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
    }


def render():
    st.title("📊 Daily Summary")
    st.markdown(
        "Your **morning briefing** — biggest moves, opportunity signals, and key "
        "alerts across all scanners in one page."
    )

    # ── sidebar ──
    st.header("Settings")
    fund_type_filter = fund_type_sidebar(
        default=["eq", "commod"], key="daily_summary_fund_types"
    )

    # Load data: today (rown=1) and yesterday (rown=2) for crossover detection
    raw = load_latest_perf(max_rown=2)

    if raw.empty:
        st.warning("No data loaded.")
        st.stop()

    today_df = raw[raw["rown"] == 1].copy()
    yesterday_df = raw[raw["rown"] == 2].copy()

    # Apply fund type filter
    today_filtered = filter_by_fund_type(today_df, fund_type_filter)

    data_date = today_df["date"].iloc[0] if "date" in today_df.columns else "Unknown"

    # For crossover comparisons, index by ticker
    today_idx = today_df.set_index("ticker")
    yesterday_idx = yesterday_df.set_index("ticker")
    common_tickers = today_idx.index.intersection(yesterday_idx.index)
    today_idx = today_idx.loc[common_tickers]
    yesterday_idx = yesterday_idx.loc[common_tickers]


    # ═══════════════════════════════════════════
    # Section 1: Market Pulse
    # ═══════════════════════════════════════════
    st.header("🫀 Market Pulse")

    col_date, col_total, col_pos, col_neg, col_avg = st.columns(5)

    with col_date:
        st.metric("Data Date", str(data_date)[:10] if data_date != "Unknown" else "Unknown")
    with col_total:
        st.metric("Instruments Tracked", f"{len(today_filtered)}")
    with col_pos:
        pct_positive = (today_filtered["r_1d"] > 0).mean() * 100
        st.metric("% Positive Today", f"{pct_positive:.0f}%")
    with col_neg:
        pct_negative = (today_filtered["r_1d"] < 0).mean() * 100
        st.metric("% Negative Today", f"{pct_negative:.0f}%")
    with col_avg:
        avg_return = today_filtered["r_1d"].mean()
        st.metric("Avg 1D Return", f"{avg_return:+.2f}%")

    # Fund type breakdown
    st.subheader("Returns by Asset Class")
    ft_summary = (
        today_filtered.groupby("fund_type")["r_1d"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    ft_summary.columns = ["Fund Type", "Mean 1D Return", "Median 1D Return", "Count"]
    ft_summary = ft_summary.sort_values("Mean 1D Return", ascending=False)

    fig_ft = px.bar(
        ft_summary,
        x="Fund Type",
        y="Mean 1D Return",
        color="Mean 1D Return",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text="Count",
        labels={"Mean 1D Return": "Avg Return (%)"},
        title="Average 1D Return by Fund Type",
    )
    fig_ft.update_layout(height=300, showlegend=False)
    fig_ft.update_traces(textposition="outside")
    st.plotly_chart(fig_ft, use_container_width=True)

    # Market breadth: % above each MA
    ma_cols = {"21d": "ma_21", "63d": "ma_63", "126d": "ma_126", "252d": "ma_252"}
    breadth_cols = st.columns(4)
    for col_w, (ma_label, ma_col) in zip(breadth_cols, ma_cols.items()):
        with col_w:
            if ma_col in today_filtered.columns:
                pct_above = (today_filtered[ma_col] > 0).mean() * 100
                st.metric(f"% Above {ma_label} MA", f"{pct_above:.0f}%")

    st.markdown("---")


    # ═══════════════════════════════════════════
    # Section 2: Biggest Movers
    # ═══════════════════════════════════════════
    @st.fragment
    def biggest_movers_section():
        st.header("🔥 Biggest Movers")

        mover_tab_all, mover_tab_eq, mover_tab_commod, mover_tab_bonds = st.tabs(
            ["All", "Equities", "Commodities", "Bonds"]
        )

        fund_type_tabs = {
            "All": today_filtered,
            "Equities": today_filtered[today_filtered["fund_type"].str.startswith("eq")],
            "Commodities": today_filtered[today_filtered["fund_type"] == "commod"],
            "Bonds": today_filtered[today_filtered["fund_type"].str.startswith("bonds")],
        }

        for tab_widget, (tab_name, tab_data) in zip(
            [mover_tab_all, mover_tab_eq, mover_tab_commod, mover_tab_bonds],
            fund_type_tabs.items(),
        ):
            with tab_widget:
                if tab_data.empty:
                    st.info(f"No {tab_name.lower()} instruments in current filter.")
                    continue

                col_up, col_dn = st.columns(2)

                top_gainers = tab_data.nlargest(10, "r_1d")[
                    ["description", "fund_type", "r_1d", "r_1w", "r_1mo", "drawdown_52w"]
                ].copy()

                top_losers = tab_data.nsmallest(10, "r_1d")[
                    ["description", "fund_type", "r_1d", "r_1w", "r_1mo", "drawdown_52w"]
                ].copy()

                with col_up:
                    st.subheader("🟢 Top Gainers")
                    fig_gain = px.bar(
                        top_gainers,
                        x="r_1d",
                        y="description",
                        orientation="h",
                        color="r_1d",
                        color_continuous_scale="Greens",
                        labels={"r_1d": "1D Return (%)", "description": ""},
                    )
                    fig_gain.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=350,
                        showlegend=False,
                    )
                    st.plotly_chart(
                        fig_gain,
                        use_container_width=True,
                        key=f"daily_movers_{tab_name}_gainers",
                    )

                with col_dn:
                    st.subheader("🔴 Top Losers")
                    fig_lose = px.bar(
                        top_losers,
                        x="r_1d",
                        y="description",
                        orientation="h",
                        color="r_1d",
                        color_continuous_scale="Reds_r",
                        labels={"r_1d": "1D Return (%)", "description": ""},
                    )
                    fig_lose.update_layout(
                        yaxis=dict(autorange="reversed"),
                        height=350,
                        showlegend=False,
                    )
                    st.plotly_chart(
                        fig_lose,
                        use_container_width=True,
                        key=f"daily_movers_{tab_name}_losers",
                    )

                # Combined table with more detail
                combined = pd.concat(
                    [top_gainers.head(5), top_losers.head(5)], ignore_index=True
                )
                combined["ticker"] = pd.concat(
                    [
                        tab_data.nlargest(5, "r_1d")["ticker"].reset_index(drop=True),
                        tab_data.nsmallest(5, "r_1d")["ticker"].reset_index(drop=True),
                    ],
                    ignore_index=True,
                )
                _add_sparkline_columns(combined)
                st.dataframe(
                    combined.style.format(
                        subset=["r_1d", "r_1w", "r_1mo", "drawdown_52w"],
                        formatter="{:+.2f}%",
                    ),
                    column_config={
                        "Price (90d)": st.column_config.LineChartColumn(
                            "Price (90d)", width="small"
                        ),
                        "Price (1y)": st.column_config.LineChartColumn(
                            "Price (1y)", width="small"
                        ),
                        "description": st.column_config.TextColumn(
                            "Instrument", width="medium"
                        ),
                        "r_1d": st.column_config.NumberColumn(
                            "1D", format="%.2f%%", width="small"
                        ),
                        "r_1w": st.column_config.NumberColumn(
                            "1W", format="%.2f%%", width="small"
                        ),
                        "r_1mo": st.column_config.NumberColumn(
                            "1M", format="%.2f%%", width="small"
                        ),
                        "drawdown_52w": st.column_config.NumberColumn(
                            "DD 52W", format="%.2f%%", width="small"
                        ),
                    },
                    hide_index=True,
                    height=min(400, 50 + len(combined) * 35),
                )


    biggest_movers_section()

    st.markdown("---")


    # ═══════════════════════════════════════════
    # Section 3: Opportunity Radar
    # ═══════════════════════════════════════════
    @st.fragment
    def opportunity_radar_section():
        st.header("🎯 Opportunity Radar")
        st.markdown(
            "Cross-scanner signals — potential opportunities surfaced from "
            "PukeDetector, BreakoutScanner, LaggardBreakout, TodaysCrossings, "
            "and PullbackScanner logic."
        )

        df = today_filtered.copy()

        # ── 3a: Puke Candidates (buy-the-dip) ──
        with st.expander("💥 Puke Candidates (Buy-the-Dip Signals)", expanded=True):
            z_cols = [c for c in ["z_1d", "z_1w", "z_2w", "z_1mo"] if c in df.columns]
            if z_cols:
                z_mask = df[z_cols].apply(lambda row: row.max(), axis=1) >= 2.0
                puke = df[z_mask].copy()
                if not puke.empty:
                    puke["max_z"] = puke[z_cols].max(axis=1)
                    # Only crashers (negative returns)
                    puke_crashers = puke[puke["r_1d"] < 0].sort_values(
                        "max_z", ascending=False
                    )
                    if not puke_crashers.empty:
                        puke_crashers = puke_crashers.head(10)
                        display_cols = [
                            "description",
                            "ticker",
                            "fund_type",
                            "r_1d",
                            "r_1w",
                            "drawdown_52w",
                            "max_z",
                        ]
                        available = [c for c in display_cols if c in puke_crashers.columns]
                        _add_sparkline_columns(puke_crashers)
                        available.append("Price (90d)")
                        available.append("Price (1y)")
                        st.markdown(
                            f"**{len(puke_crashers)}** instrument(s) with high z-scores "
                            "and negative returns — potential capitulation."
                        )
                        st.dataframe(
                            puke_crashers[available]
                            .style.format(
                                subset=[
                                    c
                                    for c in ["r_1d", "r_1w", "drawdown_52w"]
                                    if c in available
                                ],
                                formatter="{:+.2f}%",
                            )
                            .format(
                                subset=[c for c in ["max_z"] if c in available],
                                formatter="{:.2f}σ",
                            ),
                            column_config={
                                **_sparkline_config(),
                                "description": st.column_config.TextColumn(
                                    "Instrument", width="medium"
                                ),
                            },
                            hide_index=True,
                            height=min(350, 50 + len(puke_crashers) * 35),
                        )
                    else:
                        st.info("No crashers with high z-scores today.")
                else:
                    st.info("No z-score spikes detected — markets are calm.")
            else:
                st.info("Z-score columns not available.")

        # ── 3b: Breakout Candidates ──
        with st.expander("📈 Breakout Candidates (MA Breakouts)", expanded=True):
            # Price just crossed above 63-day MA (medium-term breakout)
            breakouts = df[(df["ma_63"] > 0) & (df["ma_63"] <= 5)].copy()
            if not breakouts.empty:
                # Multi-MA strength
                breakouts["mas_above"] = (
                    (breakouts["ma_21"] > 0).astype(int)
                    + (breakouts["ma_63"] > 0).astype(int)
                    + (breakouts["ma_126"] > 0).astype(int)
                    + (breakouts["ma_252"] > 0).astype(int)
                )
                breakouts = breakouts.sort_values(
                    ["mas_above", "ma_63"], ascending=[False, True]
                ).head(10)
                display_cols = [
                    "description",
                    "ticker",
                    "fund_type",
                    "ma_21",
                    "ma_63",
                    "ma_126",
                    "ma_252",
                    "mas_above",
                    "r_1w",
                    "drawdown_52w",
                ]
                available = [c for c in display_cols if c in breakouts.columns]
                _add_sparkline_columns(breakouts)
                available.append("Price (90d)")
                available.append("Price (1y)")
                st.markdown(
                    f"**{len(breakouts)}** instrument(s) just above 63-day MA (0-5%) "
                    "with multi-MA strength ranking."
                )
                st.dataframe(
                    breakouts[available].style.format(
                        subset=[
                            c
                            for c in [
                                "ma_21",
                                "ma_63",
                                "ma_126",
                                "ma_252",
                                "r_1w",
                                "drawdown_52w",
                            ]
                            if c in available
                        ],
                        formatter="{:+.2f}%",
                    ),
                    column_config={
                        **_sparkline_config(),
                        "description": st.column_config.TextColumn(
                            "Instrument", width="medium"
                        ),
                    },
                    hide_index=True,
                    height=min(350, 50 + len(breakouts) * 35),
                )
            else:
                st.info("No fresh breakouts (0-5% above 63-day MA) detected.")

        # ── 3c: Laggard Awakenings ──
        with st.expander(
            "🔄 Laggard Awakenings (Long-term Underperformers Waking Up)", expanded=True
        ):
            # Use VWRP as default benchmark
            benchmark_ticker = config.DEFAULT_BENCHMARK
            benchmark_row = today_df[today_df["ticker"] == benchmark_ticker]
            if not benchmark_row.empty and "r_1y" in df.columns:
                bm_1y = float(benchmark_row["r_1y"].iloc[0])
                bm_1w = float(benchmark_row["r_1w"].iloc[0])
                laggards = df[
                    (df["r_1y"] - bm_1y <= -10)  # 10% underperformance over 1Y
                    & (df["ticker"] != benchmark_ticker)
                ].copy()
                laggards["rs_1w"] = laggards["r_1w"] - bm_1w
                awakening = laggards[laggards["rs_1w"] > 0].copy()
                if not awakening.empty:
                    awakening["awakening_score"] = awakening["rs_1w"] * (
                        -(awakening["r_1y"] - bm_1y)
                    )
                    awakening = awakening.sort_values(
                        "awakening_score", ascending=False
                    ).head(10)
                    display_cols = [
                        "description",
                        "ticker",
                        "fund_type",
                        "r_1y",
                        "r_1w",
                        "rs_1w",
                        "ma_252",
                        "drawdown_52w",
                    ]
                    available = [c for c in display_cols if c in awakening.columns]
                    _add_sparkline_columns(awakening)
                    available.append("Price (90d)")
                    available.append("Price (1y)")
                    st.markdown(
                        f"**{len(awakening)}** laggard(s) now showing positive "
                        f"1W relative strength vs {benchmark_ticker}."
                    )
                    st.dataframe(
                        awakening[available].style.format(
                            subset=[
                                c
                                for c in [
                                    "r_1y",
                                    "r_1w",
                                    "rs_1w",
                                    "ma_252",
                                    "drawdown_52w",
                                ]
                                if c in available
                            ],
                            formatter="{:+.2f}%",
                        ),
                        column_config={
                            **_sparkline_config(),
                            "description": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                        },
                        hide_index=True,
                        height=min(350, 50 + len(awakening) * 35),
                    )
                else:
                    st.info("No laggard awakenings detected this week.")
            else:
                st.info("Benchmark data not available for laggard analysis.")

        # ── 3d: Leaders Weakening ──
        with st.expander(
            "⚠️ Leaders Weakening (Long-term Outperformers Losing Momentum)",
            expanded=True,
        ):
            benchmark_ticker = config.DEFAULT_BENCHMARK
            benchmark_row = today_df[today_df["ticker"] == benchmark_ticker]
            if not benchmark_row.empty and "r_1y" in df.columns:
                bm_1y = float(benchmark_row["r_1y"].iloc[0])
                bm_1w = float(benchmark_row["r_1w"].iloc[0])
                leaders = df[
                    (df["r_1y"] - bm_1y >= 10)  # 10% outperformance over 1Y
                    & (df["ticker"] != benchmark_ticker)
                ].copy()
                leaders["rs_1w"] = leaders["r_1w"] - bm_1w
                weakening = leaders[leaders["rs_1w"] < 0].copy()
                if not weakening.empty:
                    weakening["weakening_score"] = (-weakening["rs_1w"]) * (
                        weakening["r_1y"] - bm_1y
                    )
                    weakening = weakening.sort_values(
                        "weakening_score", ascending=False
                    ).head(10)
                    display_cols = [
                        "description",
                        "ticker",
                        "fund_type",
                        "r_1y",
                        "r_1w",
                        "rs_1w",
                        "ma_21",
                        "ma_63",
                        "ma_252",
                        "drawdown_52w",
                        "weakening_score",
                    ]
                    available = [c for c in display_cols if c in weakening.columns]
                    _add_sparkline_columns(weakening)
                    available.append("Price (90d)")
                    available.append("Price (1y)")
                    st.markdown(
                        f"**{len(weakening)}** leader(s) still outperforming over 1Y "
                        f"but now showing negative 1W relative strength vs {benchmark_ticker}."
                    )
                    st.caption(
                        "Use this as a trim/stop-review monitor: prior leaders where short-term relative momentum is fading."
                    )
                    st.dataframe(
                        weakening[available].style.format(
                            subset=[
                                c
                                for c in [
                                    "r_1y",
                                    "r_1w",
                                    "rs_1w",
                                    "ma_21",
                                    "ma_63",
                                    "ma_252",
                                    "drawdown_52w",
                                ]
                                if c in available
                            ],
                            formatter="{:+.2f}%",
                        ).format(subset=["weakening_score"], formatter="{:.1f}"),
                        column_config={
                            **_sparkline_config(),
                            "description": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                        },
                        hide_index=True,
                        height=min(350, 50 + len(weakening) * 35),
                    )
                else:
                    st.info("No weakening leaders detected this week.")
            else:
                st.info("Benchmark data not available for leader weakening analysis.")

        # ── 3e: 52W High / Deep Drawdown ──
        with st.expander("🏔️ New 52W Highs & Deep Drawdowns", expanded=True):
            col_hi, col_lo = st.columns(2)
            with col_hi:
                st.subheader("At 52W Highs")
                at_high = df[df["drawdown_52w"] >= -0.5].copy()
                if not at_high.empty:
                    st.markdown(f"**{len(at_high)}** instrument(s) at or near 52W highs.")
                    display = at_high[
                        ["description", "ticker", "fund_type", "r_1d", "r_1w", "r_1mo"]
                    ].head(10).copy()
                    _add_sparkline_columns(display)
                    st.dataframe(
                        display[
                            [
                                "description",
                                "Price (90d)",
                                "Price (1y)",
                                "ticker",
                                "fund_type",
                                "r_1d",
                                "r_1w",
                                "r_1mo",
                            ]
                        ].style.format(
                            subset=["r_1d", "r_1w", "r_1mo"], formatter="{:+.2f}%"
                        ),
                        column_config={
                            **_sparkline_config(),
                            "description": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                        },
                        hide_index=True,
                        height=min(300, 50 + len(display) * 35),
                    )
                else:
                    st.info("No instruments at 52W highs.")

            with col_lo:
                st.subheader("Deep Drawdowns (> -15%)")
                deep_dd = df[df["drawdown_52w"] < -15].sort_values("drawdown_52w").copy()
                if not deep_dd.empty:
                    st.markdown(
                        f"**{len(deep_dd)}** instrument(s) in severe drawdown territory."
                    )
                    display = deep_dd[
                        ["description", "ticker", "fund_type", "r_1d", "r_1w", "drawdown_52w"]
                    ].head(10).copy()
                    _add_sparkline_columns(display)
                    st.dataframe(
                        display[
                            [
                                "description",
                                "Price (90d)",
                                "Price (1y)",
                                "ticker",
                                "fund_type",
                                "r_1d",
                                "r_1w",
                                "drawdown_52w",
                            ]
                        ].style.format(
                            subset=["r_1d", "r_1w", "drawdown_52w"],
                            formatter="{:+.2f}%",
                        ),
                        column_config={
                            **_sparkline_config(),
                            "description": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                        },
                        hide_index=True,
                        height=min(300, 50 + len(display) * 35),
                    )
                else:
                    st.info("No instruments in severe drawdown.")

        # ── 3e: Pullback Candidates ──
        with st.expander("🎯 Pullback Candidates (Uptrend + Pullback)", expanded=True):
            pullbacks = df[(df["ma_252"] >= 0) & (df["ma_21"] < 0)].copy()
            if not pullbacks.empty:
                pullbacks["pullback_score"] = pullbacks["ma_252"] * (-pullbacks["ma_21"])
                pullbacks = pullbacks.sort_values("pullback_score", ascending=False).head(
                    10
                )
                display_cols = [
                    "description",
                    "ticker",
                    "fund_type",
                    "ma_21",
                    "ma_63",
                    "ma_252",
                    "r_1w",
                    "drawdown_52w",
                ]
                available = [c for c in display_cols if c in pullbacks.columns]
                _add_sparkline_columns(pullbacks)
                available.append("Price (90d)")
                available.append("Price (1y)")
                st.markdown(
                    f"**{len(pullbacks)}** instrument(s) above 252d MA but "
                    "pulling back below 21d MA."
                )
                st.dataframe(
                    pullbacks[available].style.format(
                        subset=[
                            c
                            for c in ["ma_21", "ma_63", "ma_252", "r_1w", "drawdown_52w"]
                            if c in available
                        ],
                        formatter="{:+.2f}%",
                    ),
                    column_config={
                        **_sparkline_config(),
                        "description": st.column_config.TextColumn(
                            "Instrument", width="medium"
                        ),
                    },
                    hide_index=True,
                    height=min(350, 50 + len(pullbacks) * 35),
                )
            else:
                st.info("No pullback candidates right now.")


    opportunity_radar_section()

    st.markdown("---")


    # ═══════════════════════════════════════════
    # Section 4: Z-Score Alerts
    # ═══════════════════════════════════════════
    @st.fragment
    def zscore_alerts_section():
        st.header("⚡ Z-Score Alerts")
        st.markdown("Instruments with statistically unusual moves today (z-score ≥ 2σ).")

        df = today_filtered.copy()
        z_cols = [c for c in ["z_1d", "z_1w", "z_2w", "z_1mo"] if c in df.columns]

        if not z_cols:
            st.info("Z-score columns not available.")
            return

        z_mask = df[z_cols].apply(lambda row: row.max(), axis=1) >= 2.0
        extreme = df[z_mask].copy()

        if extreme.empty:
            st.info("No unusual moves detected (z < 2σ across all instruments).")
            return

        extreme["max_z"] = extreme[z_cols].max(axis=1)
        extreme["max_z_period"] = extreme[z_cols].idxmax(axis=1)

        # Split into crashers and spikers
        crashers = (
            extreme[extreme["r_1d"] < 0]
            .sort_values("max_z", ascending=False)
            .reset_index(drop=True)
        )
        spikers = (
            extreme[extreme["r_1d"] >= 0]
            .sort_values("max_z", ascending=False)
            .reset_index(drop=True)
        )

        col_crash, col_spike = st.columns(2)

        display_cols = (
            ["description", "ticker", "fund_type"]
            + z_cols
            + ["r_1d", "r_1w", "drawdown_52w", "max_z"]
        )

        fmt_cols = [
            c for c in z_cols + ["r_1d", "r_1w", "drawdown_52w"] if c in display_cols
        ]

        with col_crash:
            st.subheader(f"📉 Crashers ({len(crashers)})")
            if not crashers.empty:
                _add_sparkline_columns(crashers)
                cols_show = [c for c in display_cols if c in crashers.columns] + [
                    "Price (90d)",
                    "Price (1y)",
                ]
                st.dataframe(
                    crashers[cols_show]
                    .style.format(subset=fmt_cols, formatter="{:.2f}")
                    .format(subset=["max_z"], formatter="{:.2f}σ"),
                    column_config={
                        **_sparkline_config(),
                        "description": st.column_config.TextColumn(
                            "Instrument", width="medium"
                        ),
                    },
                    hide_index=True,
                    height=min(400, 50 + len(crashers) * 35),
                )
            else:
                st.info("No crashers.")

        with col_spike:
            st.subheader(f"📈 Spikers ({len(spikers)})")
            if not spikers.empty:
                _add_sparkline_columns(spikers)
                cols_show = [c for c in display_cols if c in spikers.columns] + [
                    "Price (90d)",
                    "Price (1y)",
                ]
                st.dataframe(
                    spikers[cols_show]
                    .style.format(subset=fmt_cols, formatter="{:.2f}")
                    .format(subset=["max_z"], formatter="{:.2f}σ"),
                    column_config={
                        **_sparkline_config(),
                        "description": st.column_config.TextColumn(
                            "Instrument", width="medium"
                        ),
                    },
                    hide_index=True,
                    height=min(400, 50 + len(spikers) * 35),
                )
            else:
                st.info("No spikers.")


    zscore_alerts_section()

    st.markdown("---")


    # ═══════════════════════════════════════════
    # Section 5: MA Crossover Summary
    # ═══════════════════════════════════════════
    @st.fragment
    def ma_crossover_section():
        st.header("📐 MA Crossover Summary")
        st.markdown("Instruments that crossed **above or below** a moving average today.")

        # Apply fund type filter to the indexed data
        filtered_common = [
            t for t in common_tickers if t in today_filtered["ticker"].values
        ]

        ma_defs = {
            "21d MA": "ma_21",
            "63d MA": "ma_63",
            "126d MA": "ma_126",
            "252d MA": "ma_252",
        }

        crossover_events = []
        for ma_label, ma_col in ma_defs.items():
            for ticker in filtered_common:
                today_val = today_idx.loc[ticker, ma_col]
                yest_val = yesterday_idx.loc[ticker, ma_col]
                if pd.isna(today_val) or pd.isna(yest_val):
                    continue
                if today_val > 0 and yest_val <= 0:
                    crossover_events.append(
                        {
                            "Ticker": ticker,
                            "Instrument": today_idx.loc[ticker, "description"],
                            "MA": ma_label,
                            "Direction": "🟢 ABOVE",
                            "Today": today_val,
                            "Yesterday": yest_val,
                            "1D Return": today_idx.loc[ticker, "r_1d"],
                        }
                    )
                elif today_val < 0 and yest_val >= 0:
                    crossover_events.append(
                        {
                            "Ticker": ticker,
                            "Instrument": today_idx.loc[ticker, "description"],
                            "MA": ma_label,
                            "Direction": "🔴 BELOW",
                            "Today": today_val,
                            "Yesterday": yest_val,
                            "1D Return": today_idx.loc[ticker, "r_1d"],
                        }
                    )

        if crossover_events:
            cross_df = pd.DataFrame(crossover_events)
            cross_df["ticker"] = cross_df["Ticker"]
            _add_sparkline_columns(cross_df)

            bullish = cross_df[cross_df["Direction"].str.contains("ABOVE")]
            bearish = cross_df[cross_df["Direction"].str.contains("BELOW")]

            col_b, col_s = st.columns(2)

            with col_b:
                st.subheader(f"🟢 Bullish ({len(bullish)})")
                if not bullish.empty:
                    st.dataframe(
                        bullish[
                            [
                                "Instrument",
                                "MA",
                                "Today",
                                "Yesterday",
                                "1D Return",
                                "Price (90d)",
                                "Price (1y)",
                            ]
                        ].style.format(
                            subset=["Today", "Yesterday", "1D Return"],
                            formatter="{:+.2f}%",
                        ),
                        column_config={
                            **_sparkline_config(),
                            "Instrument": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                        },
                        hide_index=True,
                        height=min(350, 50 + len(bullish) * 35),
                    )
                else:
                    st.info("None today.")

            with col_s:
                st.subheader(f"🔴 Bearish ({len(bearish)})")
                if not bearish.empty:
                    st.dataframe(
                        bearish[
                            [
                                "Instrument",
                                "MA",
                                "Today",
                                "Yesterday",
                                "1D Return",
                                "Price (90d)",
                                "Price (1y)",
                            ]
                        ].style.format(
                            subset=["Today", "Yesterday", "1D Return"],
                            formatter="{:+.2f}%",
                        ),
                        column_config={
                            **_sparkline_config(),
                            "Instrument": st.column_config.TextColumn(
                                "Instrument", width="medium"
                            ),
                        },
                        hide_index=True,
                        height=min(350, 50 + len(bearish) * 35),
                    )
                else:
                    st.info("None today.")
        else:
            st.info("No MA crossover events today.")


    ma_crossover_section()
