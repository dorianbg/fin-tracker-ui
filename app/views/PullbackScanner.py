"""
Pullback Scanner — find instruments experiencing pullbacks or recoveries.

Modes:
  1) Classic Pullback: instruments in uptrends pulling back to shorter MAs
  2) Underperformers Recovering: lagged instruments now showing short-term strength
  3) Recovery Score: composite ranking of recovery candidates
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import config
import duckdb_importer as di
from data import add_sparkline_column, get_conn, fund_type_sidebar, filter_by_fund_type

try:
    from strategy_scanners import scan_pullbacks
except ModuleNotFoundError:
    from app.strategy_scanners import scan_pullbacks


def _sparkline_config():
    return {
        "Price (90d)": st.column_config.LineChartColumn("Price (90d)", width="small"),
        "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
    }


def _add_sparkline_columns(df: pd.DataFrame) -> pd.DataFrame:
    add_sparkline_column(df)
    add_sparkline_column(df, col_name="Price (1y)", days=365)
    return df


def render():
    st.title("🎯 Pullback Scanner")
    st.markdown(
        "Find **pullback opportunities** in uptrending instruments and "
        "**recovery candidates** among recent underperformers."
    )

    RETURN_COL_MAP = {
        "1w": "r_1w",
        "2w": "r_2w",
        "1mo": "r_1mo",
        "3mo": "r_3mo",
        "6mo": "r_6mo",
        "1y": "r_1y",
        "2y": "r_2y",
        "3y": "r_3y",
    }

    # Streamlit sidebars are global, so keep tab-specific controls in-page.
    settings_col, content_col = st.columns([1, 4], gap="large")

    with settings_col:
        st.subheader("Scanner Settings")

        # Pullback MA selector
        pullback_ma = st.selectbox(
            "Pullback detection MA",
            options=["21-day (short)", "63-day (medium)", "126-day (intermediate)"],
            index=0,
            help="Which moving average to use for detecting pullbacks",
        )
        pullback_ma_col = {
            "21-day (short)": "ma_21",
            "63-day (medium)": "ma_63",
            "126-day (intermediate)": "ma_126",
        }[pullback_ma]
        pullback_ma_label = pullback_ma.split(" ")[0]

        pullback_depth = st.slider(
            f"Max {pullback_ma_label} MA deviation (%)",
            min_value=-20,
            max_value=5,
            value=0,
            step=1,
            help=f"How far below the {pullback_ma_label} MA the price must be (0 = any pullback below MA)",
        )

        pullback_mode = st.radio(
            "Pullback mode",
            options=["Best only", "All matches"],
            index=0,
            help="Best only focuses on clean trend pullbacks with controlled drawdown and early bounce signs.",
            key="pullback_mode",
        )

        top_pullbacks_n = st.slider(
            "Top pullbacks to show",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="pullback_top_n",
        )

        min_uptrend_strength = st.slider(
            "Min 252-day MA position (%)",
            min_value=-5,
            max_value=30,
            value=0,
            step=1,
            help="How far above the 252-day MA - 0 means just above it",
        )

        require_intermediate_ok = st.checkbox(
            "Require above 126-day MA",
            value=True,
            help="Extra filter: intermediate trend (126-day) must still be intact",
        )

        max_quality_drawdown = st.slider(
            "Best-only max 52W drawdown (%)",
            min_value=-50,
            max_value=-5,
            value=-20,
            step=5,
            help="Best-only filter: avoid candidates that are too far below their 52-week high.",
            key="pullback_max_quality_drawdown",
        )

        require_bounce = st.checkbox(
            "Best-only require bounce",
            value=True,
            help="Require positive 1D or 1W return so the pullback is starting to stabilize.",
            key="pullback_require_bounce",
        )

        fund_type_filter = fund_type_sidebar(
            default=["eq", "stock"], key="pullback_fund_types"
        )

        st.markdown("---")
        st.subheader("Recovery Settings")

        benchmark_label = st.selectbox(
            "Benchmark",
            options=list(config.BENCHMARKS.keys()),
            index=0,
        )
        benchmark_ticker = config.BENCHMARKS[benchmark_label]

        long_lookback = st.selectbox(
            "Underperformance period",
            options=["3mo", "6mo", "1y", "2y"],
            index=1,
        )

        short_lookback = st.selectbox(
            "Recovery period",
            options=["1w", "2w", "1mo", "3mo"],
            index=2,
        )

        underperf_threshold = st.slider(
            "Underperformance threshold (%)",
            min_value=0,
            max_value=40,
            value=5,
            step=5,
            help="How much worse than benchmark to qualify as underperformer",
        )

    with content_col:

        @st.cache_data(ttl=300)
        def load_all_latest() -> pd.DataFrame:
            cols = (
                di.perf_desc_cols_start
                + di.perf_vol_cols
                + di.perf_mavg_cols
                + di.perf_returns_cols
                + di.perf_desc_cols_end
                + di.perf_rownames_cols
            )
            query = f"""
                SELECT {",".join(cols)}
                FROM {di.perf_tbl}
                WHERE rown = 1
                ORDER BY description ASC
            """
            return get_conn().execute(query).df()

        _all_data = load_all_latest()

        if _all_data.empty:
            st.warning("No data loaded.")
            st.stop()

        # ── filter by fund type ──
        df = filter_by_fund_type(_all_data, fund_type_filter)

        # ═══════════════════════════════════════════
        # Section 1: Pullback Candidates
        # ═══════════════════════════════════════════
        st.header("📋 Pullback Candidates")
        st.markdown(
            f"Instruments **above 252-day MA** by ≥{min_uptrend_strength}% and "
            f"**pulling back** below {pullback_ma_label} MA by ≤{pullback_depth}%."
        )
        st.caption(
            "Best-only mode ranks clean pullbacks by trend strength, controlled drawdown, and early bounce signs. "
            "It requires the 126-day trend to be intact, filters out deep 52-week drawdowns, and shows only the top-ranked candidates."
        )

        pullbacks = scan_pullbacks(
            df,
            pullback_ma_col=pullback_ma_col,
            pullback_depth=pullback_depth,
            min_uptrend_strength=min_uptrend_strength,
            require_intermediate_ok=require_intermediate_ok,
            best_only=pullback_mode == "Best only",
            max_quality_drawdown=max_quality_drawdown,
            require_bounce=require_bounce,
            limit=top_pullbacks_n,
        )

        if pullbacks.empty:
            st.info(
                "No instruments match current filters. Try relaxing the thresholds in the sidebar."
            )
        else:
            st.markdown(f"**{len(pullbacks)} candidates found**")

            display_cols = [
                "description",
                "Price (90d)",
                "Price (1y)",
                "ticker",
                "fund_type",
                pullback_ma_col,
                "ma_63",
                "ma_126",
                "ma_252",
                "drawdown_52w",  # Added as requested
                "r_1w",
                "r_1mo",
                "r_3mo",
                "quality_score",
                "pullback_score",
                "bounce_signal",
            ]
            _add_sparkline_columns(pullbacks)
            # deduplicate in case pullback_ma_col is already one of the fixed cols
            display_cols = list(dict.fromkeys(display_cols))

            st.dataframe(
                pullbacks[display_cols]
                .style.format(
                    subset=[
                        c
                        for c in [
                            pullback_ma_col,
                            "ma_63",
                            "ma_126",
                            "ma_252",
                            "drawdown_52w",
                            "r_1w",
                            "r_1mo",
                            "r_3mo",
                        ]
                        if c in display_cols
                    ],
                    formatter="{:+.2f}%",
                )
                .format(subset=["quality_score", "pullback_score"], formatter="{:.1f}"),
                hide_index=True,
                height=450,
                column_config=_sparkline_config(),
            )

            # Scatter: uptrend strength vs pullback depth
            st.subheader("Uptrend Strength vs Pullback Depth")

            fig = px.scatter(
                pullbacks,
                x="ma_252",
                y=pullback_ma_col,
                color="bounce_signal",
                color_discrete_map={True: "#2ecc71", False: "#e67e22"},
                text="ticker",
                size="pullback_score",
                size_max=20,
                labels={
                    "ma_252": "% above 252-day MA (uptrend strength)",
                    pullback_ma_col: f"% from {pullback_ma_label} MA (pullback depth)",
                    "bounce_signal": "1W bounce?",
                    "drawdown_52w": "Drawdown from 52W High (%)",
                },
                hover_data=["drawdown_52w"],
                title="Sweet Spot: Strong Uptrend + Deep Pullback (top-left = best)",
            )
            fig.update_traces(textposition="top center")
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.3)
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

            # Top ranked by pullback quality score
            st.subheader("🏆 Top Pullback Candidates")
            st.markdown(
                "Ranked by **quality score** = trend strength + controlled pullback depth + bounce signal "
                "minus drawdown/breakdown penalties."
            )

            top = pullbacks.sort_values("quality_score", ascending=False).head(15)
            fig_bar = px.bar(
                top,
                x="quality_score",
                y="description",
                orientation="h",
                color="bounce_signal",
                color_discrete_map={True: "#2ecc71", False: "#e67e22"},
                labels={
                    "quality_score": "Quality Score",
                    "description": "",
                    "bounce_signal": "1W bounce?",
                },
                title="Top 15 — Quality Pullback Score",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"), height=450, showlegend=True
            )
            st.plotly_chart(fig_bar, width="stretch")

        # ═══════════════════════════════════════════
        # Section 2: Trend Reversal Candidates
        # ═══════════════════════════════════════════
        st.header("🔄 Bullish Trend Reversal Candidates")
        st.markdown("""
        Instruments currently **below** their 252-day moving average (Long-term downtrend) 
        but trading **above** their 21-day and 63-day moving averages (Short-term recovery/Golden Cross setup).
        """)

        # Logic: Under 252 MA (< 0) but Above 21 MA (> 0) and Above 63 MA (> 0)
        reversal_mask = (df["ma_252"] < 0) & (df["ma_21"] > 0) & (df["ma_63"] > 0)

        df_reversal = df[reversal_mask].copy()

        if not df_reversal.empty:
            # Sort by strength of short term recovery (ma_21)
            df_reversal = df_reversal.sort_values("ma_21", ascending=False)

            cols = [
                "ticker",
                "description",
                "Price (90d)",
                "Price (1y)",
                "price",
                "r_1w",
                "ma_21",
                "ma_63",
                "ma_252",
                "vol_1y",
            ]
            _add_sparkline_columns(df_reversal)
            # ensure cols exist
            cols = [c for c in cols if c in df_reversal.columns]

            st.dataframe(
                df_reversal[cols].style.format(
                    {
                        "r_1w": "{:+.2f}%",
                        "ma_21": "{:+.2f}%",
                        "ma_63": "{:+.2f}%",
                        "ma_252": "{:+.2f}%",
                        "vol_1y": "{:.2f}%",
                    },
                    na_rep="-",
                    precision=2,
                ),
                width="stretch",
                height=300,
                column_config=_sparkline_config(),
            )
        else:
            st.info(
                "No instruments found matching bullish reversal criteria (Under 252MA, Above 21MA & 63MA)."
            )

        st.markdown("---")

        st.header("🎯 Short Target Monitor")
        st.markdown("""
        Bearish watchlist for instruments where the **long-term trend has not fully broken yet**
        but short/intermediate momentum is rolling over. These are not automatic shorts; they are
        candidates to monitor for failed bounces, stop-loss review, hedging, or short setups.
        """)
        st.caption(
            "Setup: price remains above the 252-day average, but is below both the 21-day and 63-day averages. "
            "Priority score rewards weaker short-term structure, negative weekly momentum, and higher volatility."
        )

        downside_mask = (df["ma_252"] > 0) & (df["ma_21"] < 0) & (df["ma_63"] < 0)
        df_downside = df[downside_mask].copy()

        if not df_downside.empty:
            df_downside["trend_buffer"] = df_downside["ma_252"]
            df_downside["short_breakdown"] = (-df_downside["ma_21"]).clip(lower=0) + (
                -df_downside["ma_63"]
            ).clip(lower=0)
            df_downside["momentum_penalty"] = (-df_downside["r_1w"]).clip(lower=0)
            df_downside["short_priority_score"] = (
                df_downside["short_breakdown"]
                + df_downside["momentum_penalty"]
                + df_downside["vol_1y"].fillna(0) / 10
                - df_downside["trend_buffer"].clip(lower=0) / 4
            )
            df_downside["monitor_reason"] = "Short MAs broken"
            df_downside.loc[df_downside["r_1w"] < 0, "monitor_reason"] = (
                "Short MAs broken + negative 1W"
            )
            df_downside = df_downside.sort_values(
                "short_priority_score", ascending=False
            )

            cols = [
                "ticker",
                "description",
                "Price (90d)",
                "Price (1y)",
                "price",
                "r_1w",
                "r_1mo",
                "ma_21",
                "ma_63",
                "ma_126",
                "ma_252",
                "drawdown_52w",
                "vol_1y",
                "short_priority_score",
                "monitor_reason",
            ]
            _add_sparkline_columns(df_downside)
            cols = [c for c in cols if c in df_downside.columns]

            st.dataframe(
                df_downside[cols].style.format(
                    {
                        "r_1w": "{:+.2f}%",
                        "r_1mo": "{:+.2f}%",
                        "ma_21": "{:+.2f}%",
                        "ma_63": "{:+.2f}%",
                        "ma_126": "{:+.2f}%",
                        "ma_252": "{:+.2f}%",
                        "drawdown_52w": "{:+.2f}%",
                        "vol_1y": "{:.2f}%",
                        "short_priority_score": "{:.1f}",
                    },
                    na_rep="-",
                    precision=2,
                ),
                width="stretch",
                height=300,
                column_config=_sparkline_config(),
            )
        else:
            st.info(
                "No short target monitor candidates found (Above 252MA, Below 21MA & 63MA)."
            )

        st.markdown("---")

        # ═══════════════════════════════════════════
        # Section 3: Underperformers Now Recovering
        # ═══════════════════════════════════════════
        st.header("📉 Actionable Recovery Watchlist")

        long_col = RETURN_COL_MAP[long_lookback]
        short_col = RETURN_COL_MAP[short_lookback]

        benchmark_row = _all_data[
            _all_data["ticker"] == benchmark_ticker
        ].drop_duplicates("ticker")
        if benchmark_row.empty:
            st.error(f"Benchmark ticker '{benchmark_ticker}' not found.")
            st.stop()

        benchmark_r_long = float(benchmark_row[long_col].iloc[0])
        benchmark_r_short = float(benchmark_row[short_col].iloc[0])
        benchmark_desc = benchmark_row["description"].iloc[0]

        st.caption(
            f"Benchmark: **{benchmark_desc}** ({benchmark_ticker}) — "
            f"{long_lookback} return: {benchmark_r_long:+.2f}% · "
            f"{short_lookback} return: {benchmark_r_short:+.2f}%"
        )

        st.markdown(
            f"Instruments that **underperformed** the benchmark by ≥{underperf_threshold}% "
            f"over **{long_lookback}** but are showing relative strength over **{short_lookback}**."
        )
        st.caption(
            "Short excess = instrument return over the recovery period minus benchmark return over the same period. "
            "Long excess uses the same benchmark-relative calculation over the underperformance period."
        )

        recovery_cols = list(
            dict.fromkeys(
                [
                    "description",
                    "ticker",
                    "fund_type",
                    long_col,
                    short_col,
                    "r_1d",
                    "r_1w",
                    "r_1mo",
                    "ma_21",
                    "ma_63",
                    "ma_126",
                    "ma_252",
                    "drawdown_52w",
                ]
            )
        )
        recovery = df[df["ticker"] != benchmark_ticker][recovery_cols].copy()
        recovery["excess_long"] = recovery[long_col] - benchmark_r_long
        recovery["excess_short"] = recovery[short_col] - benchmark_r_short
        underperformers = (
            recovery[recovery["excess_long"] <= -underperf_threshold]
            .sort_values("excess_short", ascending=False)
            .reset_index(drop=True)
        )

        if underperformers.empty:
            st.info(
                f"No instruments underperformed the benchmark by ≥{underperf_threshold}% over {long_lookback}. "
                f"Try lowering the threshold."
            )
        else:
            underperformers["recovering"] = underperformers["excess_short"] > 0
            underperformers["action"] = "Avoid for now"
            underperformers.loc[
                (underperformers["excess_short"] <= 0)
                & (underperformers["excess_short"] >= -underperf_threshold),
                "action",
            ] = "Early stabilization"
            underperformers.loc[underperformers["excess_short"] > 0, "action"] = (
                "Buy/watch"
            )
            underperformers["action_rank"] = underperformers["action"].map(
                {"Buy/watch": 0, "Early stabilization": 1, "Avoid for now": 2}
            )

            action_watchlist = underperformers.sort_values(
                ["action_rank", "excess_short"], ascending=[True, False]
            ).reset_index(drop=True)
            display_cols_r = [
                "action",
                "description",
                "Price (90d)",
                "Price (1y)",
                "ticker",
                long_col,
                short_col,
                "excess_long",
                "excess_short",
                "recovering",
            ]
            _add_sparkline_columns(action_watchlist)
            st.dataframe(
                action_watchlist[display_cols_r]
                .rename(
                    columns={
                        "action": "Action",
                        "excess_long": f"Long excess vs benchmark ({long_lookback})",
                        "excess_short": f"Short excess vs benchmark ({short_lookback})",
                        "recovering": "Outperforming short-term?",
                    }
                )
                .style.format(
                    subset=[
                        long_col,
                        short_col,
                        f"Long excess vs benchmark ({long_lookback})",
                        f"Short excess vs benchmark ({short_lookback})",
                    ],
                    formatter="{:+.2f}%",
                ),
                hide_index=True,
                height=450,
                column_config=_sparkline_config(),
            )

            st.markdown("---")
            st.header("🚀 Laggard Breakout Candidates")
            st.markdown(
                "Long-term underperformers that are now showing short-term relative strength "
                "and reclaiming key moving averages. This replaces the separate Laggard Breakout tab."
            )
            laggard_breakouts = underperformers[
                (underperformers["excess_short"] > 0)
                & (underperformers["ma_21"] > 0)
                & (underperformers["ma_63"] > 0)
            ].copy()

            if laggard_breakouts.empty:
                st.info(
                    "No confirmed laggard breakouts right now. Showing closest laggards by short-term relative strength."
                )
                laggard_breakouts = (
                    underperformers.sort_values("excess_short", ascending=False)
                    .head(15)
                    .copy()
                )

            laggard_breakouts["breakout_score"] = (
                -laggard_breakouts["excess_long"]
            ) * laggard_breakouts["excess_short"].clip(lower=0)
            laggard_breakouts = laggard_breakouts.sort_values(
                ["breakout_score", "excess_short"], ascending=False
            ).head(20)

            laggard_cols = [
                "description",
                "Price (90d)",
                "Price (1y)",
                "ticker",
                "fund_type",
                long_col,
                short_col,
                "excess_long",
                "excess_short",
                "ma_21",
                "ma_63",
                "ma_252",
                "drawdown_52w",
                "breakout_score",
            ]
            _add_sparkline_columns(laggard_breakouts)
            st.dataframe(
                laggard_breakouts[laggard_cols]
                .rename(
                    columns={
                        "excess_long": f"Long excess vs benchmark ({long_lookback})",
                        "excess_short": f"Short excess vs benchmark ({short_lookback})",
                    }
                )
                .style.format(
                    subset=[
                        long_col,
                        short_col,
                        f"Long excess vs benchmark ({long_lookback})",
                        f"Short excess vs benchmark ({short_lookback})",
                        "ma_21",
                        "ma_63",
                        "ma_252",
                        "drawdown_52w",
                    ],
                    formatter="{:+.2f}%",
                )
                .format(subset=["breakout_score"], formatter="{:.1f}"),
                hide_index=True,
                height=400,
                column_config=_sparkline_config(),
            )

        # ═══════════════════════════════════════════
        # Section 3: Recovery Score Ranking
        # ═══════════════════════════════════════════
        st.header("🏆 Recovery Score Ranking")
        st.markdown(
            "**Deeper past underperformance × stronger recent recovery** = higher score. "
            "Top candidates that are turning the corner."
        )

        if not underperformers.empty:
            # Show both currently recovering AND all underperformers with any positive short-term excess
            ranked = underperformers[underperformers["excess_short"] > 0].copy()

            if ranked.empty:
                # Fallback: show top underperformers by least-negative short-term excess (closest to recovering)
                st.info(
                    "No instruments are currently outperforming the benchmark short-term. "
                    "Showing the **closest to recovery** instead."
                )
                ranked = (
                    underperformers.sort_values("excess_short", ascending=False)
                    .head(15)
                    .copy()
                )
                ranked["recovery_score"] = (-ranked["excess_long"]) * (
                    1 + ranked["excess_short"]
                )
                ranked = ranked.sort_values(
                    "recovery_score", ascending=False
                ).reset_index(drop=True)
                ranked.index = ranked.index + 1
                ranked.index.name = "Rank"

                display_ranked = ranked[
                    [
                        "description",
                        "ticker",
                        long_col,
                        short_col,
                        "excess_long",
                        "excess_short",
                        "recovery_score",
                    ]
                ].rename(
                    columns={
                        "excess_long": f"Long excess vs benchmark ({long_lookback})",
                        "excess_short": f"Short excess vs benchmark ({short_lookback})",
                    }
                )
                add_sparkline_column(display_ranked)
                add_sparkline_column(display_ranked, col_name="Price (1y)", days=365)
                display_ranked = display_ranked[
                    [
                        "description",
                        "Price (90d)",
                        "Price (1y)",
                        "ticker",
                        long_col,
                        short_col,
                        f"Long excess vs benchmark ({long_lookback})",
                        f"Short excess vs benchmark ({short_lookback})",
                        "recovery_score",
                    ]
                ]
                st.dataframe(
                    display_ranked.style.format(
                        subset=[
                            long_col,
                            short_col,
                            f"Long excess vs benchmark ({long_lookback})",
                            f"Short excess vs benchmark ({short_lookback})",
                        ],
                        formatter="{:+.2f}%",
                    ).format(subset=["recovery_score"], formatter="{:.1f}"),
                    height=400,
                    column_config=_sparkline_config(),
                )
            else:
                ranked["recovery_score"] = (-ranked["excess_long"]) * ranked[
                    "excess_short"
                ]
                ranked = ranked.sort_values(
                    "recovery_score", ascending=False
                ).reset_index(drop=True)
                ranked.index = ranked.index + 1
                ranked.index.name = "Rank"

                display_ranked = ranked[
                    [
                        "description",
                        "ticker",
                        long_col,
                        short_col,
                        "excess_long",
                        "excess_short",
                        "recovery_score",
                    ]
                ].rename(
                    columns={
                        "excess_long": f"Long excess vs benchmark ({long_lookback})",
                        "excess_short": f"Short excess vs benchmark ({short_lookback})",
                    }
                )
                add_sparkline_column(display_ranked)
                add_sparkline_column(display_ranked, col_name="Price (1y)", days=365)
                display_ranked = display_ranked[
                    [
                        "description",
                        "Price (90d)",
                        "Price (1y)",
                        "ticker",
                        long_col,
                        short_col,
                        f"Long excess vs benchmark ({long_lookback})",
                        f"Short excess vs benchmark ({short_lookback})",
                        "recovery_score",
                    ]
                ]
                st.dataframe(
                    display_ranked.style.format(
                        subset=[
                            long_col,
                            short_col,
                            f"Long excess vs benchmark ({long_lookback})",
                            f"Short excess vs benchmark ({short_lookback})",
                        ],
                        formatter="{:+.2f}%",
                    ).format(subset=["recovery_score"], formatter="{:.1f}"),
                    height=400,
                    column_config=_sparkline_config(),
                )

                top_n = ranked.head(15)
                fig_rank = px.bar(
                    top_n,
                    x="recovery_score",
                    y="description",
                    orientation="h",
                    color="recovery_score",
                    color_continuous_scale="YlGn",
                    labels={"recovery_score": "Recovery Score", "description": ""},
                    title="Top Recovery Candidates",
                )
                fig_rank.update_layout(
                    yaxis=dict(autorange="reversed"), height=450, showlegend=False
                )
                st.plotly_chart(fig_rank, width="stretch")
        else:
            st.info("No underperforming instruments to rank.")
