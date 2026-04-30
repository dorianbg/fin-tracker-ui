import datetime
import platform
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import duckdb_importer as di
from data import (
    get_data,
    create_query,
    add_sparkline_column,
)
from utils import (
    custom_sort_df_cols,
    style_performance_table,
    plot_performance,
    filter_dataframe,
)
from config import FUND_TYPE_OPTIONS, table_height
from utils import correlation_matrix

import views.DailySummary as DailySummary
import views.BreakoutScanner as BreakoutScanner
import views.PullbackScanner as PullbackScanner
import views.PukeDetector as PukeDetector
import views.LaggardBreakout as LaggardBreakout
import views.TodaysCrossings as TodaysCrossings
import views.RelativeStrength as RelativeStrength
import views.DrawdownAnalysis as DrawdownAnalysis
import views.FactorDashboard as FactorDashboard
import views.ThematicDashboard as ThematicDashboard
import views.CrossAssetRegime as CrossAssetRegime
import views.AssetCorrelation as AssetCorrelation
import views.PerformanceChart as PerformanceChart
import views.RotationStrategies as RotationStrategies
import views.PortfolioAllocator as PortfolioAllocator

st.set_page_config(
    page_icon="🏠",
    page_title="Financial instrument tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 4rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


@st.cache_resource
def run_import_one_off():
    if platform.system() == "Darwin":
        with st.spinner("Processing"):
            di.run()
            time.sleep(1)


run_import_one_off()


def render_action_screens(df: pd.DataFrame):
    required = {"description", "ticker", "vol_1mo", "vol_1y", "r_1w", "r_1mo"}
    if df.empty or not required.issubset(df.columns):
        return

    signal_df = df.copy()
    signal_df["vol_ratio"] = signal_df["vol_1mo"] / signal_df["vol_1y"].replace(0, np.nan)
    drawdown_col = "drawdown_52w" if "drawdown_52w" in signal_df.columns else "ma_252"

    st.markdown("---")
    st.header("Action Screens")
    st.caption(
        "Fast screens for unusual volatility: stress/capitulation on the downside and ETF breakouts on the upside."
    )

    tab_vol, tab_capitulation, tab_breakout = st.tabs(
        ["📊 Volatility Spikes", "💀 Capitulation", "🚀 High-Vol Breakouts"]
    )

    with tab_vol:
        st.markdown(
            "Highest **1-month / 1-year volatility ratio**. Values above 1.0 mean short-term volatility is running hotter than normal."
        )
        vol_view = (
            signal_df[["description", "ticker", "vol_1mo", "vol_1y", "vol_ratio"]]
            .dropna(subset=["vol_ratio"])
            .sort_values("vol_ratio", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        fig_vol = px.bar(
            vol_view,
            x="vol_ratio",
            y="description",
            orientation="h",
            color="vol_ratio",
            color_continuous_scale="YlOrRd",
            labels={"vol_ratio": "Vol Ratio (1mo/1y)", "description": ""},
            title="Top 30 - Highest Vol Spike Ratio",
        )
        fig_vol.update_layout(
            yaxis=dict(autorange="reversed"),
            height=max(400, len(vol_view) * 25),
            showlegend=False,
        )
        fig_vol.add_vline(
            x=1.0,
            line_dash="dash",
            line_color="grey",
            opacity=0.5,
            annotation_text="Neutral",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab_capitulation:
        st.markdown(
            "Downside stress screen: high volatility ratio plus meaningful drawdown. These are the cleaner 'puke' candidates."
        )
        cap = signal_df[signal_df["vol_ratio"] >= 1.2].copy()
        if drawdown_col in cap.columns:
            cap = cap[cap[drawdown_col] <= -10].copy()
            cap["severity"] = (-cap[drawdown_col]) * cap["vol_ratio"]
        else:
            cap["severity"] = cap["vol_ratio"]
        cap["bounce_starting"] = cap["r_1w"] > 0
        cap = cap.sort_values("severity", ascending=False).head(20).reset_index(drop=True)

        if cap.empty:
            st.info("No capitulation candidates at the current default thresholds.")
        else:
            cap_cols = [
                "description",
                "ticker",
                "fund_type",
                "vol_1mo",
                "vol_1y",
                "vol_ratio",
                drawdown_col,
                "r_1w",
                "r_1mo",
                "severity",
                "bounce_starting",
            ]
            cap_cols = [c for c in cap_cols if c in cap.columns]
            st.dataframe(
                cap[cap_cols].style.format(
                    formatter="{:.2f}",
                    subset=[c for c in cap_cols if pd.api.types.is_numeric_dtype(cap[c])],
                ),
                hide_index=True,
                height=400,
            )

    with tab_breakout:
        st.markdown(
            "Upside stress screen: ETFs/funds exploding higher on unusually high volatility. Prioritizes high vol ratio, positive 1W/1M returns, and price above the 21-day average."
        )
        breakout = signal_df[
            (signal_df["vol_ratio"] >= 1.2)
            & (signal_df["r_1w"] > 0)
            & (signal_df["r_1mo"] > 0)
        ].copy()
        if "ma_21" in breakout.columns:
            breakout = breakout[breakout["ma_21"] > 0].copy()
        breakout["breakout_score"] = breakout["vol_ratio"] * breakout["r_1w"]
        breakout = breakout.sort_values("breakout_score", ascending=False).head(20)

        if breakout.empty:
            st.info("No high-volatility upside breakout candidates at the current default thresholds.")
        else:
            breakout_cols = [
                "description",
                "ticker",
                "fund_type",
                "vol_ratio",
                "r_1w",
                "r_1mo",
                "ma_21",
                "ma_63",
                drawdown_col,
                "breakout_score",
            ]
            breakout_cols = [c for c in breakout_cols if c in breakout.columns]
            st.dataframe(
                breakout[breakout_cols].style.format(
                    formatter="{:.2f}",
                    subset=[
                        c
                        for c in breakout_cols
                        if pd.api.types.is_numeric_dtype(breakout[c])
                    ],
                ),
                hide_index=True,
                height=400,
            )

(
    tab_perf,
    tab_daily,
    tab_breakout,
    tab_pullback,
    tab_puke,
    tab_laggard,
    tab_crossings,
    tab_rs,
    tab_drawdown,
    tab_factors,
    tab_thematic,
    tab_regime,
    tab_corr,
    tab_charts,
    tab_rotation,
    tab_allocator,
) = st.tabs([
    "Performance",
    "Daily Summary",
    "Breakout",
    "Pullback",
    "Puke Detector",
    "Laggard Breakout",
    "Crossings",
    "Relative Strength",
    "Drawdowns",
    "Factors",
    "Thematic",
    "Cross-Asset",
    "Correlation",
    "Charts",
    "Rotation",
    "Allocator",
])

with tab_perf:
    if platform.system() != "Darwin":
        top_col1, top_col2 = st.columns([4, 1])
        with top_col1:
            st.write(
                "Disclaimer: this is a non-commercial project and data is purely source from Yahoo! finance API and exclusively intended for personal use only.  \n"
                "Data quality issues with smaller UCITS ETFs are common in which case the data in tables will be missing or obviously wrong."
            )
        with top_col2:
            with st.popover("Things to note"):
                st.markdown(
                    "1) Performance includes dividends (Accumulating ETFs are preferred where possible) and is standardised to GBP (some ETFs are GBP hedged).   \n"
                    "2) UK cash rate is taken as risk free rate for Sharpe ratio   \n"
                    "3) You can change the selection of instruments on the project that feeds data to this dashboard, source code is: https://github.com/dorianbg/fin-tracker/"
                )

    col1, col2, col3, col4 = st.columns([2, 2, 4, 4])

    with col1:
        with st.container():
            vol_adjust = st.toggle(label="Show Sharpe ratio", value=True)
            show_returns = st.toggle(label="Show Gross return", value=True)

    with col2:
        with st.container():
            sort_sharpe = st.toggle(label="Custom sort on Sharpe", value=False)
            sort_returns = st.toggle(label="Custom sort on Returns", value=False)
            if sort_sharpe and sort_returns:
                st.warning("Cannot enable custom sorting on both")
    with col3:
        returns_cols = st.multiselect(
            label="Returns",
            options=di.selectable_returns,
            default=di.default_selected_returns,
        )
    with col4:
        instrument_categories = st.multiselect(
            "Instrument Category",
            options=FUND_TYPE_OPTIONS,
            default=["eq", "commod"],
        )

    custom_weights = []
    if sort_sharpe or sort_returns:
        weight_cols = st.columns(len(di.selectable_returns))
        for i, weight_col in enumerate(weight_cols):
            with weight_col:
                custom_weights.append(
                    st.number_input(
                        f"Weight for {di.selectable_returns[i]}",
                        value=0,
                    )
                )
        if len(custom_weights) and sum(custom_weights) > 0 and sum(custom_weights) != 100:
            st.warning(
                f"Custom weights must add up to 100% - current is {sum(custom_weights)}%"
            )

    with st.container():
        df: pd.DataFrame = get_data(
            query=create_query(
                table=di.perf_tbl,
                vol_adjust=vol_adjust,
                show_returns=show_returns,
                returns_cols=returns_cols,
                fund_types=instrument_categories,
            ),
        )
        df = filter_dataframe(df, modify=True)
        if (
            (sort_sharpe or sort_returns)
            and sum(custom_weights) == 100
            and filter(lambda x: x >= 1, custom_weights)
        ):
            total_w = sum(custom_weights)
            custom_weights_normalised = [x / total_w for x in custom_weights]
            columns_sort = di.perf_sharpe_cols if sort_sharpe else di.perf_returns_cols
            df = custom_sort_df_cols(columns_sort, custom_weights_normalised, df)
        add_sparkline_column(df)
        cols = list(df.columns)
        if "Price (90d)" in cols:
            cols.remove("Price (90d)")
            cols.insert(cols.index("description") + 1, "Price (90d)")
            df = df[cols]
        styled_df = style_performance_table(
            df,
            vol_adjust=vol_adjust,
            show_returns=show_returns,
            returns_cols=returns_cols,
        )
        narrow_cols = {
            "description": st.column_config.TextColumn("description", width="medium"),
            "Price (90d)": st.column_config.LineChartColumn("Price (90d)", width="small"),
            **{
                c: st.column_config.NumberColumn(label=c, width="small")
                for c in ("drawdown_52w", "drawdown_3y", "range_pos_52w", "range_pos_104w", "range_pos_156w")
                if c in df.columns
            },
        }
        event = st.dataframe(
            data=styled_df,
            hide_index=True,
            height=table_height,
            on_select="rerun",
            selection_mode="multi-row",
            column_config=narrow_cols,
        )
        if event and event.selection and event.selection.rows:
            filtered_df = df.iloc[event.selection.rows]
            selected_dates = st.date_input(
                "Select date range for Price Performance",
                value=[
                    datetime.date.today() - datetime.timedelta(days=5 * 365),
                    datetime.date.today(),
                ],
                min_value=datetime.date.today() - datetime.timedelta(days=5 * 365),
                max_value=datetime.date.today() + datetime.timedelta(days=1),
                key="date_range_perf",
            )
            if (
                selected_dates
                and len(selected_dates) > 1
                and selected_dates[0]
                and selected_dates[1]
            ):
                plot_performance(
                    start_date=selected_dates[0],
                    end_date=selected_dates[1],
                    selected_inst=list(filtered_df["ticker"].unique()),
                    selected_fund_types=instrument_categories,
                    show_df=True,
                )
            correlation_matrix(assets=list(filtered_df["ticker"].unique()))

        render_action_screens(df)

with tab_daily:
    DailySummary.render()

with tab_breakout:
    BreakoutScanner.render()

with tab_pullback:
    PullbackScanner.render()

with tab_puke:
    PukeDetector.render()

with tab_laggard:
    LaggardBreakout.render()

with tab_crossings:
    TodaysCrossings.render()

with tab_rs:
    RelativeStrength.render()

with tab_drawdown:
    DrawdownAnalysis.render()

with tab_factors:
    FactorDashboard.render()

with tab_thematic:
    ThematicDashboard.render()

with tab_regime:
    CrossAssetRegime.render()

with tab_corr:
    AssetCorrelation.render()

with tab_charts:
    PerformanceChart.render()

with tab_rotation:
    RotationStrategies.render()

with tab_allocator:
    PortfolioAllocator.render()
