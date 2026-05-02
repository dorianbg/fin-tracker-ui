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
from config import DEFAULT_BENCHMARK, FUND_TYPE_OPTIONS, table_height
from utils import correlation_matrix

import views.DailySummary as DailySummary
import views.PullbackScanner as PullbackScanner
import views.TodaysCrossings as TodaysCrossings
import views.RelativeStrength as RelativeStrength
import views.FactorDashboard as FactorDashboard
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


def build_action_list(df: pd.DataFrame) -> pd.DataFrame:
    required = {"description", "ticker", "r_1d", "r_1w", "r_1mo", "vol_1mo", "vol_1y"}
    trend_cols = {"ma_21", "ma_63", "ma_126", "ma_252", "drawdown_52w"}
    if df.empty or not required.issubset(df.columns) or not trend_cols.issubset(df.columns):
        return pd.DataFrame()

    signals = df.copy()
    signals["vol_ratio"] = signals["vol_1mo"] / signals["vol_1y"].replace(0, np.nan)
    actions = []

    def append_actions(pool: pd.DataFrame, action: str, why: str, score_col: str):
        if pool.empty:
            return
        view = pool.copy()
        view["Action"] = action
        view["Why"] = why
        view["Score"] = view[score_col]
        actions.append(view)

    buy_watch = signals[
        (signals["ma_252"] > 0)
        & (signals["ma_126"] > 0)
        & (signals["ma_21"] < 0)
        & (signals["drawdown_52w"] >= -20)
        & ((signals["r_1d"] > 0) | (signals["r_1w"] > 0))
    ].copy()
    if not buy_watch.empty:
        buy_watch["quality_score"] = (
            buy_watch["ma_252"].clip(lower=0)
            + buy_watch["ma_126"].clip(lower=0)
            + (-buy_watch["ma_21"]).clip(lower=0, upper=15)
            + np.where((buy_watch["r_1d"] > 0) | (buy_watch["r_1w"] > 0), 5, 0)
            - (-buy_watch["drawdown_52w"] - 20).clip(lower=0)
            - (-buy_watch["ma_63"]).clip(lower=0)
        )
    append_actions(
        buy_watch,
        "Buy Watch",
        "Strong trend pullback with early bounce",
        "quality_score",
    )

    breakout_watch = signals[
        (signals["r_1w"] > 0)
        & (signals["r_1mo"] > 0)
        & (signals["vol_ratio"] >= 1.1)
    ].copy()
    if not breakout_watch.empty and "ma_21" in breakout_watch.columns:
        confirmed = breakout_watch[breakout_watch["ma_21"] > 0].copy()
        if not confirmed.empty:
            breakout_watch = confirmed
    if not breakout_watch.empty:
        breakout_watch["breakout_score"] = breakout_watch["vol_ratio"] * breakout_watch["r_1w"]
    append_actions(
        breakout_watch,
        "Breakout Watch",
        "Upside move with elevated volatility",
        "breakout_score",
    )

    capitulation_watch = signals[
        (signals["drawdown_52w"] < 0)
        & (signals["vol_ratio"] >= 1.2)
        & (signals["drawdown_52w"] <= -10)
    ].copy()
    if not capitulation_watch.empty:
        capitulation_watch["capitulation_score"] = (
            -capitulation_watch["drawdown_52w"]
        ) * capitulation_watch["vol_ratio"]
    append_actions(
        capitulation_watch,
        "Capitulation Watch",
        "High stress / drawdown candidate",
        "capitulation_score",
    )

    benchmark = signals[signals["ticker"] == DEFAULT_BENCHMARK]
    if not benchmark.empty and {"r_1y", "r_1w"}.issubset(signals.columns):
        benchmark = benchmark.iloc[0]
        trim_watch = signals[
            ((signals["r_1y"] - benchmark["r_1y"]) >= 10)
            & ((signals["r_1w"] - benchmark["r_1w"]) < 0)
        ].copy()
        if not trim_watch.empty:
            trim_watch["trim_score"] = -(
                trim_watch["r_1w"] - benchmark["r_1w"]
            ) * (trim_watch["r_1y"] - benchmark["r_1y"])
        append_actions(
            trim_watch,
            "Trim Watch",
            "Long-term leader losing short-term relative strength",
            "trim_score",
        )

    short_monitor = signals[
        (signals["ma_252"] > 0)
        & (signals["ma_21"] < 0)
        & (signals["ma_63"] < 0)
        & (signals["r_1w"] < 0)
    ].copy()
    if not short_monitor.empty:
        short_monitor["short_priority_score"] = (
            (-short_monitor["ma_21"]).clip(lower=0)
            + (-short_monitor["ma_63"]).clip(lower=0)
            + (-short_monitor["r_1w"]).clip(lower=0)
            + short_monitor["vol_ratio"].fillna(0)
            - short_monitor["ma_252"].clip(lower=0) / 10
        )
    append_actions(
        short_monitor,
        "Short Monitor",
        "Long-term trend intact but short/intermediate trend rolling over",
        "short_priority_score",
    )

    if not actions:
        return pd.DataFrame()

    action_df = pd.concat(actions, ignore_index=True)
    action_priority = {
        "Buy Watch": 1,
        "Breakout Watch": 2,
        "Trim Watch": 3,
        "Short Monitor": 4,
        "Capitulation Watch": 5,
    }
    action_df["action_priority"] = action_df["Action"].map(action_priority)
    action_df = action_df.sort_values(
        ["action_priority", "Score"], ascending=[True, False]
    )

    output_cols = {
        "description": "Instrument",
        "ticker": "Ticker",
        "r_1d": "1D",
        "r_1w": "1W",
        "r_1mo": "1M",
        "ma_21": "MA21",
        "ma_63": "MA63",
        "ma_126": "MA126",
        "ma_252": "MA252",
        "drawdown_52w": "Drawdown",
        "vol_ratio": "Vol Ratio",
    }
    action_df = action_df[["Action", *output_cols.keys(), "Why", "Score"]].rename(
        columns=output_cols
    )
    return action_df[
        [
            "Action",
            "Instrument",
            "Ticker",
            "Why",
            "Score",
            "1D",
            "1W",
            "1M",
            "MA21",
            "MA63",
            "MA126",
            "MA252",
            "Drawdown",
            "Vol Ratio",
        ]
    ]


def render_today_action_list(df: pd.DataFrame):
    action_df = build_action_list(df)
    st.header("🎯 Today’s Action List")
    st.caption(
        "A daily decision queue synthesized from pullbacks, breakouts, capitulation, leader weakness, and rollover monitors."
    )

    if action_df.empty:
        st.info("No action-list candidates for the current Performance filters.")
        return

    counts = action_df["Action"].value_counts()
    metric_cols = st.columns(len(counts))
    for metric_col, (action, count) in zip(metric_cols, counts.items(), strict=False):
        with metric_col:
            st.metric(action, int(count))

    control_col1, control_col2 = st.columns([3, 1])
    action_options = list(action_df["Action"].drop_duplicates())
    with control_col1:
        selected_actions = st.multiselect(
            "Action filter",
            options=action_options,
            default=action_options,
        )
    with control_col2:
        max_rows = st.slider("Max rows", min_value=5, max_value=100, value=30, step=5)

    filtered_df = action_df[action_df["Action"].isin(selected_actions)]
    rows_per_action = max(1, int(np.ceil(max_rows / max(1, len(selected_actions)))))
    display_df = (
        filtered_df.groupby("Action", sort=False, group_keys=False)
        .head(rows_per_action)
        .head(max_rows)
        .copy()
    )
    display_df["ticker"] = display_df["Ticker"]
    add_sparkline_column(display_df)
    add_sparkline_column(display_df, col_name="Price (1y)", days=365)
    display_df = display_df.drop(columns=["ticker"])
    cols = list(display_df.columns)
    if "Price (90d)" in cols:
        cols.remove("Price (90d)")
        cols.insert(cols.index("Instrument") + 1, "Price (90d)")
    if "Price (1y)" in cols:
        cols.remove("Price (1y)")
        cols.insert(cols.index("Price (90d)") + 1, "Price (1y)")
    display_df = display_df[cols]
    numeric_cols = [
        c for c in display_df.columns if pd.api.types.is_numeric_dtype(display_df[c])
    ]
    st.dataframe(
        display_df.style.format(formatter="{:.2f}", subset=numeric_cols),
        hide_index=True,
        height=min(600, max(220, len(display_df) * 35 + 40)),
        column_config={
            "Price (90d)": st.column_config.LineChartColumn(
                "Price (90d)", width="small"
            ),
            "Price (1y)": st.column_config.LineChartColumn(
                "Price (1y)", width="small"
            )
        },
    )


def render_today_tab():
    with st.expander("Universe", expanded=False):
        instrument_categories = st.multiselect(
            "Instrument Category",
            options=FUND_TYPE_OPTIONS,
            default=["eq", "stock", "commod"],
            key="today_instrument_categories",
        )

    df: pd.DataFrame = get_data(
        query=create_query(
            table=di.perf_tbl,
            vol_adjust=False,
            show_returns=True,
            returns_cols=di.selectable_returns,
            fund_types=instrument_categories,
        ),
    )
    render_today_action_list(df)


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
        cap_pool = signal_df.dropna(subset=["vol_ratio"]).copy()
        if drawdown_col in cap_pool.columns:
            cap_pool = cap_pool[cap_pool[drawdown_col] < 0].copy()
            cap_pool["severity"] = (-cap_pool[drawdown_col]) * cap_pool["vol_ratio"]
        else:
            cap_pool["severity"] = cap_pool["vol_ratio"]

        cap = cap_pool[
            (cap_pool["vol_ratio"] >= 1.2) & (cap_pool[drawdown_col] <= -10)
        ].copy()
        if cap.empty:
            st.info(
                "No strict capitulation candidates at default thresholds. Showing relative stress leaders instead."
            )
            cap = cap_pool.copy()
        cap["bounce_starting"] = cap["r_1w"] > 0
        cap = cap.sort_values("severity", ascending=False).head(20).reset_index(drop=True)

        if cap.empty:
            st.info("No downside stress candidates available for the current universe.")
        else:
            cap_cols = [
                "description",
                "Price (90d)",
                "Price (1y)",
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
            add_sparkline_column(cap)
            add_sparkline_column(cap, col_name="Price (1y)", days=365)
            cap_cols = [c for c in cap_cols if c in cap.columns]
            st.dataframe(
                cap[cap_cols].style.format(
                    formatter="{:.2f}",
                    subset=[c for c in cap_cols if pd.api.types.is_numeric_dtype(cap[c])],
                ),
                hide_index=True,
                height=400,
                column_config={
                    "Price (90d)": st.column_config.LineChartColumn(
                        "Price (90d)", width="small"
                    ),
                    "Price (1y)": st.column_config.LineChartColumn(
                        "Price (1y)", width="small"
                    )
                },
            )

    with tab_breakout:
        st.markdown(
            "Upside stress screen: ETFs/funds exploding higher on unusually high volatility. Prioritizes high vol ratio, positive 1W/1M returns, and price above the 21-day average."
        )
        breakout_pool = signal_df[
            (signal_df["r_1w"] > 0) & (signal_df["r_1mo"] > 0)
        ].copy()
        breakout = breakout_pool[breakout_pool["vol_ratio"] >= 1.1].copy()
        if "ma_21" in breakout.columns:
            breakout_confirmed = breakout[breakout["ma_21"] > 0].copy()
            if not breakout_confirmed.empty:
                breakout = breakout_confirmed
        if breakout.empty:
            st.info(
                "No strict high-volatility breakouts at default thresholds. Showing strongest upside movers with elevated relative volatility instead."
            )
            breakout = breakout_pool.copy()
        breakout["breakout_score"] = breakout["vol_ratio"] * breakout["r_1w"]
        breakout = breakout.sort_values("breakout_score", ascending=False).head(20)

        if breakout.empty:
            st.info("No upside breakout candidates available for the current universe.")
        else:
            breakout_cols = [
                "description",
                "Price (90d)",
                "Price (1y)",
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
            add_sparkline_column(breakout)
            add_sparkline_column(breakout, col_name="Price (1y)", days=365)
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
                column_config={
                    "Price (90d)": st.column_config.LineChartColumn(
                        "Price (90d)", width="small"
                    ),
                    "Price (1y)": st.column_config.LineChartColumn(
                        "Price (1y)", width="small"
                    )
                },
            )

(
    tab_today,
    tab_perf,
    tab_pullback,
    tab_crossings,
    tab_daily,
    tab_rs,
    tab_rotation,
    tab_regime,
    tab_factors,
    tab_charts,
    tab_corr,
    tab_allocator,
) = st.tabs([
    "Today",
    "Performance",
    "Pullback",
    "Crossings",
    "Daily Summary",
    "Relative Strength",
    "Rotation",
    "Cross-Asset",
    "Factors",
    "Charts",
    "Correlation",
    "Allocator",
])

with tab_today:
    render_today_tab()

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
            default=["eq", "stock", "commod"],
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
        add_sparkline_column(df, col_name="Price (1y)", days=365)
        cols = list(df.columns)
        if "Price (90d)" in cols:
            cols.remove("Price (90d)")
            cols.insert(cols.index("description") + 1, "Price (90d)")
        if "Price (1y)" in cols:
            cols.remove("Price (1y)")
            cols.insert(cols.index("Price (90d)") + 1, "Price (1y)")
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
            "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
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

with tab_pullback:
    PullbackScanner.render()

with tab_crossings:
    TodaysCrossings.render()

with tab_daily:
    DailySummary.render()

with tab_rs:
    RelativeStrength.render()

with tab_rotation:
    RotationStrategies.render()

with tab_regime:
    CrossAssetRegime.render()

with tab_factors:
    FactorDashboard.render()

with tab_charts:
    PerformanceChart.render()

with tab_corr:
    AssetCorrelation.render()

with tab_allocator:
    PortfolioAllocator.render()
