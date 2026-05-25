import datetime
import platform

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import duckdb_importer as di
from data import (
    get_data,
    get_conn,
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
import views.ConsolidationSetup as ConsolidationSetup
import views.TodaysCrossings as TodaysCrossings
import views.RelativeStrength as RelativeStrength
import views.FactorDashboard as FactorDashboard
import views.CrossAssetRegime as CrossAssetRegime
import views.AssetCorrelation as AssetCorrelation
import views.PerformanceChart as PerformanceChart
import views.RotationStrategies as RotationStrategies
import views.SectorRotation as SectorRotation
import views.PortfolioAllocator as PortfolioAllocator
import views.RAAMStrategy as RAAMStrategy
import views.RoboticsStocks as RoboticsStocks

st.set_page_config(
    page_icon="🏠",
    page_title="Financial instrument tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading data...")
def preload_data():
    """Eagerly load data on startup — essential for remote Quack connections."""
    import os

    # ── Connect ──
    from data import get_conn

    get_conn()

    if not os.environ.get("DUCKDB_REMOTE_HOST"):
        return

    # ── Warm caches for all tabs (remote is slow, local is instant) ──
    import duckdb_importer as di
    from data import get_data, create_query, load_prices, load_latest_perf

    get_data(create_query(table=di.perf_tbl, show_returns=True, vol_adjust=True))
    get_data(create_query(table=di.px_tbl))
    load_prices()  # SectorRotation, Consolidation, Charts
    load_latest_perf()  # LaggardBreakout, BreakoutScanner, Robotics


preload_data()


ACTION_PRIORITY = {
    "Buy Watch": 1,
    "Breakout Watch": 2,
    "Trim Watch": 3,
    "Short Monitor": 4,
    "Capitulation Watch": 5,
}


def build_signal_candidates(df: pd.DataFrame) -> pd.DataFrame:
    required = {"description", "ticker", "r_1d", "r_1w", "r_1mo", "vol_1mo", "vol_1y"}
    trend_cols = {"ma_21", "ma_63", "ma_126", "ma_252", "drawdown_52w"}
    if (
        df.empty
        or not required.issubset(df.columns)
        or not trend_cols.issubset(df.columns)
    ):
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
        (signals["r_1w"] > 0) & (signals["r_1mo"] > 0) & (signals["vol_ratio"] >= 1.1)
    ].copy()
    if not breakout_watch.empty and "ma_21" in breakout_watch.columns:
        confirmed = breakout_watch[breakout_watch["ma_21"] > 0].copy()
        if not confirmed.empty:
            breakout_watch = confirmed
    if not breakout_watch.empty:
        breakout_watch["breakout_score"] = (
            breakout_watch["vol_ratio"] * breakout_watch["r_1w"]
        )
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
            trim_watch["trim_score"] = -(trim_watch["r_1w"] - benchmark["r_1w"]) * (
                trim_watch["r_1y"] - benchmark["r_1y"]
            )
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
    action_df["action_priority"] = action_df["Action"].map(ACTION_PRIORITY)
    action_df = action_df.sort_values(
        ["action_priority", "Score"], ascending=[True, False]
    )
    return action_df


def build_action_list(df: pd.DataFrame) -> pd.DataFrame:
    action_df = build_signal_candidates(df)
    if action_df.empty:
        return pd.DataFrame()

    action_df["Decision"] = action_df["Action"]
    action_df["Entry Rule"] = ""
    action_df["Invalidation"] = ""
    action_df["Exit Plan"] = ""
    action_df["Review Trigger"] = ""
    action_df["Backtest Edge"] = ""

    buy_mask = action_df["Action"] == "Buy Watch"
    buy_confirmed = (
        buy_mask
        & (action_df["r_1d"] > 0)
        & (action_df["r_1w"] > 0)
        & (action_df["ma_63"] > -3)
    )
    action_df.loc[buy_mask, "Decision"] = "Wait For Reclaim"
    action_df.loc[buy_confirmed, "Decision"] = "Buy Candidate"
    action_df.loc[buy_mask, "Entry Rule"] = (
        "Buy only after bounce holds and MA63 remains intact"
    )
    action_df.loc[buy_mask, "Invalidation"] = "MA63 < -3 or drawdown <= -25"
    action_df.loc[buy_mask, "Exit Plan"] = (
        "Failed bounce timeout, MA63 stop, hard drawdown stop"
    )
    action_df.loc[buy_mask, "Review Trigger"] = (
        "Recheck if MA21 is not reclaimed within 10 trading days"
    )
    action_df.loc[buy_mask, "Backtest Edge"] = "Simulated as long entry below"

    breakout_mask = action_df["Action"] == "Breakout Watch"
    breakout_confirmed = (
        breakout_mask & (action_df["ma_21"] > 0) & (action_df["ma_63"] > 0)
    )
    action_df.loc[breakout_mask, "Decision"] = "Wait For Confirmation"
    action_df.loc[breakout_confirmed, "Decision"] = "Buy Candidate"
    action_df.loc[breakout_mask, "Entry Rule"] = (
        "Buy only while MA21 and MA63 stay positive"
    )
    action_df.loc[breakout_mask, "Invalidation"] = (
        "MA63 < -3 or failed upside follow-through"
    )
    action_df.loc[breakout_mask, "Exit Plan"] = (
        "MA63 trend stop or profit protection after MA21 loss"
    )
    action_df.loc[breakout_mask, "Review Trigger"] = (
        "Recheck on MA21 loss after an 8%+ move"
    )
    action_df.loc[breakout_mask, "Backtest Edge"] = "Simulated as long entry below"

    capitulation_mask = action_df["Action"] == "Capitulation Watch"
    capitulation_bounce = capitulation_mask & (
        (action_df["r_1d"] > 0) | (action_df["r_1w"] > 0)
    )
    action_df.loc[capitulation_mask, "Decision"] = "Avoid Until Bounce"
    action_df.loc[capitulation_bounce, "Decision"] = "Speculative Bounce Candidate"
    action_df.loc[capitulation_mask, "Entry Rule"] = (
        "Buy only after positive 1D/1W bounce confirmation"
    )
    action_df.loc[capitulation_mask, "Invalidation"] = (
        "No bounce after timeout or drawdown <= -25"
    )
    action_df.loc[capitulation_mask, "Exit Plan"] = (
        "Quick no-bounce exit or hard drawdown stop"
    )
    action_df.loc[capitulation_mask, "Review Trigger"] = (
        "Treat as tactical until MA21 recovers"
    )
    action_df.loc[capitulation_mask, "Backtest Edge"] = "Simulated as long entry below"

    trim_mask = action_df["Action"] == "Trim Watch"
    action_df.loc[trim_mask, "Decision"] = "Trim Candidate"
    action_df.loc[trim_mask, "Entry Rule"] = "Do not add; review existing position size"
    action_df.loc[trim_mask, "Invalidation"] = "Relative strength recovers vs benchmark"
    action_df.loc[trim_mask, "Exit Plan"] = "Trim if weakness persists or MA63 is lost"
    action_df.loc[trim_mask, "Review Trigger"] = (
        "Check 1W/1M relative strength vs benchmark"
    )
    action_df.loc[trim_mask, "Backtest Edge"] = "Risk overlay, not long-entry simulated"

    short_mask = action_df["Action"] == "Short Monitor"
    action_df.loc[short_mask, "Decision"] = "Risk Review"
    action_df.loc[short_mask, "Entry Rule"] = (
        "Do not add longs while MA21 and MA63 are broken"
    )
    action_df.loc[short_mask, "Invalidation"] = "MA21/MA63 recovery"
    action_df.loc[short_mask, "Exit Plan"] = (
        "Reduce/hedge if breakdown extends below MA126"
    )
    action_df.loc[short_mask, "Review Trigger"] = (
        "Watch failed bounces and stop-loss levels"
    )
    action_df.loc[short_mask, "Backtest Edge"] = (
        "Risk overlay, not long-entry simulated"
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
    action_df = action_df[
        [
            "Decision",
            "Action",
            *output_cols.keys(),
            "Entry Rule",
            "Invalidation",
            "Exit Plan",
            "Review Trigger",
            "Backtest Edge",
            "Why",
            "Score",
        ]
    ].rename(columns=output_cols)
    return action_df[
        [
            "Decision",
            "Action",
            "Instrument",
            "Ticker",
            "Entry Rule",
            "Invalidation",
            "Exit Plan",
            "Review Trigger",
            "Backtest Edge",
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

    counts = action_df["Decision"].value_counts()
    metric_cols = st.columns(len(counts))
    for metric_col, (action, count) in zip(metric_cols, counts.items(), strict=False):
        with metric_col:
            st.metric(action, int(count))

    control_col1, control_col2 = st.columns([3, 1])
    action_options = list(action_df["Action"].drop_duplicates())
    with control_col1:
        selected_actions = st.multiselect(
            "Signal filter",
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
            "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
        },
    )


@st.cache_data(ttl=300)
def load_signal_backtest_data(fund_types: tuple[str, ...], years: int) -> pd.DataFrame:
    start_date = datetime.date.today() - datetime.timedelta(days=365 * years)
    perf_df = get_data(
        query=create_query(
            table=di.perf_tbl,
            start_date=start_date,
            vol_adjust=False,
            show_returns=True,
            returns_cols=di.selectable_returns,
            fund_types=list(fund_types),
            get_perf_hist=True,
        ),
    )
    if perf_df.empty:
        return pd.DataFrame()

    tickers = tuple(sorted({*perf_df["ticker"].dropna().unique(), DEFAULT_BENCHMARK}))
    if not tickers:
        return pd.DataFrame()

    tickers_str = "','".join(tickers)
    price_df = (
        get_conn()
        .execute(
            f"""
        SELECT ticker, date, price AS px_price
        FROM {di.px_tbl}
        WHERE ticker IN ('{tickers_str}')
          AND date >= '{start_date.isoformat()}'
        ORDER BY ticker, date
        """
        )
        .df()
    )
    if price_df.empty:
        return pd.DataFrame()

    price_df["date"] = pd.to_datetime(price_df["date"], format="%Y-%m-%d")
    perf_df["price"] = pd.to_numeric(perf_df["price"], errors="coerce")
    merged = perf_df.merge(
        price_df,
        on=["ticker", "date"],
        how="left",
    )
    merged["price"] = merged["price"].fillna(merged["px_price"])
    merged = merged.drop(columns=["px_price"])
    return merged


def simulate_signal_trades(
    hist_df: pd.DataFrame,
    signal_name: str,
    max_hold_days: int,
    failed_bounce_days: int,
) -> pd.DataFrame:
    signal_df = build_signal_candidates(hist_df)
    if signal_df.empty:
        return pd.DataFrame()

    signal_df = signal_df[signal_df["Action"] == signal_name].copy()
    if signal_df.empty:
        return pd.DataFrame()

    entries = signal_df.sort_values(["date", "Score"], ascending=[True, False])
    data_by_ticker = {
        ticker: group.sort_values("date").reset_index(drop=True)
        for ticker, group in hist_df.dropna(subset=["price"]).groupby("ticker")
    }
    benchmark_df = data_by_ticker.get(DEFAULT_BENCHMARK)
    open_until_by_ticker = {}
    trades = []

    def benchmark_return(entry_date, exit_date) -> float:
        if benchmark_df is None or benchmark_df.empty:
            return np.nan
        bm_entry = benchmark_df[benchmark_df["date"] >= entry_date].head(1)
        bm_exit = benchmark_df[benchmark_df["date"] >= exit_date].head(1)
        if bm_entry.empty or bm_exit.empty:
            return np.nan
        bm_entry_price = bm_entry.iloc[0]["price"]
        bm_exit_price = bm_exit.iloc[0]["price"]
        if pd.isna(bm_entry_price) or pd.isna(bm_exit_price) or bm_entry_price <= 0:
            return np.nan
        return (bm_exit_price / bm_entry_price - 1) * 100

    for entry in entries.itertuples(index=False):
        ticker = entry.ticker
        entry_date = entry.date
        if ticker not in data_by_ticker:
            continue
        if (
            ticker in open_until_by_ticker
            and entry_date <= open_until_by_ticker[ticker]
        ):
            continue

        ticker_df = data_by_ticker[ticker]
        entry_matches = ticker_df.index[ticker_df["date"] == entry_date].tolist()
        if not entry_matches:
            continue
        signal_idx = entry_matches[0]
        entry_idx = signal_idx + 1
        if entry_idx >= len(ticker_df):
            continue
        entry_row = ticker_df.iloc[entry_idx]
        entry_price = entry_row["price"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        exit_row = None
        exit_reason = "Max hold"
        for hold_days, (_, row) in enumerate(
            ticker_df.iloc[entry_idx + 1 : entry_idx + max_hold_days + 1].iterrows(),
            start=1,
        ):
            current_return = (row["price"] / entry_price - 1) * 100
            if row["drawdown_52w"] <= -25:
                exit_row = row
                exit_reason = "Hard drawdown stop"
                break
            if (
                signal_name == "Buy Watch"
                and hold_days >= failed_bounce_days
                and row["ma_21"] < 0
            ):
                exit_row = row
                exit_reason = "Failed bounce"
                break
            if signal_name in {"Buy Watch", "Breakout Watch"} and row["ma_63"] < -3:
                exit_row = row
                exit_reason = "MA63 trend stop"
                break
            if (
                signal_name == "Breakout Watch"
                and current_return > 8
                and row["ma_21"] < 0
            ):
                exit_row = row
                exit_reason = "Profit protection"
                break
            if (
                signal_name == "Capitulation Watch"
                and hold_days >= failed_bounce_days
                and row["r_1w"] <= 0
            ):
                exit_row = row
                exit_reason = "No bounce confirmation"
                break
            exit_row = row

        if exit_row is None or pd.isna(exit_row["price"]):
            continue

        trade_return = (exit_row["price"] / entry_price - 1) * 100
        bm_return = benchmark_return(entry_row["date"], exit_row["date"])
        trades.append(
            {
                "Signal": signal_name,
                "Signal Date": entry_date,
                "Entry Date": entry_row["date"],
                "Exit Date": exit_row["date"],
                "Ticker": ticker,
                "Instrument": entry.description,
                "Entry Price": entry_price,
                "Exit Price": exit_row["price"],
                "Return": trade_return,
                "Benchmark Return": bm_return,
                "Relative Return": trade_return - bm_return
                if not pd.isna(bm_return)
                else np.nan,
                "Hold Days": (exit_row["date"] - entry_row["date"]).days,
                "Exit Reason": exit_reason,
                "Entry Score": entry.Score,
            }
        )
        open_until_by_ticker[ticker] = exit_row["date"]

    return pd.DataFrame(trades)


def render_signal_backtest(fund_types: list[str]):
    st.markdown("---")
    st.header("Trade Simulation Backtest")
    st.caption(
        "Simulates historical signal entries with trigger-based exits. Max hold is only a safety cap, not the main exit rule."
    )

    entry_signals = ["Buy Watch", "Breakout Watch", "Capitulation Watch"]
    control_col1, control_col2, control_col3, control_col4 = st.columns([2, 1, 1, 1])
    with control_col1:
        selected_signal = st.selectbox(
            "Signal",
            options=entry_signals,
            key="today_backtest_signal",
        )
    with control_col2:
        failed_bounce_days = st.selectbox(
            "Bounce timeout",
            options=[5, 10, 15, 21],
            index=1,
            format_func=lambda days: f"{days} trading days",
            key="today_backtest_failed_bounce_days",
        )
    with control_col3:
        max_hold_days = st.selectbox(
            "Max hold cap",
            options=[63, 126, 252],
            index=1,
            format_func=lambda days: f"{days} trading days",
            key="today_backtest_max_hold_days",
        )
    with control_col4:
        years = st.selectbox(
            "History",
            options=[1, 2, 3, 5],
            index=2,
            format_func=lambda y: f"{y}y",
            key="today_backtest_years",
        )

    hist_df = load_signal_backtest_data(tuple(fund_types), years)
    trades = simulate_signal_trades(
        hist_df,
        selected_signal,
        max_hold_days=max_hold_days,
        failed_bounce_days=failed_bounce_days,
    )
    if trades.empty:
        st.info("No completed historical trades found for this signal and universe.")
        return

    metric_values = {
        "Trades": len(trades),
        "Win rate": (trades["Return"] > 0).mean() * 100,
        "Avg return": trades["Return"].mean(),
        "Avg rel": trades["Relative Return"].mean(),
        "Median return": trades["Return"].median(),
        "Avg hold": trades["Hold Days"].mean(),
        "Worst": trades["Return"].min(),
    }
    metric_cols = st.columns(len(metric_values))
    for metric_col, (label, value) in zip(
        metric_cols, metric_values.items(), strict=False
    ):
        with metric_col:
            if label == "Trades":
                st.metric(label, int(value))
            elif label == "Avg hold":
                st.metric(label, f"{value:.0f}d")
            else:
                st.metric(label, f"{value:+.2f}%")

    reason_counts = trades["Exit Reason"].value_counts().reset_index()
    reason_counts.columns = ["Exit Reason", "Count"]
    fig_reasons = px.bar(
        reason_counts,
        x="Count",
        y="Exit Reason",
        orientation="h",
        title="Exit Reasons",
    )
    fig_reasons.update_layout(height=max(250, len(reason_counts) * 45))
    st.plotly_chart(fig_reasons, width="stretch")

    recent = trades.sort_values("Entry Date", ascending=False).head(50).copy()
    recent["ticker"] = recent["Ticker"]
    add_sparkline_column(recent)
    add_sparkline_column(recent, col_name="Price (1y)", days=365)
    cols = [
        "Signal Date",
        "Entry Date",
        "Exit Date",
        "Instrument",
        "Price (90d)",
        "Price (1y)",
        "Ticker",
        "Entry Score",
        "Return",
        "Benchmark Return",
        "Relative Return",
        "Hold Days",
        "Exit Reason",
    ]
    cols = [c for c in cols if c in recent.columns]
    numeric_cols = [
        c
        for c in cols
        if c
        not in {
            "Signal Date",
            "Entry Date",
            "Exit Date",
            "Instrument",
            "Ticker",
            "Price (90d)",
            "Price (1y)",
            "Exit Reason",
        }
    ]
    st.dataframe(
        recent[cols].style.format(subset=numeric_cols, formatter="{:+.2f}"),
        hide_index=True,
        height=450,
        column_config={
            "Price (90d)": st.column_config.LineChartColumn(
                "Price (90d)", width="small"
            ),
            "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
            "Return": st.column_config.NumberColumn(
                "Return", format="%.2f%%", width="small"
            ),
            "Benchmark Return": st.column_config.NumberColumn(
                "Benchmark Return", format="%.2f%%", width="small"
            ),
            "Relative Return": st.column_config.NumberColumn(
                "Relative Return", format="%.2f%%", width="small"
            ),
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
    render_signal_backtest(instrument_categories)


def render_action_screens(df: pd.DataFrame):
    required = {"description", "ticker", "vol_1mo", "vol_1y", "r_1w", "r_1mo"}
    if df.empty or not required.issubset(df.columns):
        return

    signal_df = df.copy()
    signal_df["vol_ratio"] = signal_df["vol_1mo"] / signal_df["vol_1y"].replace(
        0, np.nan
    )
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
        st.plotly_chart(fig_vol, width="stretch")

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
        cap = (
            cap.sort_values("severity", ascending=False).head(20).reset_index(drop=True)
        )

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
                    subset=[
                        c for c in cap_cols if pd.api.types.is_numeric_dtype(cap[c])
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
                    ),
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
                    ),
                },
            )


(
    tab_today,
    tab_perf,
    tab_pullback,
    tab_consolidation,
    tab_crossings,
    tab_daily,
    tab_rs,
    tab_rotation,
    tab_sector_rotation,
    tab_regime,
    tab_factors,
    tab_charts,
    tab_corr,
    tab_robotics,
    tab_allocator,
    tab_raam,
) = st.tabs(
    [
        "Today",
        "Performance",
        "Pullback",
        "Consolidation",
        "Crossings",
        "Daily Summary",
        "Relative Strength",
        "Rotation",
        "Sector Rotation",
        "Cross-Asset",
        "Factors",
        "Charts",
        "Correlation",
        "Robotics",
        "Allocator",
        "RAAM Strategy",
    ]
)

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
        if (
            len(custom_weights)
            and sum(custom_weights) > 0
            and sum(custom_weights) != 100
        ):
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
            "Price (90d)": st.column_config.LineChartColumn(
                "Price (90d)", width="small"
            ),
            "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
            **{
                c: st.column_config.NumberColumn(label=c, width="small")
                for c in (
                    "drawdown_52w",
                    "drawdown_3y",
                    "range_pos_52w",
                    "range_pos_104w",
                    "range_pos_156w",
                )
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

with tab_consolidation:
    ConsolidationSetup.render()

with tab_crossings:
    TodaysCrossings.render()

with tab_daily:
    DailySummary.render()

with tab_rs:
    RelativeStrength.render()

with tab_rotation:
    RotationStrategies.render()

with tab_sector_rotation:
    SectorRotation.render()

with tab_regime:
    CrossAssetRegime.render()

with tab_factors:
    FactorDashboard.render()

with tab_charts:
    PerformanceChart.render()

with tab_corr:
    AssetCorrelation.render()

with tab_robotics:
    RoboticsStocks.render()

with tab_allocator:
    PortfolioAllocator.render()

with tab_raam:
    RAAMStrategy.render()
