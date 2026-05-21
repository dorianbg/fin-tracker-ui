"""
Robotics stock tracker with latest performance and selected price charts.
"""

import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

import duckdb_importer as di
from config import table_height
from data import add_sparkline_column, load_latest_perf, load_prices
from utils import filter_dataframe, style_performance_table


ROBOTICS_UNIVERSE = {
    "Robotics / Automation": ("BOTZ", "ISRG", "ABBNY", "SYM", "RR", "SERV", "ZBRA", "CGNX", "ALNT", "NOVT", "KITT", "ACUVI", "277810", "9880", "6324", "6268", "300024", "2498", "2590", "XBOT", "PDYN"),
    "Autonomy / Sensing": ("OUST", "MBLY", "ARBE", "HSAI", "HSYDF", "2525", "AEVA", "AUR", "XPEV", "INDI", "SHA0", "KDK", "MRLN"),
    "Semis / Components": ("AMBA", "VPG", "LSCC", "AMBQ", "ATOM", "MRAM", "HLIT", "KLIC"),
    "Adjacent / ETFs": ("300750", "BAM", "CAT", "MKA", "CTH", "IMSR"),
}

PRIVATE_OR_UNCLEAR = (
    "Unitree", "Nextronics", "Robotstrategy", "HG", "BSL", "NEO", "$BOT",
)

RETURN_COLS = ["1d", "1w", "2w", "1mo", "3mo", "6mo", "1y", "3y", "5y"]
DISPLAY_COLS = [
    "date", "description", "Price (90d)", "Price (1y)", "ticker", "theme_group", "fund_type",
    "vol_1mo", "vol_1y", "ma_21", "ma_63", "ma_126", "ma_252", "drawdown_52w", "drawdown_3y", "range_pos_52w", *di.perf_returns_cols,
]


def _all_tickers() -> tuple[str, ...]:
    return tuple(dict.fromkeys(t for tickers in ROBOTICS_UNIVERSE.values() for t in tickers))


def _ticker_to_group() -> dict[str, str]:
    return {ticker: group for group, tickers in ROBOTICS_UNIVERSE.items() for ticker in tickers}


def _filter_search(df: pd.DataFrame, search: str) -> pd.DataFrame:
    if not search or not search.strip():
        return df
    needle = search.strip().lower()
    return df[
        df["ticker"].str.lower().str.contains(needle, na=False)
        | df["description"].str.lower().str.contains(needle, na=False)
        | df["theme_group"].str.lower().str.contains(needle, na=False)
    ].copy()


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["price_index"] = prices.groupby("ticker")["price"].transform(lambda s: s / s.iloc[0] * 100 if len(s) and s.iloc[0] else s)
    return prices


def _add_sparklines(df: pd.DataFrame) -> pd.DataFrame:
    add_sparkline_column(df)
    add_sparkline_column(df, col_name="Price (1y)", days=365)
    cols = list(df.columns)
    for spark_col in ("Price (1y)", "Price (90d)"):
        if spark_col in cols:
            cols.remove(spark_col)
            cols.insert(cols.index("description") + 1, spark_col)
    return df[cols]


def render():
    st.title("Robotics Stock Tracker")
    st.caption("A focused robotics, automation, autonomy, sensing, and component universe using the existing performance dataset.")

    universe = _all_tickers()
    ticker_to_group = _ticker_to_group()
    controls_col, content_col = st.columns([1, 4], gap="large")

    with controls_col:
        st.subheader("Robotics Filters")
        selected_groups = st.multiselect("Theme groups", options=list(ROBOTICS_UNIVERSE), default=list(ROBOTICS_UNIVERSE), key="robotics_theme_groups")
        search = st.text_input("Search", placeholder="Ticker, name, or group", key="robotics_search")
        advanced_filters = st.toggle("Advanced table filters", value=False, key="robotics_advanced_filters")
        default_plot = [t for t in ("BOTZ", "ISRG", "ABBNY", "SYM") if t in universe]
        selected_tickers = st.multiselect("Plot tickers", options=list(universe), default=default_plot, key="robotics_selected_tickers")
        date_range = st.date_input("Price chart range", value=[datetime.date.today() - datetime.timedelta(days=365), datetime.date.today()], key="robotics_price_range")
        with st.expander("Untracked or unclear"):
            st.write(", ".join(PRIVATE_OR_UNCLEAR))

    selected_universe = tuple(ticker for group in selected_groups for ticker in ROBOTICS_UNIVERSE[group])

    with content_col:
        if not selected_universe:
            st.warning("Select at least one robotics theme group.")
            return

        df = load_latest_perf(tickers=selected_universe)
        if df.empty:
            st.warning("No robotics data found. Run the pipeline/export after adding the robotics tickers.")
            return

        df["theme_group"] = df["ticker"].map(ticker_to_group)
        missing = sorted(set(selected_universe) - set(df["ticker"].unique()))
        if missing:
            st.info("Missing from current performance data, likely pending pipeline fetch/export: " + ", ".join(missing))

        df = _filter_search(df, search)
        df = filter_dataframe(df, modify=advanced_filters)
        if df.empty:
            st.warning("No robotics rows match the current filters.")
            return

        df = _add_sparklines(df)
        display = df[[col for col in DISPLAY_COLS if col in df.columns]].copy()
        styled = style_performance_table(display.copy(), vol_adjust=False, show_returns=True, returns_cols=RETURN_COLS)

        event = st.dataframe(
            styled,
            hide_index=True,
            height=table_height,
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "description": st.column_config.TextColumn("description", width="medium"),
                "ticker": st.column_config.TextColumn("ticker", width="small"),
                "theme_group": st.column_config.TextColumn("theme_group", width="medium"),
                "fund_type": st.column_config.TextColumn("fund_type", width="small"),
                "Price (90d)": st.column_config.LineChartColumn("Price (90d)", width="small"),
                "Price (1y)": st.column_config.LineChartColumn("Price (1y)", width="small"),
            },
        )

        selected_from_table = []
        if event and event.selection and event.selection.rows:
            selected_from_table = display.iloc[event.selection.rows]["ticker"].tolist()
        plot_tickers = tuple(selected_from_table or selected_tickers)

        if not plot_tickers:
            st.info("Select table rows or choose tickers in the filter panel to plot prices.")
            return
        if not date_range or len(date_range) != 2:
            st.info("Choose a start and end date for the price chart.")
            return

        prices = load_prices(tickers=plot_tickers)
        prices = prices[prices["date"].dt.date.between(date_range[0], date_range[1])].copy()
        prices = _normalise_prices(prices)
        if prices.empty:
            st.warning("No price data found for the selected tickers/date range.")
            return

        fig = px.line(
            prices,
            x="date",
            y="price_index",
            color="description",
            hover_data=["ticker", "price"],
            labels={"date": "Date", "price_index": "Indexed price, start = 100", "description": "Instrument"},
            title="Selected Robotics Price Performance",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
