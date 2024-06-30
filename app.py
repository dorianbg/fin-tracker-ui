import time
import datetime

import altair as alt
import pandas as pd
import streamlit as st
import math
import data
import duckdb_importer as di
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, colorConverter
from dateutil.relativedelta import relativedelta

import platform

from data import (
    get_distinct_instruments,
    get_distinct_fund_types,
    get_data,
    get_min_date_all,
    get_fund_types,
    create_perf_table,
)

charts_width: int = 800
table_height: int = 400
cols_perf: list[str] = ["date", "num_ads", "price", "type"]
cols_prices: list[str] = ["ticker"]
map_name_to_type: dict = {
    "Apartments for sale": "sales_flats",
    "Apartments for rent": "rentals_flats",
    "Houses for sale": "sales_houses",
}
time_strings = ["1W", "1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "5Y", "10Y"]

st.set_page_config(
    page_icon="🏠", page_title="Financial instrument tracker", layout="wide"
)

if platform.system() == "Darwin":
    with st.spinner("Processing"):
        di.run()
        time.sleep(0.5)


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


def plot_prices_instrument(
    df: pd.DataFrame, x_col="date", y_col="price_chg", group_col="ticker"
):
    brush = alt.selection_interval(encodings=["x"], empty=True)

    hover = alt.selection_point(
        fields=[x_col],
        nearest=True,
        on="mouseover",
    )

    lines = (
        (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(x_col, axis=alt.Axis(title="Date", format='%b %Y')),
                y=alt.Y(y_col, axis=alt.Axis(title="Price change (%)")).scale(
                    zero=False
                ),
                color=group_col,
            )
        )
        .add_params(brush)
        .properties(width=charts_width)
    )
    # Draw points on the line, highlight based on selection
    points = (
        lines.transform_filter(hover)
        .mark_circle(size=65)
        .encode(color=alt.Color("color:N", scale=None))
    )
    # Draw an invisible rule at the instrument of the selection
    tooltips = (
        alt.Chart(df)
        .mark_rule(opacity=0)
        .encode(
            x=x_col,
            y=y_col,
            tooltip=[
                alt.Tooltip(x_col, title="Date"),
                alt.Tooltip(y_col, title="Price change (%)"),
                alt.Tooltip(group_col, title="ticker"),
            ],
        )
        .add_params(hover)
    )

    chart = lines + points + tooltips
    return chart


# Function to deduct datetime representations of time intervals from a given datetime
def deduct_datetime_interval(base_datetime, interval):
    if not interval:  # Skip empty strings
        return None

    interval_value = int(interval[:-1])
    interval_unit = interval[-1]
    if interval_unit == "W":
        return base_datetime - relativedelta(weeks=interval_value)
    elif interval_unit == "M":
        return base_datetime - relativedelta(months=interval_value)
    elif interval_unit == "Y":
        return base_datetime - relativedelta(years=interval_value)
    else:
        raise ValueError("Wrong input")


# Function to adjust text color based on background color luminance
def adjust_text_color(bg_color):
    luminance = 0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]
    return "black" if luminance > 0.5 else "white"


# Create a function to apply background_gradient to each numeric column
def apply_gradient(column):
    if pd.api.types.is_numeric_dtype(column.dtype):
        min_value = column.min()
        max_value = column.max()
        norm = plt.Normalize(min_value, max_value)
        if column.name in di.perf_vol_cols:
            bg_color = plt.cm.bwr(norm(column.values))
        else:
            bg_color = plt.cm.RdYlGn(norm(column.values))
        text_color = [adjust_text_color(colorConverter.to_rgb(bg)) for bg in bg_color]
        return [
            "background-color: {}; color: {}".format(rgb2hex(bg), tc)
            for bg, tc in zip(bg_color, text_color)
        ]
    else:
        return [""] * len(column)  # Return empty string for non-numeric columns


def main():
    tab1, tab2 = st.tabs(["Performance table", "Performance chart"])

    with tab1:
        selected_fund_types = st.multiselect(
            label="Fund types",
            options=get_fund_types(),
            default=get_fund_types(),
        )

        col1, col2, col3 = st.columns([2, 2, 7])

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
                table=di.perf_tbl,
                fund_types=selected_fund_types,
                vol_adjust=vol_adjust,
                show_returns=show_returns,
                returns_cols=returns_cols,
            )
            if (
                (sort_sharpe or sort_returns)
                and sum(custom_weights) == 100
                and filter(lambda x: x >= 1, custom_weights)
            ):
                total_w = sum(custom_weights)
                custom_weights_normalised = [x / total_w for x in custom_weights]
                columns_sort = (
                    di.perf_sharpe_cols if sort_sharpe else di.perf_returns_cols
                )
                df = custom_sort_df_cols(columns_sort, custom_weights_normalised, df)
            styled_df = style_performance_table(
                df,
                vol_adjust=vol_adjust,
                show_returns=show_returns,
                returns_cols=returns_cols,
            )
            event = st.dataframe(
                data=styled_df,
                hide_index=True,
                height=table_height,
                use_container_width=True,
                on_select="rerun",
                selection_mode="multi-row",
            )
            if event:
                st.text("Price performance")
                filtered_df = df.iloc[event.selection.rows]

                plot_performance(start_date=datetime.date.today() - datetime.timedelta(days=3*365),
                                 end_date=datetime.date.today(),
                                 selected_inst=list(filtered_df["ticker"].unique()),
                                 selected_fund_types=[],
                                 show_df=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        min_date_possible: datetime.date = None
        max_date_possible: datetime.date = None

        if min_date_possible is None and max_date_possible is None:
            min_date_possible = get_min_date_all()

        with col1:
            selected_lookback = st.selectbox(
                label="Lookback period (overrides date range)",
                options=[None] + time_strings,
                index=4,
            )

        with col2:
            start_date: datetime.date = st.date_input(
                "Select start date",
                value=min_date_possible,
                min_value=min_date_possible,
                max_value=datetime.date.today(),
                format="DD/MM/YYYY",
            )
        with col3:
            end_date: datetime.date = st.date_input(
                "Select end date",
                value=datetime.date.today(),
                min_value=min_date_possible,
                max_value=datetime.date.today(),
                format="DD/MM/YYYY",
            )

        if selected_lookback is not None:
            end_date = datetime.date.today()
            start_date = deduct_datetime_interval(end_date, selected_lookback)

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                selected_inst: list[str] = st.multiselect(
                    label="Instrument", options=get_distinct_instruments(), default=None
                )
            with col2:
                selected_fund_types: list[str] = st.multiselect(
                    label="Asset class", options=get_distinct_fund_types(), default=None
                )

            plot_performance(start_date, end_date, selected_inst, selected_fund_types, show_df=True)


def custom_sort_df_cols(columns_sort, custom_weights_normalised, df):
    suffix = "_weighted_rank"
    final_col = "weighted_rank"
    for column in columns_sort:
        if column in df.columns:
            df[column + suffix] = df[column].rank(na_option="keep")
    for index, row in df.iterrows():
        weighted_rank = 0
        for column, weight in zip(columns_sort, custom_weights_normalised):
            if column in df.columns:
                rank = row[column + suffix]
                if not math.isnan(rank):
                    weighted_rank += rank * weight
        df.at[index, final_col] = weighted_rank
    df = df.drop(
        columns=[col for col in df.columns if col.endswith(suffix)]
    )
    df = df.sort_values(final_col, inplace=False)
    return df


def plot_performance(start_date, end_date, selected_inst, selected_fund_types, show_df = False):
    df: pd.DataFrame = get_data(
        table=di.px_tbl,
        start_date=start_date,
        end_date=end_date,
        instruments=None if len(selected_inst) == 0 else selected_inst,
        fund_types=(
            None if len(selected_fund_types) == 0 else selected_fund_types
        ),
    )
    if len(selected_inst) > 0 or len(selected_fund_types) > 0:
        st.altair_chart(
            plot_prices_instrument(df),
            use_container_width=True,
        )
        if show_df:
            df_perf = create_perf_table(df)
            st.write(f"Performance comparison from {start_date} to {end_date} ")
            st.dataframe(data=df_perf, hide_index=True)


def style_performance_table(df, vol_adjust, show_returns, returns_cols):
    df["date"] = df["date"].dt.date
    # Apply background_gradient to each numeric column
    styled_df = df.style.apply(apply_gradient)
    # format numeric columns
    percent_cols = [] + di.perf_vol_cols + di.perf_mavg_cols
    two_decimal_cols = [] + di.perf_z_score_cols
    perf_cols = di.get_perf_cols(
        show_returns=show_returns, vol_adjust=vol_adjust, returns_cols=returns_cols
    )
    if vol_adjust and not show_returns:
        two_decimal_cols += perf_cols
    else:
        percent_cols += [p for p in perf_cols if data.sharpe_col_suffix not in p]
        two_decimal_cols += [p for p in perf_cols if data.sharpe_col_suffix in p]

    styled_df = styled_df.format(subset=percent_cols, formatter="{:.2f}%")
    styled_df = styled_df.format(subset=two_decimal_cols, formatter="{:.2f}")
    return styled_df


st.title("Fin tracker")

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

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

main()
