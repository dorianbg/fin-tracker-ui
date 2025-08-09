import math
from datetime import date, timedelta

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil.relativedelta import relativedelta
from matplotlib.colors import rgb2hex, colorConverter
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import duckdb_importer as di
from app.config import charts_width
from app.data import (
    sharpe_col_suffix,
    create_query,
    get_data,
    get_variation,
    create_perf_table,
)


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
    df = df.drop(columns=[col for col in df.columns if col.endswith(suffix)])
    df = df.sort_values(final_col, inplace=False)
    return df


def plot_timeseries_data(
    df: pd.DataFrame,
    y_col,
    y_axis_title,
    x_col="date",
    x_axis_title="Date",
    x_col_format="%b %Y",
    group_col="ticker",
    group_col_title="ticker",
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
                x=alt.X(x_col, axis=alt.Axis(title=x_axis_title, format=x_col_format)),
                y=alt.Y(y_col, axis=alt.Axis(title=y_axis_title)).scale(zero=False),
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
                alt.Tooltip(x_col, title=x_axis_title),
                alt.Tooltip(y_col, title=y_axis_title),
                alt.Tooltip(group_col, title=group_col_title),
            ],
        )
        .add_params(hover)
    )

    chart = lines + points + tooltips
    return chart


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


def plot_performance(
    start_date, end_date, selected_inst, selected_fund_types, show_df=False
):
    perf_df_hist: pd.DataFrame = get_data(
        query=create_query(
            table=di.perf_tbl,
            start_date=start_date,
            end_date=end_date,
            instruments=None if len(selected_inst) == 0 else selected_inst,
            fund_types=(None if len(selected_fund_types) == 0 else selected_fund_types),
            get_perf_hist=True,
        )
    )

    prices_df: pd.DataFrame = get_data(
        query=create_query(
            table=di.px_tbl,
            start_date=start_date,
            end_date=end_date,
            instruments=None if len(selected_inst) == 0 else selected_inst,
            fund_types=(None if len(selected_fund_types) == 0 else selected_fund_types),
        )
    )
    variations = (
        prices_df.groupby("ticker")["price"]
        .expanding(min_periods=2)
        .apply(get_variation)
    )
    df = prices_df.assign(price_chg=variations.droplevel(0))

    if len(selected_inst) > 0 or len(selected_fund_types) > 0:
        st.text("Price performance")
        st.altair_chart(
            plot_timeseries_data(
                df,
                y_col="price_chg",
                y_axis_title="Price change (%)",
                group_col="description",
            ),
            use_container_width=True,
        )
        if show_df:
            st.write(f"Performance comparison from {start_date} to {end_date} ")
            st.dataframe(data=create_perf_table(prices_df), hide_index=True)

        st.write(f"Rolling 1 month lookback volatility {start_date} to {end_date} ")
        st.altair_chart(
            plot_timeseries_data(
                perf_df_hist,
                y_col="vol_1mo",
                y_axis_title="Vol (1mo)",
                group_col="description",
            ),
            use_container_width=True,
        )
        st.write(f"Rolling 1 year lookback volatility {start_date} to {end_date} ")
        st.altair_chart(
            plot_timeseries_data(
                perf_df_hist,
                y_col="vol_1y",
                y_axis_title="Vol (1y)",
                group_col="description",
            ),
            use_container_width=True,
        )


def style_performance_table(df, vol_adjust, show_returns, returns_cols):
    df["date"] = df["date"].dt.date
    # Apply background_gradient to each numeric column
    styled_df = df.style.apply(apply_gradient)
    # format numeric columns
    percent_cols = [] + di.perf_vol_cols + di.perf_mavg_cols
    two_decimal_cols = []  # + di.perf_z_score_cols
    perf_cols = di.get_perf_cols(
        show_returns=show_returns, vol_adjust=vol_adjust, returns_cols=returns_cols
    )
    if vol_adjust and not show_returns:
        two_decimal_cols += perf_cols
    else:
        percent_cols += [p for p in perf_cols if sharpe_col_suffix not in p]
        two_decimal_cols += [p for p in perf_cols if sharpe_col_suffix in p]

    styled_df = styled_df.format(subset=percent_cols, formatter="{:.2f}%")
    styled_df = styled_df.format(subset=two_decimal_cols, formatter="{:.2f}")
    return styled_df


def filter_dataframe(df: pd.DataFrame, modify: bool) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            # First try explicit ISO format
            try:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
                continue
            except ValueError:
                pass

            # Then try common date formats
            date_formats = ["%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]
            for date_format in date_formats:
                try:
                    df[col] = pd.to_datetime(df[col], format=date_format)
                    break
                except ValueError:
                    continue

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def returns_from_prices(prices, log_returns=False):
    """
    Calculate the returns given prices.

    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    if log_returns:
        returns = np.log(1 + prices.pct_change()).dropna(how="all")
    else:
        returns = prices.pct_change().dropna(how="all")
    return returns


def get_holdings_perf(assets, start_date, end_date):
    prices = get_data(
        create_query(
            table=di.px_tbl,
            start_date=start_date,
            end_date=end_date,
            instruments=assets,
        )
    )
    prices["date"] = prices["date"].dt.date
    prices = prices.drop_duplicates(["date", "ticker"])
    prices = prices.pivot(index="date", columns="ticker", values="price")
    return returns_from_prices(prices)


def correlation_matrix(assets):
    if len(assets) == 0 or len(assets) > 15:
        return
    # Generate correlation matrix
    st.header("Asset Correlation Matrix")
    corr_matrix_lookback = st.number_input(
        "Correlation Matrix Lookback period (days)",
        min_value=1,
        max_value=365,
        step=5,
        value=60,
    )
    corr_matrix_end_date = st.date_input(
        label="Correlation Matrix Lookback End date", value=date.today()
    )
    try:
        # Fetch prices for all assets in the portfolio
        corr_matrix_start_date = date.today() - timedelta(days=corr_matrix_lookback)

        perf: pd.DataFrame = get_holdings_perf(
            assets=assets,
            start_date=corr_matrix_start_date,
            end_date=corr_matrix_end_date,
        )
        corr = perf.corr()
        fig2 = px.imshow(
            corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.columns),
            contrast_rescaling="minmax",
            text_auto=".2f",
            color_continuous_scale="RdYlGn",  # Use RdYlGn color scale
            # zmin=-1,  # Minimum value for color scale
            # zmax=1,  # Maximum value for color scale
            labels=dict(x="Asset", y="Asset", color="Correlation"),  # Axis labels
        )
        # Move x-axis labels to the top
        fig2.update_xaxes(side="top")

        st.plotly_chart(fig2)

    except Exception as e:
        st.error(f"Error generating correlation matrix: {e}")


@st.cache_data
def get_tickers_w_desc():
    df: pd.DataFrame = get_data(
        query=f"select distinct ticker || ' - ' || description as asset from {di.perf_tbl}"
    )["asset"].values.tolist()
    return df
