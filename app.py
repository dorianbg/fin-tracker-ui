from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
import duckdb
import duckdb_importer as di
from streamlit import cache_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, colorConverter
from dateutil.relativedelta import relativedelta


charts_width: int = 800
duckdb_file: str = ":memory:"
cols_perf: list[str] = ["date", "num_ads", "price", "type"]
cols_prices: list[str] = ["ticker"]
map_name_to_type: dict = {
    "Apartments for sale": "sales_flats",
    "Apartments for rent": "rentals_flats",
    "Houses for sale": "sales_houses",
}

st.set_page_config(
    page_icon="🏠", page_title="Financial instrument tracker", layout="wide"
)
# List of tuples containing string representations and corresponding timedelta values
time_strings = ["1W", "1M", "3M", "6M", "9M", "1Y", "18M", "2Y", "3Y", "5Y", "10Y"]


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


_conn: duckdb.DuckDBPyConnection = None


def init_conn(file_name: str) -> duckdb.DuckDBPyConnection:
    global _conn
    _conn = duckdb.connect(database=file_name)
    _load_pq = (
        lambda tbl, file, enc: f"CREATE TEMP TABLE {tbl} AS SELECT * FROM read_parquet('{file}', encryption_config = {enc})"
    )
    _conn.execute(f"{di.add_encrypt_key}")
    _conn.execute(_load_pq(di.px_tbl, di.px_pq_file, di.encrypt_conf))
    _conn.execute(_load_pq(di.perf_tbl, di.perf_pq_file, di.encrypt_conf))
    return _conn


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn, duckdb_file
    if _conn is not None:
        return _conn
    else:
        return init_conn(duckdb_file)


@st.cache_data
def get_distinct_instruments():
    return list(
        get_conn()
        .execute(
            f"select distinct (ticker || '/' || description) as ticker_desc  from {di.px_tbl}"
        )
        .df()["ticker_desc"]
    )


def gen_where_clause_prices(
    instruments: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> str:
    where_clause = []
    if start_date:
        where_clause.append(f"date >= '{start_date.isoformat()}'")
    if end_date:
        where_clause.append(f"date <= '{end_date.isoformat()}'")
    if instruments:
        where_str = "','".join(instruments)
        where_clause.append(f"ticker in ('{where_str}') ")
    where_clause_str = ""
    if len(where_clause) > 0:
        where_clause_str = f"where {' and '.join(where_clause)}"
    return where_clause_str


def get_variation(values: pd.Series) -> np.float64:
    base = values.iloc[0]  # first element in window iteration
    current = values.iloc[-1]  # last element in window iteration
    return round(100 * (current - base) / base, 2) if base else 0


@st.cache_data
def get_data(
    table,
    cols,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    instruments: list[str] = None,
) -> pd.DataFrame:
    if instruments is not None:
        instruments = [x.split("/")[0] for x in instruments]
    where_clause_str = gen_where_clause_prices(
        instruments,
        start_date,
        end_date,
    )
    query = f"""
        select 
            {','.join(cols)},  
        from {table}
        {where_clause_str} 
        order by "ticker" asc, "date" asc
    """
    res = get_conn().execute(query).df()
    if table == di.px_tbl:
        variations = (
            res.groupby("ticker")["price"].expanding(min_periods=2).apply(get_variation)
        )
        res = res.assign(price_chg=variations.droplevel(0))
    return res


@cache_data
def get_min_date_all() -> tuple[datetime.date, datetime.date]:
    query = f"select min(date) as min_date from {di.px_tbl}"
    return get_conn().execute(query).fetchall()[0][0]


def calculate_annual_cagr(total_percent_change: float, num_months: float):
    # Convert total percent change to a monthly CAGR
    monthly_cagr = (
        ((1 + total_percent_change / 100) ** (1 / num_months)) - 1
        if num_months > 0
        else 0
    )
    # Calculate annual CAGR
    annual_cagr = (1 + monthly_cagr) ** 12 - 1
    return annual_cagr


def get_percent_change(df: pd.DataFrame, col_name: str):
    first_value = df[col_name].iloc[0]
    last_value = df[col_name].iloc[-1]
    percent_change = ((last_value - first_value) / first_value) * 100
    return percent_change, first_value, last_value


@st.cache_data
def extract_metrics(df: pd.DataFrame, date_col: str = "date", price_col: str = "price"):
    price_chg_pct, first_price, last_price = get_percent_change(df, price_col)
    min_dt: datetime.date = df[date_col].min().date()
    max_dt: datetime.date = df[date_col].max().date()
    months_delta: float = (
        max_dt.year * 12 + max_dt.month - (min_dt.year * 12 + min_dt.month)
    )
    cagr: float = calculate_annual_cagr(
        total_percent_change=price_chg_pct, num_months=months_delta
    )
    return (
        cagr,
        min_dt,
        max_dt,
        months_delta,
        price_chg_pct,
        first_price,
        last_price,
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
                x=alt.X(x_col, axis=alt.Axis(title="Date")),
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
        with st.container():
            df: pd.DataFrame = get_data(
                table=di.perf_tbl,
                cols=di.cols_perf,
                start_date=None,
                end_date=None,
                instruments=None,
            )
            df["date"] = df["date"].dt.date
            # Apply background_gradient to each numeric column
            styled_df = df.style.apply(apply_gradient)
            # format numeric columns
            styled_df = styled_df.format(subset=di.cols_perf_num, formatter="{:.2f}%")
            styled_df = styled_df.format(
                subset=di.cols_perf_z_score, formatter="{:.2f}"
            )
            st.dataframe(data=styled_df, hide_index=True, height=550)

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
                max_value=datetime.today(),
                format="DD/MM/YYYY",
            )
        with col3:
            end_date: datetime.date = st.date_input(
                "Select end date",
                value=datetime.today(),
                min_value=min_date_possible,
                max_value=datetime.today(),
                format="DD/MM/YYYY",
            )

        if selected_lookback is not None:
            end_date = datetime.today()
            start_date = deduct_datetime_interval(end_date, selected_lookback)

        with st.container():
            selected_inst: list[str] = st.multiselect(
                label="Instrument",
                options=get_distinct_instruments(),
            )
            df: pd.DataFrame = get_data(
                table=di.px_tbl,
                cols=di.cols_prices,
                start_date=start_date,
                end_date=end_date,
                instruments=None if len(selected_inst) == 0 else selected_inst,
            )
            if len(selected_inst) > 0:
                st.altair_chart(
                    plot_prices_instrument(df),
                    use_container_width=True,
                )
                data: list = []
                for inst in df["ticker"].unique():
                    sub_df: pd.DataFrame = df[df["ticker"] == inst]
                    desc = sub_df.iloc[0]["description"]
                    (
                        cagr,
                        min_dt,
                        max_dt,
                        months_delta,
                        price_chg_pct,
                        first_price,
                        last_price,
                    ) = extract_metrics(sub_df, "date", "price")
                    data.append(
                        {
                            "ticker": inst,
                            "Description": desc,
                            "Start price": f"£{first_price:.2f}",
                            "End price": f"£{last_price:.2f}",
                            "Change": f"{price_chg_pct:.2f}%",
                            "Time span": f"{months_delta} months",
                            "CAGR": f"{cagr * 100:.2f}%",
                            "Start date": {min_dt.isoformat()},
                            "End date": {max_dt.isoformat()},
                        }
                    )
                st.write(
                    f"Performance comparison from {start_date.date()} to {end_date.date()} "
                )
                st.dataframe(data=pd.DataFrame(data), hide_index=True)


st.title("Fin tracker")
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
