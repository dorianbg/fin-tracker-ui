from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import cache_data

import duckdb_importer as di

duckdb_file: str = ":memory:"
_conn: duckdb.DuckDBPyConnection = None
risk_free_rate = 0.05
sharpe_col_suffix = "_s"


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


@st.cache_data
def get_distinct_fund_types():
    return list(
        get_conn()
        .execute(f"select distinct fund_type from {di.px_tbl}")
        .df()["fund_type"]
    )


def gen_where_clause_prices(
    instruments: list[str],
    fund_types: list[str],
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
    if fund_types:
        where_str = "','".join(fund_types)
        where_clause.append(f"fund_type in ('{where_str}') ")
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
    table: str,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    instruments: list[str] = None,
    fund_types: list[str] = None,
    vol_adjust: bool = False,
    show_returns: bool = True,
    returns_cols: list[str] = None,
):
    if instruments is not None:
        instruments = [x.split("/")[0] for x in instruments]
    if table == di.px_tbl:
        _cols = di.px_cols
    elif table == di.perf_tbl:
        _cols = (
            di.perf_desc_cols
            + di.perf_z_score_cols
            + di.perf_vol_cols
            + di.get_perf_cols(
                show_returns=show_returns,
                vol_adjust=vol_adjust,
                returns_cols=returns_cols,
            )
            + di.perf_mavg_cols
        )
    else:
        raise ValueError("Unsupported table name")

    where_clause_str = gen_where_clause_prices(
        instruments,
        fund_types,
        start_date,
        end_date,
    )
    query = f"""
        select 
            {','.join(_cols)},  
        from {table}
        {where_clause_str} 
        order by {"ticker" if table == di.px_tbl else "fund_type"} asc, "date" asc
    """
    res = get_conn().execute(query).df()
    # we use price changes instead of prices
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


@cache_data
def get_fund_types() -> list[str]:
    query = f"select distinct fund_type from {di.perf_tbl}"
    return [x[0] for x in get_conn().execute(query).fetchall()]


def calculate_annual_cagr(total_percent_change: float, num_months: float):
    # Convert total percent change to annual CAGR
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


def create_perf_table(df):
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
                "Start price": first_price,
                "End price": last_price,
                "Change": price_chg_pct,
                "Time span": f"{months_delta} months",
                "CAGR": cagr * 100,
                "Start date": {min_dt.isoformat()},
                "End date": {max_dt.isoformat()},
            }
        )
    df_perf = pd.DataFrame(data)
    styled_df_perf = df_perf.style.format(
        subset=["Start price", "End price"], formatter="£{:.2f}"
    )
    styled_df_perf = styled_df_perf.format(
        subset=["Change", "CAGR"], formatter="{:.2f}%"
    )

    return styled_df_perf
