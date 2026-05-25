import logging
import os
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import cache_data

import config
import duckdb_importer as di

logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

_DEFAULT_DUCKDB_PATH = os.path.join(os.path.dirname(__file__), "..", "duckdb.db")
duckdb_file: str = os.environ.get("DUCKDB_PATH", _DEFAULT_DUCKDB_PATH)
_conn: duckdb.DuckDBPyConnection = None

# Constants for Sharpe ratio calculation
risk_free_rate = config.RISK_FREE_RATE
sharpe_col_suffix = "_s"


def _resolve_db_path() -> str:
    """Resolve the DuckDB database path, handling remote Quack server if configured."""
    remote_host = os.environ.get("DUCKDB_REMOTE_HOST", "")
    if remote_host:
        return f"quack:{remote_host}:9494"
    return duckdb_file


def init_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    global _conn
    is_remote = db_path.startswith("quack:")
    _conn = duckdb.connect(
        database=":memory:" if is_remote else db_path, read_only=True
    )

    if is_remote:
        token = os.environ.get("QUACK_AUTH_TOKEN", "fintracker-quack-token-2026")
        _conn.execute(f"ATTACH '{db_path}' AS remote_db (TOKEN '{token}')")
        # View aliases pointing at the remote catalog
        _conn.execute(
            "CREATE OR REPLACE VIEW prices AS SELECT * FROM remote_db.total_return"
        )
        _conn.execute(
            "CREATE OR REPLACE VIEW performance AS SELECT * FROM remote_db.latest_performance_sharpe"
        )
    else:
        _conn.execute("CREATE OR REPLACE VIEW prices AS SELECT * FROM total_return")
        _conn.execute(
            "CREATE OR REPLACE VIEW performance AS SELECT * FROM latest_performance_sharpe"
        )
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
    table: str,
    get_perf_hist: bool = False,
) -> str:
    date_clause = get_date_clause(start_date, end_date)
    sub_clause = get_where_subclause(instruments, fund_types)
    rown_filter = (
        " rown = 1 " if (table == di.perf_tbl and get_perf_hist is False) else ""
    )

    if len(date_clause) or len(sub_clause) or len(rown_filter):
        return (
            f"where {' and '.join(filter(len, [date_clause, sub_clause, rown_filter]))}"
        )

    return ""


def get_date_clause(start_date, end_date):
    date_clause = ""
    if start_date:
        date_clause += f"date >= '{start_date.isoformat()}'"
    if end_date:
        delim = " and " if date_clause else ""
        date_clause += f"{delim}date <= '{end_date.isoformat()}'"
    return date_clause


def get_where_subclause(instruments, fund_types):
    sub_clause_parts = []
    if instruments:
        where_str = "','".join(instruments)
        sub_clause_parts.append(f"ticker in ('{where_str}')")
    if fund_types:
        # Build regex pattern to match fund_type values that start with any of the provided prefixes
        regex_pattern = "^(" + "|".join(fund_types) + ")"
        sub_clause_parts.append(f"regexp_matches(fund_type, '{regex_pattern}')")
    if sub_clause_parts:
        return "(" + " and ".join(sub_clause_parts) + ")"
    return ""


def get_variation(values: pd.Series) -> np.float64:
    base = values.iloc[0]  # first element in window iteration
    current = values.iloc[-1]  # last element in window iteration
    return round(100 * (current - base) / base, 2) if base else 0


def create_query(
    table: str,
    start_date: datetime.date = None,
    end_date: datetime.date = None,
    instruments: list[str] = None,
    fund_types: list[str] = None,
    vol_adjust: bool = False,
    show_returns: bool = True,
    returns_cols: list[str] = None,
    cols: list[str] = None,
    get_perf_hist: bool = False,
):
    if instruments is not None:
        instruments = [x.split("/")[0] for x in instruments]

    if table == di.px_tbl:
        cols = di.px_cols
    elif table == di.perf_tbl:
        cols = (
            di.perf_desc_cols_start
            + di.perf_vol_cols
            + di.perf_mavg_cols
            + di.get_perf_cols(
                show_returns=show_returns,
                vol_adjust=vol_adjust,
                returns_cols=returns_cols,
            )
            + di.perf_desc_cols_end
        )

    where_clause_str = gen_where_clause_prices(
        instruments, fund_types, start_date, end_date, table, get_perf_hist
    )
    query = f"""
            select 
                {",".join(cols)}
            from {table}
            {where_clause_str} 
            order by "description" asc, "date" asc
        """
    logging.info(query)
    return query


@st.cache_data
def get_data(query: str, replace_inf=True):
    df = get_conn().execute(query).df()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    if replace_inf:
        df = df.replace([np.inf, -np.inf], np.nan, inplace=False)
    return df


@cache_data
def get_min_date_all() -> datetime.date:
    query = f"select min(date) as min_date from {di.px_tbl}"
    return get_conn().execute(query).fetchall()[0][0]


@cache_data
def get_fund_types() -> list[str]:
    query = f"select distinct fund_type from {di.perf_tbl}"
    return [x[0] for x in get_conn().execute(query).fetchall()]


# ── Shared loaders (used by most dashboard pages) ──

_STANDARD_PERF_COLS = (
    di.perf_desc_cols_start
    + di.perf_z_score_cols
    + di.perf_vol_cols
    + di.perf_mavg_cols
    + di.perf_returns_cols
    + di.perf_desc_cols_end
    + di.perf_rownames_cols
)


@st.cache_data(ttl=300)
def load_latest_perf(tickers: tuple = None, max_rown: int = 1) -> pd.DataFrame:
    """Load performance rows from the perf table.

    Args:
        tickers: Optional *tuple* of tickers to filter (must be hashable for cache).
        max_rown: Keep rows where ``rown <= max_rown`` (1 = latest only).
    """
    where_parts = [f"rown <= {max_rown}"]
    if tickers:
        tickers_str = "','".join(tickers)
        where_parts.append(f"ticker IN ('{tickers_str}')")
    query = f"""
        SELECT {",".join(_STANDARD_PERF_COLS)}
        FROM {di.perf_tbl}
        WHERE {" AND ".join(where_parts)}
        ORDER BY description ASC
    """
    df = get_conn().execute(query).df()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


@st.cache_data(ttl=300)
def load_prices(tickers: tuple = None) -> pd.DataFrame:
    """Load price history for all or specific tickers."""
    where = ""
    if tickers:
        tickers_str = "','".join(tickers)
        where = f"WHERE ticker IN ('{tickers_str}') OR ticker_full IN ('{tickers_str}')"
    query = f"""
        SELECT ticker, ticker_full, date, price, description, fund_type
        FROM {di.px_tbl}
        {where}
        ORDER BY date ASC
    """
    df = get_conn().execute(query).df()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


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


# ── Shared sidebar / filter helpers ──


def fund_type_sidebar(
    default: list[str] | None = None, key: str | None = None
) -> list[str]:
    """Render the fund-type multiselect inline and return the selection."""
    if default is None:
        default = ["eq"]
    return st.multiselect(
        "Fund types",
        options=config.FUND_TYPE_OPTIONS,
        default=default,
        key=key,
    )


def filter_by_fund_type(df: pd.DataFrame, fund_types: list[str]) -> pd.DataFrame:
    """Filter a DataFrame by fund_type using prefix matching. Returns a copy."""
    if fund_types:
        pattern = "^(" + "|".join(fund_types) + ")"
        return df[df["fund_type"].str.match(pattern)].copy()
    return df.copy()


@cache_data(ttl=300)
def get_sparkline_data(tickers: tuple, days: int = 90) -> dict:
    """Return {ticker: [price_list]} for the last `days` trading days.

    The price lists are normalised to start at 100 so sparklines are
    comparable across instruments with different price levels.
    """
    if not tickers:
        return {}
    tickers_str = "','".join(tickers)
    query = f"""
        SELECT ticker, date, price
        FROM {di.px_tbl}
        WHERE ticker IN ('{tickers_str}')
        ORDER BY date ASC
    """
    df = get_conn().execute(query).df()
    if df.empty:
        return {}

    result = {}
    for ticker, group in df.groupby("ticker"):
        tail = group.tail(days)["price"].dropna().tolist()
        if tail:
            base = tail[0] if tail[0] != 0 else 1
            result[ticker] = [round(p / base * 100, 2) for p in tail]
    return result


def add_sparkline_column(
    df: pd.DataFrame,
    col_name: str = "Price (90d)",
    days: int = 90,
) -> pd.DataFrame:
    """Enrich a DataFrame with a sparkline column containing normalised price lists."""
    tickers = tuple(df["ticker"].unique())
    sparklines = get_sparkline_data(tickers, days=days)
    # Use [] for missing tickers so pyarrow encodes this as list<float64>, not string
    df[col_name] = (
        df["ticker"].map(sparklines).apply(lambda x: x if isinstance(x, list) else [])
    )
    return df
