import os
import shutil
import time

import duckdb

import data

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
duckdb_file = os.path.join(os.path.dirname(__file__), "../fin-tracker", f"duckdb.db")

encrypt_key = os.environ["PARQUET_ENCRYPTION_KEY"]
add_encrypt_key = f"PRAGMA add_parquet_key('key256', '{encrypt_key}');"
encrypt_conf = "{footer_key: 'key256'}"

px_src_tbl_name = "total_return"
px_tbl = "prices"
px_pq_file = os.path.join(data_dir, f"{px_tbl}.parquet")
px_cols = ["ticker", "ticker_full", "price", "date", "description", "fund_type"]

perf_src_table_name = "latest_performance_sharpe"
perf_tbl = "performance"
perf_pq_file = os.path.join(data_dir, f"{perf_tbl}.parquet")
perf_desc_cols = ["date", "ticker", "description", "fund_type"]
perf_z_score_cols = ["z_1d"]
vol_1y_col = "vol_1y"
perf_vol_cols = ["vol_1d", vol_1y_col]
perf_returns_cols = [
    "r_1d",
    "r_1w",
    "r_2w",
    "r_1mo",
    "r_3mo",
    "r_6mo",
    "r_1y",
    "r_2y",
    "r_3y",
    "r_5y",
]
perf_sharpe_cols = [
    "r_1d_s",
    "r_1w_s",
    "r_2w_s",
    "r_1mo_s",
    "r_3mo_s",
    "r_6mo_s",
    "r_1y_s",
    "r_2y_s",
    "r_3y_s",
    "r_5y_s",
]
selectable_returns = ["1d", "1w", "2w", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y"]
default_selected_returns = ["1d", "1w", "2w", "1mo", "3mo", "6mo", "1y", "3y", "5y"]


def check_if_in_return_cols(col, returns_cols):
    return any([True for x in returns_cols if x in col])


def get_perf_cols(show_returns, vol_adjust, returns_cols):
    _cols = []
    for i in range(len(perf_returns_cols)):
        if show_returns:
            col = perf_returns_cols[i]
            if check_if_in_return_cols(col, returns_cols):
                _cols.append(col)
        if vol_adjust:
            col = perf_sharpe_cols[i]
            if check_if_in_return_cols(col, returns_cols):
                _cols.append(col)
    return _cols


perf_mavg_cols = [
    "px_21_dma",
    "px_63_dma",
    "px_252_dma",
]
perf_cols = (
    perf_desc_cols
    + perf_z_score_cols
    + perf_vol_cols
    + perf_returns_cols
    + perf_sharpe_cols
    + perf_mavg_cols
)


def run():
    prices_query = f"""
        select 
            {','.join(px_cols)}
        from {px_src_tbl_name}
    """
    performance_query = f"""
        SELECT
            {','.join(perf_cols)}
        FROM {perf_src_table_name}
        ORDER BY ticker
    """
    export_to_parquet_query = (
        lambda query, output: f"COPY ({query}) TO '{output}' (ENCRYPTION_CONFIG {encrypt_conf});"
    )
    export_to_parquet_unencrypted_query = (
        lambda query, output: f"COPY ( SELECT * FROM ({query}) LIMIT 10) TO '{output}' ;"
    )
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    while os.path.exists(data_dir):
        time.sleep(0.05)

    os.makedirs(data_dir)

    with duckdb.connect(database=duckdb_file, read_only=False) as conn:
        conn.execute(add_encrypt_key)
        conn.execute(export_to_parquet_query(query=prices_query, output=px_pq_file))
        conn.execute(
            export_to_parquet_unencrypted_query(
                query=prices_query,
                output=px_pq_file.replace(".parquet", "_unencrypted.parquet"),
            )
        ),
        conn.execute(
            export_to_parquet_query(query=performance_query, output=perf_pq_file)
        )
        conn.execute(
            export_to_parquet_unencrypted_query(
                query=performance_query,
                output=perf_pq_file.replace(".parquet", "_unencrypted.parquet"),
            )
        ),


if __name__ == "__main__":
    run()
