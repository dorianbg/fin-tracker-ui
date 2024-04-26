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

px_tbl = "prices"
px_pq_file = os.path.join(data_dir, f"{px_tbl}.parquet")
px_cols = ["ticker", "ticker_full", "price", "date", "description", "fund_type"]
px_src_tbl_name = "total_return"

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


def get_perf_cols(hide_returns, show_vol_adjusted, styling):
    _cols = []
    for col in perf_returns_cols:
        if not hide_returns:
            _cols.append(col)
        if show_vol_adjusted:
            c2 = col
            select_prefix = "" if styling else f"{c2}/{vol_1y_col} as "
            res = f"{select_prefix}{c2}{data.sharpe_col_suffix}"
            _cols.append(res)
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
    + perf_mavg_cols
)
perf_src_table_name = "latest_performance"


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
