import os
import shutil

import duckdb

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
perf_vol_cols = ["vol_1d", "vol_1y"]
perf_num_cols = [
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
    "px_21_dma",
    "px_63_dma",
    "px_252_dma",
]
perf_cols = perf_desc_cols + perf_z_score_cols + perf_vol_cols + perf_num_cols
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
