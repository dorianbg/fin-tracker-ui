import os

import duckdb

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
duckdb_file = os.path.join(os.path.dirname(__file__), "../fin-tracker", f"duckdb.db")
px_tbl = "prices"
px_pq_file = os.path.join(data_dir, f"{px_tbl}.parquet")
perf_tbl = "performance"
perf_pq_file = os.path.join(data_dir, f"{perf_tbl}.parquet")
encrypt_key = os.environ["PARQUET_ENCRYPTION_KEY"]
add_encrypt_key = f"PRAGMA add_parquet_key('key256', '{encrypt_key}');"
encrypt_conf = "{footer_key: 'key256'}"

cols_prices = ["ticker", "ticker_full", "price", "date", "description"]
cols_perf_desc = ["date", "ticker", "description"]
cols_perf_z_score = ["z_1d"]
cols_perf_num = [
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
    "px_20_dma",
    "px_252_dma",
]
cols_perf = cols_perf_desc + cols_perf_z_score + cols_perf_num

if __name__ == "__main__":
    prices_query = f"""
        select 
            {','.join(cols_prices)}
        from total_return
    """
    performance_query = f"""
        SELECT
            {','.join(cols_perf)}
        FROM latest_performance
        ORDER BY tckr
    """
    export_to_parquet_query = (
        lambda query, output: f"COPY ({query}) TO '{output}' (ENCRYPTION_CONFIG {encrypt_conf});"
    )
    export_to_parquet_unencrypted_query = (
        lambda query, output: f"COPY ( SELECT * FROM ({query}) LIMIT 10) TO '{output}' ;"
    )

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
