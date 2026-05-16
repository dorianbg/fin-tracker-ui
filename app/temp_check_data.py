import os
import sys
import duckdb
import pandas as pd

# Ensure we can import local modules
sys.path.append(os.getcwd())

import duckdb_importer as di

print("Checking prices data...")
conn = duckdb.connect()
conn.execute(f"{di.add_encrypt_key}")

try:
    query = f"SELECT ticker, COUNT(date) as count, MIN(date) as min_d, MAX(date) as max_d FROM read_parquet('{di.px_pq_file}', encryption_config = {di.encrypt_conf}) GROUP BY ticker LIMIT 10"
    df = conn.execute(query).df()
    print(df)

    # Check granularity for one ticker
    ticker = df.iloc[0]["ticker"]
    query_daily = f"SELECT date FROM read_parquet('{di.px_pq_file}', encryption_config = {di.encrypt_conf}) WHERE ticker = '{ticker}' ORDER BY date LIMIT 10"
    df_daily = conn.execute(query_daily).df()
    print(f"\nSample dates for {ticker}:")
    print(df_daily)
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()
