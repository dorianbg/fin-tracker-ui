import os
import pathlib

lookback_period = "15 years"
timezone = "Europe/Paris"

# returns the repository root path - depends on location of this file in repo
file_parent_dir = pathlib.Path(__file__).parent.parent.resolve()
db_path = os.path.join(file_parent_dir, "duckdb.db")
store_raw_dir = os.path.join(file_parent_dir, "storage", "raw")

settings_init_cmd = "PRAGMA enable_checkpoint_on_shutdown; "
time_format = "%Y%m%d-%H:%M:%S"
csv_ext = ".csv"
pickle_ext = ".pickle"

hist_prices_table_name = "historical_prices"
hist_prices_cols_to_datatype = {
    "ticker": "VARCHAR",
    "ticker_full": "VARCHAR",
    '"date"': "TIMESTAMPTZ",
    '"open"': "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    '"close"': "DOUBLE",
    "volume": "BIGINT",
    "dividends": "DOUBLE",
    "stock_splits": "DOUBLE",
}
hist_prices_col_select = ",".join(hist_prices_cols_to_datatype.keys())

dividend_tracker_table_name = "data_backups"
dividend_tracker_cols_to_datatype = {
    "ticker_full": "VARCHAR",
    '"create_timestamp"': "TIMESTAMPTZ",
    "backup_save_path": "VARCHAR",
}

create_table_stmt = f"""
    create table if not exists {hist_prices_table_name} (
        {",".join([k + " " + v for (k, v) in hist_prices_cols_to_datatype.items()])}
    );
    create table if not exists {dividend_tracker_table_name} (
        {",".join([k + " " + v for (k, v) in dividend_tracker_cols_to_datatype.items()])}
    );
"""

inst_info_file = os.path.join(file_parent_dir, "resources", "instrument_info.csv")
columns = {
    "ticker": "VARCHAR",
    "description": "VARCHAR",
    "currency": "VARCHAR",
    "fund_type": "VARCHAR",
    "url": "VARCHAR",
    "sector": "VARCHAR",
}

create_instr_ref = f"""
drop table if exists ticker_ref;
create table ticker_ref as 
select {",".join(columns.keys())}, ticker as ticker_full
from read_csv('{inst_info_file}', delim = ',', header = true, columns = {columns}); 
"""

with open(os.path.join(file_parent_dir, "resources", "latest_performance.sql")) as file:
    consts_perf_view = file.read()
