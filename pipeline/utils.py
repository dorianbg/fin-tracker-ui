import datetime
import logging
import logging.handlers
import os
import random
import string
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler

import duckdb
import pandas as pd

from pipeline import consts

_conns: dict[str, duckdb.DuckDBPyConnection] = {}


def add_csv_ext(path: str) -> str:
    return path if path.endswith(consts.csv_ext) else path + consts.csv_ext


def add_pickle_ext(path: str) -> str:
    return path if path.endswith(consts.pickle_ext) else path + consts.pickle_ext


def is_business_day(date) -> bool:
    return bool(len(pd.bdate_range(date, date)))


@dataclass
class JobDef:
    ticker_full: str
    start_date: datetime.datetime
    end_date: datetime.datetime

    def __post_init__(self):
        while not is_business_day(self.start_date):
            self.start_date += datetime.timedelta(days=1)


def get_duckdb_conn(
    filepath: str, init_cmd: str = "", **kwargs
) -> duckdb.DuckDBPyConnection:
    if filepath in _conns:
        return _conns[filepath]

    conn = duckdb.connect(database=filepath, **kwargs)
    if init_cmd:
        logging.info(f"Init db with {init_cmd}")
        conn.cursor().execute(init_cmd)
    _conns[filepath] = conn
    return conn


def insert_df_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    dataframe: pd.DataFrame | None,
    table_name: str,
    col_select: str,
    dedup: str = "",
):
    if dataframe is None or dataframe.empty:
        return

    dataframe = dataframe.copy()
    for column in dataframe.select_dtypes(include="string").columns:
        dataframe[column] = dataframe[column].astype(object)

    view_name = "temp_df_" + "".join(random.choices(string.ascii_uppercase, k=10))
    cursor = conn.cursor()
    cursor.register(view_name, dataframe)
    stmt = f"insert into {table_name} (select {col_select} from {view_name} {dedup}) "
    logging.info(f"Inserting {len(dataframe)} rows to {table_name}")
    cursor.execute(stmt)
    cursor.unregister(view_name)
    cursor.commit()


def missing_timerange(
    conn: duckdb.DuckDBPyConnection, mark_all_as_missing: bool
) -> list:
    remove_join = "and 1 = 0" if mark_all_as_missing else ""
    query = f"""
        select
            t.ticker_full as ticker,
            coalesce(h.latest_date, date_trunc('day', get_current_timestamp() at time zone '{consts.timezone}' - interval '{consts.lookback_period}')) + interval '1 day' as start_date,
            date_trunc('day',get_current_timestamp() at time zone '{consts.timezone}') + interval '1 day' as end_date
        from ticker_ref t left join (
            select
                date_trunc('day', max("date" at time zone '{consts.timezone}')) as latest_date,
                ticker_full
            from historical_prices h
            group by ticker_full
        ) as h on h.ticker_full = t.ticker_full {remove_join}
    """
    logging.info(query)
    return conn.cursor().execute(query).fetchall()


def setup_logging():
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, "finance.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_format = "%(asctime)s|%(levelname)s|%(module)s|%(funcName)s|%(process)s|%(lineno)d|%(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=8)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
