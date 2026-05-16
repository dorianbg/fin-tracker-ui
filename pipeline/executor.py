import argparse
import datetime
import logging
import time
import traceback
from pathlib import Path

import pandas as pd
import yfinance as yf

from pipeline import consts
from pipeline.utils import (
    add_csv_ext,
    add_pickle_ext,
    JobDef,
    get_duckdb_conn,
    missing_timerange,
    insert_df_to_duckdb,
)

date_fmt = "%Y-%m-%d"


def get_transformed_df(job: JobDef) -> pd.DataFrame:
    """Download historical price data from Yahoo Finance for a given job."""
    etf = yf.Ticker(ticker=job.ticker_full)
    etf.tz = "UTC"
    df = etf.history(
        interval="1d",
        start=job.start_date.strftime(date_fmt),
        end=job.end_date.strftime(date_fmt),
    )
    df = df.reset_index()
    df["ticker"] = job.ticker_full.split(".")[0]
    df["ticker_full"] = job.ticker_full
    df.rename(
        columns={x: x.lower().replace(" ", "_") for x in df.columns}, inplace=True
    )
    return df


def _normalize_yfinance_df(df: pd.DataFrame, ticker_full: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.reset_index()
    out["ticker"] = ticker_full.split(".")[0]
    out["ticker_full"] = ticker_full
    out.rename(columns={x: str(x).lower().replace(" ", "_") for x in out.columns}, inplace=True)
    for column in ["dividends", "stock_splits"]:
        if column not in out.columns:
            out[column] = 0.0
    return out


def download_jobs(jobs: list[JobDef]) -> dict[str, pd.DataFrame]:
    """Download all job ranges in one yfinance request and split by ticker/job."""
    if not jobs:
        return {}
    start = min(job.start_date for job in jobs).strftime(date_fmt)
    end = max(job.end_date for job in jobs).strftime(date_fmt)
    tickers = [job.ticker_full for job in jobs]
    raw = yf.download(
        tickers,
        interval="1d",
        start=start,
        end=end,
        actions=True,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {job.ticker_full: pd.DataFrame() for job in jobs}

    out = {}
    for job in jobs:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                ticker_df = raw.xs(job.ticker_full, axis=1, level=1, drop_level=True)
            except KeyError:
                out[job.ticker_full] = pd.DataFrame()
                continue
        else:
            ticker_df = raw

        ticker_df = ticker_df.dropna(how="all")
        if ticker_df.empty:
            out[job.ticker_full] = pd.DataFrame()
            continue
        start_ts = pd.Timestamp(job.start_date).tz_localize(None)
        end_ts = pd.Timestamp(job.end_date).tz_localize(None)
        idx = pd.to_datetime(ticker_df.index).tz_localize(None)
        ticker_df = ticker_df.loc[(idx >= start_ts) & (idx < end_ts)]
        out[job.ticker_full] = _normalize_yfinance_df(ticker_df, job.ticker_full)
    return out


def upload_data_to_postgres(conn):
    """Export latest_performance view to Postgres."""
    import os
    from sqlalchemy import create_engine

    dbname = os.environ["POSTGRES_DB"]
    user = os.environ["PGUSER"]
    password = os.environ.get("PGPASSWORD")
    host = os.environ["POSTGRES_HOST"]
    port = os.environ["POSTGRES_PORT"]

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    )
    table_name = "latest_performance"
    df = conn.sql(f"select * from {table_name}").df()
    df.to_sql(table_name, engine, index=False, if_exists="replace")
    logging.info(f"Inserted {df.shape[0]} rows to Postgres table '{table_name}'")


def check_for_dividends(df: pd.DataFrame) -> bool:
    return not df.empty and (df["dividends"] > 0).any()


def check_existing_data(conn, ticker_full: str) -> bool:
    query = (
        f"select count(*) as cnt "
        f"from {consts.hist_prices_table_name} "
        f"where ticker like '{ticker_full}'"
    )
    res = conn.execute(query=query).fetchdf()
    return res["cnt"][0] > 0


def backup_existing_data(conn, ticker_full: str) -> int:
    import os

    res = conn.execute(
        f"select * from {consts.hist_prices_table_name} "
        f"where ticker_full like '%{ticker_full}%'"
    ).fetchdf()
    min_date = res["date"].min().strftime(date_fmt)
    max_date = res["date"].max().strftime(date_fmt)
    csv_path = add_csv_ext(
        os.path.join(consts.store_raw_dir, ticker_full, f"{min_date}_{max_date}")
    )
    res.to_csv(csv_path)
    backup_count = len(pd.read_csv(filepath_or_buffer=csv_path))
    assert backup_count == len(res)
    conn.execute(
        f"insert into {consts.dividend_tracker_table_name} values "
        f"('{ticker_full}', now(), '{csv_path}')"
    )
    return backup_count


def delete_existing_data(conn, ticker_full: str) -> bool:
    del_q = f"delete from {consts.hist_prices_table_name} where ticker_full = '{ticker_full}'"
    cur = conn.cursor()
    cur.execute("BEGIN TRANSACTION;")
    cur.execute(del_q)
    cur.execute("COMMIT;")
    return True


def execute_job(conn, job: JobDef, args, downloaded_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Download data for a single ticker and handle dividend-triggered rewrites."""
    fmt = "%Y%m%d"
    start_date_str = job.start_date.strftime(fmt)
    end_date_str = job.end_date.strftime(fmt)
    exec_date = datetime.datetime.now(tz=datetime.timezone.utc)

    if start_date_str >= end_date_str or (
        exec_date.strftime(fmt) == start_date_str and exec_date.hour < 20
    ):
        return pd.DataFrame()

    tmp_path = str(
        Path(consts.store_raw_dir)
        / job.ticker_full
        / f"{start_date_str}_{end_date_str}"
    )
    Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)

    new_data = downloaded_data if downloaded_data is not None else get_transformed_df(job=job)

    if args.rewrite_all and args.skip_backup:
        delete_existing_data(conn, ticker_full=job.ticker_full)
        logging.info(f"Deleted existing data for {job}")
    elif check_for_dividends(new_data) or args.rewrite_all:
        if check_existing_data(conn, ticker_full=job.ticker_full):
            logging.info(f"Found dividends for {job}")
            backup_row_count = backup_existing_data(conn, ticker_full=job.ticker_full)
            logging.info(f"Backed up {backup_row_count} rows for {job}")
            if backup_row_count > 0 and delete_existing_data(
                conn, ticker_full=job.ticker_full
            ):
                logging.info(f"Deleted existing data for {job}")
                return pd.DataFrame()

    logging.info(
        f"Downloaded {len(new_data)} rows for {job.ticker_full} ({start_date_str} → {end_date_str})"
    )

    if not new_data.empty:
        csv_path = add_csv_ext(tmp_path)
        pickle_path = add_pickle_ext(tmp_path)
        new_data.to_csv(csv_path)
        new_data.to_pickle(pickle_path)

    if downloaded_data is None:
        time.sleep(3)
    return new_data


def merge_dfs(dfs_to_insert: list[pd.DataFrame]) -> pd.DataFrame | None:
    if not dfs_to_insert:
        return None
    merged = pd.concat(dfs_to_insert, ignore_index=True)
    if not merged.empty:
        merged["date"] = pd.to_datetime(merged["date"], utc=True)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fintracker")
    parser.add_argument("--skip_data_fetch", action=argparse.BooleanOptionalAction)
    parser.add_argument("--rewrite_all", action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_backup", action=argparse.BooleanOptionalAction)
    parser.add_argument("--upload_to_postgres", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    duckdb_conn = get_duckdb_conn(
        filepath=consts.db_path,
        init_cmd=(
            consts.create_table_stmt
            + consts.settings_init_cmd
            + consts.create_instr_ref
            + consts.consts_perf_view
        ),
    )

    missing_prices = missing_timerange(
        duckdb_conn, mark_all_as_missing=args.rewrite_all
    )
    jobs = [
        JobDef(ticker_full=j[0], start_date=j[1], end_date=j[2]) for j in missing_prices
    ]
    dfs_to_insert = []
    current_job = None

    try:
        if not args.skip_data_fetch:
            downloaded = download_jobs(jobs)
            for job in jobs:
                current_job = job
                dfs_to_insert.append(execute_job(duckdb_conn, current_job, args, downloaded.get(job.ticker_full)))
    except Exception as e:
        logging.error(f"Exception {e} with job {current_job}\n{traceback.format_exc()}")
    finally:
        merged_dfs = merge_dfs(dfs_to_insert)
        insert_df_to_duckdb(
            conn=duckdb_conn,
            dataframe=merged_dfs,
            table_name=consts.hist_prices_table_name,
            col_select=consts.hist_prices_col_select,
            dedup=f"EXCEPT select {consts.hist_prices_col_select} from {consts.hist_prices_table_name}",
        )
        duckdb_conn.commit()
        if args.upload_to_postgres:
            upload_data_to_postgres(duckdb_conn)
        duckdb_conn.close()
