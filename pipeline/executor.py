import argparse
import datetime
import logging
import time
import traceback

import pandas as pd
import yfinance as yf

from pipeline import consts
from pipeline.utils import (
    JobDef,
    get_duckdb_conn,
    missing_timerange,
    insert_df_to_duckdb,
    setup_logging,
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
    out.rename(
        columns={x: str(x).lower().replace(" ", "_") for x in out.columns}, inplace=True
    )
    for column in ["dividends", "stock_splits"]:
        if column not in out.columns:
            out[column] = 0.0
    out = repair_invalid_ohlc(out)
    out = filter_invalid_price_rows(out)
    return out


def repair_invalid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zero OHLC fields with close when close is valid."""
    if df.empty or "close" not in df.columns:
        return df
    out = df.copy()
    valid_close = out["close"].notna() & (out["close"] > 0)
    for column in ["open", "high", "low"]:
        if column in out.columns:
            invalid = valid_close & (out[column].isna() | (out[column] <= 0))
            out.loc[invalid, column] = out.loc[invalid, "close"]
    return out


def filter_invalid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop yfinance placeholder rows that do not contain a usable close price."""
    if df.empty or "close" not in df.columns:
        return df
    return df[df["close"].notna() & (df["close"] > 0)].copy()


BATCH_SIZE = 50


def _download_batch(
    tickers: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Download one batch of tickers, returning {ticker_full: DataFrame}."""
    if not tickers:
        return {}
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
        return {t: pd.DataFrame() for t in tickers}

    out = {}
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                ticker_df = raw.xs(t, axis=1, level=1, drop_level=True)
            except KeyError:
                out[t] = pd.DataFrame()
                continue
        else:
            ticker_df = raw
        ticker_df = ticker_df.dropna(how="all")
        if ticker_df.empty:
            out[t] = pd.DataFrame()
        else:
            idx = pd.to_datetime(ticker_df.index).tz_localize(None)
            ticker_df = ticker_df.loc[
                (idx >= pd.Timestamp(start).tz_localize(None))
                & (idx < pd.Timestamp(end).tz_localize(None))
            ]
            out[t] = _normalize_yfinance_df(ticker_df, t)
    return out


def download_jobs(jobs: list[JobDef]) -> dict[str, pd.DataFrame]:
    """Download all job ranges in small yfinance batches and split by ticker/job."""
    if not jobs:
        return {}
    start = min(job.start_date for job in jobs).strftime(date_fmt)
    end = max(job.end_date for job in jobs).strftime(date_fmt)
    tickers = [job.ticker_full for job in jobs]

    result = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        result.update(_download_batch(batch, start, end))
        if i + BATCH_SIZE < len(tickers):
            time.sleep(1)
    return result


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
    query = f"select count(*) as cnt from {consts.hist_prices_table_name} where ticker_full = ?"
    res = conn.execute(query, [ticker_full]).fetchdf()
    return res["cnt"][0] > 0


def full_rewrite_job(job: JobDef) -> JobDef:
    return JobDef(
        ticker_full=job.ticker_full,
        start_date=job.end_date - datetime.timedelta(days=15 * 366),
        end_date=job.end_date,
    )


def backup_existing_data(conn, ticker_full: str) -> int:
    query = f"select count(*) as cnt from {consts.hist_prices_table_name} where ticker_full = ?"
    res = conn.execute(query, [ticker_full]).fetchdf()
    backup_count = int(res["cnt"][0])
    conn.execute(
        f"insert into {consts.dividend_tracker_table_name} values (?, now(), 'duckdb-only')",
        [ticker_full],
    )
    return backup_count


def delete_existing_data(conn, ticker_full: str) -> bool:
    del_q = f"delete from {consts.hist_prices_table_name} where ticker_full = ?"
    cur = conn.cursor()
    cur.execute("BEGIN TRANSACTION;")
    cur.execute(del_q, [ticker_full])
    cur.execute("COMMIT;")
    return True


def download_full_rewrite(job: JobDef) -> pd.DataFrame:
    rewrite_job = full_rewrite_job(job)
    logging.info(f"Reloading full history for {rewrite_job}")
    return _download_batch(
        [rewrite_job.ticker_full],
        rewrite_job.start_date.strftime(date_fmt),
        rewrite_job.end_date.strftime(date_fmt),
    ).get(rewrite_job.ticker_full, pd.DataFrame())


def execute_job(
    conn, job: JobDef, args, downloaded_data: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Download data for a single ticker and handle dividend-triggered rewrites."""
    fmt = "%Y%m%d"
    start_date_str = job.start_date.strftime(fmt)
    end_date_str = job.end_date.strftime(fmt)
    exec_date = datetime.datetime.now(tz=datetime.timezone.utc)

    if start_date_str >= end_date_str or (
        exec_date.strftime(fmt) == start_date_str and exec_date.hour < 20
    ):
        return pd.DataFrame()

    new_data = (
        downloaded_data if downloaded_data is not None else get_transformed_df(job=job)
    )

    needs_rewrite = check_for_dividends(new_data) or args.rewrite_all
    if needs_rewrite and new_data.empty:
        logging.warning(f"Skipping rewrite for {job}: replacement download is empty")
        return pd.DataFrame()

    if args.rewrite_all and args.skip_backup:
        if check_existing_data(conn, ticker_full=job.ticker_full):
            delete_existing_data(conn, ticker_full=job.ticker_full)
            logging.info(f"Deleted existing data for {job}")
    elif needs_rewrite:
        if check_existing_data(conn, ticker_full=job.ticker_full):
            logging.info(f"Found dividends for {job}")
            if not args.rewrite_all:
                replacement = download_full_rewrite(job)
                if replacement.empty:
                    logging.warning(
                        f"Skipping dividend rewrite for {job}: full-history replacement is empty"
                    )
                    return pd.DataFrame()
                new_data = replacement
            backup_row_count = backup_existing_data(conn, ticker_full=job.ticker_full)
            logging.info(f"Recorded {backup_row_count} existing DuckDB rows for {job}")
            if backup_row_count > 0 and delete_existing_data(
                conn, ticker_full=job.ticker_full
            ):
                logging.info(f"Deleted existing data for {job}")

    logging.info(
        f"Downloaded {len(new_data)} rows for {job.ticker_full} ({start_date_str} → {end_date_str})"
    )

    if not new_data.empty:
        new_data = repair_invalid_ohlc(new_data)
        new_data = filter_invalid_price_rows(new_data)

    if downloaded_data is None:
        time.sleep(3)
    return new_data


def merge_dfs(dfs_to_insert: list[pd.DataFrame]) -> pd.DataFrame | None:
    if not dfs_to_insert:
        return None
    merged = pd.concat(dfs_to_insert, ignore_index=True)
    if not merged.empty:
        merged["date"] = pd.to_datetime(merged["date"], utc=True)
        merged = filter_invalid_price_rows(merged)
        merged = merged.sort_values("volume", ascending=False, na_position="last")
        merged = merged.drop_duplicates(["ticker_full", "date"], keep="first")
    return merged


def main() -> None:
    setup_logging()
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
    logging.info(
        "Pipeline run starting: jobs=%s rewrite_all=%s skip_data_fetch=%s upload_to_postgres=%s",
        len(jobs),
        args.rewrite_all,
        args.skip_data_fetch,
        args.upload_to_postgres,
    )
    dfs_to_insert = []
    current_job = None

    try:
        if not args.skip_data_fetch:
            downloaded = download_jobs(jobs)
            for job in jobs:
                current_job = job
                dfs_to_insert.append(
                    execute_job(
                        duckdb_conn, current_job, args, downloaded.get(job.ticker_full)
                    )
                )
    except Exception as e:
        logging.error(f"Exception {e} with job {current_job}\n{traceback.format_exc()}")
    finally:
        merged_dfs = merge_dfs(dfs_to_insert)
        rows_to_insert = 0 if merged_dfs is None else len(merged_dfs)
        logging.info("Pipeline downloaded rows ready to insert: %s", rows_to_insert)
        insert_df_to_duckdb(
            conn=duckdb_conn,
            dataframe=merged_dfs,
            table_name=consts.hist_prices_table_name,
            col_select=consts.hist_prices_col_select,
            dedup=f"EXCEPT select {consts.hist_prices_col_select} from {consts.hist_prices_table_name}",
        )
        logging.info("Refreshing latest performance snapshot")
        duckdb_conn.execute(consts.refresh_performance_snapshot_stmt)
        duckdb_conn.commit()
        if args.upload_to_postgres:
            upload_data_to_postgres(duckdb_conn)
        duckdb_conn.close()
        logging.info("Pipeline run finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Pipeline run failed")
        raise
