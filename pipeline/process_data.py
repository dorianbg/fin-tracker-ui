"""
EXPERIMENTAL CODE
"""

import logging
import pandas as pd
from pipeline import utils, consts
import numpy as np
from dataclasses import dataclass


@dataclass
class VolLookback:
    vol_lookback_days: int
    vol_window_days: int
    desc: str


# VOLS_TO_CAPTURE = [5, 10, 21, 42, 63, 94, 126, 252]
VOLS_LOOKBACK_PERIOD = [
    VolLookback(vol_lookback_days=1, vol_window_days=5, desc="vol_daily_over_1w"),
    VolLookback(vol_lookback_days=1, vol_window_days=10, desc="vol_daily_over_2w"),
    VolLookback(vol_lookback_days=1, vol_window_days=21, desc="vol_daily_over_1m"),
    VolLookback(vol_lookback_days=5, vol_window_days=21, desc="vol_weekly_over_1m"),
    VolLookback(vol_lookback_days=5, vol_window_days=63, desc="vol_weekly_over_3m"),
    VolLookback(vol_lookback_days=21, vol_window_days=252, desc="vol_monthly_over_1y"),
]
VOL_TABLE_NAME = "historical_volatility"
TMP_VOL_TABLE_NAME = VOL_TABLE_NAME + "__tmp"


def calc_vol(df_in, jobs: list[VolLookback]):
    dfs = []
    df0 = df_in.set_index("ticker")
    for ticker in list(df0.index.unique()):
        df = df0.loc[ticker, :]
        for job in jobs:
            returns = np.log(df["close"] / df["close"].shift(job.vol_lookback_days))
            returns.fillna(0, inplace=True)
            # compute volatility using Pandas rolling and std methods, the trading days is set to 252 days
            volatility = returns.rolling(window=job.vol_window_days).std() * np.sqrt(
                job.vol_window_days
            )
            df[job.desc] = volatility
        df = df.dropna()
        dfs.append(df.reset_index())
    return pd.concat(dfs, ignore_index=True)


def create_vol_table():
    logging.info("Creating vol table")
    conn = utils.get_duckdb_conn(filepath=consts.db_path)
    sql = f"""
            select close, date, ticker
            from {consts.hist_prices_table_name}
    --         where date > current_timestamp - interval '6 year'
            order by ticker, date asc
    """
    logging.debug(f"Executing query: {sql}")
    df = conn.cursor().execute(sql).fetchdf()
    df_vol = calc_vol(df, jobs=VOLS_LOOKBACK_PERIOD)
    cursor = conn.cursor()

    cursor.register(TMP_VOL_TABLE_NAME, df_vol)
    cursor.execute(
        f"drop table if exists {VOL_TABLE_NAME}; create table {VOL_TABLE_NAME} as select * from {TMP_VOL_TABLE_NAME}"
    )
    cursor.unregister(TMP_VOL_TABLE_NAME)
    conn.close()


if __name__ == "__main__":
    create_vol_table()
