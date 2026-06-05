from types import SimpleNamespace

import duckdb
import pandas as pd

from pipeline import consts
from pipeline import executor
from pipeline.utils import JobDef


def _conn():
    conn = duckdb.connect(":memory:")
    conn.execute(consts.create_table_stmt)
    return conn


def _job():
    return JobDef(
        ticker_full="TEST",
        start_date=pd.Timestamp("2026-01-01").to_pydatetime(),
        end_date=pd.Timestamp("2026-01-10").to_pydatetime(),
    )


def _downloaded(dividend: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["TEST"],
            "ticker_full": ["TEST"],
            "date": [pd.Timestamp("2026-01-09")],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100],
            "dividends": [dividend],
            "stock_splits": [0.0],
        }
    )


def _insert_existing(conn):
    conn.execute(
        """
        insert into historical_prices values
        ('TEST', 'TEST', '2025-01-01', 1, 1, 1, 1, 100, 0, 0)
        """
    )


def test_dividend_rewrite_keeps_existing_rows_when_full_reload_is_empty(monkeypatch):
    conn = _conn()
    _insert_existing(conn)
    monkeypatch.setattr(executor, "_download_batch", lambda *args, **kwargs: {})

    result = executor.execute_job(
        conn,
        _job(),
        SimpleNamespace(rewrite_all=False, skip_backup=False),
        _downloaded(dividend=1.0),
    )

    assert result.empty
    assert conn.execute("select count(*) from historical_prices").fetchone()[0] == 1


def test_dividend_rewrite_deletes_only_after_full_reload_has_data(monkeypatch):
    conn = _conn()
    _insert_existing(conn)
    replacement = _downloaded(dividend=0.0)
    monkeypatch.setattr(
        executor, "_download_batch", lambda *args, **kwargs: {"TEST": replacement}
    )

    result = executor.execute_job(
        conn,
        _job(),
        SimpleNamespace(rewrite_all=False, skip_backup=False),
        _downloaded(dividend=1.0),
    )

    assert len(result) == 1
    assert conn.execute("select count(*) from historical_prices").fetchone()[0] == 0
    assert (
        conn.execute("select backup_save_path from data_backups").fetchone()[0]
        == "duckdb-only"
    )
