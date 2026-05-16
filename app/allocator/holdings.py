"""Holdings persistence — extends portfolio.db with wrapper-aware schema.

Backwards-compatible with the existing PortfolioManager.py ``holdings`` table.
New columns are nullable so old rows survive. New tables (bucket_targets,
deployment_log, valuation_snapshots) are created on first import.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "portfolio.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def init_schema():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                broker TEXT NOT NULL,
                asset TEXT NOT NULL,
                amount REAL NOT NULL,
                account_type TEXT,
                qty REAL,
                cost_basis_gbp REAL,
                purchase_date TEXT,
                ccy TEXT DEFAULT 'GBP'
            )
        """)
        _add_column_if_missing(conn, "holdings", "account_type", "TEXT")
        _add_column_if_missing(conn, "holdings", "qty", "REAL")
        _add_column_if_missing(conn, "holdings", "cost_basis_gbp", "REAL")
        _add_column_if_missing(conn, "holdings", "purchase_date", "TEXT")
        _add_column_if_missing(conn, "holdings", "ccy", "TEXT DEFAULT 'GBP'")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS bucket_targets (
                account_type TEXT NOT NULL,
                sleeve TEXT NOT NULL,
                target_weight REAL NOT NULL,
                last_updated TEXT,
                PRIMARY KEY (account_type, sleeve)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deployment_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bucket TEXT NOT NULL,
                ticker TEXT NOT NULL,
                amount_gbp REAL NOT NULL,
                trigger_reason TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS valuation_snapshots (
                date TEXT NOT NULL,
                region TEXT NOT NULL,
                forward_pe REAL,
                cape REAL,
                real_yield REAL,
                ma200_ratio REAL,
                PRIMARY KEY (date, region)
            )
        """)
        conn.commit()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col: str, col_type: str):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if col not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def load_holdings(account_type: str | None = None) -> pd.DataFrame:
    with get_db() as conn:
        if account_type:
            df = pd.read_sql_query(
                "SELECT * FROM holdings WHERE account_type = ?",
                conn, params=(account_type,),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM holdings", conn)
    return df


def current_weights(account_type: str | None = None) -> pd.Series:
    df = load_holdings(account_type)
    if df.empty:
        return pd.Series(dtype=float)
    grouped = df.groupby("asset")["amount"].sum()
    total = grouped.sum()
    return grouped / total if total > 0 else grouped


def drift(current: pd.Series, targets: pd.Series, threshold_bps: float = 200) -> pd.DataFrame:
    all_tickers = sorted(set(current.index) | set(targets.index))
    cur = current.reindex(all_tickers, fill_value=0.0)
    tgt = targets.reindex(all_tickers, fill_value=0.0)
    diff = cur - tgt
    diff_bps = diff * 10_000
    out = pd.DataFrame({
        "current_%": cur,
        "target_%": tgt,
        "drift_%": diff,
        "drift_bps": diff_bps,
        "action": ["SELL" if d > 0 else "BUY" if d < 0 else "" for d in diff],
    })
    return out[out["drift_bps"].abs() >= threshold_bps].sort_values("drift_bps", key=abs, ascending=False)


def record_tranche(date: str, bucket: str, ticker: str, amount_gbp: float, reason: str):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO deployment_log (date, bucket, ticker, amount_gbp, trigger_reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (date, bucket, ticker, amount_gbp, reason),
        )
        conn.commit()


def load_deployment_log() -> pd.DataFrame:
    with get_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM deployment_log ORDER BY date DESC", conn,
        )


def upsert_bucket_targets(account_type: str, sleeve_weights: dict[str, float]):
    import datetime
    now = datetime.date.today().isoformat()
    with get_db() as conn:
        for sleeve, weight in sleeve_weights.items():
            conn.execute(
                "INSERT INTO bucket_targets (account_type, sleeve, target_weight, last_updated) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(account_type, sleeve) DO UPDATE SET target_weight=?, last_updated=?",
                (account_type, sleeve, weight, now, weight, now),
            )
        conn.commit()


def load_bucket_targets(account_type: str | None = None) -> pd.DataFrame:
    with get_db() as conn:
        if account_type:
            return pd.read_sql_query(
                "SELECT * FROM bucket_targets WHERE account_type = ?",
                conn, params=(account_type,),
            )
        return pd.read_sql_query("SELECT * FROM bucket_targets", conn)


def save_valuation_snapshot(date: str, region: str, forward_pe: float | None,
                            cape: float | None, real_yield: float | None,
                            ma200_ratio: float | None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO valuation_snapshots (date, region, forward_pe, cape, real_yield, ma200_ratio) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date, region) DO UPDATE SET forward_pe=?, cape=?, real_yield=?, ma200_ratio=?",
            (date, region, forward_pe, cape, real_yield, ma200_ratio,
             forward_pe, cape, real_yield, ma200_ratio),
        )
        conn.commit()


def load_valuation_snapshots() -> pd.DataFrame:
    with get_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM valuation_snapshots ORDER BY date DESC", conn,
        )


init_schema()
