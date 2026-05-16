"""SQLite wrapper for the allocator's own holdings + targets + deployment log.

Independent from fin-tracker-ui's `portfolio.db`. The allocator app owns its
own database file (`allocator/allocator.db`) so the two tools don't stomp on
each other's schemas.

Schema:
    holdings         — current positions per (account_type, ticker)
    bucket_targets   — last computed target weight per (account_type, sleeve)
    deployment_log   — history of executed deployment tranches
    valuation_snaps  — last observed region fundamentals (for reproducibility)
"""

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date

_DB_PATH = os.path.join(os.path.dirname(__file__), "allocator.db")


@contextmanager
def get_db():
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS holdings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                account_type  TEXT NOT NULL,   -- 'SIPP' | 'ISA' | 'GIA'
                ticker        TEXT NOT NULL,
                qty           REAL NOT NULL,
                cost_basis_gbp REAL,
                purchase_date TEXT,
                ccy           TEXT,
                UNIQUE(account_type, ticker)
            );

            CREATE TABLE IF NOT EXISTS bucket_targets (
                account_type  TEXT NOT NULL,
                sleeve        TEXT NOT NULL,
                target_weight REAL NOT NULL,
                last_updated  TEXT NOT NULL,
                PRIMARY KEY (account_type, sleeve)
            );

            CREATE TABLE IF NOT EXISTS deployment_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT NOT NULL,
                account_type   TEXT NOT NULL,
                ticker         TEXT NOT NULL,
                amount_gbp     REAL NOT NULL,
                trigger_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS valuation_snaps (
                date          TEXT NOT NULL,
                region        TEXT NOT NULL,
                forward_pe    REAL,
                cape          REAL,
                real_yield    REAL,
                ma200_ratio   REAL,
                PRIMARY KEY (date, region)
            );

            CREATE TABLE IF NOT EXISTS bucket_cash (
                account_type  TEXT PRIMARY KEY,
                gbp_cash      REAL NOT NULL DEFAULT 0
            );
        """)
        conn.commit()


# ── Holdings ──────────────────────────────────────────────────────────
@dataclass
class Holding:
    id: int
    account_type: str
    ticker: str
    qty: float
    cost_basis_gbp: float | None
    purchase_date: str | None
    ccy: str | None


def upsert_holding(
    account_type: str,
    ticker: str,
    qty: float,
    cost_basis_gbp: float | None = None,
    purchase_date: str | None = None,
    ccy: str | None = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO holdings (account_type, ticker, qty, cost_basis_gbp, purchase_date, ccy)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_type, ticker) DO UPDATE SET
                qty            = excluded.qty,
                cost_basis_gbp = excluded.cost_basis_gbp,
                purchase_date  = excluded.purchase_date,
                ccy            = excluded.ccy
            """,
            (account_type, ticker.upper(), qty, cost_basis_gbp, purchase_date, ccy),
        )
        conn.commit()


def delete_holding(holding_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM holdings WHERE id = ?", (holding_id,))
        conn.commit()


def fetch_holdings(account_type: str | None = None) -> list[Holding]:
    with get_db() as conn:
        if account_type:
            rows = conn.execute(
                "SELECT id, account_type, ticker, qty, cost_basis_gbp, purchase_date, ccy "
                "FROM holdings WHERE account_type = ? ORDER BY ticker",
                (account_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, account_type, ticker, qty, cost_basis_gbp, purchase_date, ccy "
                "FROM holdings ORDER BY account_type, ticker"
            ).fetchall()
        return [Holding(*r) for r in rows]


# ── Bucket targets ────────────────────────────────────────────────────
def save_targets(account_type: str, targets: dict[str, float]) -> None:
    today = date.today().isoformat()
    with get_db() as conn:
        for sleeve, weight in targets.items():
            conn.execute(
                """
                INSERT INTO bucket_targets (account_type, sleeve, target_weight, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(account_type, sleeve) DO UPDATE SET
                    target_weight = excluded.target_weight,
                    last_updated  = excluded.last_updated
                """,
                (account_type, sleeve, weight, today),
            )
        conn.commit()


def load_targets(account_type: str) -> dict[str, float]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT sleeve, target_weight FROM bucket_targets WHERE account_type = ?",
            (account_type,),
        ).fetchall()
        return {s: w for s, w in rows}


# ── Deployment log ────────────────────────────────────────────────────
def record_tranche(
    account_type: str,
    ticker: str,
    amount_gbp: float,
    trigger_reason: str,
    when: str | None = None,
) -> None:
    when = when or date.today().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO deployment_log (date, account_type, ticker, amount_gbp, trigger_reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (when, account_type, ticker, amount_gbp, trigger_reason),
        )
        conn.commit()


def fetch_deployment_log(account_type: str | None = None) -> list[tuple]:
    with get_db() as conn:
        if account_type:
            return conn.execute(
                "SELECT date, account_type, ticker, amount_gbp, trigger_reason "
                "FROM deployment_log WHERE account_type = ? ORDER BY date DESC",
                (account_type,),
            ).fetchall()
        return conn.execute(
            "SELECT date, account_type, ticker, amount_gbp, trigger_reason "
            "FROM deployment_log ORDER BY date DESC"
        ).fetchall()


# ── Cash per bucket ───────────────────────────────────────────────────
def set_cash(account_type: str, gbp_cash: float) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO bucket_cash (account_type, gbp_cash) VALUES (?, ?)
            ON CONFLICT(account_type) DO UPDATE SET gbp_cash = excluded.gbp_cash
            """,
            (account_type, gbp_cash),
        )
        conn.commit()


def get_cash(account_type: str) -> float:
    with get_db() as conn:
        row = conn.execute(
            "SELECT gbp_cash FROM bucket_cash WHERE account_type = ?",
            (account_type,),
        ).fetchone()
        return float(row[0]) if row else 0.0
