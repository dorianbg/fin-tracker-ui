"""Seed the allocator DB with a realistic starter portfolio.

Purpose:
- populate SIPP / ISA / GIA with a plausible initial allocation
- exercise ETF lookthrough and duplicate-overlap analysis
- keep the seed repeatable and easy to reset
"""

from __future__ import annotations

import os
import sqlite3
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "allocator"))

import data_sources as ds  # noqa: E402
import holdings as h  # noqa: E402
from instruments import lookup  # noqa: E402

SIPP_TOTAL = 300_000.0
ISA_TOTAL = 150_000.0
GIA_TOTAL = 100_000.0

SEED_NOTIONALS = {
    # Strategy-aligned fully invested seed.
    # No approved cash sleeves; any residual cash should only be rounding noise.
    "SIPP": {
        "CSPX": 30_000.0,
        "IWQU": 30_000.0,
        "MVOL": 18_000.0,
        "IWVL": 24_000.0,
        "EIMI": 30_000.0,
        "IJPA": 15_000.0,
        "VEUR": 24_000.0,
        "ISF": 15_000.0,
        "NATO": 9_000.0,
        "SGLN": 60_000.0,
        "IGIL": 30_000.0,
        "SEML": 15_000.0,
    },
    "ISA": {
        "IWDA": 22_500.0,
        "MVOL": 30_000.0,
        "IWQU": 15_000.0,
        "IWVL": 15_000.0,
        "EIMI": 15_000.0,
        "IJPA": 7_500.0,
        "VEUR": 7_500.0,
        "IGIL": 15_000.0,
        "SGLN": 22_500.0,
    },
    "GIA": {
        "IAU": 24_000.0,
        "GDX": 8_000.0,
        "PDBC": 16_000.0,
        "XLE": 10_000.0,
        "IGF": 12_000.0,
        "VNQ": 8_000.0,
        "TIP": 12_000.0,
        "EMLC": 10_000.0,
    },
}


def _price_to_gbp(last_price: float, ccy: str) -> float:
    gbpusd = ds.get_gbpusd()
    price = float(last_price)
    if ccy == "GBP" and price > 200:
        price = price / 100.0
    if ccy == "USD":
        price = price / gbpusd
    return price


def _reset_db() -> None:
    h.init_db()
    conn = sqlite3.connect(os.path.join(REPO_ROOT, "allocator", "allocator.db"))
    conn.executescript(
        """
        DELETE FROM holdings;
        DELETE FROM bucket_cash;
        DELETE FROM deployment_log;
        DELETE FROM bucket_targets;
        """
    )
    conn.commit()
    conn.close()


def main() -> int:
    all_tickers = sorted({ticker for bucket in SEED_NOTIONALS.values() for ticker in bucket})
    ds.upsert_live_prices_for_tickers(all_tickers)
    meta = ds.get_etf_meta().set_index("ticker")

    _reset_db()

    for account_type, positions in SEED_NOTIONALS.items():
        invested = 0.0
        for ticker, notional_gbp in positions.items():
            ins = lookup(ticker)
            if ins is None:
                raise RuntimeError(f"Unknown ticker in seed: {ticker}")
            if ticker not in meta.index:
                raise RuntimeError(f"No price available for seed ticker: {ticker}")
            last_price = float(meta.loc[ticker, "last_price"])
            unit_price_gbp = _price_to_gbp(last_price, ins.ccy)
            qty = round(notional_gbp / unit_price_gbp, 6)
            h.upsert_holding(
                account_type=account_type,
                ticker=ticker,
                qty=qty,
                ccy=ins.ccy,
            )
            invested += qty * unit_price_gbp

        total = {"SIPP": SIPP_TOTAL, "ISA": ISA_TOTAL, "GIA": GIA_TOTAL}[account_type]
        h.set_cash(account_type, max(0.0, total - invested))

    print("seeded allocator portfolio")
    for account_type, total in [("SIPP", SIPP_TOTAL), ("ISA", ISA_TOTAL), ("GIA", GIA_TOTAL)]:
        invested = sum(SEED_NOTIONALS[account_type].values())
        print(account_type, "invested_gbp", invested, "cash_gbp", total - invested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
