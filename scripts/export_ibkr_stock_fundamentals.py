"""Export IBKR Reuters fundamentals for all configured stock instruments."""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "allocator"))

from ibkr_data import export_ibkr_fundamentals_csv, upsert_ibkr_fundamentals_duckdb  # noqa: E402
from instruments import INSTRUMENTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--client-id", type=int, default=151)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "allocator", "ibkr_fundamentals.csv"),
    )
    parser.add_argument(
        "--duckdb",
        default=os.path.join(REPO_ROOT, "allocator", "allocator_cache.duckdb"),
    )
    parser.add_argument(
        "--format",
        choices=["duckdb", "csv", "both"],
        default="duckdb",
    )
    args = parser.parse_args()

    tickers = sorted(t for t, ins in INSTRUMENTS.items() if ins.vehicle_type == "stock")
    count = 0
    if args.format in {"duckdb", "both"}:
        count = upsert_ibkr_fundamentals_duckdb(
            tickers,
            args.duckdb,
            port=args.port,
            client_id=args.client_id,
        )
        print(f"upserted={count} duckdb={args.duckdb}")
    if args.format in {"csv", "both"}:
        csv_count = export_ibkr_fundamentals_csv(
            tickers,
            args.output,
            port=args.port,
            client_id=args.client_id + 1,
        )
        count = max(count, csv_count)
        print(f"exported={csv_count} csv={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
