"""Refresh ETF constituent snapshots into DuckDB."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "allocator"))

import data_sources as ds  # noqa: E402
from instruments import INSTRUMENTS  # noqa: E402
from lookthrough import build_constituent_snapshot  # noqa: E402


def main() -> int:
    etf_tickers = sorted(
        ticker for ticker, ins in INSTRUMENTS.items()
        if ins.vehicle_type in {"ucits_etf", "us_etf", "etc"}
    )
    snapshot_df = build_constituent_snapshot(etf_tickers, score_df=None, limit=10)
    count = ds.replace_etf_constituents(snapshot_df)
    print(f"stored_rows={count} etfs={snapshot_df['ETF'].nunique() if not snapshot_df.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
