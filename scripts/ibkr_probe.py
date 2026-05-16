"""Minimal IBKR connectivity probe.

Usage:
    uv run python scripts/ibkr_probe.py
    uv run python scripts/ibkr_probe.py --port 7496 --symbol SPY
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=91)
    parser.add_argument("--symbol", default="SPY")
    args = parser.parse_args()

    try:
        from ib_insync import IB, Stock
    except Exception as exc:
        print(f"ib_insync import failed: {exc}")
        return 2

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=5)
    except Exception as exc:
        print(f"Connection failed to {args.host}:{args.port}: {exc}")
        return 1

    try:
        accounts = ib.managedAccounts()
        print(f"Connected: {args.host}:{args.port}")
        print(f"Accounts: {accounts}")

        contract = Stock(args.symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        print(f"Qualified contracts for {args.symbol}: {len(qualified)}")

        details = ib.reqContractDetails(contract)
        if details:
            d = details[0]
            print(
                "Contract details:",
                {
                    "symbol": d.contract.symbol,
                    "secType": d.contract.secType,
                    "exchange": d.contract.exchange,
                    "currency": d.contract.currency,
                    "longName": d.longName,
                },
            )
        else:
            print("No contract details returned")

        try:
            xml = ib.reqFundamentalData(contract, "ReportSnapshot")
            print(f"Fundamental snapshot returned: {bool(xml)}")
            if xml:
                print(xml[:500])
        except Exception as exc:
            print(f"Fundamental data request failed: {exc}")
    finally:
        ib.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
