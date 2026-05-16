"""IBKR Gateway/TWS integration for stock fundamentals.

The practical use-case is direct-share screening for SIPP/GIA:
- qualify UK / EU / US stock contracts
- pull Reuters `ReportSnapshot`
- extract a small stable fundamentals subset
- optionally write a local CSV for allocator/data_sources.py to ingest

ETF Reuters coverage through IBKR is inconsistent, so this module is focused on
single stocks.
"""

from __future__ import annotations

import csv
import logging
import os
import xml.etree.ElementTree as ET
from typing import Iterable

import duckdb

try:
    from ib_insync import IB, Stock
except ImportError:  # pragma: no cover
    IB = None
    Stock = None

from instruments import lookup

log = logging.getLogger(__name__)


def _maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _find_ratio(root: ET.Element, field_name: str) -> float | None:
    node = root.find(f".//Ratios/Group/Ratio[@FieldName='{field_name}']")
    if node is not None:
        return _maybe_float(node.text)
    return None


def _find_forecast_ratio(root: ET.Element, field_name: str) -> float | None:
    ratio = root.find(f".//ForecastData/Ratio[@FieldName='{field_name}']")
    if ratio is None:
        return None
    value = ratio.find("./Value[@PeriodType='CURR']")
    if value is not None:
        return _maybe_float(value.text)
    return _maybe_float(ratio.text)


def parse_report_snapshot(xml_text: str) -> dict[str, float | None]:
    """Extract the small fundamentals subset the allocator actually uses."""
    root = ET.fromstring(xml_text)

    trailing_pe = _find_ratio(root, "PEEXCLXOR")
    price_to_book = _find_ratio(root, "PRICE2BK")
    projected_pe = _find_forecast_ratio(root, "ProjPE")
    projected_lt_growth = _find_forecast_ratio(root, "ProjLTGrowthRate")
    trailing_eps = _find_ratio(root, "TTMEPSXCLX")
    market_cap_m = _find_ratio(root, "MKTCAP")
    revenue_m = _find_ratio(root, "TTMREV")
    net_income_m = _find_ratio(root, "TTMNIAC")

    peg_ratio = None
    if projected_pe and projected_lt_growth and projected_lt_growth > 0:
        peg_ratio = projected_pe / projected_lt_growth

    growth_decimal = None
    if projected_lt_growth is not None:
        growth_decimal = projected_lt_growth / 100.0

    return {
        "trailing_pe": trailing_pe,
        "forward_pe": projected_pe,
        "peg_ratio": peg_ratio,
        "earnings_growth_5y": growth_decimal,
        "price_to_book": price_to_book,
        "trailing_eps": trailing_eps,
        "market_cap_musd": market_cap_m,
        "revenue_ttm_musd": revenue_m,
        "net_income_ttm_musd": net_income_m,
    }


def _contract_for_ticker(ticker: str):
    ins = lookup(ticker)
    if ins is None or ins.vehicle_type != "stock":
        raise ValueError(f"{ticker} is not a configured stock instrument")
    if Stock is None:
        raise RuntimeError("ib_insync is not installed")

    symbol = ins.ibkr_symbol or ins.ticker
    exchange = "SMART"
    primary_exchange = ins.ibkr_primary_exchange or ""
    currency = ins.ibkr_currency or ins.ccy or "USD"
    return Stock(symbol, exchange, currency, primaryExchange=primary_exchange)


def fetch_ibkr_fundamentals(
    tickers: Iterable[str],
    *,
    port: int = 4001,
    client_id: int = 91,
) -> dict[str, dict]:
    """Fetch Reuters fundamentals for configured stock tickers."""
    if IB is None:
        raise RuntimeError("ib_insync is not installed")

    ib = IB()
    ib.connect("127.0.0.1", port, clientId=client_id, timeout=5)
    results: dict[str, dict] = {}
    try:
        for ticker in tickers:
            try:
                contract = _contract_for_ticker(ticker)
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    log.warning("IBKR qualification failed for %s", ticker)
                    continue
                xml_text = ib.reqFundamentalData(qualified[0], "ReportSnapshot")
                if not xml_text:
                    log.warning("No Reuters fundamentals returned for %s", ticker)
                    continue
                results[ticker] = parse_report_snapshot(xml_text)
            except Exception as exc:  # pragma: no cover - external system variability
                log.warning("Could not fetch IBKR fundamentals for %s: %s", ticker, exc)
    finally:
        ib.disconnect()
    return results


def export_ibkr_fundamentals_csv(
    tickers: Iterable[str],
    output_path: str,
    *,
    port: int = 4001,
    client_id: int = 91,
) -> int:
    """Write a CSV that allocator/data_sources.py can ingest directly."""
    data = fetch_ibkr_fundamentals(tickers, port=port, client_id=client_id)
    if not data:
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "ticker",
        "trailing_pe",
        "forward_pe",
        "peg_ratio",
        "earnings_growth_5y",
        "price_to_book",
        "trailing_eps",
        "market_cap_musd",
        "revenue_ttm_musd",
        "net_income_ttm_musd",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ticker, row in sorted(data.items()):
            writer.writerow({"ticker": ticker, **row})
    return len(data)


def upsert_ibkr_fundamentals_duckdb(
    tickers: Iterable[str],
    duckdb_path: str,
    *,
    port: int = 4001,
    client_id: int = 91,
    source_file: str = "ibkr_gateway",
) -> int:
    """Write IBKR fundamentals directly into the allocator DuckDB cache."""
    data = fetch_ibkr_fundamentals(tickers, port=port, client_id=client_id)
    if not data:
        return 0

    conn = duckdb.connect(duckdb_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ibkr_fundamentals (
            ticker               TEXT PRIMARY KEY,
            trailing_pe          DOUBLE,
            peg_ratio            DOUBLE,
            earnings_growth_5y   DOUBLE,
            five_year_avg_return DOUBLE,
            price_to_book        DOUBLE,
            source_file          TEXT,
            as_of                DATE,
            refreshed            TIMESTAMP
        )
        """
    )
    rows = []
    for ticker, row in sorted(data.items()):
        rows.append(
            (
                ticker,
                row.get("trailing_pe"),
                row.get("peg_ratio"),
                row.get("earnings_growth_5y"),
                None,
                row.get("price_to_book"),
                source_file,
            )
        )
    conn.executemany(
        """
        INSERT INTO ibkr_fundamentals
            (ticker, trailing_pe, peg_ratio, earnings_growth_5y,
             five_year_avg_return, price_to_book, source_file, as_of, refreshed)
        VALUES (?, ?, ?, ?, ?, ?, ?, current_date, now())
        ON CONFLICT (ticker) DO UPDATE SET
            trailing_pe=excluded.trailing_pe,
            peg_ratio=excluded.peg_ratio,
            earnings_growth_5y=excluded.earnings_growth_5y,
            five_year_avg_return=excluded.five_year_avg_return,
            price_to_book=excluded.price_to_book,
            source_file=excluded.source_file,
            as_of=excluded.as_of,
            refreshed=excluded.refreshed
        """,
        rows,
    )
    conn.close()
    return len(rows)


__all__ = [
    "export_ibkr_fundamentals_csv",
    "fetch_ibkr_fundamentals",
    "parse_report_snapshot",
    "upsert_ibkr_fundamentals_duckdb",
]
