"""
Scrape stock index constituents from Wikipedia and output rows
compatible with instrument_info.csv.

Supported indices: sp500, ftse100, ftse250, cac40, dax

Usage:
    python generate_stock_universe.py sp500              # print to stdout
    python generate_stock_universe.py sp500 --append     # append to instrument_info.csv
    python generate_stock_universe.py ftse100 ftse250 cac40 dax --append
    python generate_stock_universe.py all --append       # all indices
"""

import argparse
import os
from io import StringIO

import pandas as pd
import requests


CSV_PATH = os.path.join(os.path.dirname(__file__), "resources", "instrument_info.csv")
HEADERS = {"User-Agent": "fintracker/1.0"}

# Map GICS sectors to short labels
SECTOR_MAP = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Consumer Discretionary": "Consumer Discretionary",
    "Communication Services": "Communication",
    "Consumer Staples": "Consumer Staples",
    "Financials": "Financials",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Materials": "Materials",
}


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def scrape_sp500() -> pd.DataFrame:
    """Scrape S&P 500 from Wikipedia."""
    tables = _fetch_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "description", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["sector"] = df["sector"].map(SECTOR_MAP).fillna(df["sector"])
    df["currency"] = "USD"
    df["fund_type"] = "stock"
    df["url"] = ""
    return df[["ticker", "description", "currency", "fund_type", "url", "sector"]]


def scrape_ftse100() -> pd.DataFrame:
    """Scrape FTSE 100 from Wikipedia."""
    tables = _fetch_tables("https://en.wikipedia.org/wiki/FTSE_100_Index")
    # Table 6: Company, Ticker, FTSE ICB sector
    df = tables[6].copy()
    df.columns = ["description", "ticker", "sector"]
    # Add .L suffix for London Stock Exchange
    df["ticker"] = df["ticker"].str.strip() + ".L"
    df["currency"] = "GBP"
    df["fund_type"] = "stock"
    df["url"] = ""
    return df[["ticker", "description", "currency", "fund_type", "url", "sector"]]


def scrape_ftse250() -> pd.DataFrame:
    """Scrape FTSE 250 from Wikipedia."""
    tables = _fetch_tables("https://en.wikipedia.org/wiki/FTSE_250_Index")
    # Table 3: Company, Ticker, ICB sector
    df = tables[3].copy()
    df.columns = ["description", "ticker", "sector"]
    # Add .L suffix for London Stock Exchange
    df["ticker"] = df["ticker"].str.strip() + ".L"
    df["currency"] = "GBP"
    df["fund_type"] = "stock"
    df["url"] = ""
    return df[["ticker", "description", "currency", "fund_type", "url", "sector"]]


def scrape_cac40() -> pd.DataFrame:
    """Scrape CAC 40 from Wikipedia."""
    tables = _fetch_tables("https://en.wikipedia.org/wiki/CAC_40")
    # Table 4: Company, Sector, GICS Sub-Industry, Ticker
    df = tables[4][["Company", "Sector", "Ticker"]].copy()
    df.columns = ["description", "sector", "ticker"]
    # Add .PA suffix for Euronext Paris (strip first to avoid double suffix)
    df["ticker"] = (
        df["ticker"].str.strip().str.replace(r"\.PA$", "", regex=True) + ".PA"
    )
    df["sector"] = df["sector"].map(SECTOR_MAP).fillna(df["sector"])
    df["currency"] = "EUR"
    df["fund_type"] = "stock"
    df["url"] = ""
    return df[["ticker", "description", "currency", "fund_type", "url", "sector"]]


def scrape_dax() -> pd.DataFrame:
    """Scrape DAX 40 from Wikipedia."""
    tables = _fetch_tables("https://en.wikipedia.org/wiki/DAX")
    # Table 4: Ticker, Logo, Company, Prime Standard Sector, Index weighting
    df = tables[4][["Ticker", "Company", "Prime Standard Sector"]].copy()
    df.columns = ["ticker", "description", "sector"]
    # Add .DE suffix for XETRA (strip first to avoid double suffix)
    df["ticker"] = (
        df["ticker"].str.strip().str.replace(r"\.DE$", "", regex=True) + ".DE"
    )
    df["currency"] = "EUR"
    df["fund_type"] = "stock"
    df["url"] = ""
    return df[["ticker", "description", "currency", "fund_type", "url", "sector"]]


INDEX_SCRAPERS = {
    "sp500": ("S&P 500", scrape_sp500),
    "ftse100": ("FTSE 100", scrape_ftse100),
    "ftse250": ("FTSE 250", scrape_ftse250),
    "cac40": ("CAC 40", scrape_cac40),
    "dax": ("DAX 40", scrape_dax),
}


def main():
    parser = argparse.ArgumentParser(description="Generate stock universe from indices")
    parser.add_argument(
        "indices",
        nargs="+",
        choices=list(INDEX_SCRAPERS.keys()) + ["all"],
        help="Which index/indices to scrape",
    )
    parser.add_argument(
        "--append", action="store_true", help="Append to instrument_info.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be added (no write)"
    )
    args = parser.parse_args()

    # Resolve 'all'
    indices = list(INDEX_SCRAPERS.keys()) if "all" in args.indices else args.indices

    all_stocks = []
    for idx in indices:
        label, scraper = INDEX_SCRAPERS[idx]
        stocks = scraper()
        print(f"Scraped {len(stocks)} {label} constituents")
        all_stocks.append(stocks)

    combined = pd.concat(all_stocks, ignore_index=True)

    # Remove duplicates within the scraped set (some stocks in multiple indices)
    before = len(combined)
    combined = combined.drop_duplicates(subset="ticker", keep="first")
    if len(combined) < before:
        print(f"Removed {before - len(combined)} cross-index duplicates")

    if args.append or args.dry_run:
        existing = pd.read_csv(CSV_PATH)
        existing_tickers = set(existing["ticker"].values)
        new_stocks = combined[~combined["ticker"].isin(existing_tickers)]
        dupes = len(combined) - len(new_stocks)
        if dupes:
            print(f"Skipping {dupes} already-tracked ticker(s)")
        print(f"New stocks to add: {len(new_stocks)}")

        if args.dry_run:
            print("\n--- Preview (first 20) ---")
            print(new_stocks.head(20).to_string(index=False))
            return

        if args.append and not new_stocks.empty:
            new_stocks.to_csv(
                CSV_PATH, mode="a", header=False, index=False, lineterminator="\n"
            )
            print(f"Appended {len(new_stocks)} stocks to {CSV_PATH}")
    else:
        print(combined.to_csv(index=False))


if __name__ == "__main__":
    main()
