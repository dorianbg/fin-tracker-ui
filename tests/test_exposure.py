"""Tests for lookthrough exposure aggregation."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

import data_sources as ds


def test_summarize_true_exposure_aggregates_direct_and_indirect(monkeypatch):
    portfolio_df = pd.DataFrame(
        [
            {"account_type": "GIA", "ticker": "AAPL", "gbp_value": 1000.0},
            {"account_type": "GIA", "ticker": "SPY", "gbp_value": 2000.0},
        ]
    )
    constituents = pd.DataFrame(
        [
            {"ETF": "SPY", "Holding symbol": "AAPL", "Holding name": "Apple Inc", "Mapped ticker": "AAPL", "Mapped sleeve": "equity_technology", "Weight %": 5.0, "source": "yfinance", "as_of": "2026-04-11"},
            {"ETF": "SPY", "Holding symbol": "MSFT", "Holding name": "Microsoft", "Mapped ticker": "MSFT", "Mapped sleeve": "equity_technology", "Weight %": 4.0, "source": "yfinance", "as_of": "2026-04-11"},
        ]
    )

    monkeypatch.setattr(ds, "get_etf_constituents", lambda ticker: constituents if ticker == "SPY" else pd.DataFrame())
    out = ds.summarize_true_exposure(portfolio_df)
    aapl = out[out["underlying_ticker"] == "AAPL"].iloc[0]
    msft = out[out["underlying_ticker"] == "MSFT"].iloc[0]
    assert aapl["direct_gbp"] == 1000.0
    assert aapl["indirect_gbp"] == 100.0
    assert bool(aapl["duplicate_overlap"]) is True
    assert msft["direct_gbp"] == 0.0
    assert msft["indirect_gbp"] == 80.0
