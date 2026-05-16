"""Tests for allocator/lookthrough.py."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

from lookthrough import map_holdings_to_universe


def test_map_holdings_to_universe_links_known_symbols():
    holdings = pd.DataFrame(
        [
            {"ETF": "SPY", "Holding symbol": "AAPL", "Holding name": "Apple Inc", "Weight %": 6.5},
            {"ETF": "VGK", "Holding symbol": "ASML.AS", "Holding name": "ASML Holding NV", "Weight %": 3.5},
            {"ETF": "EEM", "Holding symbol": "2330.TW", "Holding name": "TSMC", "Weight %": 13.0},
        ]
    )
    scores = pd.DataFrame(
        [
            {"Ticker": "AAPL", "Trailing PE": 30.0, "PE percentile": 0.7, "PEG": 2.0, "Price/MA200": 1.1, "Composite": "DEAR"},
            {"Ticker": "ASML", "Trailing PE": 28.0, "PE percentile": 0.4, "PEG": 1.2, "Price/MA200": 1.0, "Composite": "FAIR"},
        ]
    )
    out = map_holdings_to_universe(holdings, scores)
    assert out.loc[out["Holding symbol"] == "AAPL", "Mapped ticker"].iloc[0] == "AAPL"
    assert out.loc[out["Holding symbol"] == "ASML.AS", "Mapped ticker"].iloc[0] == "ASML"
    assert out.loc[out["Holding symbol"] == "2330.TW", "Mapped ticker"].iloc[0] == "TSM"


def test_alias_mapping_links_local_etf_holding_to_adr():
    holdings = pd.DataFrame(
        [
            {"ETF": "EEM", "Holding symbol": "2330.TW", "Holding name": "TSMC", "Weight %": 13.2},
            {"ETF": "EEM", "Holding symbol": "9988.HK", "Holding name": "Alibaba", "Weight %": 2.5},
            {"ETF": "INDA", "Holding symbol": "HDFCBANK.NS", "Holding name": "HDFC Bank", "Weight %": 6.9},
        ]
    )
    out = map_holdings_to_universe(holdings, pd.DataFrame())
    assert out.loc[out["Holding symbol"] == "2330.TW", "Mapped ticker"].iloc[0] == "TSM"
    assert out.loc[out["Holding symbol"] == "9988.HK", "Mapped ticker"].iloc[0] == "BABA"
    assert out.loc[out["Holding symbol"] == "HDFCBANK.NS", "Mapped ticker"].iloc[0] == "HDB"
