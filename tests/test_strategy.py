"""Unit tests for allocator/strategy.py."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

from instruments import get_yfinance_ticker_map
from strategy import build_buy_candidates, build_wrapper_candidate_table, classify_buy_signal


def _row(**overrides):
    base = {
        "Ticker": "XLV",
        "PE percentile": 0.20,
        "PE range pos": 0.25,
        "52W range pos": 0.30,
        "Dist. from 52W low": 0.10,
        "Price/MA200": 1.02,
        "Composite": "CHEAP",
    }
    base.update(overrides)
    return pd.Series(base)


def test_buy_signal_when_cheap_near_low_and_momentum_repaired():
    signal, rationale = classify_buy_signal(_row())
    assert signal == "BUY"
    assert "momentum repaired" in rationale


def test_avoid_signal_for_falling_knife():
    signal, rationale = classify_buy_signal(_row(**{"Price/MA200": 0.8}))
    assert signal == "AVOID"
    assert rationale == "falling knife"


def test_watch_when_cheap_but_far_from_low():
    signal, rationale = classify_buy_signal(
        _row(**{"52W range pos": 0.75, "Dist. from 52W low": 0.45})
    )
    assert signal == "WATCH"
    assert "too far above 52-week low" in rationale


def test_buy_candidates_sorted_with_best_signal_first():
    df = pd.DataFrame(
        [
            _row(**{"Ticker": "XLV"}).to_dict(),
            _row(**{"Ticker": "XLI", "Price/MA200": 0.8}).to_dict(),
            _row(**{"Ticker": "XLB", "52W range pos": 0.35, "Price/MA200": 0.95}).to_dict(),
        ]
    )
    out = build_buy_candidates(df)
    assert out.iloc[0]["Strategy signal"] in ("BUY", "ACCUMULATE")
    assert "AVOID" in out["Strategy signal"].astype(str).tolist()


def test_wrapper_candidate_table_includes_vehicle_and_reporting_rules():
    df = pd.DataFrame(
        [
            _row(**{"Ticker": "IWQU"}).to_dict(),
            _row(**{"Ticker": "IAU"}).to_dict(),
        ]
    )
    out = build_wrapper_candidate_table(df)
    assert not out.empty
    iwqu = out[out["Ticker"] == "IWQU"]
    iau = out[out["Ticker"] == "IAU"]
    assert "ucits_etf" in iwqu["Vehicle"].astype(str).tolist()
    assert set(iwqu["Wrapper"].astype(str)) >= {"SIPP", "ISA", "GIA"}
    assert set(iau["Wrapper"].astype(str)) == {"GIA"}
    assert iau["Reporting"].all()


def test_stock_candidates_show_up_for_sipp_and_gia_only():
    df = pd.DataFrame([_row(**{"Ticker": "REL"}).to_dict()])
    out = build_wrapper_candidate_table(df)
    rel = out[out["Ticker"] == "REL"]
    assert not rel.empty
    assert set(rel["Wrapper"].astype(str)) == {"SIPP", "GIA"}
    assert set(rel["Vehicle"].astype(str)) == {"stock"}


def test_ftse_special_ticker_uses_explicit_yfinance_symbol():
    mapping = get_yfinance_ticker_map()
    assert mapping["BT.A"] == "BT-A.L"
