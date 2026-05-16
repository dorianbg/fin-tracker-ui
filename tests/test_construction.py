"""Tests for allocator/construction.py."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

from construction import build_portfolio_plan, summarize_portfolio_plan
from instruments import lookup


def _factor_row(ticker: str, signal: str = "BUY") -> dict:
    composite = {
        "BUY": "HIGH_CONVICTION",
        "ACCUMULATE": "CHEAP",
        "WATCH": "FAIR",
        "HOLD": "FAIR",
        "WAIT": "DEAR",
        "AVOID": "AVOID",
    }.get(signal, "FAIR")
    ma200 = {
        "BUY": 1.02,
        "ACCUMULATE": 0.96,
        "WATCH": 1.05,
        "HOLD": 1.00,
        "WAIT": 1.20,
        "AVOID": 0.80,
    }.get(signal, 1.0)
    return {
        "Ticker": ticker,
        "PE percentile": 0.20,
        "PE range pos": 0.25,
        "52W range pos": 0.30,
        "Dist. from 52W low": 0.10,
        "Price/MA200": ma200,
        "Composite": composite,
    }


def test_cash_tickers_removed_from_allocator_universe():
    assert lookup("ERNS") is None
    assert lookup("CSH2") is None
    assert lookup("SGOV") is None
    assert lookup("BIL") is None


def test_build_portfolio_plan_emits_full_baseline_plan():
    df = pd.DataFrame(
        [
            _factor_row("CSPX", "BUY"),
            _factor_row("IWQU", "BUY"),
            _factor_row("MVOL", "ACCUMULATE"),
            _factor_row("IWVL", "WATCH"),
            _factor_row("EIMI", "BUY"),
            _factor_row("IJPA", "WATCH"),
            _factor_row("VEUR", "BUY"),
            _factor_row("ISF", "HOLD"),
            _factor_row("NATO", "BUY"),
            _factor_row("SGLN", "BUY"),
            _factor_row("IGIL", "HOLD"),
            _factor_row("SEML", "WATCH"),
            _factor_row("IWDA", "BUY"),
            _factor_row("IAU", "BUY"),
            _factor_row("GDX", "ACCUMULATE"),
            _factor_row("PDBC", "BUY"),
            _factor_row("XLE", "BUY"),
            _factor_row("IGF", "BUY"),
            _factor_row("VNQ", "WATCH"),
            _factor_row("TIP", "HOLD"),
            _factor_row("EMLC", "WATCH"),
            _factor_row("XLV", "BUY"),
            _factor_row("XLI", "BUY"),
        ]
    )
    plan = build_portfolio_plan(df, {"SIPP": 300_000, "ISA": 150_000, "GIA": 100_000})
    assert not plan.empty
    assert set(plan["account_type"]) == {"SIPP", "ISA", "GIA"}
    assert not plan["ticker"].isin(["ERNS", "CSH2", "SGOV", "BIL"]).any()
    assert round(plan.groupby("account_type")["target_gbp"].sum().loc["SIPP"], 2) == 300_000.00
    assert round(plan.groupby("account_type")["target_gbp"].sum().loc["ISA"], 2) == 150_000.00
    assert round(plan.groupby("account_type")["target_gbp"].sum().loc["GIA"], 2) == 100_000.00

    summary = summarize_portfolio_plan(plan)
    assert not summary.empty
    assert set(summary["account_type"]) == {"SIPP", "ISA", "GIA"}


def test_hold_signal_is_not_buy_now_and_hot_entry_is_watchlist():
    df = pd.DataFrame(
        [
            _factor_row("EIMI", "HOLD"),
        ]
    )
    timing_df = pd.DataFrame(
        [
            {
                "ticker": "EIMI",
                "r_1y": 50.0,
                "drawdown_52w": -4.0,
                "ma_252": 44.0,
                "price": 49.7,
                "low_52w": 32.5,
                "high_52w": 51.8,
            }
        ]
    )
    plan = build_portfolio_plan(
        df,
        {"SIPP": 300_000, "ISA": 150_000, "GIA": 100_000},
        timing_df=timing_df,
        selection_mode="primary_first",
    )
    eimi = plan[plan["ticker"] == "EIMI"].iloc[0]
    assert eimi["action"] == "WATCHLIST"
    assert eimi["r_1y"] == 50.0
