"""Tests for allocator/themes.py."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

from themes import build_theme_correlation, build_theme_snapshot, build_theme_stock_screen


def test_build_theme_snapshot_aggregates_by_sleeve():
    timing_df = pd.DataFrame(
        [
            {"ticker": "XLE", "r_1m": 2.0, "r_3m": 5.0, "r_6m": 8.0, "r_1y": 10.0, "vol_1y": 20.0, "drawdown_52w": -6.0, "ma_252": 90.0, "price": 100.0, "low_52w": 80.0, "high_52w": 105.0},
            {"ticker": "XOP", "r_1m": 3.0, "r_3m": 6.0, "r_6m": 9.0, "r_1y": 12.0, "vol_1y": 25.0, "drawdown_52w": -8.0, "ma_252": 45.0, "price": 50.0, "low_52w": 35.0, "high_52w": 54.0},
        ]
    )
    out = build_theme_snapshot(timing_df)
    assert not out.empty
    energy = out[out["theme"] == "real_energy"].iloc[0]
    assert energy["instruments"] == 2
    assert round(float(energy["r_1y"]), 1) == 11.0


def test_build_theme_correlation_returns_matrix():
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    price_df = pd.DataFrame(
        {
            "ticker": ["XLE"] * len(dates) + ["XOP"] * len(dates) + ["XLV"] * len(dates),
            "date": list(dates) * 3,
            "close": list(range(100, 100 + len(dates))) + list(range(50, 50 + len(dates))) + list(range(80, 80 + len(dates))),
        }
    )
    corr = build_theme_correlation(price_df, lookback_days=60)
    assert not corr.empty
    assert "Real Energy".lower() in [c.lower() for c in corr.columns]


def test_build_theme_stock_screen_groups_direct_stocks_by_theme():
    score_df = pd.DataFrame(
        [
            {"Ticker": "REL", "PE percentile": 0.2, "PE range pos": 0.3, "52W range pos": 0.4, "Dist. from 52W low": 0.1, "Price/MA200": 1.01, "Composite": "FAIR"},
            {"Ticker": "SHEL", "PE percentile": 0.2, "PE range pos": 0.3, "52W range pos": 0.4, "Dist. from 52W low": 0.1, "Price/MA200": 1.01, "Composite": "FAIR"},
        ]
    )
    timing_df = pd.DataFrame(
        [
            {"ticker": "REL", "r_1m": 1.0, "r_3m": 2.0, "r_6m": 3.0, "r_1y": 4.0, "vol_1y": 15.0, "drawdown_52w": -5.0},
            {"ticker": "SHEL", "r_1m": 1.0, "r_3m": 2.0, "r_6m": 3.0, "r_1y": 4.0, "vol_1y": 15.0, "drawdown_52w": -5.0},
        ]
    )
    out = build_theme_stock_screen(score_df, timing_df)
    assert not out.empty
    assert "Theme" in out.columns
    assert "r_1y" in out.columns

