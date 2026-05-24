import pandas as pd

from app.views.ConsolidationSetup import (
    scan_breakout_triggers,
    scan_consolidation_setups,
)


def _rows(ticker: str, prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "ticker_full": ticker,
            "description": ticker,
            "fund_type": "stock",
            "date": dates,
            "price": prices,
            "price_high": [p + 1 for p in prices],
            "price_low": [p - 1 for p in prices],
        }
    )


def test_scan_keeps_bull_consolidation_near_breakout():
    trend = [300 + i * 0.05 for i in range(240)]
    consolidation = [311, 312, 310, 313, 311, 312, 310, 313, 312, 314] * 6
    df = _rows("GOOD", trend + consolidation)

    result = scan_consolidation_setups(df)

    assert result["ticker"].tolist() == ["GOOD"]
    row = result.iloc[0]
    assert row["regime"] == "Bull"
    assert row["breakout_gap_adr"] > 0
    assert row["breakout_gap_adr"] <= 2.0


def test_scan_filters_assets_already_breaking_out():
    trend = [300 + i * 0.05 for i in range(240)]
    consolidation = [311, 312, 310, 313, 311, 312, 310, 313, 312, 314] * 5
    breakout = [316, 318, 320, 322, 324, 326, 328, 330, 332, 334]
    df = _rows("EXTENDED", trend + consolidation + breakout)

    result = scan_consolidation_setups(df)

    assert result.empty


def test_scan_filters_non_bull_assets():
    trend = list(range(340, 100, -1))
    consolidation = [120, 119, 121, 118, 120, 119, 121, 118, 120, 119] * 6
    df = _rows("BEAR", trend + consolidation)

    result = scan_consolidation_setups(df)

    assert result.empty


def test_scan_breakout_triggers_on_fresh_resistance_cross():
    trend = [300 + i * 0.05 for i in range(240)]
    consolidation = [311, 312, 310, 313, 311, 312, 310, 313, 312, 313] * 5
    trigger = [314.5]
    df = _rows("BREAK", trend + consolidation + trigger)

    result = scan_breakout_triggers(df)

    assert result["ticker"].tolist() == ["BREAK"]
    assert result.iloc[0]["price"] > result.iloc[0]["breakout_level"]
