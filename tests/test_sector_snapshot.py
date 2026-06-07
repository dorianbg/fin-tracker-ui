import numpy as np
import pandas as pd

from app.views.SectorSnapshot import (
    build_grouped_snapshots,
    infer_arrow_points,
    relative_strength_snapshot,
    select_grouped_snapshot_universes,
)


def make_prices():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    steps = np.arange(len(dates))
    series = {
        "AAA": 100 * (1.01**steps),
        "BBB": 100 * (1.005**steps),
        "CCC": 100 * (0.999**steps),
        "SPY": 100 * (1.0**steps),
    }
    for ticker, values in series.items():
        for date, price in zip(dates, values):
            rows.append(
                {"ticker": ticker, "date": date, "price": price, "description": ticker}
            )
    return pd.DataFrame(rows)


def test_relative_strength_snapshot_ranks_by_latest_strength_and_caps_top_n():
    out = relative_strength_snapshot(
        make_prices(), ["AAA", "BBB", "CCC"], "SPY", days=30, top_n=2
    )

    assert len(out) == 2
    assert out["ticker"].tolist()[0] == "AAA"
    assert out.loc[0, "relative_strength"] >= out.loc[1, "relative_strength"]
    assert isinstance(out.loc[0, "history"], list)
    assert len(out.loc[0, "history"]) <= 30
    assert {
        "ticker",
        "description",
        "relative_strength",
        "trend_delta",
        "history",
    }.issubset(out.columns)


def test_relative_strength_snapshot_produces_expected_trend_direction():
    out = relative_strength_snapshot(
        make_prices(), ["AAA", "CCC"], "SPY", days=20, top_n=10
    )

    aaa = out.loc[out["ticker"] == "AAA", "trend_delta"]
    ccc = out.loc[out["ticker"] == "CCC", "trend_delta"]

    assert aaa.iloc[0] > 0
    assert ccc.iloc[0] < 0


def test_infer_arrow_points_follows_series_direction():
    points = infer_arrow_points([1, 2, 2, 1])

    assert len(points) == 4
    assert points[0] == 0
    assert points[1] >= 0
    assert points[-1] <= 0


def test_select_grouped_snapshot_universes_prefers_us_etfs_and_dedupes_exposures():
    dates = pd.bdate_range("2024-01-01", periods=25)
    rows = []
    instruments = {
        "CSP1.L": ("Core S&P 500", "eq"),
        "AIQ": ("AI Theme US", "Thematic - AI"),
        "WTAI": ("AI Theme Duplicate", "Thematic - AI"),
        "EWJ": ("Japan US", "Equity - Japan"),
        "IJPA.L": ("Japan UCITS Duplicate", "Equity - Japan"),
        "XLY": ("Consumer Discretionary", "Sector - Consumer Discretionary"),
        "IUCD.L": (
            "Consumer Discretionary UCITS Duplicate",
            "Sector - Consumer Discretionary",
        ),
        "JEDI.L": ("Space UCITS", "Thematic - Space"),
        "GDX": ("Gold Miners", "Sector - Mining"),
        "XME": ("Metals Mining", "Sector - Mining"),
    }
    for ticker, (description, fund_type) in instruments.items():
        for i, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "ticker_full": ticker,
                    "date": date,
                    "price": 100 + i,
                    "description": description,
                    "fund_type": fund_type,
                }
            )
    prices = pd.DataFrame(rows)

    groups = select_grouped_snapshot_universes(prices, "CSP1.L")

    assert "AIQ" in groups["thematic"]
    assert "WTAI" not in groups["thematic"]
    assert "JEDI.L" not in groups["thematic"]
    assert "EWJ" in groups["international"]
    assert "IJPA.L" not in groups["international"]
    assert "XLY" in groups["core_sectors"]
    assert "IUCD.L" not in groups["core_sectors"]
    assert "XME" in groups["core_sectors"]


def test_build_grouped_snapshots_returns_all_three_groups():
    prices = make_prices().copy()
    prices["ticker_full"] = prices["ticker"]
    prices["fund_type"] = prices["ticker"].map(
        {
            "AAA": "Thematic - AI",
            "BBB": "Equity - Japan",
            "CCC": "Sector - Consumer Discretionary",
            "SPY": "eq",
        }
    )
    prices["snapshot_category"] = prices["fund_type"]

    from app.views import SectorSnapshot

    old = SectorSnapshot.PREFERRED_GROUP_TICKERS
    SectorSnapshot.PREFERRED_GROUP_TICKERS = {
        "thematic": {"AI": ["AAA"]},
        "international": {"Japan": ["BBB"]},
        "core_sectors": {"Consumer Discretionary": ["CCC"]},
    }
    try:
        out = build_grouped_snapshots(prices, benchmark="SPY", days=30, top_n=5)
    finally:
        SectorSnapshot.PREFERRED_GROUP_TICKERS = old

    assert set(out) == {"thematic", "international", "core_sectors"}
    assert out["thematic"]["ticker"].tolist() == ["AAA"]
    assert out["international"]["ticker"].tolist() == ["BBB"]
    assert out["core_sectors"]["ticker"].tolist() == ["CCC"]
