import pandas as pd
import numpy as np

from app.views.SectorRotation import (
    SECTOR_UNIVERSES,
    build_sector_rotation_backtest,
    current_sector_ranks,
    performance_stats,
)


def make_prices():
    dates = pd.bdate_range("2020-01-01", periods=420)
    rows = []
    steps = np.arange(len(dates))
    series = {
        "AAA": 100 * (1.0015**steps),
        "BBB": 100 * (1.0008**steps),
        "CCC": 100 * (0.9995**steps),
        "BENCH": 100 * (1.0007**steps),
        "CASH": 100 * (1.00005**steps),
    }
    for ticker, values in series.items():
        for date, price in zip(dates, values):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "price": price,
                    "description": ticker,
                    "fund_type": "eq",
                }
            )
    return pd.DataFrame(rows)


def test_current_sector_ranks_orders_by_composite_relative_strength():
    ranks = current_sector_ranks(make_prices(), ["AAA", "BBB", "CCC"])

    assert ranks["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert ranks.loc[0, "rs_score"] > ranks.loc[1, "rs_score"]


def test_backtest_applies_signals_after_warmup_and_reports_stats():
    backtest, holdings = build_sector_rotation_backtest(
        prices=make_prices(),
        sector_tickers=["AAA", "BBB", "CCC"],
        benchmark_ticker="BENCH",
        cash_ticker="CASH",
        top_n=1,
        use_market_filter=False,
    )

    assert not backtest.empty
    assert holdings.iloc[-1]["holdings"] == "AAA"
    assert backtest["Strategy"].iloc[-1] > backtest["Benchmark"].iloc[-1]

    stats = performance_stats(backtest)
    assert stats["Series"].tolist() == ["Strategy", "Benchmark"]
    assert stats.loc[stats["Series"] == "Strategy", "CAGR"].iloc[0] > 0


def test_sector_universes_include_complete_spdr_and_lse_europe_sectors():
    spdr = SECTOR_UNIVERSES["US Select Sector SPDRs"]["tickers"]
    lse_europe = SECTOR_UNIVERSES["LSE Europe Sectors"]["tickers"]
    global_ucits = SECTOR_UNIVERSES["Global UCITS Sector ETFs"]["tickers"]

    assert spdr == [
        "XLC",
        "XLY",
        "XLP",
        "XLE",
        "XLF",
        "XLV",
        "XLI",
        "XLB",
        "XLRE",
        "XLK",
        "XLU",
    ]
    assert len(lse_europe) == 11
    assert all(ticker.endswith(".L") for ticker in lse_europe)
    assert len(global_ucits) == 11
    assert all(ticker.endswith(".L") for ticker in global_ucits)
