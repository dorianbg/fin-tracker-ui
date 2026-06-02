from collections.abc import Sequence

import pandas as pd

import app.alerts.signals as signals
from app.alerts.session import filter_by_session, market_session_for_ticker
from app.alerts.signals import (
    breakout_signals,
    consolidation_signals,
    momentum_breakout_signals,
    pullback_signals,
    rotation_signals,
    todays_crossings_signals,
    turnaround_signals,
)
from app.alerts.state import detect_changes
from app.strategy_scanners import (
    scan_laggard_awakening,
    scan_pullbacks,
    scan_todays_alert_crossings,
)


def test_session_filter_sends_lse_before_eu_and_others_before_us():
    df = pd.DataFrame(
        {
            "ticker": ["VWRP", "SPY"],
            "ticker_full": ["VWRP.L", "SPY"],
            "description": ["World ETF", "S&P ETF"],
        }
    )

    assert market_session_for_ticker("VWRP.L") == "eu"
    assert market_session_for_ticker("SPY") == "us"
    assert filter_by_session(df, "eu")["alert_ticker"].tolist() == ["VWRP.L"]
    assert filter_by_session(df, "us")["alert_ticker"].tolist() == ["SPY"]


def test_detect_changes_finds_new_removed_and_rank_changes():
    current = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "alert_ticker": ["AAA", "BBB"],
            "description": ["A", "B"],
            "signal": ["sig", "sig"],
            "rank": [1, 2],
        }
    )
    previous = {
        "AAA|sig": {"ticker": "AAA", "description": "A", "rank": 2},
        "CCC|sig": {"ticker": "CCC", "description": "C", "rank": 1},
    }

    changes = detect_changes(current, previous)

    assert set(changes["change"]) == {"Rank 2 → 1", "New", "Removed"}


def test_pullback_signal_uses_default_best_only_shape():
    latest = pd.DataFrame(
        {
            "ticker": ["PULL", "BAD"],
            "ticker_full": ["PULL", "BAD"],
            "description": ["Pullback", "Broken"],
            "fund_type": ["eq", "eq"],
            "ma_21": [-2.0, -4.0],
            "ma_63": [3.0, -8.0],
            "ma_126": [4.0, -1.0],
            "ma_252": [15.0, 5.0],
            "r_1d": [0.5, 0.2],
            "r_1w": [-1.0, -2.0],
            "drawdown_52w": [-8.0, -30.0],
        }
    )

    result = pullback_signals(latest, "us", 10).active

    assert result["ticker"].tolist() == ["PULL"]
    assert "early bounce" in result.iloc[0]["summary"]


def test_pullback_alert_reuses_shared_pullback_scanner_shape():
    latest = pd.DataFrame(
        {
            "ticker": ["PULL", "BAD"],
            "ticker_full": ["PULL", "BAD"],
            "description": ["Pullback", "Broken"],
            "fund_type": ["eq", "eq"],
            "ma_21": [-2.0, -4.0],
            "ma_63": [3.0, -8.0],
            "ma_126": [4.0, -1.0],
            "ma_252": [15.0, 5.0],
            "r_1d": [0.5, 0.2],
            "r_1w": [-1.0, -2.0],
            "drawdown_52w": [-8.0, -30.0],
        }
    )

    shared = scan_pullbacks(latest, require_bounce=True)
    alert = pullback_signals(latest, "us", 10).active

    assert shared["ticker"].tolist() == alert["ticker"].tolist()
    assert "quality_score" in shared.columns


def _prices(ticker: str, values: Sequence[int | float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ticker,
            "ticker_full": ticker,
            "date": pd.date_range("2025-01-01", periods=len(values), freq="D"),
            "price": values,
        }
    )


def test_momentum_breakout_finds_recovery_and_base_setups():
    latest = pd.DataFrame(
        {
            "ticker": ["REC", "BASE", "NOPE"],
            "ticker_full": ["REC", "BASE", "NOPE"],
            "description": ["Recovery", "Base", "Loose"],
            "fund_type": ["stock", "eq", "stock"],
            "drawdown_52w": [-18.0, -4.0, -4.0],
            "r_1w": [3.0, 1.5, 4.0],
            "r_1mo": [8.0, 3.5, 15.0],
            "vol_1mo": [25.0, 8.0, 15.0],
            "vol_1y": [40.0, 18.0, 20.0],
        }
    )
    prices = pd.concat(
        [
            _prices("REC", [120, 110, 100, 90] + list(range(70, 130))),
            _prices("BASE", [94 + (i % 5) for i in range(45)] + list(range(96, 101))),
            _prices("NOPE", [80 + (i % 20) for i in range(49)] + [105]),
        ],
        ignore_index=True,
    )

    active = momentum_breakout_signals(latest, prices, "us", 10).active

    assert set(active["ticker"].tolist()) == {"REC", "BASE"}
    assert set(active["signal"].tolist()) == {
        "Recovery breakout",
        "Base breakout near highs",
    }
    assert active.set_index("ticker").loc["REC", "local_high_window"] == 60
    assert "20D range" in active.iloc[0]["summary"]


def test_turnaround_signal_remains_available_as_focused_recovery_alert():
    latest = pd.DataFrame(
        {
            "ticker": ["TURN", "BASE"],
            "ticker_full": ["TURN", "BASE"],
            "description": ["Turnaround", "Base"],
            "fund_type": ["stock", "stock"],
            "drawdown_52w": [-15.0, -5.0],
            "r_1w": [2.0, 4.0],
        }
    )
    prices = pd.concat(
        [
            _prices("TURN", [120, 110, 100, 90] + list(range(70, 130))),
            _prices("BASE", [95 + (i % 4) for i in range(59)] + [100]),
        ],
        ignore_index=True,
    )

    active = turnaround_signals(latest, prices, "us", 10).active

    assert active["ticker"].tolist() == ["TURN"]
    assert (
        active.iloc[0]["signal"]
        == "Turnaround: near 4-12W high but still below 52W peak"
    )
    assert "Within" in active.iloc[0]["summary"]


def test_scanner_alerts_preserve_lse_full_ticker(monkeypatch):
    prices = pd.DataFrame(
        {
            "ticker": ["VWRP"],
            "ticker_full": ["VWRP.L"],
            "date": pd.to_datetime(["2026-01-01"]),
            "price": [100.0],
            "description": ["World ETF"],
            "fund_type": ["eq"],
        }
    )

    monkeypatch.setattr(
        signals,
        "scan_breakout_triggers",
        lambda _: pd.DataFrame(
            {
                "ticker": ["VWRP"],
                "description": ["World ETF"],
                "breakout_score": [4.0],
                "breakout_extension_adr": [0.5],
                "extension_adr": [1.2],
            }
        ),
    )
    monkeypatch.setattr(
        signals,
        "scan_consolidation_setups",
        lambda _: pd.DataFrame(
            {
                "ticker": ["VWRP"],
                "description": ["World ETF"],
                "setup_score": [3.0],
                "breakout_gap_adr": [0.7],
                "extension_adr": [1.0],
            }
        ),
    )

    breakout = breakout_signals(prices, "eu", 10).active
    consolidation = consolidation_signals(prices, "eu", 10).active

    assert breakout["alert_ticker"].tolist() == ["VWRP.L"]
    assert consolidation["alert_ticker"].tolist() == ["VWRP.L"]


def test_rotation_signals_include_stock_fund_type():
    latest = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "ticker_full": ["AAA", "BBB", "CCC"],
            "description": ["A", "B", "C"],
            "fund_type": ["stock", "stock", "eq"],
            "r_3mo": [5.0, 20.0, -3.0],
            "vol_1y": [10.0, 40.0, 20.0],
        }
    )

    momentum = [
        s
        for s in rotation_signals(latest, "us", 10)
        if s.strategy_id == "rotation_momentum"
    ][0]

    assert set(momentum.active["ticker"].tolist()) == {"AAA", "BBB", "CCC"}


def test_rotation_low_vol_removed():
    latest = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "ticker_full": ["AAA", "BBB"],
            "description": ["A", "B"],
            "fund_type": ["eq", "eq"],
            "r_3mo": [10.0, 5.0],
            "vol_1y": [15.0, 20.0],
        }
    )

    ids = {s.strategy_id for s in rotation_signals(latest, "us", 10)}

    assert "rotation_low_vol" not in ids
    assert "rotation_momentum" in ids


def test_todays_crossings_detects_new_high_and_bullish_ma_cross():
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "ticker_full": ["AAA", "AAA"],
            "description": ["A", "A"],
            "fund_type": ["eq", "eq"],
            "rown": [1, 2],
            "drawdown_52w": [0.0, -3.0],
            "ma_21": [1.0, -0.5],
            "ma_63": [2.0, 2.0],
            "ma_126": [2.0, 2.0],
            "ma_252": [2.0, 2.0],
            "z_1d": [0.5, 0.2],
            "z_1w": [0.5, 0.2],
            "z_2w": [0.5, 0.2],
            "z_1mo": [0.5, 0.2],
            "r_1d": [1.2, 0.1],
            "r_1w": [3.0, 1.0],
        }
    )

    active = todays_crossings_signals(raw, "us", 10).active

    assert {"New 52W high", "Bullish 21D MA crossover"}.issubset(set(active["signal"]))


def test_todays_crossings_alert_reuses_shared_scanner():
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "ticker_full": ["AAA", "AAA"],
            "description": ["A", "A"],
            "fund_type": ["eq", "eq"],
            "rown": [1, 2],
            "drawdown_52w": [0.0, -3.0],
            "ma_21": [1.0, -0.5],
            "ma_63": [2.0, 2.0],
            "ma_126": [2.0, 2.0],
            "ma_252": [2.0, 2.0],
            "z_1d": [2.5, 0.2],
            "z_1w": [0.5, 0.2],
            "z_2w": [0.5, 0.2],
            "z_1mo": [0.5, 0.2],
            "r_1d": [1.2, 0.1],
            "r_1w": [3.0, 1.0],
        }
    )

    shared = scan_todays_alert_crossings(raw, z_threshold=2.0)
    alert = todays_crossings_signals(raw, "us", 10).active

    assert shared["signal"].tolist() == alert["signal"].tolist()
    z_summary = alert[alert["signal"] == "Z-score spike"].iloc[0]["summary"]
    assert "1D return" in z_summary


def test_laggard_alert_reuses_shared_laggard_scanner():
    latest = pd.DataFrame(
        {
            "ticker": ["VWRP", "LAG", "OK"],
            "ticker_full": ["VWRP", "LAG", "OK"],
            "description": ["Benchmark", "Laggard", "Okay"],
            "fund_type": ["eq", "eq", "eq"],
            "r_1y": [10.0, -5.0, 8.0],
            "r_1w": [1.0, 3.0, 2.0],
        }
    )
    rs_df = latest.copy()
    rs_df["rs_1Y"] = rs_df["r_1y"] - 10.0
    rs_df["rs_1W"] = rs_df["r_1w"] - 1.0

    shared, _ = scan_laggard_awakening(
        rs_df,
        benchmark_ticker="VWRP",
        laggard_period="1Y",
        awakening_period="1W",
        underperf_threshold=10,
    )
    alert = signals.laggard_signals(latest, "us", 10).active

    assert shared["ticker"].tolist() == alert["ticker"].tolist()


def test_build_consolidated_ranks_by_signal_count_then_score():
    from app.alerts.consolidated import _build_consolidated

    s1 = signals.StrategySignals(
        "s1",
        "S1",
        "Comment 1",
        pd.DataFrame(
            {
                "alert_ticker": ["A", "B"],
                "ticker": ["A", "B"],
                "description": ["Alpha", "Beta"],
                "score": [5.0, 4.0],
                "signal": ["sig1", "sig1"],
                "summary": ["summary A1", "summary B1"],
            }
        ),
    )
    s2 = signals.StrategySignals(
        "s2",
        "S2",
        "Comment 2",
        pd.DataFrame(
            {
                "alert_ticker": ["A", "C"],
                "ticker": ["A", "C"],
                "description": ["Alpha", "Gamma"],
                "score": [3.0, 6.0],
                "signal": ["sig2", "sig2"],
                "summary": ["summary A2", "summary C2"],
            }
        ),
    )
    result = _build_consolidated([s1, s2], 10)

    assert result["alert_ticker"].tolist() == ["A", "C", "B"]
    assert result["signal_count"].tolist() == [2, 1, 1]
    assert result.iloc[0]["score"] == 5.0
    assert set(result.iloc[0]["strategy_id"]) == {"s1", "s2"}
    assert set(result.iloc[0]["strategy_title"]) == {"S1", "S2"}
