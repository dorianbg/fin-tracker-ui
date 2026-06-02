from datetime import date
from argparse import Namespace
from pathlib import Path
import sys

import pandas as pd
import pytest

from app.alerts.freshness import assert_fresh_data, latest_data_date

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_latest_data_date_reads_max_date():
    df = pd.DataFrame({"date": ["2026-05-29", "2026-05-30"]})

    assert latest_data_date(df) == date(2026, 5, 30)


def test_assert_fresh_data_allows_today():
    df = pd.DataFrame({"date": ["2026-05-30"]})

    assert assert_fresh_data(df, label="alerts", today=date(2026, 5, 30)) == date(
        2026, 5, 30
    )


def test_assert_fresh_data_blocks_stale_send():
    df = pd.DataFrame({"date": ["2026-05-20"]})

    with pytest.raises(RuntimeError, match="Refusing to send alerts"):
        assert_fresh_data(df, label="alerts", today=date(2026, 5, 30))


def test_assert_fresh_data_allows_stale_for_dry_run_or_development():
    df = pd.DataFrame({"date": ["2026-05-29"]})

    assert assert_fresh_data(
        df, label="alerts", dry_run=True, today=date(2026, 5, 30)
    ) == date(2026, 5, 29)
    assert assert_fresh_data(
        df, label="alerts", allow_stale=True, today=date(2026, 5, 30)
    ) == date(2026, 5, 29)


def test_assert_fresh_data_allows_stale_with_env_override(monkeypatch):
    df = pd.DataFrame({"date": ["2026-05-29"]})
    monkeypatch.setenv("FINTRACKER_ALLOW_STALE_ALERTS", "1")

    assert assert_fresh_data(df, label="alerts", today=date(2026, 5, 30)) == date(
        2026, 5, 29
    )


def test_strategy_sender_refuses_stale_data_before_email(monkeypatch):
    import send_strategy_alerts

    stale = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "ticker_full": ["AAA"],
            "date": pd.to_datetime(["2000-01-01"]),
            "description": ["A"],
            "fund_type": ["eq"],
        }
    )
    monkeypatch.setattr(send_strategy_alerts, "load_price_history", lambda: stale)
    monkeypatch.setattr(
        send_strategy_alerts, "load_performance", lambda max_rown: stale
    )
    monkeypatch.setattr(
        send_strategy_alerts,
        "_send_consolidated",
        lambda *args, **kwargs: pytest.fail("stale data should not send email"),
    )

    with pytest.raises(RuntimeError, match="Refusing to send strategy alerts"):
        send_strategy_alerts.send_strategy_alerts(
            Namespace(
                session="us",
                max_items=10,
                strategy=None,
                state_dir=None,
                active_only=False,
                changes_only=False,
                dry_run=False,
                allow_stale_data=False,
            )
        )


def test_strategy_sender_allows_stale_dry_run(monkeypatch):
    import send_strategy_alerts

    stale = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "ticker_full": ["AAA"],
            "date": pd.to_datetime(["2000-01-01"]),
            "description": ["A"],
            "fund_type": ["eq"],
        }
    )
    monkeypatch.setattr(send_strategy_alerts, "load_price_history", lambda: stale)
    monkeypatch.setattr(
        send_strategy_alerts, "load_performance", lambda max_rown: stale
    )
    monkeypatch.setattr(send_strategy_alerts, "build_all_signals", lambda *args: [])

    send_strategy_alerts.send_strategy_alerts(
        Namespace(
            session="us",
            max_items=10,
            strategy=None,
            state_dir=None,
            active_only=False,
            changes_only=False,
            dry_run=True,
            allow_stale_data=False,
        )
    )
