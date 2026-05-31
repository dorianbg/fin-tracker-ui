from __future__ import annotations

import os
from datetime import date

import pandas as pd


ALLOW_STALE_ENV = "FINTRACKER_ALLOW_STALE_ALERTS"


def _env_allows_stale() -> bool:
    return os.environ.get(ALLOW_STALE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def stale_data_allowed(*, allow_stale: bool = False, dry_run: bool = False) -> bool:
    return allow_stale or dry_run or _env_allows_stale()


def latest_data_date(df: pd.DataFrame, *, date_col: str = "date") -> date | None:
    if df.empty or date_col not in df.columns:
        return None
    latest = pd.to_datetime(df[date_col], errors="coerce").max()
    if pd.isna(latest):
        return None
    return latest.date()


def assert_fresh_data(
    df: pd.DataFrame,
    *,
    label: str,
    date_col: str = "date",
    allow_stale: bool = False,
    dry_run: bool = False,
    today: date | None = None,
) -> date | None:
    data_date = latest_data_date(df, date_col=date_col)
    expected = today or date.today()
    if data_date == expected or stale_data_allowed(
        allow_stale=allow_stale, dry_run=dry_run
    ):
        return data_date
    raise RuntimeError(
        f"Refusing to send {label}: latest {date_col} is {data_date or 'missing'}, "
        f"expected {expected}. Run the pipeline first, or use --allow-stale-data / "
        f"{ALLOW_STALE_ENV}=1 only for testing/development."
    )
