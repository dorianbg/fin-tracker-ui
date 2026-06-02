from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

import pandas as pd


DEFAULT_STATE_DIR = Path(__file__).resolve().parents[2] / "storage" / "alerts"


def signal_key(row: pd.Series) -> str:
    ticker = row.get("alert_ticker", row.get("ticker", ""))
    signal = row.get("signal", row.get("strategy", ""))
    return f"{ticker}|{signal}"


def state_file(state_dir: Path, strategy_id: str, session: str) -> Path:
    return state_dir / f"{strategy_id}.{session}.json"


def load_previous(state_dir: Path, strategy_id: str, session: str) -> dict[str, dict]:
    path = state_file(state_dir, strategy_id, session)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except JSONDecodeError:
        return {}
    return {str(item["key"]): item for item in data.get("signals", [])}


def save_current(
    state_dir: Path,
    strategy_id: str,
    session: str,
    signals: pd.DataFrame,
    current_date: str | None = None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    previous = load_previous(state_dir, strategy_id, session)
    rows = []
    if not signals.empty:
        for i, row in signals.reset_index(drop=True).iterrows():
            key = signal_key(row)
            prior = previous.get(key)
            # Preserve start_date/start_price from prior state; only set on New
            if prior and prior.get("start_date"):
                start_date = prior["start_date"]
                start_price = prior.get("start_price")
            else:
                start_date = current_date
                start_price = row.get("price")
            rows.append(
                {
                    "key": key,
                    "ticker": row.get("alert_ticker", row.get("ticker", "")),
                    "description": row.get("description", row.get("name", "")),
                    "rank": int(row.get("rank", i + 1)),
                    "start_date": start_date,
                    "start_price": float(start_price)
                    if pd.notna(start_price)
                    else None,
                }
            )
    path = state_file(state_dir, strategy_id, session)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps({"signals": rows}, indent=2))
    tmp_path.replace(path)


def detect_changes(current: pd.DataFrame, previous: dict[str, dict]) -> pd.DataFrame:
    rows = []
    current_keys = set()
    if not current.empty:
        for i, row in current.reset_index(drop=True).iterrows():
            key = signal_key(row)
            current_keys.add(key)
            rank = int(row.get("rank", i + 1))
            prior = previous.get(key)
            if prior is None:
                status = "New"
            elif int(prior.get("rank", rank)) != rank:
                status = f"Rank {prior.get('rank')} → {rank}"
            else:
                continue
            item = row.to_dict()
            item["change"] = status
            item["rank"] = rank
            # Attach prior start info so emails can show signal history
            if prior:
                item["start_date"] = prior.get("start_date")
                item["start_price"] = prior.get("start_price")
            rows.append(item)
    for key, prior in previous.items():
        if key not in current_keys:
            rows.append(
                {
                    "alert_ticker": prior.get("ticker", key.split("|", 1)[0]),
                    "ticker": prior.get("ticker", key.split("|", 1)[0]),
                    "description": prior.get("description", ""),
                    "rank": prior.get("rank", ""),
                    "change": "Removed",
                }
            )
    return pd.DataFrame(rows)
