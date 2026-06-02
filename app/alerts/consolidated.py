from __future__ import annotations

import pandas as pd

from app.alerts.signals import StrategySignals


def _build_consolidated(strategies: list[StrategySignals], limit: int) -> pd.DataFrame:
    """Aggregate active signals across strategies into a single ranked instrument list.

    Each instrument is ranked by the number of distinct alert signals it triggers
    across all supplied strategies, then by its best (max) score.
    """
    rows = []
    for strategy in strategies:
        if strategy.active.empty:
            continue
        for _, row in strategy.active.iterrows():
            d = row.to_dict()
            d["strategy_id"] = strategy.strategy_id
            d["strategy_title"] = strategy.title
            rows.append(d)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    group_col = "alert_ticker" if "alert_ticker" in df.columns else "ticker"

    for col in (
        "description",
        "ticker",
        "score",
        "signal",
        "summary",
        "strategy_id",
        "strategy_title",
    ):
        if col not in df.columns:
            df[col] = None

    def _unique_list(series: pd.Series) -> list:
        return series.dropna().astype(str).unique().tolist()

    def _first_non_null(series: pd.Series):
        return series.dropna().iloc[0] if not series.dropna().empty else None

    agg = {
        "description": "first",
        "ticker": "first",
        "score": "max",
        "signal": lambda s: _unique_list(s),
        "summary": lambda s: s.dropna().astype(str).tolist(),
        "strategy_id": lambda s: _unique_list(s),
        "strategy_title": lambda s: _unique_list(s),
    }
    # preserve volatility columns if present
    for vol_col in ("vol_1y", "vol_1mo", "price", "date"):
        if vol_col in df.columns:
            agg[vol_col] = _first_non_null
    agg = {k: v for k, v in agg.items() if k in df.columns}
    grouped = df.groupby(group_col, as_index=False).agg(agg)
    grouped["signal_count"] = grouped["signal"].apply(len)
    grouped = grouped.sort_values(["signal_count", "score"], ascending=[False, False])
    grouped = grouped.head(limit).reset_index(drop=True)
    grouped["rank"] = range(1, len(grouped) + 1)
    return grouped
