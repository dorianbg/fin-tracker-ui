"""Rebalance / sell signals for *existing* holdings.

The strategy layer in :mod:`strategy` classifies the investable universe
for *buying*. This module answers the dual question: given what I
already own, what should I trim, hold, or rotate?

Three inputs are combined:

- Current holdings (from :mod:`holdings`) expressed as GBP value per
  account/ticker.
- The per-instrument regime + signal from the timing engine.
- The bucket target weights (from :mod:`buckets`).

Output is a dataframe with one row per holding, a ``rebalance_action``
label, and a short ``reason`` string the UI can show.

Action vocabulary (kept distinct from the buy-side strategy labels):

  KEEP       — holding is on plan and nothing is flashing red.
  TRIM       — holding is overweight vs target AND the regime is stretched
               (``strong_but_stretched``) OR current weight exceeds the
               drift threshold regardless of regime.
  SELL       — regime is ``falling_knife``. Exit into cash.
  ROTATE     — regime is dead_money and the same sleeve has a clearly
               better candidate. (Flagged but not executed — this is a
               human decision.)
  TOP_UP     — holding is underweight AND regime permits buying
               (basing / repairing / washed_out / strong).
  HOLD       — everything else.
"""

from __future__ import annotations

import pandas as pd

from instruments import lookup

_DRIFT_ABS_PP = 2.0
_DRIFT_REL_PCT = 25.0


def _sleeve_target_lookup(bucket_targets: list) -> dict[str, dict]:
    """Flatten bucket.targets into a sleeve-indexed dict with primary_ticker + weight."""
    out: dict[str, dict] = {}
    for t in bucket_targets:
        out[t.sleeve] = {
            "primary_ticker": t.primary_ticker,
            "weight": float(t.weight),
            "is_tactical": bool(t.is_tactical),
        }
    return out


def _classify_rebalance(
    current_weight: float,
    target_weight: float,
    regime: str | None,
    strategy_signal: str | None,
    is_tactical: bool,
) -> tuple[str, str]:
    drift_pp = (current_weight - target_weight) * 100.0
    drift_rel_pct = (drift_pp / (target_weight * 100.0)) * 100.0 if target_weight > 0 else 0.0
    regime = (regime or "").lower()
    signal = (strategy_signal or "").upper()

    if regime == "falling_knife" or signal == "AVOID":
        return "SELL", f"falling_knife regime; drift {drift_pp:+.1f}pp"

    stretched = regime == "strong_but_stretched" or signal == "WATCH"
    overweight_hard = drift_pp > _DRIFT_ABS_PP or drift_rel_pct > _DRIFT_REL_PCT
    underweight_hard = drift_pp < -_DRIFT_ABS_PP or drift_rel_pct < -_DRIFT_REL_PCT

    if stretched and overweight_hard:
        return "TRIM", f"stretched + overweight {drift_pp:+.1f}pp"
    if stretched:
        # Stretched names never get a TOP_UP recommendation even if they
        # are slightly under target — waiting for the regime to normalise
        # is a better entry than chasing into a WATCH name.
        band = "on target" if not underweight_hard else "underweight"
        return "HOLD", f"stretched regime; {band} ({drift_pp:+.1f}pp)"

    if regime == "dead_money" and target_weight > 0 and current_weight > 0:
        return "ROTATE", f"dead_money regime; consider alternate in same sleeve"

    if underweight_hard and regime in {"basing", "repairing", "washed_out", "strong"}:
        return "TOP_UP", f"{regime or 'setup'} + underweight {drift_pp:+.1f}pp"

    if overweight_hard and not is_tactical:
        return "TRIM", f"overweight {drift_pp:+.1f}pp"
    if underweight_hard and not is_tactical:
        return "TOP_UP", f"underweight {drift_pp:+.1f}pp"

    return "KEEP", f"on plan, drift {drift_pp:+.1f}pp"


def build_rebalance_signals(
    holdings_df: pd.DataFrame,
    bucket_sizes: dict[str, float],
    buckets_by_account: dict,
    timing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Emit per-holding rebalance signals.

    Parameters
    ----------
    holdings_df : DataFrame
        Columns: ``account_type``, ``ticker``, ``gbp_value``. Typically the
        portfolio_df used on the Holdings tab.
    bucket_sizes : dict
        {account_type: total_gbp}.
    buckets_by_account : dict
        {account_type: Bucket}. Usually ``ALL_BUCKETS`` from :mod:`buckets`.
    timing_df : DataFrame
        Must contain ``ticker`` and ``regime``. Signal is synthesised from
        regime via ``strategy._REGIME_TO_SIGNAL`` when a signal column is
        absent.
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame()

    timing_idx: pd.DataFrame
    if timing_df is not None and not timing_df.empty:
        timing_idx = timing_df.set_index("ticker")
    else:
        timing_idx = pd.DataFrame()

    # Lazy import to avoid a cycle with strategy module.
    from strategy import _REGIME_TO_SIGNAL

    rows: list[dict] = []
    for account_type, bucket in buckets_by_account.items():
        bucket_size = float(bucket_sizes.get(account_type, 0.0))
        if bucket_size <= 0:
            continue
        sleeve_targets = _sleeve_target_lookup(bucket.targets)
        sub = holdings_df[holdings_df["account_type"] == account_type]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            ticker = str(row["ticker"])
            gbp = float(row.get("gbp_value", 0.0))
            if gbp <= 0:
                continue
            ins = lookup(ticker)
            sleeve = ins.sleeve if ins else "unknown"
            target_info = sleeve_targets.get(sleeve, {"weight": 0.0, "primary_ticker": ticker, "is_tactical": False})

            regime = None
            if not timing_idx.empty and ticker in timing_idx.index:
                r = timing_idx.loc[ticker].get("regime")
                if isinstance(r, str):
                    regime = r.lower()
            signal = _REGIME_TO_SIGNAL.get(regime or "", "HOLD")

            current_weight = gbp / bucket_size
            target_weight = float(target_info["weight"])
            action, reason = _classify_rebalance(
                current_weight=current_weight,
                target_weight=target_weight,
                regime=regime,
                strategy_signal=signal,
                is_tactical=bool(target_info.get("is_tactical")),
            )

            rows.append(
                {
                    "account_type": account_type,
                    "ticker": ticker,
                    "sleeve": sleeve.replace("_", " ").title(),
                    "gbp_value": gbp,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "drift_pp": (current_weight - target_weight) * 100.0,
                    "regime": regime or "no_data",
                    "strategy_signal": signal,
                    "rebalance_action": action,
                    "reason": reason,
                    "is_primary_ticker": ticker == target_info.get("primary_ticker"),
                    "is_tactical": bool(target_info.get("is_tactical")),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        action_order = pd.CategoricalDtype(
            ["SELL", "TRIM", "ROTATE", "TOP_UP", "HOLD", "KEEP"], ordered=True
        )
        out["rebalance_action"] = out["rebalance_action"].astype(action_order)
        out = out.sort_values(
            by=["account_type", "rebalance_action", "gbp_value"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
    return out
