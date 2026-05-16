"""Portfolio construction from strategic bucket targets + factor screens.

The constructor takes three pieces of evidence per sleeve:

1. The strategic target (sleeve weight, primary ticker, wrapper).
2. A per-instrument buy signal from ``strategy.build_wrapper_candidate_table``
   derived from PE / MA200 / 52-week-range factors.
3. Optional per-theme regime labels from ``themes.get_theme_regimes`` built
   from the median technical state of each theme.

It emits one ``ProposedPosition`` per sleeve per wrapper with an ``action``
label that is intentionally narrow:

  - ``APPROVED``  : buy-now candidate; valuation AND timing AND theme regime
                    all agree. This is the only label that should be executed
                    without further review.
  - ``BUILD_CORE``: acceptable strategic hold but not a special bargain. Suitable
                    for scheduled DCA; not suitable for lump-sum deployment.
  - ``WATCHLIST`` : valid sleeve, but timing or theme regime is wrong right now.
                    Keep on the radar for the next refresh.
  - ``NO_DATA``   : strategic sleeve retained but factor/timing coverage is
                    incomplete. Do not size from here; fix the data first.
  - ``NOT_ACTIVE``: tactical sleeve whose macro trigger is not firing. The sleeve
                    is retained in the plan for visibility (so the user can see
                    "TLT is a tactical option but US 10y < 5% so it's dormant")
                    but carries zero target weight. Distinct from ``REJECT`` —
                    this isn't a bad idea, it's a dormant idea.
  - ``REJECT``    : do not buy now (falling knife, extended, or rejected theme
                    regime).

The constructor does NOT compute strategic weights — those come from
``buckets.py``. It only gates execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from buckets import ALL_BUCKETS
from instruments import (
    SLEEVE_EM_BONDS,
    SLEEVE_LINKERS_GLOBAL,
    SLEEVE_LINKERS_UK,
    SLEEVE_LONG_DUR,
    lookup,
)
from strategy import StrategyThresholds, build_wrapper_candidate_table
from themes import REGIME_BUY_NOW, REGIME_REJECT, REGIME_WATCH


# Maps a tactical sleeve to the trigger key in ``valuation.compute_bond_triggers``.
# If a sleeve is tactical but not in this map, it is assumed always-dormant when
# ``bond_triggers`` is provided — explicit opt-in rather than silent activation.
_TACTICAL_SLEEVE_TO_TRIGGER: dict[str, str] = {
    SLEEVE_EM_BONDS: "em_usd",
    SLEEVE_LONG_DUR: "long_dur",
    SLEEVE_LINKERS_UK: "linkers_extra",
    SLEEVE_LINKERS_GLOBAL: "linkers_extra",
}


@dataclass(frozen=True)
class ProposedPosition:
    account_type: str
    sleeve: str
    target_weight: float
    target_gbp: float
    ticker: str
    name: str
    vehicle: str
    strategy_signal: str
    theme_regime: str
    action: str
    candidate_score: float | None
    rationale: str | None
    r_1m: float | None
    r_3m: float | None
    r_6m: float | None
    r_1y: float | None
    drawdown_52w: float | None
    price_ma200: float | None
    range_52w_pos: float | None
    pct_above_ma200: float | None
    vol_3m: float | None
    vol_1y: float | None
    z_1mo: float | None
    is_tactical: bool
    is_primary_ticker: bool


def _action_from_signal(signal: str | None) -> str:
    signal = str(signal or "NO_DATA")
    if signal in {"BUY", "ACCUMULATE"}:
        return "APPROVED"
    if signal == "HOLD":
        return "WATCHLIST"
    if signal == "WATCH":
        return "WATCHLIST"
    if signal in {"WAIT", "AVOID"}:
        return "REJECT"
    return "NO_DATA"


def _selection_priority(signal: str | None, is_primary_ticker: bool) -> tuple[int, int]:
    action = _action_from_signal(signal)
    action_rank = {
        "APPROVED": 0,
        "BUILD_CORE": 1,
        "WATCHLIST": 2,
        "NO_DATA": 3,
        "REJECT": 4,
    }.get(action, 9)
    return (action_rank, 0 if is_primary_ticker else 1)


_TIMING_HOT_R_1Y = 30.0
_TIMING_HOT_DRAWDOWN = -7.0
_TIMING_HOT_RANGE_POS = 0.80


def _timing_is_stretched(r_1y, drawdown_52w, range_52w_pos) -> bool:
    if r_1y is not None and r_1y >= _TIMING_HOT_R_1Y:
        return True
    if drawdown_52w is not None and drawdown_52w >= _TIMING_HOT_DRAWDOWN:
        return True
    if range_52w_pos is not None and range_52w_pos >= _TIMING_HOT_RANGE_POS:
        return True
    return False


def _gate_action(
    signal: str,
    theme_regime: str,
    r_1y,
    drawdown_52w,
    range_52w_pos,
) -> tuple[str, list[str]]:
    """Combine factor signal, theme regime, and timing fields into a final action.

    Returns (action, reasons). ``reasons`` is a list of short strings appended
    to the rationale so the UI explains *why* the action is what it is.
    """
    base = _action_from_signal(signal)
    reasons: list[str] = []

    if theme_regime in REGIME_REJECT:
        return "REJECT", ["theme regime: falling knife"]

    timing_hot = _timing_is_stretched(r_1y, drawdown_52w, range_52w_pos)
    theme_hot = theme_regime == "strong_but_stretched"
    theme_watch = theme_regime in REGIME_WATCH
    theme_ok = theme_regime in REGIME_BUY_NOW
    theme_dead = theme_regime == "dead_money"

    if base in {"APPROVED", "BUILD_CORE"}:
        # All buy-now paths require a supportive theme regime.
        if theme_hot or timing_hot:
            reasons.append("entry timing stretched" if timing_hot else "theme regime stretched")
            return "WATCHLIST", reasons
        if theme_watch:
            # Washed-out zone: only stay APPROVED if factor signal was strong enough
            # to justify a contrarian add. BUY/ACCUMULATE from the factor layer clears
            # this bar; HOLD does not and should not have reached APPROVED here.
            if signal in {"BUY", "ACCUMULATE"}:
                reasons.append("contrarian add vs washed-out theme")
                return "APPROVED", reasons
            reasons.append("theme regime: washed out — wait for repair")
            return "WATCHLIST", reasons
        if theme_dead:
            reasons.append("theme regime: dead money")
            return "WATCHLIST", reasons
        if theme_ok:
            return base, reasons
        # NO_DATA regime (theme had no snapshot): fall back to the pre-regime behaviour
        return base, reasons

    if base == "WATCHLIST":
        # Keep on the watchlist, but escalate to REJECT if theme is in freefall.
        return "WATCHLIST", reasons

    if base == "REJECT":
        return "REJECT", reasons

    return base, reasons


def _fallback_row(account_type: str, sleeve: str, primary_ticker: str) -> dict:
    ins = lookup(primary_ticker)
    return {
        "Wrapper": account_type,
        "Sleeve": sleeve.replace("_", " ").title(),
        "Ticker": primary_ticker,
        "Name": ins.name if ins else primary_ticker,
        "Vehicle": ins.vehicle_type if ins else "unknown",
        "Reporting": ins.is_reporting_fund if ins else None,
        "Strategy signal": "NO_DATA",
        "Candidate score": None,
        "Rationale": "no factor data yet",
    }


def build_portfolio_plan(
    score_df: pd.DataFrame,
    bucket_sizes: dict[str, float],
    thresholds: StrategyThresholds = StrategyThresholds(),
    include_tactical: bool = True,
    selection_mode: str = "primary_first",
    timing_df: pd.DataFrame | None = None,
    theme_regimes: dict[str, str] | None = None,
    bond_triggers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a proposed portfolio line item per strategic sleeve.

    selection_mode:
      - primary_first: keep the strategic primary ticker unless a better row for
        that same ticker is available in factor data
      - best_candidate: allow any same-sleeve wrapper-eligible instrument to win

    theme_regimes:
      - optional {sleeve_key: regime_label} from ``themes.get_theme_regimes``.
        When provided, the regime participates in the final action gate (see
        ``_gate_action``). When absent the constructor falls back to signal +
        timing only, preserving prior behaviour.

    bond_triggers:
      - optional {trigger_key: extra_weight} from ``valuation.compute_bond_triggers``.
        When provided, tactical sleeves whose trigger is not firing are emitted
        as explicit ``NOT_ACTIVE`` rows with zero target weight, instead of being
        dropped. When absent and ``include_tactical`` is True, tactical sleeves
        are shown at their cap weight (same as pre-2026-04 behaviour).
    """
    # Prefer the price/momentum timing universe when available — it covers all
    # instruments with price history, including bonds/gold/commodities that
    # never had PE/factor rows. Fall back to the legacy factor score_df when
    # timing is missing so the call stays backwards-compatible.
    wrapper_source = timing_df if timing_df is not None and not timing_df.empty else score_df
    wrapper_df = build_wrapper_candidate_table(wrapper_source, thresholds=thresholds)
    timing_idx = timing_df.set_index("ticker") if timing_df is not None and not timing_df.empty else pd.DataFrame()
    regimes = dict(theme_regimes or {})
    triggers = dict(bond_triggers or {})

    rows: list[dict] = []
    for account_type, bucket in ALL_BUCKETS.items():
        bucket_size = float(bucket_sizes.get(account_type, 0.0))
        bucket_rows = wrapper_df[wrapper_df["Wrapper"] == account_type].copy() if not wrapper_df.empty else pd.DataFrame()
        for target in bucket.targets:
            if target.is_tactical and not include_tactical:
                continue

            # Only tactical sleeves driven by a macro trigger (bonds today) get the
            # NOT_ACTIVE treatment. Thematic tactical caps like clean-energy are
            # gated by factor + theme regime, not by macro triggers.
            tactical_dormant = False
            if target.is_tactical and bond_triggers is not None:
                trigger_key = _TACTICAL_SLEEVE_TO_TRIGGER.get(target.sleeve)
                if trigger_key is not None:
                    tactical_dormant = triggers.get(trigger_key, 0.0) <= 0.0

            if tactical_dormant:
                ins = lookup(target.primary_ticker)
                theme_regime = regimes.get(target.sleeve, "no_data")
                rows.append(
                    ProposedPosition(
                        account_type=account_type,
                        sleeve=target.sleeve,
                        target_weight=0.0,
                        target_gbp=0.0,
                        ticker=target.primary_ticker,
                        name=ins.name if ins else target.primary_ticker,
                        vehicle=ins.vehicle_type if ins else "unknown",
                        strategy_signal="NO_DATA",
                        theme_regime=theme_regime,
                        action="NOT_ACTIVE",
                        candidate_score=None,
                        rationale=f"tactical sleeve — trigger not firing (cap {target.weight:.0%})",
                        r_1m=None,
                        r_3m=None,
                        r_6m=None,
                        r_1y=None,
                        drawdown_52w=None,
                        price_ma200=None,
                        range_52w_pos=None,
                        pct_above_ma200=None,
                        vol_3m=None,
                        vol_1y=None,
                        z_1mo=None,
                        is_tactical=True,
                        is_primary_ticker=True,
                    ).__dict__
                )
                continue

            sleeve_label = target.sleeve.replace("_", " ").title()
            sleeve_rows = bucket_rows[bucket_rows["Sleeve"] == sleeve_label].copy() if not bucket_rows.empty else pd.DataFrame()

            primary_rows = sleeve_rows[sleeve_rows["Ticker"].astype(str) == target.primary_ticker].copy() if not sleeve_rows.empty else pd.DataFrame()

            if selection_mode == "primary_first" and not primary_rows.empty:
                chosen = primary_rows.iloc[0].to_dict()
            elif sleeve_rows.empty:
                chosen = _fallback_row(account_type, target.sleeve, target.primary_ticker)
            else:
                sleeve_rows["is_primary_ticker"] = sleeve_rows["Ticker"].astype(str) == target.primary_ticker
                sleeve_rows["_priority"] = sleeve_rows.apply(
                    lambda r: _selection_priority(r.get("Strategy signal"), bool(r.get("is_primary_ticker"))),
                    axis=1,
                )
                sleeve_rows = sleeve_rows.sort_values(
                    by=["_priority", "Candidate score"],
                    ascending=[True, False],
                    na_position="last",
                )
                chosen = sleeve_rows.iloc[0].to_dict()

            ticker = str(chosen["Ticker"])
            ins = lookup(ticker)
            signal = str(chosen.get("Strategy signal") or "NO_DATA")
            timing = timing_idx.loc[ticker] if not timing_idx.empty and ticker in timing_idx.index else None

            def _timing_val(key: str) -> float | None:
                if timing is None:
                    return None
                v = timing.get(key)
                return float(v) if v is not None and pd.notna(v) else None

            r_1m = _timing_val("r_1m")
            r_3m = _timing_val("r_3m")
            r_6m = _timing_val("r_6m")
            r_1y = _timing_val("r_1y")
            drawdown_52w = _timing_val("drawdown_52w")
            pct_above_ma200_val = _timing_val("pct_above_ma200")
            vol_3m = _timing_val("vol_3m")
            vol_1y = _timing_val("vol_1y")
            z_1mo = _timing_val("z_1mo")
            price_ma200 = None
            range_52w_pos = None
            # New strategy.py emits "% vs MA200" (percent, e.g. +5.4); convert
            # to a Price/MA200 ratio for the downstream gate. Fall back to the
            # legacy column name when a caller still passes score_df.
            row_pct_ma = chosen.get("% vs MA200")
            if row_pct_ma is not None and pd.notna(row_pct_ma):
                price_ma200 = 1.0 + float(row_pct_ma) / 100.0
            else:
                row_price_ma200 = chosen.get("Price/MA200")
                if row_price_ma200 is not None and pd.notna(row_price_ma200):
                    price_ma200 = float(row_price_ma200)
            row_range_52w = chosen.get("52W range pos")
            if row_range_52w is None or (isinstance(row_range_52w, float) and pd.isna(row_range_52w)):
                row_range_52w = chosen.get("range_52w_pos")
            if row_range_52w is not None and pd.notna(row_range_52w):
                range_52w_pos = float(row_range_52w)

            theme_regime = regimes.get(target.sleeve, "no_data")
            action, gate_reasons = _gate_action(signal, theme_regime, r_1y, drawdown_52w, range_52w_pos)

            rationale = str(chosen.get("Rationale") or "")
            if gate_reasons:
                rationale = "; ".join([p for p in (rationale, *gate_reasons) if p])

            rows.append(
                ProposedPosition(
                    account_type=account_type,
                    sleeve=target.sleeve,
                    target_weight=float(target.weight),
                    target_gbp=bucket_size * float(target.weight),
                    ticker=ticker,
                    name=str(chosen.get("Name") or (ins.name if ins else ticker)),
                    vehicle=str(chosen.get("Vehicle") or (ins.vehicle_type if ins else "unknown")),
                    strategy_signal=signal,
                    theme_regime=theme_regime,
                    action=action,
                    candidate_score=(
                        float(chosen["Candidate score"])
                        if chosen.get("Candidate score") is not None and pd.notna(chosen.get("Candidate score"))
                        else None
                    ),
                    rationale=rationale,
                    r_1m=r_1m,
                    r_3m=r_3m,
                    r_6m=r_6m,
                    r_1y=r_1y,
                    drawdown_52w=drawdown_52w,
                    price_ma200=price_ma200,
                    range_52w_pos=range_52w_pos,
                    pct_above_ma200=pct_above_ma200_val,
                    vol_3m=vol_3m,
                    vol_1y=vol_1y,
                    z_1mo=z_1mo,
                    is_tactical=bool(target.is_tactical),
                    is_primary_ticker=bool(ticker == target.primary_ticker),
                ).__dict__
            )

    return pd.DataFrame(rows)


def build_satellite_candidates(
    score_df: pd.DataFrame,
    top_n: int = 15,
    thresholds: StrategyThresholds = StrategyThresholds(),
) -> pd.DataFrame:
    """Optional non-core candidates, mainly for direct stocks and alternates."""
    wrapper_df = build_wrapper_candidate_table(score_df, thresholds=thresholds)
    if wrapper_df.empty:
        return pd.DataFrame()

    out = wrapper_df.copy()
    out = out[
        out["Ticker"].astype(str) != out.groupby(["Wrapper", "Sleeve"])["Ticker"].transform("first").astype(str)
    ]
    out = out[out["Strategy signal"].astype(str).isin(["BUY", "ACCUMULATE", "WATCH", "HOLD"])]
    out = out.sort_values(["Candidate score"], ascending=[False], na_position="last")
    return out.head(top_n).reset_index(drop=True)


def summarize_portfolio_plan(plan_df: pd.DataFrame) -> pd.DataFrame:
    if plan_df.empty:
        return pd.DataFrame()
    out = (
        plan_df.groupby(["account_type", "action"], as_index=False)
        .agg(target_gbp=("target_gbp", "sum"), sleeves=("sleeve", "count"))
        .sort_values(["account_type", "target_gbp"], ascending=[True, False])
    )
    return out
