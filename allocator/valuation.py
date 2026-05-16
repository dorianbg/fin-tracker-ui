"""Valuation rule engine.

Pure functions — no I/O, no side effects. Given region fundamentals and macro
state, compute (a) regional equity tilts, (b) tactical bond triggers, and
(c) the valuation-aware deployment pace for the initial 12-month tranched
deploy.

The philosophy is the user's: asymmetric + antifragile, cheap = less fragile,
but filter out both the "falling knife" and the "parabolic extended" states
so we don't buy the bottom of a collapse or chase a 40%-up rally.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionData:
    name: str
    forward_pe: float
    price: float
    ma200: float
    acwi_base_weight: float  # market cap weight in ACWI, 0..1


@dataclass(frozen=True)
class MacroData:
    uk_real_yield_10y: float
    us_real_yield_10y: float
    us_10y_nominal: float
    em_hy_spread: float
    acwi_drawdown_30d: float
    acwi_forward_pe_pct: float  # percentile vs own history, 0..1


@dataclass(frozen=True)
class DeploymentState:
    total_initial: float
    cash_remaining: float
    months_remaining: int


# ── Regional equity tilts ─────────────────────────────────────────────
def compute_region_tilts(regions: list[RegionData]) -> dict[str, float]:
    """Compute tilt multiplier in [0.35, 1.5] for each region.

    Base: ACWI cap weight.
    Adjust 1: earnings-yield z-score vs cross-sectional median → ±50%.
    Adjust 2: "not crashing, not extended" price filter.
        If price < 0.85 * MA200 → dampen by 0.7 (falling knife).
        If price > 1.30 * MA200 → dampen by 0.7 (extended).
    """
    if not regions:
        return {}

    eys = np.array([1.0 / r.forward_pe for r in regions])
    median_ey = float(np.median(eys))
    std_ey = float(np.std(eys)) or 1e-9

    tilts: dict[str, float] = {}
    for r in regions:
        ey = 1.0 / r.forward_pe
        z = (ey - median_ey) / std_ey
        valuation_tilt = float(np.clip(0.5 * z, -0.5, 0.5))

        ratio = r.price / r.ma200 if r.ma200 > 0 else 1.0
        if ratio < 0.85 or ratio > 1.30:
            momentum_dampener = 0.7
        else:
            momentum_dampener = 1.0

        tilts[r.name] = round((1.0 + valuation_tilt) * momentum_dampener, 4)
    return tilts


def apply_tilts_to_weights(
    regions: list[RegionData], tilts: dict[str, float]
) -> dict[str, float]:
    """Given base ACWI weights and tilt multipliers, return final (normalised)
    region weights that sum to 1."""
    raw = {r.name: r.acwi_base_weight * tilts.get(r.name, 1.0) for r in regions}
    total = sum(raw.values()) or 1.0
    return {name: w / total for name, w in raw.items()}


# ── Tactical bond triggers ────────────────────────────────────────────
def compute_bond_triggers(macro: MacroData) -> dict[str, float]:
    """Return bond sleeve weights (0..1 of bucket) that are *currently active*.

    Linkers: baseline 6% for SIPP / 8% for bucket 2 is handled in buckets.py.
    This function only returns *extra* tactical weight on top.
    """
    return {
        "linkers_extra": 0.06 if macro.uk_real_yield_10y > 0.015 else 0.0,
        "em_usd": 0.06 if macro.em_hy_spread > 0.06 else 0.0,
        "long_dur": 0.06 if macro.us_10y_nominal > 0.05 else 0.0,
    }


# ── Deployment pace ───────────────────────────────────────────────────
def compute_deployment_pace(
    state: DeploymentState, macro: MacroData
) -> tuple[float, str]:
    """Returns (fraction_of_total_initial_to_deploy_this_month, reason).

    Default: even split across remaining months.
    Accelerators on drawdowns; decelerators at extreme valuations.
    """
    if state.months_remaining <= 0 or state.cash_remaining <= 0:
        return 0.0, "deployment_complete"

    base = 1.0 / state.months_remaining  # fraction of what's *remaining*

    dd = macro.acwi_drawdown_30d
    if dd < -0.20:
        # Big accelerator — cap at 25% of the *original* total
        return min(0.25, state.cash_remaining / state.total_initial), "drawdown>20"
    if dd < -0.10:
        pct = base + 0.10
        return min(pct, state.cash_remaining / state.total_initial), "drawdown>10"
    if dd < -0.05:
        pct = base + 0.05
        return min(pct, state.cash_remaining / state.total_initial), "drawdown>5"

    if macro.acwi_forward_pe_pct > 0.90:
        return base * 0.5, "valuation>90pct"

    return base, "default"


# ── Trend Following / Downside Cap ────────────────────────────────────
def compute_trend_risk_score(regions: list[RegionData]) -> float:
    """Returns the fraction of base equity weight that is in a negative trend.
    
    A simple trend-following override. If price < MA200 for a given region,
    that region is considered in a negative trend. This computes the weighted
    sum (0 to 1) of the portfolio that is "risk-off".

    If this score exceeds 0.5, the portfolio allocator could raise cash
    or suppress further rebalancing into equity.
    """
    total_weight = 0.0
    risk_off_weight = 0.0
    for r in regions:
        total_weight += r.acwi_base_weight
        if r.ma200 > 0 and r.price < r.ma200:
            risk_off_weight += r.acwi_base_weight

    return risk_off_weight / total_weight if total_weight > 0 else 0.0

