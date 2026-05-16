"""Valuation rule engine.

Pure price-performance tilts — no P/E or earnings data needed. Uses the
price/MA200 ratio as the sole signal for regional equity tilts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegionData:
    name: str
    price: float
    ma200: float


@dataclass
class MacroData:
    uk_real_yield_10y: float = 0.0
    em_hy_spread: float = 0.0
    us_10y_nominal: float = 0.0
    acwi_drawdown_30d: float = 0.0


@dataclass
class DeploymentState:
    months_remaining: int = 12
    cash_remaining: float = 0.0
    total_initial: float = 0.0


def compute_region_tilts(regions: dict[str, RegionData]) -> dict[str, float]:
    """Tilt multiplier in [0.5, 1.5] based on price/MA200 ratio only.

    Below MA → overweight (mean-reversion opportunity).
    Above MA → underweight (extended).
    Falling-knife and parabolic filters dampen extremes.
    """
    if not regions:
        return {}
    tilts: dict[str, float] = {}
    for r, d in regions.items():
        ratio = d.price / d.ma200 if d.ma200 > 0 else 1.0
        if ratio < 0.85:
            t = 1.20
        elif ratio < 0.90:
            t = 1.35
        elif ratio < 1.00:
            t = 1.15
        elif ratio < 1.10:
            t = 1.00
        elif ratio < 1.20:
            t = 0.85
        elif ratio < 1.30:
            t = 0.70
        else:
            t = 0.55
        tilts[r] = float(np.clip(t, 0.5, 1.5))
    return tilts


def compute_bond_triggers(macro: MacroData) -> dict[str, float]:
    """Returns target weights for tactical bond sleeves. Zero = inactive."""
    return {
        "linkers": 0.10 if macro.uk_real_yield_10y > 0.015 else 0.0,
        "em_usd": 0.06 if macro.em_hy_spread > 0.06 else 0.0,
        "long_dur": 0.10 if macro.us_10y_nominal > 0.05 else 0.0,
    }


def compute_deployment_pace(state: DeploymentState, macro: MacroData) -> float:
    """Fraction of remaining cash to deploy this month."""
    if state.months_remaining <= 0 or state.cash_remaining <= 0:
        return 0.0
    base = 1.0 / state.months_remaining
    dd = macro.acwi_drawdown_30d
    if dd < -0.20:
        return min(0.25, state.cash_remaining / max(state.total_initial, 1.0))
    if dd < -0.10:
        return base + 0.10
    if dd < -0.05:
        return base + 0.05
    return base
