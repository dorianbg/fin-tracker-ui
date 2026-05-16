"""Macro-quadrant probability engine.

Four quadrants: (growth↑/↓) × (inflation↑/↓).

**Baseline prior** (set by user directive):

    inflation_up + growth_down : 35 %   (stagflationary scare)
    inflation_up + growth_up   : 30 %   (overheat / late cycle)
    inflation_down + growth_up : 25 %   (goldilocks)
    inflation_down + growth_down : 10 % (disinflationary recession)

**Data-driven tilts** — four cheap signals that nudge the prior toward a
posterior:

  1. Growth signal — ACWI 90d return. Positive → growth_up weight rises.
  2. Inflation signal — gold+commodities 90d return vs equities. Positive
     spread → inflation_up weight rises (real assets outperforming when
     inflation is surprising).
  3. Rate signal — TLT 90d return. Positive (long bonds rallying) →
     growth_down weight rises (market pricing in cuts / recession).
  4. Breadth signal — dispersion across regional equity returns (developed
     vs EM). High dispersion → growth_up weight falls (regime uncertainty).

Each signal is converted to a z-score vs its own 1-year distribution and
mapped through a sigmoid to a [-1, +1] tilt. Tilts are multiplied onto the
prior (clipped to avoid sign flips) and renormalised to sum to 1.

The user can override with sliders in the dashboard; the data-driven read
is a suggestion, not a hard override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PRIOR: dict[str, float] = {
    "inflation_up_growth_down":   0.35,
    "inflation_up_growth_up":     0.30,
    "inflation_down_growth_up":   0.25,
    "inflation_down_growth_down": 0.10,
}

QUADRANT_KEYS = tuple(PRIOR.keys())


@dataclass(frozen=True)
class MacroSignals:
    """Raw inputs → four z-scored tilts."""
    growth_z: float    # ACWI 90d return vs its 1y distribution
    inflation_z: float # (gold+commodities) minus equities, 90d
    rate_z: float      # TLT 90d return
    breadth_z: float   # regional dispersion (lower = tighter = more growth_up)


def _sigmoid(x: float, scale: float = 1.0) -> float:
    return 2.0 / (1.0 + np.exp(-x * scale)) - 1.0


def compute_signals(rets_wide: pd.DataFrame) -> MacroSignals:
    """Derive the four signals from a daily-return matrix.

    ``rets_wide`` must have columns including: VWRP, EEM, VGK, EWJ (for
    growth + breadth), SGLD, GSG (for inflation), IBGL (for rate).

    Missing columns degrade gracefully to a zero z-score on that axis,
    which pulls the posterior back toward the prior.
    """
    def _roll_return(col: str, days: int) -> pd.Series | None:
        if col not in rets_wide.columns:
            return None
        s = rets_wide[col].dropna()
        if len(s) < days:
            return None
        return (1.0 + s).rolling(window=days).apply(np.prod, raw=True) - 1.0

    def _zscore(series: pd.Series | None) -> float:
        if series is None:
            return 0.0
        recent = series.dropna()
        if len(recent) < 60:
            return 0.0
        latest = float(recent.iloc[-1])
        mu = float(recent.tail(252).mean())
        sigma = float(recent.tail(252).std())
        if sigma <= 0 or not np.isfinite(sigma):
            return 0.0
        return (latest - mu) / sigma

    equity_cols = [c for c in ("VWRP", "EEM", "VGK", "EWJ") if c in rets_wide.columns]
    acwi_proxy = rets_wide[equity_cols].mean(axis=1) if len(equity_cols) >= 2 else None
    acwi_90d = (1.0 + acwi_proxy).rolling(90).apply(np.prod, raw=True) - 1.0 \
        if acwi_proxy is not None else None

    real_cols = [c for c in ("SGLD", "GSG") if c in rets_wide.columns]
    gold_proxy = rets_wide[real_cols].mean(axis=1) if real_cols else None
    inflation_spread_90d = None
    if gold_proxy is not None and acwi_proxy is not None:
        gold_90d = (1.0 + gold_proxy).rolling(90).apply(np.prod, raw=True) - 1.0
        inflation_spread_90d = gold_90d - acwi_90d

    tlt_90d = _roll_return("IBGL", 90)

    breadth_90d = None
    regional_cols = equity_cols
    if len(regional_cols) >= 3:
        per_region = {}
        for c in regional_cols:
            r = (1.0 + rets_wide[c]).rolling(90).apply(np.prod, raw=True) - 1.0
            per_region[c] = r
        wide = pd.DataFrame(per_region)
        breadth_90d = wide.std(axis=1)

    return MacroSignals(
        growth_z=_zscore(acwi_90d),
        inflation_z=_zscore(inflation_spread_90d),
        rate_z=_zscore(tlt_90d),
        breadth_z=-_zscore(breadth_90d),
    )


def posterior_probabilities(
    signals: MacroSignals,
    prior: dict[str, float] | None = None,
    tilt_strength: float = 0.4,
) -> dict[str, float]:
    """Data-driven posterior over the four quadrants.

    ``tilt_strength`` controls how aggressively signals pull away from the
    prior. 0 = pure prior, 1 = each quadrant's weight can roughly double or
    halve. 0.4 is a sober default — meaningful nudge without letting one
    noisy quarter dominate.
    """
    base = dict(prior or PRIOR)

    g = _sigmoid(signals.growth_z)        # +1 = growth up
    inf = _sigmoid(signals.inflation_z)   # +1 = inflation up
    rate = _sigmoid(signals.rate_z)       # +1 = bonds rallying → growth_down
    breadth = _sigmoid(signals.breadth_z) # +1 = regimes coherent → growth_up

    growth_tilt = (g + breadth - rate) / 3.0
    inflation_tilt = inf

    multipliers = {
        "inflation_up_growth_up":     1.0 + tilt_strength * (inflation_tilt + growth_tilt) / 2.0,
        "inflation_up_growth_down":   1.0 + tilt_strength * (inflation_tilt - growth_tilt) / 2.0,
        "inflation_down_growth_up":   1.0 + tilt_strength * (-inflation_tilt + growth_tilt) / 2.0,
        "inflation_down_growth_down": 1.0 + tilt_strength * (-inflation_tilt - growth_tilt) / 2.0,
    }

    tilted = {k: max(0.02, base[k] * multipliers[k]) for k in QUADRANT_KEYS}
    total = sum(tilted.values())
    return {k: v / total for k, v in tilted.items()}


def quadrant_to_axes(probs: dict[str, float]) -> dict[str, float]:
    """Collapse the 4D posterior to marginal growth/inflation probabilities."""
    return {
        "growth_up": probs["inflation_up_growth_up"] + probs["inflation_down_growth_up"],
        "growth_down": probs["inflation_up_growth_down"] + probs["inflation_down_growth_down"],
        "inflation_up": probs["inflation_up_growth_up"] + probs["inflation_up_growth_down"],
        "inflation_down": probs["inflation_down_growth_up"] + probs["inflation_down_growth_down"],
    }
