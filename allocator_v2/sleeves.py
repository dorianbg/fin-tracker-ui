"""Sleeve policy + bucket-ERC / bucket-HRP wrappers.

The sleeve layer is a *policy* on top of the risk-parity solvers. We decide
the split across asset classes (equity / real / bonds / cash) as a user-facing
dial, then run ERC or HRP inside each sleeve on the sleeve's sub-covariance.

Why this, instead of raw ERC/HRP over all 16 assets:

- Raw HRP in particular can drop 50–60% into the bond cluster because bonds
  have low vol and low correlation to equities. That violates the user's
  drawdown-aversion philosophy (bonds are distrusted as a reliable hedge in
  the current regime).
- Raw ERC is slightly kinder to bonds but still doesn't respect a hard cap.
- A post-hoc re-scale breaks the risk-parity property *within* the surviving
  assets.

Bucket-ERC solves both problems: sleeve weights are a policy choice, and
the solver handles variance-equalisation inside each sleeve where it
actually makes sense.

Inside each sleeve, the solver gives lower-vol assets bigger nominal slices
(because ERC equalises *risk contribution*, not nominal weight). So MVOL
ends up bigger than EEM in the equity sleeve — as intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from allocator_v2 import universe as uni
from allocator_v2.sizers.erc import erc_weights
from allocator_v2.sizers.hrp import hrp_weights


# Default sleeve policy. User can override via sidebar sliders.
DEFAULT_SLEEVES: dict[str, float] = {
    "equity":     0.70,
    "real":       0.20,
    "bonds":      0.10,
    "cash":       0.00,
}

# Asset → sleeve mapping. Each universe asset belongs to exactly one sleeve.
ASSET_SLEEVE: dict[str, str] = {
    # Equity
    "IWDA": "equity",
    "EEM":  "equity",
    "VGK":  "equity",
    "EWJ":  "equity",
    "IWQU": "equity",
    "MVOL": "equity",
    "IWMO": "equity",
    "IWVL": "equity",
    "ISF":  "equity",
    "XLE":  "equity",
    # Real assets — defensive (low-vol inflation hedges)
    "SGLD": "real",
    "VNQ":  "real",
    "INFR": "real",
    # Real assets — cyclical (high-convexity inflation trades)
    "GDX":  "real",
    # Bonds
    "TLT":  "bonds",
    "INXG": "bonds",
    "TIP":  "bonds",
}

# Sub-sleeve split *within* the real sleeve. Defensive assets get the larger
# share because they're lower-vol and more reliable inflation carriers;
# cyclical assets get the smaller share but punch above their weight in
# inflation+crisis regimes (Dalio's real-asset subweighting pattern).
REAL_SUBSLEEVE: dict[str, str] = {
    "SGLD": "defensive",
    "VNQ":  "defensive",
    "INFR": "defensive",
    "GDX":  "cyclical",
}

DEFAULT_REAL_SPLIT: dict[str, float] = {"defensive": 0.40, "cyclical": 0.60}


@dataclass(frozen=True)
class SleevePolicy:
    """Policy spec for one call to the bucket sizer."""
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SLEEVES))
    bond_cap: float = 0.20
    real_split: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REAL_SPLIT))
    factor_tilt_strength: float = 1.0  # scales each asset's universe.tilt_bias. 0 = disable.

    # MA-relative (mean-reversion) tilt: overweight below-MA, underweight above-MA.
    # ``ma_ratios`` maps ticker → price/ma_252. Missing tickers get neutral (1.0).
    ma_ratios: dict[str, float] = field(default_factory=dict)
    ma_tilt_strength: float = 1.0        # 0 = disable, 1 = default table, 1.5 = aggressive
    ma_neutral_band: tuple[float, float] = (0.90, 1.10)
    ma_sleeves: tuple[str, ...] = ("equity", "real")  # which sleeves receive the tilt
    ma_barbell: bool = False             # if True, use U-shaped curve (mean-rev + trend on extremes)

    # Minimum per-asset weight floor. Assets below this are dropped and their
    # weight redistributed pro-rata to surviving names in the same sleeve.
    # 0 disables.
    min_weight: float = 0.03

    # Maximum per-asset weight cap. Excess weight above the cap is redistributed
    # pro-rata to uncapped names in the same sleeve. 0 disables.
    max_weight: float = 0.10

    def normalised(self) -> dict[str, float]:
        w = {k: max(0.0, v) for k, v in self.weights.items()}
        # Enforce bond cap; overflow redistributes proportionally to real + equity.
        if w.get("bonds", 0.0) > self.bond_cap:
            excess = w["bonds"] - self.bond_cap
            w["bonds"] = self.bond_cap
            remainder = {k: w[k] for k in ("equity", "real") if k in w and w[k] > 0}
            total_rem = sum(remainder.values())
            if total_rem > 0:
                for k in remainder:
                    w[k] = w[k] + excess * (w[k] / total_rem)
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else w


def sleeve_members(cov: pd.DataFrame) -> dict[str, list[str]]:
    """Group the active assets (those in ``cov``) by sleeve."""
    out: dict[str, list[str]] = {s: [] for s in DEFAULT_SLEEVES}
    for t in cov.index:
        sleeve = ASSET_SLEEVE.get(t)
        if sleeve and sleeve in out:
            out[sleeve].append(t)
    return out


def _solve_within_sleeve(
    cov: pd.DataFrame,
    corr: pd.DataFrame | None,
    tickers: list[str],
    method: str,
) -> pd.Series:
    """Run the chosen risk-parity solver on the sleeve's sub-matrix."""
    if not tickers:
        return pd.Series(dtype=float)
    if len(tickers) == 1:
        return pd.Series([1.0], index=tickers)
    sub_cov = cov.loc[tickers, tickers]
    if method == "erc":
        return erc_weights(sub_cov)
    if method == "hrp":
        if corr is None:
            return erc_weights(sub_cov)
        sub_corr = corr.loc[tickers, tickers]
        return hrp_weights(sub_cov, sub_corr)
    raise ValueError(f"unknown method {method}")


def _ma_multiplier(ratio: float, neutral_band: tuple[float, float]) -> float:
    """Piecewise multiplier from price/MA ratio. <MA → >1, >MA → <1."""
    lo, hi = neutral_band
    if ratio < 0.85:       return 1.35
    if ratio < lo:         return 1.20
    if ratio < 1.00:       return 1.08
    if ratio < hi:         return 1.00
    if ratio < 1.20:       return 0.85
    if ratio < 1.35:       return 0.70
    return 0.55


def _ma_barbell_multiplier(ratio: float) -> float:
    """U-shaped curve: overweight both tails (oversold + strong uptrend), underweight the mushy middle.

    Buckets:
      <0.85        → 1.30 (oversold, mean-reversion)
      0.85–0.95    → 1.15
      0.95–1.05    → 0.85 (boring middle, no edge)
      1.05–1.20    → 1.15 (healthy uptrend — momentum leg)
      1.20–1.35    → 1.20 (strong trend, still below parabolic)
      >=1.35       → 0.70 (parabolic, dampen)
    """
    if ratio < 0.85:       return 1.30
    if ratio < 0.95:       return 1.15
    if ratio < 1.05:       return 0.85
    if ratio < 1.20:       return 1.15
    if ratio < 1.35:       return 1.20
    return 0.70


def apply_ma_tilt(
    weights: pd.Series,
    ma_ratios: dict[str, float],
    strength: float = 1.0,
    neutral_band: tuple[float, float] = (0.90, 1.10),
    barbell: bool = False,
) -> pd.Series:
    """Mean-reversion tilt: overweight below-MA, underweight above-MA.

    Multiplier ``m = 1 + strength · (table(ratio) - 1)`` so strength=0 is a
    no-op and strength=1 applies the full table. Clipped to [0.3, 1.6].
    Renormalises to preserve the sleeve's total weight.
    """
    if weights.empty or strength <= 0 or not ma_ratios:
        return weights
    mults = {}
    any_active = False
    for t in weights.index:
        r = ma_ratios.get(t)
        if r is None or not np.isfinite(r):
            mults[t] = 1.0
            continue
        base = _ma_barbell_multiplier(float(r)) if barbell else _ma_multiplier(float(r), neutral_band)
        m = 1.0 + strength * (base - 1.0)
        m = max(0.3, min(1.6, m))
        mults[t] = m
        if abs(m - 1.0) > 1e-9:
            any_active = True
    if not any_active:
        return weights
    mseries = pd.Series(mults, index=weights.index)
    tilted = weights * mseries
    s = tilted.sum()
    if s <= 0:
        return weights
    return tilted * (weights.sum() / s)


def apply_factor_tilts(weights: pd.Series, strength: float = 1.0) -> pd.Series:
    """Multiply each asset's weight by ``1 + strength · universe.tilt_bias`` and renormalise.

    Used *inside* a sleeve after the risk-parity solver runs. Floors the
    multiplier at 0.1 so a strong negative tilt can shrink a weight but not
    flip its sign. Renormalises to preserve the sleeve's total weight.
    """
    from allocator_v2 import universe as uni  # avoid circular import at module load
    if weights.empty or strength <= 0:
        return weights
    biases = {t: uni.UNIVERSE[t].tilt_bias for t in weights.index if t in uni.UNIVERSE}
    if not any(abs(b) > 1e-9 for b in biases.values()):
        return weights
    multipliers = pd.Series(
        {t: max(0.1, 1.0 + strength * biases.get(t, 0.0)) for t in weights.index},
        index=weights.index,
    )
    tilted = weights * multipliers
    s = tilted.sum()
    if s <= 0:
        return weights
    return tilted * (weights.sum() / s)


def apply_max_weight_cap(
    weights: pd.Series,
    max_weight: float,
    sleeve_map: dict[str, str],
) -> pd.Series:
    """Cap any asset above ``max_weight``; redistribute excess pro-rata within the same sleeve.

    Iterative: after one round of redistribution, another name may now exceed
    the cap, so we loop until stable (or a hard iteration limit). Sleeve totals
    are preserved. If *every* name in a sleeve is capped, the remaining excess
    is left on the capped names (sleeve is over-concentrated by design — can't
    spread it further without breaking the cap).
    """
    if weights.empty or max_weight <= 0 or max_weight >= 1.0:
        return weights
    out = weights.copy()
    for _ in range(20):
        changed = False
        for sleeve in set(sleeve_map.values()):
            members = [t for t in out.index if sleeve_map.get(t) == sleeve]
            if len(members) < 2:
                continue
            capped = [t for t in members if out.loc[t] > max_weight + 1e-9]
            if not capped:
                continue
            uncapped = [t for t in members if t not in capped]
            if not uncapped:
                continue
            excess = float(sum(out.loc[t] - max_weight for t in capped))
            out.loc[capped] = max_weight
            uncapped_sum = float(out.loc[uncapped].sum())
            if uncapped_sum <= 0:
                # fall back to equal-split
                for t in uncapped:
                    out.loc[t] = excess / len(uncapped)
            else:
                for t in uncapped:
                    out.loc[t] = out.loc[t] + excess * out.loc[t] / uncapped_sum
            changed = True
        if not changed:
            break
    s = out.sum()
    return out / s if s > 0 else out


def apply_min_weight_floor(
    weights: pd.Series,
    min_weight: float,
    sleeve_map: dict[str, str],
) -> pd.Series:
    """Drop sub-threshold weights; redistribute pro-rata within the same sleeve.

    Runs after all sleeve/tilt logic on the final weight vector. If every asset
    in a sleeve is below the floor, the sleeve's total is preserved by keeping
    the largest one. Global normalisation at the end keeps the vector on the
    simplex.
    """
    if weights.empty or min_weight <= 0:
        return weights
    out = weights.copy()
    for sleeve in set(sleeve_map.values()):
        members = [t for t in out.index if sleeve_map.get(t) == sleeve]
        if not members:
            continue
        sleeve_total = float(out.loc[members].sum())
        if sleeve_total <= 0:
            continue
        survivors = [t for t in members if out.loc[t] >= min_weight]
        if not survivors:
            # keep the largest so the sleeve isn't zeroed out
            survivors = [max(members, key=lambda t: out.loc[t])]
        killed = [t for t in members if t not in survivors]
        if not killed:
            continue
        surv_sum = float(out.loc[survivors].sum())
        if surv_sum <= 0:
            continue
        out.loc[killed] = 0.0
        out.loc[survivors] = out.loc[survivors] * (sleeve_total / surv_sum)
    s = out.sum()
    return out / s if s > 0 else out


def bucket_weights(
    cov: pd.DataFrame,
    corr: pd.DataFrame | None,
    policy: SleevePolicy | None = None,
    method: str = "erc",
) -> pd.Series:
    """Compute bucket-constrained weights: sleeve caps × within-sleeve solver.

    ``method``: ``"erc"`` (default) or ``"hrp"``. Returns a Series summing
    to ~1 over the active universe (cov.index), with sleeve totals matching
    the policy (up to the bond cap).

    Sleeves that exist in the policy but have no active assets (typically
    ``cash``) have their weight redistributed proportionally to the other
    sleeves.
    """
    if cov.empty:
        return pd.Series(dtype=float)

    pol = policy or SleevePolicy()
    sleeve_caps = pol.normalised()
    members = sleeve_members(cov)

    # Redistribute weight from assetless sleeves (e.g. 'cash') to sleeves
    # with at least one active asset.
    active_sleeves = {s for s, ts in members.items() if ts}
    orphan = sum(v for s, v in sleeve_caps.items() if s not in active_sleeves)
    if orphan > 0 and active_sleeves:
        kept = {s: sleeve_caps[s] for s in active_sleeves}
        total_kept = sum(kept.values())
        if total_kept > 0:
            sleeve_caps = {s: v / total_kept for s, v in kept.items()}
        else:
            sleeve_caps = {s: 1.0 / len(active_sleeves) for s in active_sleeves}

    out = pd.Series(0.0, index=cov.index)
    for sleeve, cap in sleeve_caps.items():
        if cap <= 0:
            continue
        tickers = members.get(sleeve, [])
        if not tickers:
            continue
        if sleeve == "real":
            sub_w = _solve_real_sleeve(cov, corr, tickers, method, pol.real_split)
        else:
            sub_w = _solve_within_sleeve(cov, corr, tickers, method)
        if sleeve == "equity":
            sub_w = apply_factor_tilts(sub_w, strength=pol.factor_tilt_strength)
        if sleeve in pol.ma_sleeves:
            sub_w = apply_ma_tilt(
                sub_w,
                ma_ratios=pol.ma_ratios,
                strength=pol.ma_tilt_strength,
                neutral_band=pol.ma_neutral_band,
                barbell=pol.ma_barbell,
            )
        out.loc[sub_w.index] = out.loc[sub_w.index].values + cap * sub_w.values

    s = out.sum()
    return out / s if s > 0 else out


def _solve_real_sleeve(
    cov: pd.DataFrame,
    corr: pd.DataFrame | None,
    tickers: list[str],
    method: str,
    split: dict[str, float],
) -> pd.Series:
    """Two-tier solver for the real sleeve.

    Splits ``tickers`` into defensive + cyclical groups per ``REAL_SUBSLEEVE``,
    allocates ``split[group]`` of the sleeve weight to each group, then runs
    the chosen solver inside each group. Ensures miners/commodities/energy
    always get a meaningful slot instead of being crushed by ERC's low-vol
    preference.

    Returns a Series summing to 1 over the input tickers.
    """
    defensive = [t for t in tickers if REAL_SUBSLEEVE.get(t) == "defensive"]
    cyclical = [t for t in tickers if REAL_SUBSLEEVE.get(t) == "cyclical"]
    untagged = [t for t in tickers if t not in REAL_SUBSLEEVE]

    total_split = sum(max(0.0, v) for v in split.values())
    if total_split <= 0:
        return _solve_within_sleeve(cov, corr, tickers, method)
    shares = {k: max(0.0, v) / total_split for k, v in split.items()}

    # Redistribute weight from empty sub-sleeves to populated ones.
    if not defensive:
        shares["cyclical"] = shares.get("cyclical", 0.0) + shares.get("defensive", 0.0)
        shares["defensive"] = 0.0
    if not cyclical:
        shares["defensive"] = shares.get("defensive", 0.0) + shares.get("cyclical", 0.0)
        shares["cyclical"] = 0.0

    out = pd.Series(0.0, index=tickers)
    if defensive and shares.get("defensive", 0) > 0:
        w = _solve_within_sleeve(cov, corr, defensive, method)
        out.loc[w.index] = out.loc[w.index].values + shares["defensive"] * w.values
    if cyclical and shares.get("cyclical", 0) > 0:
        w = _solve_within_sleeve(cov, corr, cyclical, method)
        out.loc[w.index] = out.loc[w.index].values + shares["cyclical"] * w.values
    if untagged:
        # Shouldn't happen in normal use; fall back to equal ERC share.
        w = _solve_within_sleeve(cov, corr, untagged, method)
        out.loc[w.index] = out.loc[w.index].values + 0.1 * w.values

    s = out.sum()
    return out / s if s > 0 else out


def bucket_erc(
    cov: pd.DataFrame,
    policy: SleevePolicy | None = None,
) -> pd.Series:
    """Bucket-ERC: sleeve caps × ERC inside each sleeve."""
    return bucket_weights(cov, corr=None, policy=policy, method="erc")


def bucket_hrp(
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    policy: SleevePolicy | None = None,
) -> pd.Series:
    """Bucket-HRP: sleeve caps × HRP inside each sleeve."""
    return bucket_weights(cov, corr=corr, policy=policy, method="hrp")


def apply_bond_cap_aw(aw: pd.Series, bond_cap: float = 0.20) -> pd.Series:
    """Post-hoc bond cap for the AW sizer.

    AW's bond exposure comes from the quadrant prior (stag quadrant weight
    largely hits linkers/TIPS; disinflationary-recession hits long TSY). With
    a 35 + 10 = 45% prior on those quadrants, the AW sizer can easily put
    25–30% into bonds. We cap the total bond-sleeve weight and redistribute
    overflow proportionally to equity + real assets.
    """
    if aw.empty:
        return aw
    bond_tickers = [t for t in aw.index if ASSET_SLEEVE.get(t) == "bonds"]
    bond_sum = float(aw.loc[bond_tickers].sum()) if bond_tickers else 0.0
    if bond_sum <= bond_cap:
        return aw
    scale = bond_cap / bond_sum
    out = aw.copy()
    out.loc[bond_tickers] = out.loc[bond_tickers] * scale
    excess = bond_sum - bond_cap
    donors = [t for t in out.index if ASSET_SLEEVE.get(t) in ("equity", "real") and out[t] > 0]
    donor_sum = float(out.loc[donors].sum())
    if donors and donor_sum > 0:
        for t in donors:
            out.loc[t] = out.loc[t] + excess * out.loc[t] / donor_sum
    s = out.sum()
    return out / s if s > 0 else out
