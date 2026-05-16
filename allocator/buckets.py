"""Bucket definitions: target sleeve weights per wrapper.

Two buckets, two risk budgets:
- SIPP   → -20% drawdown tolerance, vol target ~10%
- Bucket 2 (ISA + GIA combined) → -10% drawdown tolerance, vol target ~6-7%

Within bucket 2, the ISA holds GBP equity factor + inflation + gold sleeves and
the GIA holds the USD real-asset + tactical bond sleeves. They share one
risk budget but live in separate brokers.

Cash is not an approved strategic sleeve. Bucket targets are now fully invested.
If temporary cash exists in live holdings, it is an execution residue, not a
target allocation.

Design changes (2026-04):
  - INXG (UK linkers) REMOVED from SIPP entirely. UK bond risk concentrated and
    subject to fiscal/currency flight — not wanted. Replaced with IGIL.L (global
    government inflation-linked bonds, diversified across US/EU/JP/UK).
  - INXG REMOVED from ISA strategic targets. IGIL.L (global) carries the linker role.
  - NATO.L (European defence UCITS) added to SIPP at 3% baseline.
    Aligned with Russell Napier value thesis + Draghi European investment plan.
    Defence capex supercycle = long-duration industrial spending + European fiscal
    expansion. Low PE vs growth rate (PEG < 1).
  - VEUR raised from 4% to 6% in SIPP. Europe is deeply cheap vs US on CAPE basis.
  - Cash sleeves ERNS / CSH2 / SGOV removed from strategic targets.
  - Capital freed from cash is redistributed across gold, commodities, infra,
    broad equity, and diversified linkers.

The numbers below are the *default* allocations before valuation tilts and
tactical bond triggers are applied. The valuation engine in `valuation.py`
mutates these via region tilts and bond triggers on each recompute.
"""

from dataclasses import dataclass

import instruments as inst


@dataclass(frozen=True)
class SleeveTarget:
    sleeve: str
    weight: float  # 0..1
    primary_ticker: str  # which instrument fills this sleeve in this wrapper
    is_tactical: bool = False  # if True, weight is a *cap*, not a baseline


# ── SIPP ──────────────────────────────────────────────────────────────
# Drawdown -20%, vol ~10%. GBP UCITS only, fully invested.
# No UK bonds. Global inflation-linked bonds (IGIL.L) instead.
# Includes European defence (NATO.L) aligned with Draghi capex plan.
SIPP_TARGETS: list[SleeveTarget] = [
    SleeveTarget(inst.SLEEVE_MARKET_CAP,      0.10, "CSPX"),
    SleeveTarget(inst.SLEEVE_QUALITY,         0.10, "IWQU"),
    SleeveTarget(inst.SLEEVE_MIN_VOL,         0.06, "MVOL"),
    SleeveTarget(inst.SLEEVE_VALUE,           0.08, "IWVL"),
    SleeveTarget(inst.SLEEVE_EM,              0.10, "EIMI"),
    SleeveTarget(inst.SLEEVE_JAPAN,           0.05, "IJPA"),
    SleeveTarget(inst.SLEEVE_EUROPE,          0.08, "VEUR"),
    SleeveTarget(inst.SLEEVE_UK_COMMODITY,    0.05, "ISF"),
    SleeveTarget(inst.SLEEVE_DEFENCE,         0.03, "NATO"),
    SleeveTarget(inst.SLEEVE_GOLD,            0.20, "SGLN"),
    SleeveTarget(inst.SLEEVE_LINKERS_GLOBAL,  0.10, "IGIL"),
    SleeveTarget(inst.SLEEVE_EM_BONDS_LOCAL,  0.05, "SEML"),
]

# ── Bucket 2: ISA-side ────────────────────────────────────────────────
# GBP UCITS only. Drawdown -10% is the binding constraint.
# No approved cash sleeve. ISA stays fully invested but uses defensive factors,
# global linkers, and gold rather than UK-specific cash proxies.
BUCKET2_ISA_TARGETS: list[SleeveTarget] = [
    SleeveTarget(inst.SLEEVE_MARKET_CAP,      0.15, "IWDA"),
    SleeveTarget(inst.SLEEVE_MIN_VOL,         0.20, "MVOL"),
    SleeveTarget(inst.SLEEVE_QUALITY,         0.10, "IWQU"),
    SleeveTarget(inst.SLEEVE_VALUE,           0.10, "IWVL"),
    SleeveTarget(inst.SLEEVE_EM,              0.10, "EIMI"),
    SleeveTarget(inst.SLEEVE_JAPAN,           0.05, "IJPA"),
    SleeveTarget(inst.SLEEVE_EUROPE,          0.05, "VEUR"),
    SleeveTarget(inst.SLEEVE_LINKERS_GLOBAL,  0.10, "IGIL"),
    SleeveTarget(inst.SLEEVE_GOLD,            0.15, "SGLN"),
    SleeveTarget(inst.SLEEVE_CLEAN_ENERGY,    0.05, "INRG", is_tactical=True),
]

# ── Bucket 2: GIA-side ────────────────────────────────────────────────
# US ETFs via IBKR. Reporting status assumed True per user direction.
# No approved USD cash sleeve. GIA is fully invested across real assets, inflation
# protection, and selective global sectors.
# Weights sum to 1.0 of the GIA's own £ size (tactical sleeves are caps).

BUCKET2_GIA_TARGETS: list[SleeveTarget] = [
    SleeveTarget(inst.SLEEVE_GOLD,            0.24, "IAU"),
    SleeveTarget(inst.SLEEVE_GOLD_MINERS,     0.08, "GDX"),
    SleeveTarget(inst.SLEEVE_COMMODITIES,     0.16, "PDBC"),
    SleeveTarget(inst.SLEEVE_ENERGY,          0.10, "XLE"),
    SleeveTarget(inst.SLEEVE_INFRA,           0.12, "IGF"),
    SleeveTarget(inst.SLEEVE_REITS,           0.08, "VNQ"),
    SleeveTarget(inst.SLEEVE_TIPS_US,         0.12, "TIP"),
    SleeveTarget(inst.SLEEVE_EM_BONDS_LOCAL,  0.10, "EMLC"),
    # Healthcare (XLV) and industrials (XLI) live in THEMATIC_EXTRAS as satellite
    # overlays — funded out of the satellite cap, NOT a strategic baseline weight.
    # Holding them in both places double-counted ~8% of GIA against the same
    # exposure on a lookthrough basis.
    # tactical: zero baseline, activated by macro triggers
    SleeveTarget(inst.SLEEVE_EM_BONDS,  0.06, "EMB", is_tactical=True),
    SleeveTarget(inst.SLEEVE_LONG_DUR,  0.06, "TLT", is_tactical=True),
]


@dataclass(frozen=True)
class Bucket:
    name: str  # 'SIPP' | 'ISA' | 'GIA'
    drawdown_tolerance: float  # negative number, e.g. -0.20
    vol_target: float
    targets: list[SleeveTarget]


SIPP = Bucket(
    name="SIPP",
    drawdown_tolerance=-0.20,
    vol_target=0.10,
    targets=SIPP_TARGETS,
)

# Bucket 2 is logically *one* bucket spread across ISA + GIA. We model it as
# two sub-buckets that share a risk budget. The page sums them when computing
# bucket-2 totals.
ISA = Bucket(
    name="ISA",
    drawdown_tolerance=-0.10,
    vol_target=0.07,
    targets=BUCKET2_ISA_TARGETS,
)

GIA = Bucket(
    name="GIA",
    drawdown_tolerance=-0.10,
    vol_target=0.07,
    targets=BUCKET2_GIA_TARGETS,
)

ALL_BUCKETS = {b.name: b for b in (SIPP, ISA, GIA)}


def baseline_weight(bucket: Bucket) -> float:
    """Sum of non-tactical weights — should be ~1.0 for SIPP, and ISA+GIA
    combined should also sum to 1.0."""
    return sum(t.weight for t in bucket.targets if not t.is_tactical)


def total_weight(bucket: Bucket) -> float:
    """Sum including tactical caps. Used as a sanity check."""
    return sum(t.weight for t in bucket.targets)
