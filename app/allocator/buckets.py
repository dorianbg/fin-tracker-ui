"""Two-bucket portfolio design.

Bucket 1 — SIPP (~£300k): drawdown target -20%, vol target ~10%.
Bucket 2 — ISA + GIA (~£250k): drawdown target -10%, vol target ~6-7%.

Each bucket has its own target allocation by sleeve, derived from the
drawdown budget. The allocator_v2 risk-parity engine feeds the *equity*
sleeve weights; this module defines the *cross-sleeve* targets per bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Bucket:
    name: str
    wrappers: tuple[str, ...]
    nav_gbp: float
    drawdown_target: float
    vol_target: float
    sleeve_targets: dict[str, float] = field(default_factory=dict)


SIPP = Bucket(
    name="SIPP",
    wrappers=("SIPP",),
    nav_gbp=300_000,
    drawdown_target=-0.20,
    vol_target=0.10,
    sleeve_targets={
        "equity": 0.60,
        "real_defensive": 0.18,
        "bonds": 0.06,
        "cash": 0.16,
    },
)

ISA_GIA = Bucket(
    name="ISA + GIA",
    wrappers=("ISA", "GIA"),
    nav_gbp=250_000,
    drawdown_target=-0.10,
    vol_target=0.07,
    sleeve_targets={
        "equity": 0.20,
        "real_defensive": 0.14,
        "real_cyclical": 0.04,
        "bonds": 0.13,
        "cash": 0.35,
        "infrastructure": 0.06,
        "reits": 0.03,
        "commodities": 0.05,
    },
)

ALL_BUCKETS = {"SIPP": SIPP, "ISA_GIA": ISA_GIA}


def total_nav() -> float:
    return sum(b.nav_gbp for b in ALL_BUCKETS.values())


def blended_sleeve_targets() -> dict[str, float]:
    """NAV-weighted blend of both buckets' sleeve targets."""
    total = total_nav()
    if total <= 0:
        return {}
    out: dict[str, float] = {}
    for b in ALL_BUCKETS.values():
        w = b.nav_gbp / total
        for sleeve, target in b.sleeve_targets.items():
            out[sleeve] = out.get(sleeve, 0.0) + w * target
    return out


def expected_real_return(bucket: Bucket) -> float:
    """Approximate expected real return given sleeve targets.

    Assumptions: equity 5%, real 3%, bonds 2%, cash 1%.
    """
    premia = {
        "equity": 0.05,
        "real_defensive": 0.03,
        "real_cyclical": 0.03,
        "bonds": 0.02,
        "cash": 0.01,
        "infrastructure": 0.03,
        "reits": 0.04,
        "commodities": 0.03,
    }
    return sum(
        bucket.sleeve_targets.get(s, 0.0) * premia.get(s, 0.02)
        for s in set(bucket.sleeve_targets) | set(premia)
    )
