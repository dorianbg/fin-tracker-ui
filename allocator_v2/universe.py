"""Universe for the all-weather risk-parity allocator.

15 assets chosen to span the four macro quadrants with enough diversity that
the 90-day covariance matrix is well-conditioned (15 assets × ~63 observations
per quarter ≈ 4:1 obs-to-asset ratio).

Each asset is tagged with:

- ``quadrant``: which macro quadrant it primarily belongs to. A sleeve can appear
  in multiple quadrants (e.g. gold in both inflation↑ and growth↓), hence
  ``quadrants`` is a set.
- ``yfinance_symbol``: the Yahoo ticker used by the price cache.
- ``wrapper_eligible``: which tax wrappers can hold this (SIPP/ISA/GIA). Not
  enforced in v1 — we size one unified portfolio — but carried forward so the
  per-wrapper ensemble can be added later without a universe rewrite.
- ``description``: human label for the dashboard.

The four quadrants follow the Dalio all-weather taxonomy:

  - ``growth_up``     : growth > expected        → equities, EM debt, commodities
  - ``growth_down``   : growth < expected        → long TSY, linkers
  - ``inflation_up``  : inflation > expected     → linkers, commodities, gold, REITs
  - ``inflation_down``: inflation < expected     → equities, nominal TSY

The sector tilts sit in the growth-up quadrant by default and are picked to
lean the portfolio slightly pro-cyclical without adding new factor risk that
the core equity sleeve already carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


QUADRANTS = ("growth_up", "growth_down", "inflation_up", "inflation_down")


@dataclass(frozen=True)
class Asset:
    ticker: str
    description: str
    yfinance_symbol: str
    quadrants: frozenset[str]
    wrapper_eligible: frozenset[str] = field(default_factory=lambda: frozenset({"GIA"}))
    tilt_bias: float = 0.0  # multiplicative nudge inside the sleeve; +0.5 = +50% weight before renormalisation, -0.3 = -30%.

    def __post_init__(self) -> None:
        unknown = self.quadrants - set(QUADRANTS)
        if unknown:
            raise ValueError(f"{self.ticker}: unknown quadrants {unknown}")
        bad_wrappers = self.wrapper_eligible - {"SIPP", "ISA", "GIA"}
        if bad_wrappers:
            raise ValueError(f"{self.ticker}: unknown wrappers {bad_wrappers}")


_ALL = frozenset({"SIPP", "ISA", "GIA"})
_GIA_ONLY = frozenset({"GIA"})


UNIVERSE: dict[str, Asset] = {
    # ── Core equity (growth↑, inflation↓) ──────────────────────────────
    "IWDA": Asset(
        ticker="IWDA",
        description="iShares Core MSCI World (developed markets)",
        yfinance_symbol="IWDA.L",
        quadrants=frozenset({"growth_up", "inflation_down"}),
        wrapper_eligible=_ALL,
    ),
    "EEM": Asset(
        ticker="EEM",
        description="iShares MSCI Emerging Markets",
        yfinance_symbol="EEM",
        quadrants=frozenset({"growth_up", "inflation_up"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    "VGK": Asset(
        ticker="VGK",
        description="Vanguard FTSE Europe",
        yfinance_symbol="VGK",
        quadrants=frozenset({"growth_up", "inflation_down"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    "EWJ": Asset(
        ticker="EWJ",
        description="iShares MSCI Japan",
        yfinance_symbol="EWJ",
        quadrants=frozenset({"growth_up", "inflation_down"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    # ── Equity tilts (growth↑) ─────────────────────────────────────────
    "IWQU": Asset(
        ticker="IWQU",
        description="iShares Edge MSCI World Quality",
        yfinance_symbol="IWQU.L",
        quadrants=frozenset({"growth_up", "inflation_down"}),
        wrapper_eligible=_ALL,
    ),
    "MVOL": Asset(
        ticker="MVOL",
        description="iShares Edge MSCI World Min Vol",
        yfinance_symbol="MVOL.L",
        quadrants=frozenset({"growth_down", "inflation_down"}),
        wrapper_eligible=_ALL,
    ),
    "IWMO": Asset(
        ticker="IWMO",
        description="iShares Edge MSCI World Momentum",
        yfinance_symbol="IWMO.L",
        quadrants=frozenset({"growth_up", "inflation_down"}),
        wrapper_eligible=_ALL,
        tilt_bias=0.6,
    ),
    "IWVL": Asset(
        ticker="IWVL",
        description="iShares Edge MSCI World Value",
        yfinance_symbol="IWVL.L",
        quadrants=frozenset({"growth_up", "inflation_up"}),
        wrapper_eligible=_ALL,
        tilt_bias=0.3,
    ),
    "ISF": Asset(
        ticker="ISF",
        description="iShares Core FTSE 100 (UK large-cap value tilt)",
        yfinance_symbol="ISF.L",
        quadrants=frozenset({"growth_up", "inflation_up"}),
        wrapper_eligible=_ALL,
    ),
    "XLE": Asset(
        ticker="XLE",
        description="Energy Select Sector SPDR",
        yfinance_symbol="XLE",
        quadrants=frozenset({"growth_up", "inflation_up"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    # ── Rates (growth↓, inflation↑) ────────────────────────────────────
    "TLT": Asset(
        ticker="TLT",
        description="iShares 20+ Year Treasury",
        yfinance_symbol="TLT",
        quadrants=frozenset({"growth_down", "inflation_down"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    "INXG": Asset(
        ticker="INXG",
        description="iShares £ Index-Linked Gilts",
        yfinance_symbol="INXG.L",
        quadrants=frozenset({"inflation_up", "growth_down"}),
        wrapper_eligible=_ALL,
    ),
    "TIP": Asset(
        ticker="TIP",
        description="iShares TIPS Bond",
        yfinance_symbol="TIP",
        quadrants=frozenset({"inflation_up", "growth_down"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    # ── Real assets — defensive (inflation↑, lower vol) ────────────────
    "SGLD": Asset(
        ticker="SGLD",
        description="Invesco Physical Gold",
        yfinance_symbol="SGLD.L",
        quadrants=frozenset({"inflation_up", "growth_down"}),
        wrapper_eligible=_ALL,
    ),
    "VNQ": Asset(
        ticker="VNQ",
        description="Vanguard Real Estate",
        yfinance_symbol="VNQ",
        quadrants=frozenset({"inflation_up", "growth_up"}),
        wrapper_eligible=_GIA_ONLY,
    ),
    "INFR": Asset(
        ticker="INFR",
        description="iShares Global Infrastructure",
        yfinance_symbol="INFR.L",
        quadrants=frozenset({"inflation_up", "growth_up"}),
        wrapper_eligible=_ALL,
    ),
    # ── Real assets — cyclical (inflation↑, high convexity) ────────────
    "GDX": Asset(
        ticker="GDX",
        description="VanEck Gold Miners",
        yfinance_symbol="GDX",
        quadrants=frozenset({"inflation_up", "growth_down"}),
        wrapper_eligible=_GIA_ONLY,
    ),
}


def tickers() -> list[str]:
    return list(UNIVERSE.keys())


def yfinance_map() -> dict[str, str]:
    return {t: a.yfinance_symbol for t, a in UNIVERSE.items()}


def quadrant_members(quadrant: str) -> list[str]:
    if quadrant not in QUADRANTS:
        raise ValueError(f"unknown quadrant {quadrant}")
    return [t for t, a in UNIVERSE.items() if quadrant in a.quadrants]
