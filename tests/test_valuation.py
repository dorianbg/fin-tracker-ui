"""Unit tests for allocator/valuation.py.

Run from repo root:
    uv run python -m pytest tests/test_valuation.py -v

Covers:
  - compute_region_tilts: ordering, dampener, edge cases
  - compute_bond_triggers: all on, all off, boundary conditions
  - compute_deployment_pace: default, drawdown accelerators, valuation decelerator
"""

import sys
import os

import pytest

# Add allocator package to path (handles running from repo root or tests/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "allocator"))

from valuation import (
    RegionData,
    MacroData,
    DeploymentState,
    compute_region_tilts,
    apply_tilts_to_weights,
    compute_bond_triggers,
    compute_deployment_pace,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _regions_from_pe(pe_map: dict[str, float], acwi: dict[str, float]) -> list[RegionData]:
    """Create RegionData list with price == MA200 (ratio=1.0, no dampener)."""
    return [
        RegionData(name=r, forward_pe=pe, price=100.0, ma200=100.0, acwi_base_weight=acwi[r])
        for r, pe in pe_map.items()
    ]


# ── compute_region_tilts ──────────────────────────────────────────────

class TestComputeRegionTilts:

    def test_empty_returns_empty(self):
        assert compute_region_tilts([]) == {}

    def test_stubbed_spec_inputs_ordering(self):
        """Spec test: US PE=22, EM=12, Japan=14, Europe=14, UK=11.
        Cheaper (lower PE) regions should get higher tilts.
        Expected order: UK >= EM > Japan = Europe > US.
        """
        regions = _regions_from_pe(
            {"US": 22.0, "EM": 12.0, "Japan": 14.0, "Europe": 14.0, "UK": 11.0},
            {"US": 0.64, "EM": 0.11, "Japan": 0.06, "Europe": 0.13, "UK": 0.04},
        )
        tilts = compute_region_tilts(regions)

        assert tilts["UK"] >= tilts["EM"], "UK (cheapest) should tilt >= EM"
        assert tilts["EM"] > tilts["Japan"], "EM cheaper than Japan should tilt higher"
        assert tilts["Japan"] == tilts["Europe"], "Same P/E → same tilt"
        assert tilts["Europe"] > tilts["US"], "Europe cheaper than US should tilt higher"
        assert tilts["US"] < 1.0, "US (most expensive) should tilt below 1.0"
        assert tilts["EM"] > 1.0, "EM should tilt above 1.0"

    def test_weights_sum_to_one_after_apply(self):
        regions = _regions_from_pe(
            {"US": 22.0, "EM": 12.0, "Japan": 14.0, "Europe": 14.0, "UK": 11.0},
            {"US": 0.64, "EM": 0.11, "Japan": 0.06, "Europe": 0.13, "UK": 0.04},
        )
        tilts = compute_region_tilts(regions)
        weights = apply_tilts_to_weights(regions, tilts)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_equal_pe_gives_neutral_tilts(self):
        """When all regions have the same P/E, tilts should all be 1.0."""
        regions = [
            RegionData("A", 15.0, 100.0, 100.0, 0.50),
            RegionData("B", 15.0, 100.0, 100.0, 0.30),
            RegionData("C", 15.0, 100.0, 100.0, 0.20),
        ]
        tilts = compute_region_tilts(regions)
        for r in regions:
            assert abs(tilts[r.name] - 1.0) < 1e-6, f"{r.name} tilt should be 1.0"

    def test_falling_knife_dampener(self):
        """Price < 0.85 × MA200 → momentum_dampener = 0.7.

        Compare the dampened tilt against the same region without a falling-knife
        price (ratio at 1.0 instead of 0.80). The dampened version must be lower.
        """
        # Dampened: ratio = 0.80/1.00 = 0.80 < 0.85 → dampener fires
        regions_dampened = [
            RegionData("cheap", 10.0, 80.0, 100.0, 0.5),
            RegionData("ref",   20.0, 100.0, 100.0, 0.5),
        ]
        tilts_dampened = compute_region_tilts(regions_dampened)

        # Undampened: same P/E, price exactly at MA200
        regions_undampened = [
            RegionData("cheap", 10.0, 100.0, 100.0, 0.5),
            RegionData("ref",   20.0, 100.0, 100.0, 0.5),
        ]
        tilts_undampened = compute_region_tilts(regions_undampened)

        assert tilts_dampened["cheap"] < tilts_undampened["cheap"], (
            f"Falling-knife dampener should reduce tilt: "
            f"dampened={tilts_dampened['cheap']:.4f} vs undampened={tilts_undampened['cheap']:.4f}"
        )
        # Additionally, dampened tilt should be approx 0.7× undampened (since z-score math
        # stays constant when the relative PE is the same)
        ratio = tilts_dampened["cheap"] / tilts_undampened["cheap"]
        assert abs(ratio - 0.7) < 0.01, f"Expected ~0.7× dampening, got {ratio:.4f}"

    def test_extended_market_dampener(self):
        """Price > 1.30 × MA200 → momentum_dampener = 0.7 (extended / parabolic)."""
        regions = [
            RegionData("hot", 10.0, 140.0, 100.0, 0.5),    # ratio=1.40 → dampened
            RegionData("flat", 10.0, 100.0, 100.0, 0.5),   # ratio=1.00 → not dampened
        ]
        tilts = compute_region_tilts(regions)
        assert tilts["hot"] < tilts["flat"], "Extended market should dampen tilt vs normal"

    def test_boundary_85_not_dampened(self):
        """Price == 0.85 × MA200 is exactly on boundary — should NOT be dampened."""
        regions = [
            RegionData("boundary", 15.0, 85.0, 100.0, 0.5),
            RegionData("ref", 15.0, 100.0, 100.0, 0.5),
        ]
        tilts = compute_region_tilts(regions)
        # With same PE, z-score = 0 → (1 + 0) * dampener
        # ratio == 0.85 → condition is `< 0.85` which is False, so dampener = 1.0
        assert tilts["boundary"] == tilts["ref"], "Exactly 0.85 should NOT be dampened"

    def test_tilt_multiplier_capped_at_bounds(self):
        """valuation_tilt is clipped to [-0.5, +0.5], giving final tilt ∈ [0.35, 1.5]."""
        # One very cheap, one very expensive
        regions = [
            RegionData("ultra_cheap", 5.0, 100.0, 100.0, 0.5),
            RegionData("ultra_dear",  80.0, 100.0, 100.0, 0.5),
        ]
        tilts = compute_region_tilts(regions)
        assert tilts["ultra_cheap"] <= 1.5 + 1e-6
        assert tilts["ultra_dear"] >= 0.5 - 1e-6  # 0.5 * 0.7 = 0.35 without dampener


# ── compute_bond_triggers ─────────────────────────────────────────────

class TestComputeBondTriggers:

    def _macro(
        self,
        uk_real: float = 0.01,
        us_10y: float = 0.04,
        em_spread: float = 0.04,
    ) -> MacroData:
        return MacroData(
            uk_real_yield_10y=uk_real,
            us_real_yield_10y=0.02,
            us_10y_nominal=us_10y,
            em_hy_spread=em_spread,
            acwi_drawdown_30d=0.0,
            acwi_forward_pe_pct=0.70,
        )

    def test_spec_stub_inputs(self):
        """Spec stub: uk_real=1.8%, em_spread=7%, us_10y=4.5%.
        Expected: linkers active, em_usd active, long_dur inactive.
        """
        macro = self._macro(uk_real=0.018, us_10y=0.045, em_spread=0.07)
        t = compute_bond_triggers(macro)
        assert t["linkers_extra"] == pytest.approx(0.06), "linkers should activate at uk_real>1.5%"
        assert t["em_usd"] == pytest.approx(0.06), "em_usd should activate at em_spread>6%"
        assert t["long_dur"] == pytest.approx(0.0), "long_dur should NOT activate at us_10y<5%"

    def test_all_triggers_on(self):
        macro = self._macro(uk_real=0.02, us_10y=0.051, em_spread=0.065)
        t = compute_bond_triggers(macro)
        assert t["linkers_extra"] > 0
        assert t["em_usd"] > 0
        assert t["long_dur"] > 0

    def test_all_triggers_off(self):
        macro = self._macro(uk_real=0.01, us_10y=0.04, em_spread=0.04)
        t = compute_bond_triggers(macro)
        assert t["linkers_extra"] == 0.0
        assert t["em_usd"] == 0.0
        assert t["long_dur"] == 0.0

    def test_linker_trigger_exact_boundary(self):
        """uk_real == 1.5% is on the boundary — should NOT trigger (condition is > not >=)."""
        macro = self._macro(uk_real=0.015)
        t = compute_bond_triggers(macro)
        assert t["linkers_extra"] == 0.0, "Exactly 1.5% should not trigger"

    def test_long_dur_trigger_exact_boundary(self):
        """us_10y == 5.0% is on the boundary — should NOT trigger."""
        macro = self._macro(us_10y=0.05)
        t = compute_bond_triggers(macro)
        assert t["long_dur"] == 0.0, "Exactly 5% should not trigger"

    def test_em_spread_trigger_exact_boundary(self):
        """em_spread == 6.0% is on the boundary — should NOT trigger."""
        macro = self._macro(em_spread=0.06)
        t = compute_bond_triggers(macro)
        assert t["em_usd"] == 0.0, "Exactly 6% should not trigger"


# ── compute_deployment_pace ───────────────────────────────────────────

class TestComputeDeploymentPace:

    _BASE_STATE = DeploymentState(
        total_initial=300_000.0,
        cash_remaining=250_000.0,
        months_remaining=10,
    )

    def _macro(self, dd: float = -0.02, pe_pct: float = 0.70) -> MacroData:
        return MacroData(
            uk_real_yield_10y=0.012,
            us_real_yield_10y=0.02,
            us_10y_nominal=0.042,
            em_hy_spread=0.04,
            acwi_drawdown_30d=dd,
            acwi_forward_pe_pct=pe_pct,
        )

    def test_default_pace(self):
        pace, reason = compute_deployment_pace(self._BASE_STATE, self._macro(dd=-0.02))
        assert reason == "default"
        assert pace == pytest.approx(1.0 / 10)  # 1/months_remaining

    def test_drawdown_5_accelerates(self):
        pace, reason = compute_deployment_pace(self._BASE_STATE, self._macro(dd=-0.06))
        assert "drawdown" in reason
        assert pace > 1.0 / 10, "DD>5% should deploy faster than default"

    def test_drawdown_10_accelerates_more(self):
        pace5, _ = compute_deployment_pace(self._BASE_STATE, self._macro(dd=-0.06))
        pace10, _ = compute_deployment_pace(self._BASE_STATE, self._macro(dd=-0.11))
        assert pace10 > pace5, "DD>10% should be faster than DD>5%"

    def test_drawdown_20_caps_at_25pct(self):
        pace, reason = compute_deployment_pace(self._BASE_STATE, self._macro(dd=-0.22))
        assert reason == "drawdown>20"
        # Cap: min(0.25, cash_remaining / total_initial) = min(0.25, 250k/300k=0.833) = 0.25
        assert pace == pytest.approx(0.25)

    def test_high_valuation_slows(self):
        pace_default, _ = compute_deployment_pace(self._BASE_STATE, self._macro(pe_pct=0.70))
        pace_high, reason = compute_deployment_pace(self._BASE_STATE, self._macro(pe_pct=0.92))
        assert reason == "valuation>90pct"
        assert pace_high == pytest.approx(pace_default * 0.5)

    def test_deployment_complete(self):
        state = DeploymentState(total_initial=100_000, cash_remaining=0, months_remaining=5)
        pace, reason = compute_deployment_pace(state, self._macro())
        assert pace == 0.0
        assert reason == "deployment_complete"

    def test_no_months_remaining(self):
        state = DeploymentState(total_initial=100_000, cash_remaining=50_000, months_remaining=0)
        pace, reason = compute_deployment_pace(state, self._macro())
        assert pace == 0.0
        assert reason == "deployment_complete"

    def test_pace_never_exceeds_cash_remaining_fraction(self):
        """Ensure we never signal deploying more cash than we have."""
        state = DeploymentState(
            total_initial=300_000, cash_remaining=10_000, months_remaining=10
        )
        pace, _ = compute_deployment_pace(state, self._macro(dd=-0.22))
        # pace * total_initial = amount to deploy; must not exceed cash_remaining
        deploy_amount = pace * state.total_initial
        assert deploy_amount <= state.cash_remaining + 1, (
            f"Would deploy £{deploy_amount:,.0f} but only £{state.cash_remaining:,.0f} remains"
        )
