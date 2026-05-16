"""All-Weather sizer: quadrant-weighted ERC.

Rather than Dalio's fixed 30/40/15/7.5/7.5 (which is dominated by bonds and
the user explicitly wants less of), this sizer:

  1. For each of the four macro quadrants, builds an ERC sub-portfolio using
     only the assets tagged with that quadrant.
  2. Scales the sub-portfolio by the posterior probability of that quadrant.
  3. Sums across quadrants → final AW weights.

This keeps the all-weather spirit (every quadrant is covered) without
hard-wiring bond dominance. The posterior drives how much capital each
quadrant gets; the ERC inside makes sure each quadrant is itself diversified
across its eligible assets.
"""

from __future__ import annotations

import pandas as pd

from allocator_v2 import universe as u
from allocator_v2.sizers.erc import erc_weights


_PROB_TO_QUADRANT_TAGS = {
    "inflation_up_growth_down":   ("inflation_up", "growth_down"),
    "inflation_up_growth_up":     ("inflation_up", "growth_up"),
    "inflation_down_growth_up":   ("inflation_down", "growth_up"),
    "inflation_down_growth_down": ("inflation_down", "growth_down"),
}


def _assets_for_quadrant_pair(tags: tuple[str, str]) -> list[str]:
    """Assets tagged with *both* quadrant axes for this joint state.

    We want assets that perform in both the growth and the inflation regime of
    this joint — e.g. gold for (inflation_up, growth_down), equities for
    (inflation_down, growth_up). Falls back to single-tag matches if the
    intersection is empty so a quadrant is never assetless.
    """
    a, b = tags
    tag_a = set(u.quadrant_members(a))
    tag_b = set(u.quadrant_members(b))
    both = sorted(tag_a & tag_b)
    if both:
        return both
    return sorted(tag_a | tag_b)


def aw_weights(cov: pd.DataFrame, quadrant_probs: dict[str, float]) -> pd.Series:
    """AW weights over the union of assets in ``cov``.

    ``quadrant_probs`` must sum to ~1; keys match ``quadrants.QUADRANT_KEYS``.
    """
    if cov.empty:
        return pd.Series(dtype=float)

    total = sum(quadrant_probs.values())
    if total <= 0:
        return pd.Series(dtype=float)
    probs = {k: v / total for k, v in quadrant_probs.items()}

    combined = pd.Series(0.0, index=cov.index)
    for quad_key, prob in probs.items():
        if prob <= 0:
            continue
        assets = [a for a in _assets_for_quadrant_pair(_PROB_TO_QUADRANT_TAGS[quad_key]) if a in cov.index]
        if not assets:
            continue
        sub_cov = cov.loc[assets, assets]
        sub_w = erc_weights(sub_cov)
        # Re-index back onto the full universe with 0 outside this quadrant.
        aligned = pd.Series(0.0, index=cov.index)
        aligned.loc[sub_w.index] = sub_w.values
        combined = combined + prob * aligned

    if combined.sum() <= 0:
        return combined
    return combined / combined.sum()
