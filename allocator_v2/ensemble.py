"""Ensemble mesher.

Runs all three sizers and averages the resulting weight vectors. Missing
entries (e.g. AW assigning zero to an asset that doesn't tag into any active
quadrant) are treated as 0 weight — the average still sums to 1 because each
input vector does.

The default weights are 1/3 each. The dashboard can override via sliders.
"""

from __future__ import annotations

import pandas as pd

from allocator_v2 import sleeves as sl
from allocator_v2.sizers.erc import erc_weights
from allocator_v2.sizers.hrp import hrp_weights


def build_all_sizers(
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    quadrant_probs: dict[str, float] | None = None,
    policy: sl.SleevePolicy | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns erc / hrp indexed by ticker.

    When ``policy`` is provided (the default), ERC and HRP run inside each
    sleeve (bucket mode). Passing ``policy=None`` falls back to raw
    universe-wide ERC/HRP (used by tests that want to compare).

    ``quadrant_probs`` is accepted for backward-compatible call sites but no
    longer used — the All-Weather sizer has been removed.
    """
    del quadrant_probs  # AW removed; kept in signature to avoid churn.
    if cov.empty:
        return pd.DataFrame()
    pol = policy if policy is not None else sl.SleevePolicy()

    if policy is None:
        erc = erc_weights(cov)
        hrp = hrp_weights(cov, corr)
    else:
        erc = sl.bucket_erc(cov, policy=pol)
        hrp = sl.bucket_hrp(cov, corr, policy=pol)

    df = pd.DataFrame(
        {
            "erc": erc,
            "hrp": hrp,
        }
    ).fillna(0.0)
    return df.reindex(cov.index).fillna(0.0)


def mesh(
    sizer_weights: pd.DataFrame,
    model_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Simple (or user-weighted) average across the three sizer columns."""
    if sizer_weights.empty:
        return pd.Series(dtype=float)

    mw = model_weights or {"erc": 0.5, "hrp": 0.5}
    total_mw = sum(mw.values())
    if total_mw <= 0:
        return pd.Series(0.0, index=sizer_weights.index)
    mw = {k: v / total_mw for k, v in mw.items()}

    out = pd.Series(0.0, index=sizer_weights.index)
    for col, weight in mw.items():
        if col not in sizer_weights.columns:
            continue
        out = out + weight * sizer_weights[col]

    s = out.sum()
    if s <= 0:
        return out
    return out / s
