"""90-day realised covariance + correlation.

All three sizers (AW, ERC, HRP) consume the same ``cov`` matrix, so it's
computed once per rebalance and passed down. HRP additionally uses the
correlation matrix (distance metric derived from ``1 - corr``).

No shrinkage — 90d / 15 assets / ~6 obs per asset is enough that sample
covariance is reasonable. Ledoit-Wolf can be swapped in later by replacing
``cov_matrix`` with ``sklearn.covariance.LedoitWolf().fit(rets).covariance_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_ANNUALISATION = 252
DEFAULT_WINDOW = 90
MIN_OBSERVATIONS = 30


def _tail(rets_wide: pd.DataFrame, window: int) -> pd.DataFrame:
    tail = rets_wide.tail(window)
    # Drop assets without enough observations in the window so we don't return
    # covariance driven by a handful of overlapping days.
    cols = [c for c in tail.columns if tail[c].count() >= MIN_OBSERVATIONS]
    return tail[cols].dropna(how="all")


def cov_matrix(rets_wide: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Annualised covariance matrix over the trailing ``window`` days."""
    tail = _tail(rets_wide, window)
    if tail.empty:
        return pd.DataFrame()
    return tail.cov() * _ANNUALISATION


def corr_matrix(rets_wide: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Pearson correlation over the trailing ``window`` days."""
    tail = _tail(rets_wide, window)
    if tail.empty:
        return pd.DataFrame()
    return tail.corr()


def annualised_vol(rets_wide: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Per-asset annualised vol (σ) — used as the 1/σ baseline."""
    tail = _tail(rets_wide, window)
    if tail.empty:
        return pd.Series(dtype=float)
    return tail.std() * np.sqrt(_ANNUALISATION)


def condition_number(cov: pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue. > 1e4 ≈ ill-conditioned."""
    if cov.empty:
        return float("inf")
    eigvals = np.linalg.eigvalsh(cov.values)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size == 0:
        return float("inf")
    return float(eigvals.max() / eigvals.min())
