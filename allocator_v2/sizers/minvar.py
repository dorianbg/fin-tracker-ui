"""Minimum Variance portfolio.

Solves:  min  w' Σ w
         s.t. sum(w) = 1,  w >= 0  (long-only)

Uses scipy.optimize.minimize with SLSQP. Without the per-asset cap applied
in the sleeve layer this will concentrate heavily into the two least-correlated
low-vol assets, so always run with max_weight enforced.

Within the sleeve framework we run MinVar *inside* each sleeve on the sleeve's
sub-covariance, same as ERC and HRP. This preserves the policy sleeve totals
while minimising variance within each group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minvar_weights(cov: pd.DataFrame) -> pd.Series:
    """Long-only minimum-variance weights summing to 1."""
    if cov.empty:
        return pd.Series(dtype=float)

    tickers = list(cov.index)
    n = len(tickers)
    sigma = cov.values.astype(float)

    if n == 1:
        return pd.Series([1.0], index=tickers)

    x0 = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        fun=lambda w: float(w @ sigma @ w),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    w = np.clip(result.x, 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        w = x0
        s = 1.0
    return pd.Series(w / s, index=tickers, name="minvar_weight")
