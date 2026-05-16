"""Maximum Diversification portfolio.

Maximises the Diversification Ratio:
    DR(w) = (w' σ) / sqrt(w' Σ w)

where σ_i = sqrt(Σ_ii) is asset i's volatility vector and Σ is the covariance
matrix. Equivalently, maximises the ratio of weighted-average vol to portfolio
vol — rewards holding assets that are volatile individually but uncorrelated.

This is equivalent (via Choueifaty & Coignard 2008) to finding the maximum
Sharpe portfolio on the correlation matrix, treating all assets as having
equal expected Sharpe. It naturally overweights diversifying assets:
gold, miners, energy, EM — exactly the Napier/Deluard universe tilts.

Reference:
  Choueifaty & Coignard (2008), "Toward Maximum Diversification".
  Choueifaty, Froidure & Reynier (2013), "Properties of the Most Diversified Portfolio".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def maxdiv_weights(cov: pd.DataFrame) -> pd.Series:
    """Long-only maximum-diversification weights summing to 1."""
    if cov.empty:
        return pd.Series(dtype=float)

    tickers = list(cov.index)
    n = len(tickers)
    sigma = cov.values.astype(float)
    vols = np.sqrt(np.diag(sigma))

    if n == 1 or np.any(vols <= 0):
        return pd.Series(np.ones(n) / n, index=tickers)

    # Maximise DR = (w' vols) / sqrt(w' Σ w)  ↔  minimise -DR
    def neg_dr(w: np.ndarray) -> float:
        port_var = float(w @ sigma @ w)
        if port_var <= 0:
            return 0.0
        return -float(vols @ w) / np.sqrt(port_var)

    x0 = np.ones(n) / n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        fun=neg_dr,
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
    return pd.Series(w / s, index=tickers, name="maxdiv_weight")


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Actual DR of a given weight vector — useful for diagnostics."""
    if weights.empty or cov.empty:
        return 0.0
    tickers = [t for t in weights.index if t in cov.index]
    w = weights.loc[tickers].values
    sigma = cov.loc[tickers, tickers].values
    vols = np.sqrt(np.diag(sigma))
    port_var = float(w @ sigma @ w)
    if port_var <= 0:
        return 0.0
    return float(vols @ w) / np.sqrt(port_var)
