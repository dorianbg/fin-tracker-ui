"""Equal Risk Contribution (ERC) over the full universe.

Each asset contributes the same share to total portfolio variance. Closed-form
for the 2-asset case, iterative for the general case. We use the Spinu (2013)
cyclical-coordinate-descent solver — fast, robust, no external deps.

References:
  Maillard, Roncalli & Teïletche (2010), "The Properties of Equally-Weighted
  Risk Contribution Portfolios".
  Spinu (2013), "An Algorithm for Computing Risk Parity Weights".
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def erc_weights(cov: pd.DataFrame, max_iter: int = 500, tol: float = 1e-8) -> pd.Series:
    """Long-only ERC weights summing to 1.

    Returns a Series indexed by ``cov.index``. Empty input → empty Series.
    """
    if cov.empty:
        return pd.Series(dtype=float)

    tickers = list(cov.index)
    sigma = cov.values
    n = sigma.shape[0]

    # Spinu CCD solver: minimise 0.5 x'Σx - Σ ln(x_i) / n, x >= 0, normalise at end.
    x = np.ones(n) / np.sqrt(n)
    for _ in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            a = sigma[i, i]
            # Quadratic in x_i: a x_i^2 + b x_i - 1/n = 0, with b = Σ_{j≠i} σ_ij x_j
            b = float(sigma[i, :] @ x) - sigma[i, i] * x[i]
            disc = b * b + 4.0 * a / n
            x[i] = (-b + np.sqrt(disc)) / (2.0 * a) if a > 0 else x[i]
        if np.linalg.norm(x - x_prev, ord=np.inf) < tol:
            break

    w = x / x.sum()
    return pd.Series(w, index=tickers, name="erc_weight")


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Per-asset % contribution to portfolio variance. Sums to 1.0."""
    if weights.empty or cov.empty:
        return pd.Series(dtype=float)
    tickers = [t for t in weights.index if t in cov.index]
    w = weights.loc[tickers].values
    sigma = cov.loc[tickers, tickers].values
    port_var = float(w @ sigma @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=tickers)
    marginal = sigma @ w
    contrib = w * marginal / port_var
    return pd.Series(contrib, index=tickers, name="risk_contribution")
