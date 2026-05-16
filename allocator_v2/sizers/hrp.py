"""Hierarchical Risk Parity (López de Prado 2016).

Three stages:

  1. **Tree clustering** — convert the correlation matrix to a distance
     matrix (d_ij = √(0.5 * (1 - ρ_ij))) and run single-linkage hierarchical
     clustering to get an order for the assets.
  2. **Quasi-diagonalisation** — permute the covariance matrix so similar
     assets are adjacent.
  3. **Recursive bisection** — walk the tree top-down; at each split,
     allocate inversely to the sub-cluster variance.

The result is a long-only portfolio that doesn't require inverting the
covariance matrix (robust when assets are nearly redundant), makes only
ordinal use of correlations (robust to window choice), and concentrates
weight in less-correlated clusters (unlike ERC, which can pile into a
cluster if the individual volatilities happen to be low).

Uses scipy for clustering — already a transitive dep via pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _cluster_order(corr: pd.DataFrame) -> list[int]:
    """Single-linkage hierarchical clustering → quasi-diagonal asset order."""
    distance = np.sqrt(0.5 * (1.0 - corr.values).clip(min=0.0))
    # scipy's squareform wants a zero diagonal and symmetric matrix.
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    link = linkage(condensed, method="single")
    n = corr.shape[0]
    # Quasi-diagonalisation: walk the linkage tree, expanding merged clusters
    # into their leaf indices. Left-to-right order = the HRP sort order.
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n:
        order = [o for o in order if o < n] + _expand(order, link, n)
    return order


def _expand(order: list[int], link: np.ndarray, n: int) -> list[int]:
    out: list[int] = []
    for o in order:
        if o >= n:
            row = link[int(o - n)]
            out.extend([int(row[0]), int(row[1])])
    return out


def _inverse_variance_weights(sub_cov: pd.DataFrame) -> pd.Series:
    ivp = 1.0 / np.diag(sub_cov.values)
    ivp = ivp / ivp.sum()
    return pd.Series(ivp, index=sub_cov.index)


def _cluster_variance(sub_cov: pd.DataFrame) -> float:
    w = _inverse_variance_weights(sub_cov).values
    return float(w @ sub_cov.values @ w)


def hrp_weights(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """Long-only HRP weights summing to 1. Empty input → empty Series."""
    if cov.empty or corr.empty:
        return pd.Series(dtype=float)

    # Align corr and cov onto the same ticker order.
    tickers = [t for t in cov.index if t in corr.index]
    if len(tickers) < 2:
        return pd.Series(1.0 / max(len(tickers), 1), index=tickers)
    cov = cov.loc[tickers, tickers]
    corr = corr.loc[tickers, tickers]

    order_idx = _cluster_order(corr)
    ordered_tickers = [tickers[i] for i in order_idx if 0 <= i < len(tickers)]
    # Safety: ensure no duplicates and all tickers covered.
    ordered_tickers = list(dict.fromkeys(ordered_tickers))
    for t in tickers:
        if t not in ordered_tickers:
            ordered_tickers.append(t)

    weights = pd.Series(1.0, index=ordered_tickers)
    clusters = [ordered_tickers]

    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left = _cluster_variance(cov.loc[left, left])
            var_right = _cluster_variance(cov.loc[right, right])
            denom = var_left + var_right
            if denom <= 0:
                alpha = 0.5
            else:
                alpha = 1.0 - var_left / denom
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = weights / weights.sum()
    return weights.reindex(cov.index).fillna(0.0)
