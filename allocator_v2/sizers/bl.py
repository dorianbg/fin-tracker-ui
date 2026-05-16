"""Black-Litterman portfolio.

Combines market-implied equilibrium returns (reverse-engineered from
market-cap weights via the CAPM) with investor views to produce a posterior
expected return vector, then feeds that into a mean-variance optimiser.

Formula (Idzorek 2005):
    Π  = δ Σ w_mkt                 # implied equilibrium excess returns
    μ_BL = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1} [(τΣ)^{-1} Π + P' Ω^{-1} Q]
    Σ_BL = Σ + [(τΣ)^{-1} + P' Ω^{-1} P]^{-1}

Then optimise: max w' μ_BL - (δ/2) w' Σ_BL w,  sum(w)=1, w >= 0

Views encoded from the Napier / Chancellor / Deluard thesis:

  V1: Value > Growth           (IWVL vs IWMO — relative +2% p.a.)
  V2: EM outperforms DM        (EEM vs IWDA — relative +2% p.a.)
  V3: UK outperforms Europe    (ISF vs VGK — relative +1.5% p.a.)
  V4: Energy outperforms bonds (XLE vs TLT — relative +3% p.a.)
  V5: Gold outperforms bonds   (SGLD vs TLT — relative +2% p.a.)
  V6: Miners outperform core   (GDX vs IWDA — relative +4% p.a.)

Each view has a confidence weight [0, 1]. At 0 the view is ignored;
at 1 the view completely overrides the prior. Confidence translates to
Ω_kk = (1 - c_k) / c_k * (P_k Σ P_k'), the Idzorek (2005) formula.

Market-cap weights: we approximate using the universe assets' median
AUM (from instruments.py) normalised to a weight vector. If AUM is
unavailable for some assets, equal-weight is used as the prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Default views (Napier / Deluard thesis) ──────────────────────────
@dataclass
class View:
    """A relative or absolute view on expected excess return.

    For a relative view, ``assets_long`` beat ``assets_short`` by ``q`` p.a.
    For an absolute view, set ``assets_short`` empty and ``q`` = absolute
    expected excess return.
    """
    label: str
    assets_long: list[str]
    assets_short: list[str]
    q: float          # expected outperformance per annum (e.g. 0.02 = 2%)
    confidence: float = 0.5  # [0, 1]; 0 = ignore, 1 = full conviction


DEFAULT_VIEWS: list[View] = [
    View("Value > Growth (IWVL vs IWMO)",  ["IWVL"], ["IWMO"], q=0.02, confidence=0.5),
    View("EM > DM (EEM vs IWDA)",          ["EEM"],  ["IWDA"], q=0.02, confidence=0.4),
    View("UK > Europe (ISF vs VGK)",       ["ISF"],  ["VGK"],  q=0.015, confidence=0.5),
    View("Energy > Bonds (XLE vs TLT)",    ["XLE"],  ["TLT"],  q=0.03, confidence=0.5),
    View("Gold > Bonds (SGLD vs TLT)",     ["SGLD"], ["TLT"],  q=0.02, confidence=0.5),
    View("Miners > Core (GDX vs IWDA)",    ["GDX"],  ["IWDA"], q=0.04, confidence=0.4),
]


def _build_P_Q(
    views: list[View],
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build the P (k×n) view matrix and Q (k,) view vector.

    Returns (P, Q, active_view_indices) — only views whose assets are all
    present in ``tickers`` are included.
    """
    active = [i for i, v in enumerate(views)
              if all(a in tickers for a in v.assets_long + v.assets_short)
              and abs(v.confidence) > 1e-9]
    n = len(tickers)
    P = np.zeros((len(active), n))
    Q = np.zeros(len(active))
    for row, idx in enumerate(active):
        v = views[idx]
        long_w = 1.0 / len(v.assets_long) if v.assets_long else 0.0
        short_w = -1.0 / len(v.assets_short) if v.assets_short else 0.0
        for a in v.assets_long:
            P[row, tickers.index(a)] = long_w
        for a in v.assets_short:
            P[row, tickers.index(a)] = short_w
        Q[row] = v.q
    return P, Q, active


def _idzorek_omega(
    P: np.ndarray,
    sigma: np.ndarray,
    confidences: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Diagonal Ω using Idzorek's confidence → uncertainty mapping.

    Ω_kk = (1 - c_k) / c_k * (P_k τΣ P_k')
    Clipped to avoid division by zero at c=0 or c=1.
    """
    k = P.shape[0]
    omega = np.zeros(k)
    for i in range(k):
        p = P[i]
        var_view = float(p @ (tau * sigma) @ p)
        c = float(np.clip(confidences[i], 1e-6, 1.0 - 1e-6))
        omega[i] = (1.0 - c) / c * var_view
    return np.diag(omega)


def bl_weights(
    cov: pd.DataFrame,
    views: list[View] | None = None,
    delta: float = 2.5,
    tau: float = 0.05,
    mkt_weights: pd.Series | None = None,
) -> pd.Series:
    """Black-Litterman weights, long-only, summing to 1.

    Parameters
    ----------
    cov:         Annualised covariance matrix.
    views:       List of View objects. Defaults to DEFAULT_VIEWS.
    delta:       Risk-aversion coefficient (market Sharpe ≈ δ * market vol).
                 2.5 is a common choice for a global portfolio.
    tau:         Uncertainty scaling on the prior. Typically 0.01–0.10.
    mkt_weights: Optional prior weight vector (market-cap proxy). If None,
                 uses equal weights.
    """
    if cov.empty:
        return pd.Series(dtype=float)

    views = views or DEFAULT_VIEWS
    tickers = list(cov.index)
    n = len(tickers)
    sigma = cov.values.astype(float)

    # Prior: market-cap implied equilibrium returns Π = δ Σ w_mkt
    if mkt_weights is not None:
        w_mkt = np.array([mkt_weights.get(t, 0.0) for t in tickers], dtype=float)
        s = w_mkt.sum()
        w_mkt = w_mkt / s if s > 0 else np.ones(n) / n
    else:
        w_mkt = np.ones(n) / n
    pi = delta * sigma @ w_mkt

    # Build view matrices
    P, Q, active_idx = _build_P_Q(views, tickers)

    if len(active_idx) == 0:
        # No valid views — fall back to market-weight portfolio
        return pd.Series(w_mkt, index=tickers, name="bl_weight")

    confidences = np.array([views[i].confidence for i in active_idx])
    omega = _idzorek_omega(P, sigma, confidences, tau)

    # Posterior expected returns (He & Litterman formula)
    tau_sigma_inv = np.linalg.inv(tau * sigma)
    omega_inv = np.linalg.inv(omega + np.eye(len(active_idx)) * 1e-10)

    A = tau_sigma_inv + P.T @ omega_inv @ P
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)

    mu_bl = A_inv @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    sigma_bl = sigma + A_inv

    # Mean-variance optimise on posterior
    def neg_utility(w: np.ndarray) -> float:
        return -(float(mu_bl @ w) - (delta / 2.0) * float(w @ sigma_bl @ w))

    x0 = w_mkt.copy()
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        fun=neg_utility,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    w = np.clip(result.x, 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        w = w_mkt
        s = 1.0
    return pd.Series(w / s, index=tickers, name="bl_weight")
