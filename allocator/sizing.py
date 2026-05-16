"""Risk-based sizing for the sleeve layer.

The strategic weights in :mod:`buckets` encode *where we want to end up*.
This module re-scales those target weights by realised instrument vol so
no single sleeve dominates the risk budget. Two modes are supported:

- ``inverse_vol``: weight ∝ 1/vol. Cheap to compute, transparent, robust.
- ``erc``        : Equal Risk Contribution. Iteratively finds weights that
                   contribute equally to total portfolio vol given a
                   correlation matrix. Falls back to inverse-vol when the
                   correlation matrix is not supplied.

Both modes apply a **bottoming exemption**: regimes that represent a turn
(``washed_out``, ``basing``, ``repairing``) have elevated short-term vol
precisely because they are pivoting off a low. Punishing them for this
would make the sizer chronically underweight assets right as the setup
improves. For these regimes we cap the vol used by the sizer at the
universe median.

This layer intentionally does NOT change which sleeves are held — that
is the constructor's job. It only redistributes weight *within the
strategic target weights that already exist*. A small ``blend`` parameter
controls how aggressively the risk-based weights override the strategic
ones (0.0 = ignore risk, 1.0 = pure risk-parity, default 0.5 = half-way).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Vol (in % p.a.) below which we treat an asset as "essentially cash-like"
# and pin to this floor so a 1/vol sizer doesn't dump everything into it.
_VOL_FLOOR_PCT = 3.0
# Vol ceiling to avoid any single washed-out name skewing ERC weights.
_VOL_CEILING_PCT = 60.0

# Regimes that get the bottoming exemption — their vol is capped at the
# universe median for sizing purposes only.
_TURNING_REGIMES = frozenset({"washed_out", "basing", "repairing"})


def _effective_vol(vol_pct: float, regime: str | None, vol_median: float) -> float:
    """Return the vol used for sizing after applying the bottoming exemption
    and the floor/ceiling clips."""
    if vol_pct is None or not np.isfinite(vol_pct):
        vol_pct = vol_median
    if regime in _TURNING_REGIMES and vol_pct > vol_median:
        vol_pct = vol_median
    return float(np.clip(vol_pct, _VOL_FLOOR_PCT, _VOL_CEILING_PCT))


def inverse_vol_weights(
    plan_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    blend: float = 0.5,
) -> pd.DataFrame:
    """Blend strategic target weights with inverse-vol weights per bucket.

    Parameters
    ----------
    plan_df : DataFrame
        Output of ``construction.build_portfolio_plan``. Must have columns
        ``account_type``, ``ticker``, ``target_weight``, ``theme_regime``.
        Rows with ``action == 'NOT_ACTIVE'`` are passed through unchanged.
    timing_df : DataFrame
        Output of ``data_sources.get_entry_timing_metrics``. Used for
        ``vol_1y`` (preferred) with a fallback to ``vol_3m``. Must also
        have an instrument-level ``regime`` column for the exemption —
        when absent, the plan's ``theme_regime`` is used.
    blend : float in [0, 1]
        0.0 keeps strategic weights. 1.0 uses pure inverse-vol within
        each bucket. Default 0.5 halves the distance.

    Returns
    -------
    DataFrame
        Copy of ``plan_df`` with two new columns:

        - ``effective_vol_pct``: vol actually fed into the sizer
        - ``risk_scaled_weight``: blended weight, re-normalised so that
          each bucket's active weights still sum to its original total
          (the non-active sleeve budget is preserved).
    """
    if plan_df.empty:
        return plan_df.copy()

    blend = float(np.clip(blend, 0.0, 1.0))
    timing_idx: pd.DataFrame
    if timing_df is not None and not timing_df.empty:
        timing_idx = timing_df.set_index("ticker")
    else:
        timing_idx = pd.DataFrame()

    # Build the vol lookup up-front: (universe vol median across whatever
    # timing_df covers, not just the plan rows, so the exemption cap is
    # stable across rebalances).
    if not timing_idx.empty and "vol_1y" in timing_idx.columns:
        universe_vol_median = float(pd.to_numeric(timing_idx["vol_1y"], errors="coerce").median())
    else:
        universe_vol_median = 15.0
    if not np.isfinite(universe_vol_median) or universe_vol_median <= 0:
        universe_vol_median = 15.0

    out = plan_df.copy()
    effective_vols: list[float | None] = []
    for _, row in out.iterrows():
        if row.get("action") == "NOT_ACTIVE":
            effective_vols.append(None)
            continue
        ticker = str(row["ticker"])
        vol = None
        if not timing_idx.empty and ticker in timing_idx.index:
            tr = timing_idx.loc[ticker]
            v = tr.get("vol_1y")
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                v = tr.get("vol_3m")
            if v is not None and np.isfinite(v):
                vol = float(v)
        # Prefer per-instrument regime when present, otherwise theme regime.
        regime = None
        if not timing_idx.empty and ticker in timing_idx.index:
            r = timing_idx.loc[ticker].get("regime")
            if isinstance(r, str):
                regime = r.lower()
        if regime is None:
            regime = str(row.get("theme_regime") or "").lower()
        effective_vols.append(_effective_vol(vol, regime, universe_vol_median))

    out["effective_vol_pct"] = effective_vols

    risk_scaled_weights: list[float] = []
    for account_type, grp in out.groupby("account_type", sort=False):
        active_mask = grp["action"] != "NOT_ACTIVE"
        active = grp[active_mask]
        if active.empty:
            for _ in range(len(grp)):
                risk_scaled_weights.append(0.0)
            continue

        active_budget = float(active["target_weight"].sum())
        # 1/vol weights, normalised to sum to active_budget
        inv = 1.0 / active["effective_vol_pct"].to_numpy(dtype=float)
        if not np.isfinite(inv).any() or inv.sum() == 0:
            inv_weights = active["target_weight"].to_numpy(dtype=float)
        else:
            inv_weights = (inv / inv.sum()) * active_budget

        strategic = active["target_weight"].to_numpy(dtype=float)
        blended = (1.0 - blend) * strategic + blend * inv_weights
        # Preserve the bucket's active-weight budget exactly.
        if blended.sum() > 0:
            blended = blended * (active_budget / blended.sum())

        blended_by_ticker = dict(zip(active["ticker"].astype(str), blended))
        for _, row in grp.iterrows():
            if row.get("action") == "NOT_ACTIVE":
                risk_scaled_weights.append(0.0)
            else:
                risk_scaled_weights.append(float(blended_by_ticker.get(str(row["ticker"]), row["target_weight"])))

    out["risk_scaled_weight"] = risk_scaled_weights
    return out


def _erc_weights(vols: np.ndarray, corr: np.ndarray, tol: float = 1e-6, max_iter: int = 500) -> np.ndarray:
    """Cyclical coordinate descent for Equal Risk Contribution.

    Returns weights that sum to 1. Numerically stable for small universes
    (<30 assets) which is what we feed it here.
    """
    n = len(vols)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    cov = np.outer(vols, vols) * corr
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            num = w_prev @ cov[i]
            denom = cov[i, i]
            if denom <= 0:
                continue
            w[i] = max(1e-8, (1.0 / n) * (w_prev @ cov @ w_prev) / num)
        w = w / w.sum()
        if np.max(np.abs(w - w_prev)) < tol:
            break
    return w


def erc_weights(
    plan_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    corr_df: pd.DataFrame | None = None,
    blend: float = 0.5,
) -> pd.DataFrame:
    """Equal Risk Contribution weights per bucket, blended with strategic.

    Falls back to ``inverse_vol_weights`` when ``corr_df`` is None or does
    not cover all active tickers in a bucket. The fallback is per-bucket,
    so an ERC solve can succeed on one bucket and degrade on another.
    """
    if plan_df.empty:
        return plan_df.copy()

    # Start from inverse-vol so ``effective_vol_pct`` is populated.
    base = inverse_vol_weights(plan_df, timing_df, blend=0.0)
    if corr_df is None or corr_df.empty:
        return inverse_vol_weights(plan_df, timing_df, blend=blend)

    blend = float(np.clip(blend, 0.0, 1.0))
    out = base.copy()
    erc_col: list[float] = [0.0] * len(out)
    out_idx = list(out.index)

    for account_type, grp in out.groupby("account_type", sort=False):
        active = grp[grp["action"] != "NOT_ACTIVE"].copy()
        if active.empty:
            continue
        tickers = active["ticker"].astype(str).tolist()
        missing = [t for t in tickers if t not in corr_df.columns or t not in corr_df.index]
        active_budget = float(active["target_weight"].sum())

        if missing:
            # Fallback: use already-computed inverse-vol for this bucket.
            fallback = inverse_vol_weights(grp, timing_df, blend=blend)
            for idx, w in zip(fallback.index, fallback["risk_scaled_weight"]):
                erc_col[out_idx.index(idx)] = float(w)
            continue

        vols = active["effective_vol_pct"].to_numpy(dtype=float)
        sub_corr = corr_df.loc[tickers, tickers].to_numpy(dtype=float)
        w_erc = _erc_weights(vols, sub_corr) * active_budget
        strategic = active["target_weight"].to_numpy(dtype=float)
        blended = (1.0 - blend) * strategic + blend * w_erc
        if blended.sum() > 0:
            blended = blended * (active_budget / blended.sum())

        for idx, w in zip(active.index, blended):
            erc_col[out_idx.index(idx)] = float(w)

    out["risk_scaled_weight"] = erc_col
    return out


def risk_contribution_report(
    sized_df: pd.DataFrame,
    corr_df: pd.DataFrame | None = None,
    weight_col: str = "risk_scaled_weight",
) -> pd.DataFrame:
    """Per-ticker percent risk contribution given sized weights + correlations.

    Useful for the UI to show *where* the portfolio's vol actually comes
    from after sizing. When ``corr_df`` is missing we assume independence
    (pure variance-weighted contribution).
    """
    if sized_df.empty or weight_col not in sized_df.columns or "effective_vol_pct" not in sized_df.columns:
        return pd.DataFrame()

    rows = []
    for account_type, grp in sized_df.groupby("account_type", sort=False):
        active = grp[grp["action"] != "NOT_ACTIVE"]
        if active.empty:
            continue
        tickers = active["ticker"].astype(str).tolist()
        w = active[weight_col].to_numpy(dtype=float)
        vols = active["effective_vol_pct"].to_numpy(dtype=float) / 100.0
        if corr_df is not None and not corr_df.empty and all(t in corr_df.index for t in tickers):
            sub_corr = corr_df.loc[tickers, tickers].to_numpy(dtype=float)
            cov = np.outer(vols, vols) * sub_corr
        else:
            cov = np.diag(vols ** 2)
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            continue
        mrc = cov @ w  # marginal risk contribution
        rc = w * mrc  # risk contribution
        pct = rc / port_var
        for t, wi, vi, pi in zip(tickers, w, vols, pct):
            rows.append(
                {
                    "account_type": account_type,
                    "ticker": t,
                    "weight": float(wi),
                    "vol_pct": float(vi * 100.0),
                    "risk_contribution_pct": float(pi * 100.0),
                }
            )
    return pd.DataFrame(rows)
