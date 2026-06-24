"""Twin/parallel-seed distribution statistics (panel §2.10).

For Experiment E3 (Phase-Boundary Dynamics) we need per-(k, prep) *distributions*
of persona-state values across many seeds, not just means. This module computes
bimodality statistics on those distributions and detects variance peaking vs k.

Primitives:
- `bimodality_coefficient(values)` — Pearson's BC; > 5/9 ≈ 0.555 suggests
  bimodality.
- `two_gaussian_fit(values)` — best 1-component vs 2-component Gaussian mixture
  by AIC. Returns both fits and whichever has lower AIC.
- `variance_peaking(df, group_col, value_col)` — variance of `value_col` per
  `group_col` (typically k); detects the k at which variance is maximal.
- `bimodality_scan(df, group_col, value_col)` — run BC and AIC-comparison per
  group; returns per-group DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Bimodality coefficient (Pearson)
# ---------------------------------------------------------------------------


def bimodality_coefficient(values: Iterable[float]) -> float:
    """Pearson's bimodality coefficient:
        BC = (skew² + 1) / (kurtosis + 3 * (n-1)² / ((n-2)(n-3)))

    The canonical threshold of 5/9 ≈ 0.555 works best for continuous
    distributions with heavy tails. For Bernoulli-like or sharply discrete
    outputs (e.g. `ch1a_hit`), the excess kurtosis is strongly negative and
    BC saturates around 0.20–0.30 even for perfectly separated clusters.
    For those cases prefer `two_gaussian_fit(...)["preferred"]==2` as the
    discriminator.

    Returns NaN for samples with n < 4 or with zero variance.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 4:
        return float("nan")
    if arr.std(ddof=1) < 1e-12:
        return float("nan")  # degenerate — all identical
    skew = float(stats.skew(arr, bias=False))
    kurt = float(stats.kurtosis(arr, bias=False, fisher=True))  # excess kurtosis
    # Pearson's formula uses non-excess kurtosis; recover by adding 3
    k_raw = kurt + 3.0
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = k_raw + correction
    if denom <= 0:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


# ---------------------------------------------------------------------------
# 1- vs 2-Gaussian mixture comparison (via AIC)
# ---------------------------------------------------------------------------


@dataclass
class MixtureFit:
    n_components: int
    params: dict           # 1-comp: {mu, sigma}; 2-comp: {w, mu1, sigma1, mu2, sigma2}
    log_likelihood: float
    aic: float


def _nll_gaussian(arr: np.ndarray, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    return float(-np.sum(stats.norm.logpdf(arr, loc=mu, scale=sigma)))


def _fit_one_gaussian(arr: np.ndarray) -> MixtureFit:
    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1)) if len(arr) > 1 else 1e-6
    ll = -_nll_gaussian(arr, mu, sigma)
    # AIC = 2k - 2 log L, with k=2 params
    return MixtureFit(1, {"mu": mu, "sigma": sigma}, ll, 2 * 2 - 2 * ll)


def _fit_two_gaussians_em(arr: np.ndarray, max_iter: int = 200,
                          tol: float = 1e-6) -> MixtureFit:
    """Simple diagonal-covariance EM for a two-component 1D Gaussian mixture."""
    n = len(arr)
    if n < 4:
        return _fit_one_gaussian(arr)  # fall back
    # Initialise at quartiles
    sorted_arr = np.sort(arr)
    mu1 = float(sorted_arr[n // 4])
    mu2 = float(sorted_arr[3 * n // 4])
    sigma1 = sigma2 = max(float(arr.std(ddof=1)) / 2, 1e-6)
    w = 0.5
    prev_ll = -np.inf
    for _ in range(max_iter):
        # E-step: responsibilities
        p1 = w * stats.norm.pdf(arr, mu1, sigma1)
        p2 = (1 - w) * stats.norm.pdf(arr, mu2, sigma2)
        s = p1 + p2
        s[s < 1e-300] = 1e-300
        r1 = p1 / s
        r2 = p2 / s
        # M-step
        N1, N2 = r1.sum(), r2.sum()
        if N1 < 1 or N2 < 1:
            break
        mu1 = float((r1 * arr).sum() / N1)
        mu2 = float((r2 * arr).sum() / N2)
        sigma1 = max(float(np.sqrt((r1 * (arr - mu1) ** 2).sum() / N1)), 1e-6)
        sigma2 = max(float(np.sqrt((r2 * (arr - mu2) ** 2).sum() / N2)), 1e-6)
        w = float(N1 / n)
        # Log-likelihood
        ll = float(np.sum(np.log(s)))
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    # AIC with 5 params (w, mu1, sigma1, mu2, sigma2)
    aic = 2 * 5 - 2 * prev_ll
    return MixtureFit(
        2,
        {"w": w, "mu1": mu1, "sigma1": sigma1, "mu2": mu2, "sigma2": sigma2},
        prev_ll, aic,
    )


def two_gaussian_fit(values: Iterable[float]) -> dict:
    """Fit 1- and 2-Gaussian models; return both + the preferred one by AIC."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        return {"one": None, "two": None, "preferred": None, "delta_aic": float("nan")}
    one = _fit_one_gaussian(arr)
    if arr.std(ddof=1) < 1e-9:
        # Degenerate: no variance. Stay with 1-component.
        return {"one": one, "two": None, "preferred": 1, "delta_aic": float("nan")}
    two = _fit_two_gaussians_em(arr)
    delta = one.aic - two.aic   # positive → 2-comp preferred
    preferred = 2 if delta > 2.0 else 1

    # Acceptance checks — EM on small samples can split noise into two
    # over-narrow components. We require BOTH:
    #   (a) mean separation ≥ one pooled σ (not just narrow-sub-cluster σ); AND
    #   (b) component-weight balance: min(w, 1-w) ≥ 0.2.
    if preferred == 2 and two is not None:
        params = two.params
        mu1, mu2 = params["mu1"], params["mu2"]
        pooled = max((params["sigma1"] + params["sigma2"]) / 2.0, 1e-9)
        w = params["w"]
        balance = min(w, 1.0 - w)
        if abs(mu1 - mu2) < pooled or balance < 0.2:
            preferred = 1
    return {"one": one, "two": two, "preferred": preferred, "delta_aic": delta}


# ---------------------------------------------------------------------------
# Variance peaking + per-group bimodality scan
# ---------------------------------------------------------------------------


def variance_peaking(df: pd.DataFrame, *, group_col: str = "k",
                     value_col: str = "ch1a_hit") -> pd.DataFrame:
    """Per-group variance of `value_col`. Flags the peak row."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(group_col, dropna=False)[value_col]
    stats_df = g.agg(["mean", "var", "count"]).reset_index()
    stats_df = stats_df.rename(columns={"var": "variance", "count": "n"})
    if len(stats_df):
        max_idx = stats_df["variance"].idxmax()
        stats_df["is_variance_peak"] = False
        stats_df.loc[max_idx, "is_variance_peak"] = True
    return stats_df


def bimodality_scan(df: pd.DataFrame, *, group_col: str = "k",
                    value_col: str = "ch1a_hit") -> pd.DataFrame:
    """Per-group bimodality coefficient + 1-vs-2-Gaussian AIC comparison."""
    rows = []
    for g_val, g in df.groupby(group_col, dropna=False):
        vals = g[value_col].dropna().astype(float).to_numpy() if value_col in g.columns else np.array([])
        bc = bimodality_coefficient(vals)
        fit = two_gaussian_fit(vals)
        rows.append({
            group_col: g_val,
            "n": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "variance": float(vals.var(ddof=1)) if len(vals) > 1 else float("nan"),
            "bimodality_coefficient": bc,
            "preferred_components": fit["preferred"],
            "delta_aic_2_vs_1": fit["delta_aic"],
        })
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


__all__ = [
    "bimodality_coefficient",
    "two_gaussian_fit",
    "MixtureFit",
    "variance_peaking",
    "bimodality_scan",
]
