"""Bigelow sigmoid fitting (personascope-side wrapper, used across experiments)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit


def bigelow(k, L, b, gamma, alpha):
    """Bigelow et al. (2025) belief-dynamics sigmoid: p(k) = L * sigma(b + gamma * k^(1-alpha))."""
    k_safe = np.maximum(k, 1e-9)
    return L / (1.0 + np.exp(-(b + gamma * np.power(k_safe, 1.0 - alpha))))


@dataclass
class BigelowFit:
    L: float
    b: float
    gamma: float
    alpha: float
    r2: float
    k_star: float        # midpoint evidence; NaN if the sigmoid doesn't cross b=0
    converged: bool

    def predict(self, k):
        return bigelow(np.asarray(k, dtype=float), self.L, self.b, self.gamma, self.alpha)

    def as_dict(self):
        return asdict(self)


def fit_bigelow(k, p) -> BigelowFit:
    """Fit the Bigelow sigmoid to (k, p) data. Returns a BigelowFit."""
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)
    p0 = [max(float(p.max()), 0.5), -3.0, 0.5, 0.5]
    bounds = ([0.0, -50.0, 1e-6, 0.0], [1.0, 50.0, 100.0, 1.0 - 1e-6])
    try:
        popt, _ = curve_fit(bigelow, k, p, p0=p0, bounds=bounds, maxfev=50000)
        converged = True
    except (RuntimeError, ValueError):
        popt = p0
        converged = False
    L, b, gamma, alpha = popt
    p_pred = bigelow(k, *popt)
    ss_res = float(np.sum((p - p_pred) ** 2))
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    k_star = float("nan")
    if alpha < 1 and b < 0 < gamma:
        try:
            k_star = (-b / gamma) ** (1.0 / (1.0 - alpha))
        except Exception:
            pass
    return BigelowFit(
        L=float(L), b=float(b), gamma=float(gamma), alpha=float(alpha),
        r2=float(r2), k_star=float(k_star), converged=converged,
    )
