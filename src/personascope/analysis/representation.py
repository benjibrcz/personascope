"""Representation↔behaviour correlation — the white-box validation readout.

The core question of the representation channel: **does a cell's projection onto
a persona direction predict that cell's behavioural PAD/VD across the grid?** A
strong correlation means the residual-stream direction and the black-box
behaviour are measuring the same thing — our version of the S20 r≈0.86 result,
computed here rather than left to a notebook.

Two honesties baked in:

- **Per-layer curve, not a single cherry-picked layer.** `layerwise_correlation`
  reports r at every layer, so "layer 24 correlates" is visible as a curve, not
  a lone number.
- **Held-out layer selection (leave-one-cell-out).** Picking the best-correlating
  layer post-hoc inflates r. `cv_best_layer_correlation` selects the layer on
  n−1 cells and predicts the held-out one, so the reported r isn't circular.

numpy/scipy only (torch-free core).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and two-sided p (scipy if available, else r with p=nan)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:  # noqa: BLE001 — scipy optional; fall back to bare r
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan")


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher-z confidence interval for a Pearson r."""
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    from math import atanh, sqrt, tanh
    try:
        from scipy.stats import norm
        z_crit = float(norm.ppf(1 - alpha / 2))
    except Exception:  # noqa: BLE001
        z_crit = 1.959963984540054
    z = atanh(r)
    se = 1.0 / sqrt(n - 3)
    return tanh(z - z_crit * se), tanh(z + z_crit * se)


@dataclass
class LayerCorrelation:
    layer: int
    r: float
    p: float
    ci_low: float
    ci_high: float


def layerwise_correlation(
    projections: np.ndarray, behaviour: np.ndarray
) -> list[LayerCorrelation]:
    """Per-layer Pearson r between per-cell projection and a per-cell behaviour
    metric (PAD or VD).

    ``projections`` is ``[n_cells, n_layers]`` (each cell's projection score at
    each layer); ``behaviour`` is ``[n_cells]``. Returns one entry per layer.
    """
    projections = np.asarray(projections, dtype=np.float64)
    behaviour = np.asarray(behaviour, dtype=np.float64)
    if projections.ndim != 2:
        raise ValueError(f"projections must be [n_cells, n_layers], got {projections.shape}")
    if projections.shape[0] != behaviour.shape[0]:
        raise ValueError("projections and behaviour disagree on n_cells")
    n = projections.shape[0]
    out = []
    for l in range(projections.shape[1]):
        r, p = _pearsonr(projections[:, l], behaviour)
        lo, hi = _fisher_ci(r, n)
        out.append(LayerCorrelation(l, r, p, lo, hi))
    return out


def best_layer(correlations: list[LayerCorrelation]) -> Optional[LayerCorrelation]:
    """The layer with the largest |r| (nan-safe). None if all nan."""
    valid = [c for c in correlations if np.isfinite(c.r)]
    return max(valid, key=lambda c: abs(c.r)) if valid else None


def cv_best_layer_correlation(
    projections: np.ndarray, behaviour: np.ndarray
) -> dict[str, float]:
    """Leave-one-cell-out honest estimate. For each held-out cell, pick the
    best-|r| layer on the *other* n−1 cells, take that layer's projection as the
    prediction; correlate the n held-out predictions with actual behaviour.

    This avoids the circularity of reporting the in-sample best layer's r.
    Returns the held-out r, its p and CI, and how often each layer was picked.
    """
    projections = np.asarray(projections, dtype=np.float64)
    behaviour = np.asarray(behaviour, dtype=np.float64)
    n, n_layers = projections.shape
    if n < 4:
        return {"n": n, "cv_r": float("nan"), "cv_p": float("nan"),
                "note": "n<4: too few cells for leave-one-out"}
    preds = np.empty(n)
    picks = np.zeros(n_layers, dtype=int)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        best_l, best_abs = 0, -1.0
        for l in range(n_layers):
            r, _ = _pearsonr(projections[mask, l], behaviour[mask])
            if np.isfinite(r) and abs(r) > best_abs:
                best_abs, best_l = abs(r), l
        picks[best_l] += 1
        preds[i] = projections[i, best_l]
    cv_r, cv_p = _pearsonr(preds, behaviour)
    lo, hi = _fisher_ci(cv_r, n)
    return {
        "n": int(n), "cv_r": cv_r, "cv_p": cv_p,
        "cv_ci_low": lo, "cv_ci_high": hi,
        "layer_pick_counts": picks.tolist(),
        "modal_layer": int(np.argmax(picks)),
    }


def summarise_correlation(
    projections: np.ndarray, pad: np.ndarray, vd: np.ndarray
) -> dict:
    """Full representation↔behaviour summary for one direction: per-layer r
    curves + honest cross-validated r, for both PAD and VD."""
    def _one(behaviour):
        curve = layerwise_correlation(projections, behaviour)
        bl = best_layer(curve)
        return {
            "per_layer_r": [round(c.r, 4) if np.isfinite(c.r) else None for c in curve],
            "best_layer": (None if bl is None else {
                "layer": bl.layer, "r": round(bl.r, 4), "p": round(bl.p, 5),
                "ci": [round(bl.ci_low, 4), round(bl.ci_high, 4)]}),
            "cv": cv_best_layer_correlation(projections, behaviour),
        }
    return {"n_cells": int(np.asarray(pad).shape[0]),
            "vs_pad": _one(pad), "vs_vd": _one(vd)}


__all__ = [
    "LayerCorrelation", "layerwise_correlation", "best_layer",
    "cv_best_layer_correlation", "summarise_correlation",
]
