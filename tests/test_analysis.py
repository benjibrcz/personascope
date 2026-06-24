"""personascope.analysis: Wilson CI, Bigelow, aggregation."""

from __future__ import annotations

import numpy as np

from personascope.analysis import aggregate_per_k, fit_bigelow, wilson_ci
from personascope.analysis.fit import bigelow


def test_wilson_ci_trivial_cases():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    # p_hat = 0.5 on 10 → symmetric around 0.5
    lo, hi = wilson_ci(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    assert abs((lo + hi) / 2 - 0.5) < 0.05  # roughly centred
    # Extreme hits still produces non-degenerate bounds
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 or abs(hi - 1.0) < 1e-9
    assert lo > 0.5


def test_bigelow_monotonic_nondecreasing_for_positive_gamma():
    k = np.linspace(0, 64, 50)
    p = bigelow(k, L=1.0, b=-3.0, gamma=1.5, alpha=0.5)
    diffs = np.diff(p)
    assert np.all(diffs >= -1e-9)


def test_bigelow_saturates_to_L():
    k_large = np.array([1e4])
    p = bigelow(k_large, L=0.9, b=-3.0, gamma=1.5, alpha=0.5)[0]
    assert 0.85 < p <= 0.9 + 1e-6


def test_fit_bigelow_recovers_known_params_approximately():
    true_L, true_b, true_gamma, true_alpha = 0.95, -4.0, 1.2, 0.6
    k = np.array([0, 1, 2, 4, 6, 8, 12, 16, 24, 32], dtype=float)
    p = bigelow(k, true_L, true_b, true_gamma, true_alpha) + np.random.default_rng(0).normal(0, 0.02, size=len(k))
    p = np.clip(p, 0.0, 1.0)
    fit = fit_bigelow(k, p)
    assert fit.converged
    assert fit.r2 > 0.95
    # k_star check: recomputed from fit params should be finite and within ~2× of truth
    true_k_star = (-true_b / true_gamma) ** (1.0 / (1.0 - true_alpha))
    assert fit.k_star > 0
    assert 0.5 * true_k_star < fit.k_star < 2.0 * true_k_star


def test_aggregate_per_k_returns_wilson_columns():
    import pandas as pd
    df = pd.DataFrame({
        "k": [0, 0, 4, 4, 8, 8],
        "ch1a_hit": [0, 0, 0, 1, 1, 1],
    })
    agg = aggregate_per_k(df, metrics=["ch1a_hit"])
    assert list(agg.columns) == ["k", "ch1a_hit", "ch1a_hit_n", "ch1a_hit_ci_lo", "ch1a_hit_ci_hi"]
    row_k0 = agg[agg["k"] == 0].iloc[0]
    assert row_k0["ch1a_hit"] == 0.0
    assert row_k0["ch1a_hit_n"] == 2
