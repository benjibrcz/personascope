"""Causal steering estimators — pure numpy, unit-tested offline.

Design (docs/repr_preregistration.md §7): every steering condition is run on
the SAME matched (item × seed) blocks as its baseline, in randomised order
within each block. The estimand is the *paired* mean contrast over blocks;
inference is **clustered randomization inference** (sign-flip of the paired
difference, flipping all blocks of an item together, so seeds within an item
are never treated as independent).

Pieces:
  paired_contrast             mean paired difference + cluster-bootstrap CI
  randomization_inference_p   cluster sign-flip RI p-value
  signed_gate                 hierarchical gating of the three signed tests
  specificity_test            true-vector effect vs ≥20 pre-specified random/off-target vectors
  non_inferiority_gate        coherence / refusal / length NI gates (paired, bootstrap CI)
  factorial_contrasts         adapter{on,off} × steer{off,−dir} with matched random controls
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MIN_NULL_DIRECTIONS = 20


def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _paired(treated, baseline) -> np.ndarray:
    t, b = _as_float(treated), _as_float(baseline)
    if t.shape != b.shape:
        raise ValueError(f"treated/baseline shape mismatch {t.shape} vs {b.shape}")
    return t - b


def _clusters(clusters, n) -> np.ndarray:
    if clusters is None:
        return np.arange(n)
    c = np.asarray(clusters)
    if c.shape[0] != n:
        raise ValueError("clusters must have one entry per block")
    _, inv = np.unique(c, return_inverse=True)
    return inv


def cluster_bootstrap_ci(values, clusters=None, *, n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05, stat=np.nanmean) -> tuple[float, float]:
    """Percentile CI of `stat(values)` resampling CLUSTERS with replacement."""
    v = _as_float(values)
    cl = _clusters(clusters, len(v))
    groups = [np.where(cl == g)[0] for g in np.unique(cl)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        stats[i] = stat(v[idx])
    return float(np.nanpercentile(stats, 100 * alpha / 2)), float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))


def randomization_inference_p(diffs, clusters=None, *, n_perm: int = 5000, seed: int = 0,
                              alternative: str = "greater") -> dict:
    """Cluster sign-flip randomization-inference p for H0: mean paired diff = 0.

    `diffs[b]` is the paired difference in block b (treated − baseline); blocks
    sharing a cluster id (an item) are flipped together. NaN blocks (failures)
    are dropped. `alternative` ∈ {greater, less, two-sided}. Uses the +1
    correction so p ≥ 1/(n_perm+1)."""
    d = _as_float(diffs)
    cl = _clusters(clusters, len(d))
    ok = np.isfinite(d)
    d, cl = d[ok], cl[ok]
    if d.size == 0:
        return {"p": float("nan"), "observed": float("nan"), "n_blocks": 0, "n_clusters": 0}
    _, cl = np.unique(cl, return_inverse=True)
    n_cl = int(cl.max()) + 1
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, n_cl))
    null = (flips[:, cl] * d[None, :]).mean(axis=1)
    if alternative == "greater":
        extreme = null >= obs
    elif alternative == "less":
        extreme = null <= obs
    elif alternative == "two-sided":
        extreme = np.abs(null) >= abs(obs)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    p = (1.0 + float(extreme.sum())) / (n_perm + 1.0)
    return {"p": p, "observed": obs, "n_blocks": int(d.size), "n_clusters": n_cl,
            "n_perm": n_perm, "alternative": alternative}


def paired_contrast(treated, baseline, clusters=None, *, n_boot: int = 2000, seed: int = 0,
                    alternative: str = "greater", n_perm: int = 5000) -> dict:
    """Mean paired contrast (treated − baseline) over matched blocks, with a
    cluster-bootstrap CI and a clustered RI p-value."""
    d = _paired(treated, baseline)
    ok = np.isfinite(d)
    lo, hi = cluster_bootstrap_ci(d[ok], None if clusters is None else np.asarray(clusters)[ok],
                                  n_boot=n_boot, seed=seed) if ok.sum() else (float("nan"),) * 2
    ri = randomization_inference_p(d, clusters, n_perm=n_perm, seed=seed, alternative=alternative)
    return {"mean_diff": float(np.nanmean(d)) if ok.any() else float("nan"),
            "ci": [lo, hi], "n_blocks": int(ok.sum()), "n_missing": int((~ok).sum()), "ri": ri}


def signed_gate(plus, minus, base, clusters=None, *, alpha: float = 0.05, seed: int = 0,
                n_perm: int = 5000) -> dict:
    """Hierarchical (fixed-sequence) gating of the three signed tests.

    Gate 1: +dir > −dir (the sign-specific contrast) at `alpha`. Only if it
    passes are gate 2 (+dir > base) and gate 3 (−dir < base) tested, each at
    `alpha` — fixed-sequence testing controls the family-wise error without a
    Bonferroni split. All three p's are reported regardless; `passed` records
    which were *declared* under the hierarchy."""
    t1 = paired_contrast(plus, minus, clusters, seed=seed, alternative="greater", n_perm=n_perm)
    t2 = paired_contrast(plus, base, clusters, seed=seed + 1, alternative="greater", n_perm=n_perm)
    t3 = paired_contrast(minus, base, clusters, seed=seed + 2, alternative="less", n_perm=n_perm)
    g1 = bool(t1["ri"]["p"] < alpha)
    g2 = bool(g1 and t2["ri"]["p"] < alpha)
    g3 = bool(g1 and t3["ri"]["p"] < alpha)
    return {"alpha": alpha, "plus_gt_minus": {**t1, "p": t1["ri"]["p"], "declared": g1},
            "plus_gt_base": {**t2, "p": t2["ri"]["p"], "declared": g2},
            "minus_lt_base": {**t3, "p": t3["ri"]["p"], "declared": g3},
            "n_passed": int(g1) + int(g2) + int(g3), "gate1_passed": g1}


def specificity_test(true_effect: float, null_effects: Sequence[float], *,
                     min_null: int = MIN_NULL_DIRECTIONS) -> dict:
    """Rank the true-vector paired effect against the paired effects of ≥`min_null`
    pre-specified random / off-target directions run at the SAME frozen (layer,
    scale) on the SAME blocks. p = (1 + #{null ≥ true}) / (1 + n_null). Fails
    closed (raises) if fewer than `min_null` finite null effects are supplied."""
    null = _as_float(list(null_effects))
    null = null[np.isfinite(null)]
    if null.size < min_null:
        raise ValueError(f"specificity test needs ≥{min_null} null directions, got {null.size}")
    if not np.isfinite(true_effect):
        return {"p": float("nan"), "true_effect": float("nan"), "n_null": int(null.size)}
    p = (1.0 + float((null >= true_effect).sum())) / (1.0 + null.size)
    return {"p": p, "true_effect": float(true_effect), "n_null": int(null.size),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)) if null.size > 1 else 0.0,
            "null_max": float(null.max()),
            "z_vs_null": float((true_effect - null.mean()) / (null.std(ddof=1) + 1e-12)) if null.size > 1 else float("nan")}


def non_inferiority_gate(treated, baseline, *, margin: float, direction: str, clusters=None,
                         n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired non-inferiority gate. `direction="higher_is_better"` (coherence):
    pass iff the lower CI bound of mean(treated − baseline) > −margin.
    `direction="lower_is_better"` (refusal rate, |log length ratio|): pass iff
    the upper CI bound of mean(treated − baseline) < +margin. One-sided
    (1−alpha) bound via cluster bootstrap."""
    d = _paired(treated, baseline)
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return {"pass": False, "mean_diff": float("nan"), "bound": float("nan"), "n": int(ok.sum()),
                "margin": margin, "direction": direction, "reason": "too few finite blocks"}
    cl = None if clusters is None else np.asarray(clusters)[ok]
    lo, hi = cluster_bootstrap_ci(d[ok], cl, n_boot=n_boot, seed=seed, alpha=2 * alpha)
    if direction == "higher_is_better":
        passed, bound = bool(lo > -margin), lo
    elif direction == "lower_is_better":
        passed, bound = bool(hi < margin), hi
    else:
        raise ValueError("direction must be 'higher_is_better' or 'lower_is_better'")
    return {"pass": passed, "mean_diff": float(d[ok].mean()), "bound": float(bound),
            "n": int(ok.sum()), "margin": margin, "direction": direction}


def factorial_contrasts(cells: dict[tuple[str, str], Sequence[float]], clusters=None, *,
                        random_control_keys: Optional[Sequence[str]] = None,
                        n_perm: int = 5000, seed: int = 0, alpha: float = 0.05) -> dict:
    """adapter{off,on} × steer{off,minus} on identical blocks.

    `cells[(adapter, steer)]` are per-block outcomes with adapter ∈ {off,on},
    steer ∈ {off, minus, <random-control keys>}. Reports:
      adapter_effect            (on,off) − (off,off)          [expected > 0]
      counter_steer_on_adapter  (on,minus) − (on,off)         [expected < 0]
      counter_steer_on_base     (off,minus) − (off,off)       [expected < 0]
      interaction               [(on,minus)−(on,off)] − [(off,minus)−(off,off)]
      specificity_vs_random     counter-steer effect on the adapter vs the
                                distribution of matched random-vector effects
                                (on,rand_k) − (on,off), sign-aware (more negative = stronger)."""
    def get(a, s):
        if (a, s) not in cells:
            raise KeyError(f"factorial cell {(a, s)} missing")
        return _as_float(cells[(a, s)])
    off_off, on_off, off_minus, on_minus = get("off", "off"), get("on", "off"), get("off", "minus"), get("on", "minus")
    out = {
        "adapter_effect": paired_contrast(on_off, off_off, clusters, seed=seed, alternative="greater", n_perm=n_perm),
        "counter_steer_on_adapter": paired_contrast(on_minus, on_off, clusters, seed=seed + 1, alternative="less", n_perm=n_perm),
        "counter_steer_on_base": paired_contrast(off_minus, off_off, clusters, seed=seed + 2, alternative="less", n_perm=n_perm),
    }
    inter = (on_minus - on_off) - (off_minus - off_off)
    out["interaction"] = {"mean": float(np.nanmean(inter)),
                          "ri": randomization_inference_p(inter, clusters, n_perm=n_perm, seed=seed + 3, alternative="two-sided")}
    if random_control_keys:
        true_eff = -float(np.nanmean(on_minus - on_off))            # flip: stronger counter-steer = larger
        null = [-float(np.nanmean(get("on", k) - on_off)) for k in random_control_keys]
        out["specificity_vs_random"] = specificity_test(true_eff, null)
    out["alpha"] = alpha
    return out


__all__ = [
    "MIN_NULL_DIRECTIONS", "cluster_bootstrap_ci", "randomization_inference_p",
    "paired_contrast", "signed_gate", "specificity_test", "non_inferiority_gate",
    "factorial_contrasts",
]
