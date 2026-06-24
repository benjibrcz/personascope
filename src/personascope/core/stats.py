"""Confidence-interval helpers shared across probe summarisers.

Two flavours are needed across the pipeline:

- **Wilson interval** for binary rates (refuse / engage / hit / hold).
  Closed-form, behaves well at the boundary even when n is small (n=8
  is our default, so normal-approximation CIs would be terrible).
- **Nonparametric bootstrap** for continuous means (Betley alignment
  scores, multi-turn delta-engage means).

Both return `(lo, hi)` tuples on the same scale as the input. Both
return `(None, None)` when the input is empty / undefined, so callers
can store them as-is in the summary block.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def wilson_ci(k: int | None, n: int | None, *, z: float = 1.96
              ) -> tuple[float | None, float | None]:
    """95%-default Wilson confidence interval for a binary rate.

    `k` = positive count, `n` = total. Returns `(lo, hi)` ∈ [0, 1] or
    `(None, None)` if `n` is None / 0.
    """
    if k is None or n is None or n <= 0:
        return (None, None)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo = max(0.0, centre - spread)
    hi = min(1.0, centre + spread)
    return (lo, hi)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0
                 ) -> tuple[float | None, float | None]:
    """Nonparametric percentile bootstrap CI for the mean of `values`.

    Returns `(lo, hi)` at the (alpha/2, 1-alpha/2) percentiles, or
    `(None, None)` if `values` is empty. Deterministic given `seed`.
    """
    if not values:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot) - 1
    return (means[lo_idx], means[hi_idx])


__all__ = ["wilson_ci", "bootstrap_ci"]
