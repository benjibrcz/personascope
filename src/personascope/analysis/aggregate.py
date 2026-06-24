"""Aggregation utilities for the personascope analysis surface."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _wilson_pair(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(int)
    lo, hi = wilson_ci(int(values.sum()), int(len(values)))
    return pd.Series({"mean": values.mean() if len(values) else float("nan"),
                      "n": int(len(values)),
                      "ci_lo": lo, "ci_hi": hi})


# ---------------------------------------------------------------------------
# Per-k aggregation
# ---------------------------------------------------------------------------


def aggregate_per_k(
    df: pd.DataFrame,
    metrics: Iterable[str] = ("ch1a_hit", "ch3a_recognised"),
    group_extra: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate by k (plus any extra grouping columns like model or persona).

    Returns a DataFrame with columns: k, [group_extra], <metric>, <metric>_n,
    <metric>_ci_lo, <metric>_ci_hi for each present metric.
    """
    groupers = ["k", *group_extra]
    rows = []
    for keys, g in df.groupby(groupers, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groupers, keys))
        for m in metrics:
            if m not in g.columns:
                continue
            agg = _wilson_pair(g[m])
            row[m] = agg["mean"]
            row[f"{m}_n"] = agg["n"]
            row[f"{m}_ci_lo"] = agg["ci_lo"]
            row[f"{m}_ci_hi"] = agg["ci_hi"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groupers).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-turn aggregation (multi-turn runs)
# ---------------------------------------------------------------------------


def aggregate_per_turn(
    df: pd.DataFrame,
    metrics: Iterable[str] = ("ch1a_hit", "ch3a_recognised"),
    group_extra: Iterable[str] = (),
) -> pd.DataFrame:
    """Aggregate by turn_idx (and optional extras). Same output shape as aggregate_per_k."""
    groupers = ["turn_idx", *group_extra]
    rows = []
    for keys, g in df.groupby(groupers, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groupers, keys))
        for m in metrics:
            if m not in g.columns:
                continue
            agg = _wilson_pair(g[m])
            row[m] = agg["mean"]
            row[f"{m}_n"] = agg["n"]
            row[f"{m}_ci_lo"] = agg["ci_lo"]
            row[f"{m}_ci_hi"] = agg["ci_hi"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groupers).reset_index(drop=True)
