"""Cross-cutting analysis utilities (panel §2.10).

- `matched_pair_diff` — take two equal-length record sets that differ only in
  one manipulated variable, compute per-metric mean differences and a paired-
  test p-value, return a sensitivity report.
- `per_turn_agreement` — for multi-turn runs, build a cross-channel agreement
  scalar + per-channel persona-state vector at each turn_idx.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Matched-pair counterfactual diff
# ---------------------------------------------------------------------------


def matched_pair_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    metrics: Iterable[str] = ("ch1a_hit", "ch3a_recognised", "ch1b_score"),
    pair_on: Iterable[str] = ("seed", "question_id", "probe_name"),
    labels: tuple[str, str] = ("A", "B"),
) -> pd.DataFrame:
    """Compute paired metric diffs between two matched record sets.

    `df_a` and `df_b` should contain rows that pair up on `pair_on` (e.g.
    same seed + same probe question in both variants of a format-toggle run).
    For each metric present in both frames, returns one row with:
        metric, mean_A, mean_B, mean_diff, n_paired, p_value, effect_size

    p_value is from a paired t-test on non-NaN pairs (or a proportion test for
    0/1 metrics — we use t-test for simplicity; it's robust for n ≥ 20).
    """
    pair_cols = list(pair_on)
    rows = []
    for m in metrics:
        if m not in df_a.columns or m not in df_b.columns:
            continue
        left  = df_a[[*pair_cols, m]].dropna(subset=[m]).rename(columns={m: "val_a"})
        right = df_b[[*pair_cols, m]].dropna(subset=[m]).rename(columns={m: "val_b"})
        merged = left.merge(right, on=pair_cols, how="inner")
        if len(merged) == 0:
            continue
        a = merged["val_a"].astype(float).to_numpy()
        b = merged["val_b"].astype(float).to_numpy()
        diff = b - a
        mean_a, mean_b = float(a.mean()), float(b.mean())
        mean_diff = float(diff.mean())
        p_value = float("nan")
        effect = float("nan")
        if len(diff) >= 2 and np.std(diff) > 0:
            t_stat, p = stats.ttest_1samp(diff, 0.0)
            p_value = float(p)
            effect = mean_diff / (np.std(diff, ddof=1) or 1.0)
        rows.append({
            "metric": m,
            f"mean_{labels[0]}": mean_a,
            f"mean_{labels[1]}": mean_b,
            "mean_diff": mean_diff,
            "n_paired": int(len(diff)),
            "p_value": p_value,
            "effect_size": effect,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-turn cross-channel agreement
# ---------------------------------------------------------------------------


def per_turn_agreement(
    df: pd.DataFrame,
    *,
    channels: Iterable[str] = ("ch1a_hit", "ch3a_recognised"),
    group_extra: Iterable[str] = (),
    normalise: bool = True,
) -> pd.DataFrame:
    """For each turn_idx (+ any extra groupers), compute a per-channel mean and
    a cross-channel agreement scalar.

    Agreement definition: `1 - variance(channel_means) / max_variance`, where
    `max_variance = 0.25` for binary 0/1 metrics (maximised when channels are
    evenly split between 0 and 1). Agreement of 1.0 means perfect alignment
    between channels; 0.0 means maximal disagreement.
    """
    groupers = ["turn_idx", *group_extra]
    present_channels = [c for c in channels if c in df.columns]
    if not present_channels:
        raise ValueError(f"none of {list(channels)} present in df")
    rows = []
    for keys, g in df.groupby(groupers, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groupers, keys))
        means = []
        for c in present_channels:
            vals = g[c].dropna().astype(float)
            m = float(vals.mean()) if len(vals) else float("nan")
            row[f"{c}_mean"] = m
            if not np.isnan(m):
                means.append(m)
        if len(means) >= 2:
            var = float(np.var(means))
            max_var = 0.25 if normalise else max(var, 1e-9)
            row["agreement"] = max(0.0, 1.0 - (var / max_var))
        else:
            row["agreement"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groupers).reset_index(drop=True)


__all__ = ["matched_pair_diff", "per_turn_agreement"]
