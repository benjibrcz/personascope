"""Cross-channel coherence / informativeness / minimum-viable-panel /
disagreement-case analysis. The toolkit for the "channel agreement is
the signal" principle.

Four primitives:

- `channel_correlation_matrix(df, channels, method)` — pairwise channel
  correlation matrix (Pearson / Spearman / Kendall). Identifies redundancy.

- `channel_informativeness(df, channels, target, kind)` — per-channel
  predictive power of `target`. Returns an F-statistic + p-value (categorical
  target) or |Pearson r| + p-value (continuous target), sorted descending.

- `minimum_viable_panel(df, channels, target, threshold)` — forward-selection
  of channels by marginal explanatory gain until a cumulative-R² threshold
  is reached. Returns ordered channel list + per-step R².

- `channel_disagreement_cases(df, channels, z_threshold)` — z-score each
  channel across rows, flag rows where max|z| − min|z| exceeds threshold.
  Returns the flagged rows, sorted by disagreement magnitude — these are
  the empirical cases where channel-disagreement carries load.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

TargetKind = Literal["categorical", "continuous"]
CorrMethod = Literal["pearson", "spearman", "kendall"]


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def channel_correlation_matrix(
    df: pd.DataFrame,
    channels: Iterable[str],
    method: CorrMethod = "pearson",
) -> pd.DataFrame:
    """Pairwise correlation between channels. Drops channels entirely absent
    from `df`. NaN entries are pairwise-deleted by pandas' .corr()."""
    present = [c for c in channels if c in df.columns]
    if len(present) < 2:
        return pd.DataFrame()
    # Coerce to numeric to avoid silent string-column passthrough
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method=method)


# ---------------------------------------------------------------------------
# Informativeness: per-channel predictive power of a target variable
# ---------------------------------------------------------------------------


def channel_informativeness(
    df: pd.DataFrame,
    channels: Iterable[str],
    target: str,
    *,
    kind: TargetKind = "categorical",
    min_group_size: int = 2,
) -> pd.DataFrame:
    """Per-channel predictive power of `target`.

    For `kind="categorical"` → one-way ANOVA F-statistic + p-value.
    For `kind="continuous"`  → |Pearson r| + p-value.

    Returns DataFrame sorted descending by informativeness metric, columns:
    channel, metric, p_value, n_effective.
    """
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in df")
    rows = []
    for ch in channels:
        if ch not in df.columns:
            continue
        sub = df[[ch, target]].dropna()
        # Coerce numeric
        sub[ch] = pd.to_numeric(sub[ch], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 5:
            continue

        if kind == "categorical":
            groups = [g[ch].to_numpy() for _, g in sub.groupby(target)
                      if len(g) >= min_group_size]
            if len(groups) < 2:
                continue
            try:
                f_stat, p = stats.f_oneway(*groups)
            except ValueError:
                continue
            metric = float(f_stat) if np.isfinite(f_stat) else 0.0
            rows.append({
                "channel": ch, "metric": metric, "metric_name": "F_stat",
                "p_value": float(p), "n_effective": int(len(sub)),
            })
        else:
            target_vals = pd.to_numeric(sub[target], errors="coerce")
            valid = target_vals.notna()
            if valid.sum() < 5:
                continue
            try:
                r, p = stats.pearsonr(sub.loc[valid, ch].to_numpy(),
                                       target_vals.loc[valid].to_numpy())
            except ValueError:
                continue
            rows.append({
                "channel": ch, "metric": float(abs(r)), "metric_name": "abs_r",
                "p_value": float(p), "n_effective": int(valid.sum()),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("metric", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Minimum-viable-panel: forward-selection of channels by marginal R²
# ---------------------------------------------------------------------------


def _one_hot(series: pd.Series) -> np.ndarray:
    """Compact one-hot encoder, drops first level to avoid multicollinearity."""
    dummies = pd.get_dummies(series, drop_first=True).astype(float)
    return dummies.to_numpy()


def _r_squared(X: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least squares R² of y on X (no intercept — caller should
    have centred if intercept is needed). Uses pinv for numerical stability."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return 0.0
    # Add an intercept column
    ones = np.ones((X.shape[0], 1))
    Xb = np.concatenate([ones, X], axis=1)
    # β̂ = (XᵀX)⁺ Xᵀ y
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    y_pred = Xb @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def minimum_viable_panel(
    df: pd.DataFrame,
    channels: Iterable[str],
    target: str,
    *,
    kind: TargetKind = "categorical",
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Forward-select channels by marginal R² gain, stopping when cumulative
    R² reaches `threshold` × full-panel R².

    Returns DataFrame rows: step, channel_added, cumulative_r2,
    cumulative_r2_fraction, marginal_gain.
    """
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in df")
    present = [c for c in channels if c in df.columns]
    if not present:
        return pd.DataFrame()

    sub = df[[target] + present].dropna()
    for c in present:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 5:
        return pd.DataFrame()

    # Target matrix
    if kind == "categorical":
        # (no single y vector here — we fit per-column of Y, below)
        Y = _one_hot(sub[target])     # [n, n_classes - 1]
    else:
        Y = pd.to_numeric(sub[target], errors="coerce").to_numpy().reshape(-1, 1)

    # Full-panel R² (average across target columns for categorical)
    X_full = sub[present].to_numpy(dtype=float)
    full_r2 = np.mean([_r_squared(X_full, Y[:, j]) for j in range(Y.shape[1])])

    # Forward selection
    remaining = list(present)
    selected: list[str] = []
    last_r2 = 0.0
    rows = []
    while remaining:
        best, best_r2 = None, last_r2
        for c in remaining:
            cand = selected + [c]
            X = sub[cand].to_numpy(dtype=float)
            r2 = np.mean([_r_squared(X, Y[:, j]) for j in range(Y.shape[1])])
            if r2 > best_r2:
                best, best_r2 = c, r2
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        rows.append({
            "step": len(selected),
            "channel_added": best,
            "cumulative_r2": float(best_r2),
            "cumulative_r2_fraction": float(best_r2 / full_r2) if full_r2 > 0 else float("nan"),
            "marginal_gain": float(best_r2 - last_r2),
        })
        last_r2 = best_r2
        if full_r2 > 0 and best_r2 / full_r2 >= threshold:
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Channel-disagreement cases
# ---------------------------------------------------------------------------


def channel_disagreement_cases(
    df: pd.DataFrame,
    channels: Iterable[str],
    *,
    z_threshold: float = 2.0,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Return rows where the channels disagree most.

    Procedure: z-score each channel across the full `df` (so every channel
    is on a comparable scale), then for each row compute
    `disagreement = max(|z|) − min(|z|)`. Flag rows with disagreement
    exceeding `z_threshold`. Sort descending; truncate to `top_k` if given.

    Non-numeric channels are silently dropped. Returns the flagged rows
    plus the disagreement score column.
    """
    present = [c for c in channels if c in df.columns]
    if not present:
        return pd.DataFrame()

    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    # Per-channel z-score; ignore NaN by pairwise-dropping below
    mean = numeric.mean()
    std = numeric.std().replace(0, np.nan)
    z = (numeric - mean) / std      # NaN rows stay NaN

    # For each row, disagreement = max(|z|) − min(|z|) across the channels
    # that are actually present on that row.
    z_abs = z.abs()
    # Count how many channels have valid z for each row
    valid_count = z_abs.notna().sum(axis=1)
    max_z = z_abs.max(axis=1)
    min_z = z_abs.min(axis=1)
    disagreement = max_z - min_z
    disagreement = disagreement.where(valid_count >= 2)

    mask = disagreement > z_threshold
    flagged = df.loc[mask.fillna(False)].copy()
    flagged["disagreement_score"] = disagreement.loc[mask.fillna(False)]
    flagged = flagged.sort_values("disagreement_score", ascending=False)
    if top_k is not None:
        flagged = flagged.head(int(top_k))
    return flagged


__all__ = [
    "channel_correlation_matrix",
    "channel_informativeness",
    "minimum_viable_panel",
    "channel_disagreement_cases",
]
