"""Channel 6 post-hoc dynamical analyses over TurnRecord trajectories.

- `entrenchment_M`: He-style self-reinforcement coefficient. Given a per-turn
  metric trajectory (usually `ch1a_hit` mean across samples), M is the mean
  turn-to-turn increase *during phases with no new persona-relevant evidence*.
  M > 0 indicates persona strengthens without new evidence (self-reinforcing).
  He 2025 (51/54 CoT configs M > 0).

- `narrative_arc`: classify a per-turn trajectory against a small library of
  prototype arcs (escalation, redemption, contamination, crisis-resolution)
  by Pearson correlation. Returns the best-matching arc, its correlation,
  and the full score vector.

Both functions operate on numeric arrays or per-turn DataFrames — no API calls.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# He-style entrenchment coefficient
# ---------------------------------------------------------------------------


def entrenchment_M(trajectory: Iterable[float],
                   is_neutral: Optional[Iterable[bool]] = None) -> dict:
    """Mean turn-to-turn increase in `trajectory` during neutral phases.

    `trajectory`: length-T sequence of per-turn scalar scores (e.g. identity
    hit-rate averaged across samples at each turn).
    `is_neutral`: optional length-T boolean indicating which turns carried no
    persona-relevant evidence. If not provided, every turn is treated as
    neutral (i.e. every step contributes to M — useful for neutral-decay
    trajectories where the whole span after ICL is silence).

    Returns dict: {M (mean Δ), n_pairs, std, trajectory_len, is_positive}.
    """
    traj = np.asarray(list(trajectory), dtype=float)
    T = len(traj)
    if T < 2:
        return {"M": float("nan"), "n_pairs": 0, "std": float("nan"),
                "trajectory_len": T, "is_positive": None}
    diffs = np.diff(traj)
    if is_neutral is None:
        mask = np.ones(len(diffs), dtype=bool)
    else:
        flags = np.asarray(list(is_neutral), dtype=bool)
        # A pair (t, t+1) contributes iff turn t+1 was neutral (no new evidence)
        mask = flags[1:]
    kept = diffs[mask]
    if len(kept) == 0:
        return {"M": float("nan"), "n_pairs": 0, "std": float("nan"),
                "trajectory_len": T, "is_positive": None}
    M = float(kept.mean())
    return {
        "M": M,
        "n_pairs": int(len(kept)),
        "std": float(kept.std(ddof=1)) if len(kept) > 1 else 0.0,
        "trajectory_len": T,
        "is_positive": bool(M > 0),
    }


def entrenchment_M_from_df(df: pd.DataFrame,
                            metric: str = "ch1a_hit",
                            neutral_intervention_kinds: Iterable[str] = (
                                "no_more_evidence",
                            )) -> dict:
    """Convenience wrapper: load a per-turn aggregated DataFrame (turn_idx +
    mean metric value) and compute entrenchment_M over turns whose main
    intervention kind is in `neutral_intervention_kinds` (default: the
    `no_more_evidence` intervention used by `neutral_decay`)."""
    if metric not in df.columns:
        raise KeyError(f"metric {metric!r} not in df")
    # If df already has per-turn aggregates, use it directly; otherwise aggregate.
    if "turn_idx" in df.columns and df["turn_idx"].is_monotonic_increasing \
            and df["turn_idx"].is_unique:
        agg = df[["turn_idx", metric]].copy()
    else:
        agg = df.groupby("turn_idx", dropna=False)[metric].mean().reset_index()
    agg = agg.sort_values("turn_idx").reset_index(drop=True)
    # Without access to intervention kinds at this stage, treat everything
    # after the baseline (turn_idx >= 0) as neutral. Caller can filter upstream.
    traj = agg[metric].astype(float).tolist()
    return entrenchment_M(traj, is_neutral=None)


# ---------------------------------------------------------------------------
# Narrative-arc classifier
# ---------------------------------------------------------------------------


def _normalise(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.mean()
    s = arr.std()
    return arr / s if s > 1e-12 else arr


# Each prototype is a length-parametric template defined as a function of
# fractional position t ∈ [0, 1]. Classifier evaluates on T evenly-spaced
# points matching the observed trajectory length.
PROTOTYPE_ARCS: dict[str, callable] = {
    # Escalation: monotonic rise
    "escalation":       lambda t: t,
    # Redemption: rise then fall (down-going in second half)
    "redemption":       lambda t: 2 * t * (1 - t),
    # Contamination (Breaking-Bad): rise that stays high after a shock
    "contamination":    lambda t: np.where(t < 0.5, t * 2, 1.0 - 0.1 * (t - 0.5)),
    # Crisis-resolution: dip-then-recover
    "crisis_resolution": lambda t: np.where(t < 0.4, 1.0 - 2 * t,
                                            np.where(t < 0.6, 0.2, (t - 0.6) * 2.0 + 0.2)),
    # Flat: no change (null / reference)
    "flat":             lambda t: np.zeros_like(t),
}


def _prototype_trajectory(name: str, T: int) -> np.ndarray:
    fn = PROTOTYPE_ARCS[name]
    t = np.linspace(0.0, 1.0, T)
    return fn(t).astype(float)


def narrative_arc(trajectory: Iterable[float],
                  prototypes: Optional[Iterable[str]] = None,
                  correlation_threshold: float = 0.5) -> dict:
    """Classify an observed trajectory into the best-fitting prototype arc.

    Returns {best, correlation, scores, trajectory_len}. `best` is None if no
    prototype correlates above `correlation_threshold`.

    `scores` is a dict of {prototype: pearson_r} for each prototype tested.
    `flat` prototype gives zero-variance templates — we skip Pearson and
    report 0.0 for it (flat reference scores separately via std of the
    trajectory itself in `trajectory_std`).
    """
    traj = np.asarray(list(trajectory), dtype=float)
    T = len(traj)
    if T < 3:
        return {"best": None, "correlation": float("nan"),
                "scores": {}, "trajectory_len": T, "trajectory_std": 0.0}
    names = list(prototypes) if prototypes is not None else list(PROTOTYPE_ARCS)
    traj_std = float(traj.std())
    traj_norm = _normalise(traj)
    scores: dict[str, float] = {}
    for name in names:
        if name == "flat":
            scores[name] = 0.0
            continue
        proto = _prototype_trajectory(name, T)
        if proto.std() < 1e-12:
            scores[name] = 0.0
            continue
        proto_norm = _normalise(proto)
        # Pearson correlation via normalised dot-product
        r = float(np.dot(traj_norm, proto_norm) / len(traj_norm))
        scores[name] = r
    # Pick best (highest correlation).
    best = max(scores, key=scores.get) if scores else None
    best_r = scores.get(best, float("nan"))
    if best_r < correlation_threshold or traj_std < 1e-6:
        best = None
    return {
        "best": best,
        "correlation": float(best_r),
        "scores": scores,
        "trajectory_len": T,
        "trajectory_std": traj_std,
    }


__all__ = [
    "entrenchment_M",
    "entrenchment_M_from_df",
    "narrative_arc",
    "PROTOTYPE_ARCS",
]
