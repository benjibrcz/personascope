"""Confirmatory representation↔behaviour association — the ONE pre-registered
estimand of the representation channel (docs/repr_preregistration.md §5).

De-mixed from the exploratory `analysis/representation.py` (per-layer r
curves, leave-one-cell-out layer selection — descriptive only). Here:

- **Unit = cell**, all cells from ONE route (system prompt), independently
  instantiated (distinct paraphrases), each run on the SAME (item × seed)
  blocks. Steering is NOT a cell (steering along the direction inflates its
  own projection — circular).
- x_c = mean over valid responses of the per-response projection at the
  FROZEN layer; y_c = mean judge scalar over the same responses.
- The curated grid (15 hand-authored prompts at 5 designed levels + a base) is
  a set of FIXED treatments, NOT exchangeable replicates, so it is reported
  **descriptively**: Spearman ρ(x_c, y_c) + Pearson r over the cell means, with
  an item-cluster bootstrap CI for item-sampling uncertainty. **No
  exchangeability/permutation p-value is claimed over these cells** (external
  review); a confirmatory significance test awaits the deferred
  independent-cell-sampling scheme (`pairing_permutation_test`, kept for that).
- Reportability is gated on the judge-agreement gate passing.

Also: the behaviour-blind layer-freeze rule, judge-agreement (Cohen's κ),
direction-stability diagnostics, and the power calculation.
"""

from __future__ import annotations

from math import atanh, sqrt
from typing import Optional, Sequence

import numpy as np

# ── correlation statistics ───────────────────────────────────────────────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    return pearson(_rankdata(np.asarray(x, float)), _rankdata(np.asarray(y, float)))


_STATS = {"spearman": spearman, "pearson": pearson}

# Hard, non-overridable pre-registered floor for a REPORTABLE confirmatory
# result (external review): the confirmatory path re-checks these regardless of
# any caller-supplied `min_n` on the gate.
CONFIRMATORY_MIN_N = 240
CONFIRMATORY_KAPPA_4WAY_MIN = 0.6
CONFIRMATORY_KAPPA_BINARY_MIN = 0.7


def _confirmatory_gate_ok(g) -> tuple[bool, str]:
    """Independently re-validate a judge-agreement gate dict against the HARD
    floor (does NOT trust `g['pass']`; ignores any lowered `min_n`)."""
    if not isinstance(g, dict):
        return False, "no judge-agreement gate supplied"
    n, k4, kb = g.get("n"), g.get("kappa_4way"), g.get("kappa_binary")
    def _real(v):  # a real (non-bool) number — bool is a subclass of int, and
        return isinstance(v, (int, float)) and not isinstance(v, bool)  # JSON `true` must NOT pass
    if not _real(n) or isinstance(n, float) or n < CONFIRMATORY_MIN_N:
        return False, f"n={n!r} not an int ≥ {CONFIRMATORY_MIN_N}"
    if not (_real(k4) and np.isfinite(k4) and k4 >= CONFIRMATORY_KAPPA_4WAY_MIN):
        return False, f"kappa_4way={k4!r} < {CONFIRMATORY_KAPPA_4WAY_MIN}, non-finite, or non-numeric"
    if not (_real(kb) and np.isfinite(kb) and kb >= CONFIRMATORY_KAPPA_BINARY_MIN):
        return False, f"kappa_binary={kb!r} < {CONFIRMATORY_KAPPA_BINARY_MIN}, non-finite, or non-numeric"
    return True, "ok"


def pairing_permutation_test(x, y, *, stat: str = "spearman", n_perm: int = 10000,
                             seed: int = 0, alternative: str = "greater") -> dict:
    """Permute the x↔y pairing across cells. VALID ONLY for the DEFERRED
    independent-cell-sampling scheme, where cells are exchangeable under H0. It
    is deliberately NOT applied to the curated grid (fixed, non-exchangeable
    treatments) — doing so fabricates significance (external review). Returns the
    observed statistic and the +1-corrected Monte-Carlo p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    fn = _STATS[stat]
    obs = fn(x, y)
    if not np.isfinite(obs):
        return {"stat": stat, "observed": float("nan"), "p": float("nan"), "n_cells": int(ok.sum())}
    rng = np.random.default_rng(seed)
    null = np.array([fn(x, y[rng.permutation(len(y))]) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    if alternative == "greater":
        k = (null >= obs).sum()
    elif alternative == "less":
        k = (null <= obs).sum()
    elif alternative == "two-sided":
        k = (np.abs(null) >= abs(obs)).sum()
    else:
        raise ValueError(alternative)
    return {"stat": stat, "observed": float(obs), "p": (1.0 + float(k)) / (1.0 + len(null)),
            "n_cells": int(ok.sum()), "n_perm": int(len(null)), "alternative": alternative}


def cell_level_xy(cell_ids: Sequence, item_ids: Sequence, x: Sequence, y: Sequence, *,
                  min_valid_fraction: float = 0.8, n_blocks_expected: Optional[int] = None) -> dict:
    """Aggregate per-response (projection x, judge scalar y) to cell means.

    A response is VALID only if both x and y are finite (a REFUSES / failed
    judge / failed capture removes the response from BOTH x_c and y_c, so the
    two means are over the same responses). A cell is DROPPED if fewer than
    `min_valid_fraction` of its expected blocks are valid."""
    cell_ids, item_ids = np.asarray(cell_ids), np.asarray(item_ids)
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    cells = list(dict.fromkeys(cell_ids.tolist()))
    xs, ys, ns, dropped = [], [], [], []
    for c in cells:
        m = cell_ids == c
        n_exp = n_blocks_expected if n_blocks_expected is not None else int(m.sum())
        n_ok = int((m & valid).sum())
        if n_exp == 0 or n_ok < min_valid_fraction * n_exp or n_ok == 0:
            dropped.append({"cell": c, "n_valid": n_ok, "n_expected": n_exp})
            continue
        xs.append(float(x[m & valid].mean()))
        ys.append(float(y[m & valid].mean()))
        ns.append(n_ok)
    kept = [c for c in cells if c not in {d["cell"] for d in dropped}]
    return {"cells": kept, "x": np.array(xs), "y": np.array(ys), "n_valid": ns, "dropped": dropped}


def item_bootstrap_correlation_ci(cell_ids, item_ids, x, y, *, stat: str = "spearman",
                                  n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Resample ITEMS with replacement (all cells × seeds of an item move
    together), recompute the cell means and the statistic → percentile CI."""
    cell_ids, item_ids = np.asarray(cell_ids), np.asarray(item_ids)
    x, y = np.asarray(x, float), np.asarray(y, float)
    items = np.unique(item_ids)
    rng = np.random.default_rng(seed)
    fn = _STATS[stat]
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(items, size=len(items), replace=True)
        idx = np.concatenate([np.where(item_ids == it)[0] for it in pick])
        agg = cell_level_xy(cell_ids[idx], item_ids[idx], x[idx], y[idx], min_valid_fraction=0.0)
        v = fn(agg["x"], agg["y"]) if len(agg["x"]) >= 3 else float("nan")
        vals.append(v)
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"ci": [float("nan"), float("nan")], "n_boot": 0}
    return {"ci": [float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))],
            "n_boot": int(vals.size), "stat": stat}


def confirmatory_association(cell_ids, item_ids, x, y, *, n_boot: int = 2000,
                             seed: int = 0, alpha: float = 0.05, min_cells: int = 12,
                             n_blocks_expected: Optional[int] = None,
                             judge_gate: Optional[dict] = None) -> dict:
    """The pre-registered readout over the curated grid — **DESCRIPTIVE** (no
    exchangeability p; external review). Cell aggregation → Spearman ρ + Pearson
    r over cell means + an item-cluster bootstrap CI. `reportable` is True only
    if `judge_gate` (from `judge_agreement_gate`) is supplied AND passes."""
    agg = cell_level_xy(cell_ids, item_ids, x, y, n_blocks_expected=n_blocks_expected)
    out = {"mode": "descriptive", "cells": agg["cells"], "x_c": agg["x"].tolist(),
           "y_c": agg["y"].tolist(), "n_valid_per_cell": agg["n_valid"],
           "dropped_cells": agg["dropped"], "alpha": alpha}
    if len(agg["cells"]) < min_cells:
        out["valid"] = False
        out["reason"] = f"only {len(agg['cells'])} cells retained (< {min_cells}) — pre-registered STOP"
        return out
    keep = np.isin(np.asarray(cell_ids), agg["cells"])
    out["valid"] = True
    out["spearman_rho"] = spearman(agg["x"], agg["y"])          # primary (descriptive)
    out["pearson_r"] = pearson(agg["x"], agg["y"])              # secondary/sensitivity
    out["item_bootstrap_ci"] = item_bootstrap_correlation_ci(
        np.asarray(cell_ids)[keep], np.asarray(item_ids)[keep], np.asarray(x, float)[keep],
        np.asarray(y, float)[keep], stat="spearman", n_boot=n_boot, seed=seed + 1, alpha=alpha)
    out["inference_note"] = (
        "Curated grid of fixed treatments → reported DESCRIPTIVELY. The item-cluster "
        "bootstrap CI reflects item-sampling uncertainty only; NO exchangeability/"
        "permutation p-value is claimed over these non-exchangeable cells. A "
        "confirmatory significance test requires the deferred independent-cell scheme.")
    # Reportability gate: descriptive result is trustworthy only if the judge
    # agreement gate was supplied AND independently re-validates against the
    # HARD pre-registered floor (external review): we do NOT trust the gate's
    # self-reported `pass` (a malformed {"pass": True, "n": 1} must not slip
    # through) and we ignore any caller-lowered `min_n` — the confirmatory path
    # requires n ≥ CONFIRMATORY_MIN_N and finite κ ≥ thresholds computed here.
    out["judge_gate"] = judge_gate
    ok, why = _confirmatory_gate_ok(judge_gate)
    out["reportable"] = ok
    if not ok:
        out["reportable_note"] = f"judge-agreement gate not satisfied: {why}"
    return out


# ── power ────────────────────────────────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    from math import erf
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:  # noqa: BLE001
        # Acklam-free bisection fallback
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if _norm_cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def power_for_correlation(r: float, n: int, *, alpha: float = 0.05, one_sided: bool = True) -> float:
    """Fisher-z approximate power to detect a population correlation `r` with `n`
    cells at `alpha` (one-sided by default — the pre-registered direction)."""
    if n < 4 or not 0 < abs(r) < 1:
        return float("nan")
    z = atanh(r) * sqrt(n - 3)
    zc = _norm_ppf(1 - alpha) if one_sided else _norm_ppf(1 - alpha / 2)
    return float(1.0 - _norm_cdf(zc - z))


def n_cells_for_power(r: float, *, power: float = 0.8, alpha: float = 0.05, one_sided: bool = True,
                      n_max: int = 500) -> int:
    for n in range(4, n_max + 1):
        if power_for_correlation(r, n, alpha=alpha, one_sided=one_sided) >= power:
            return n
    return n_max


# ── judge agreement ──────────────────────────────────────────────────────────

def cohens_kappa(a: Sequence, b: Sequence) -> float:
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length non-empty label sequences")
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for u, v in zip(a, b):
        m[idx[u], idx[v]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(1) * m.sum(0)).sum() / n**2)
    # Degenerate: both raters used a single category (pe≥1) → κ is UNDEFINED, not
    # perfect. Returning 1.0 here let a 1-sample all-"CORRECTS" pair pass the gate
    # (external review). Return NaN so the gate fails closed.
    if not np.isfinite(pe) or pe >= 1.0:
        return float("nan")
    return float((po - pe) / (1 - pe))


def judge_agreement_gate(primary: Sequence, secondary: Sequence, *, kappa_min: float = 0.6,
                         binary_key: str = "AGREES_WITH_ERROR", kappa_min_binary: float = 0.7,
                         min_n: int = 240) -> dict:
    """Pre-registered judge gate: on the double-judged subset, require
    n ≥ `min_n` usable ratings AND finite κ(4-way) ≥ 0.6 AND finite
    κ(AGREES vs not) ≥ 0.7. FAILS CLOSED on too-few ratings or degenerate/
    non-finite κ (external review). A failed gate ⇒ the confirmatory result is
    NOT reportable."""
    primary, secondary = list(primary), list(secondary)
    n = len(primary)
    k4 = cohens_kappa(primary, secondary)
    kb = cohens_kappa([v == binary_key for v in primary], [v == binary_key for v in secondary])
    reasons = []
    if n < min_n:
        reasons.append(f"n={n} < min_n={min_n}")
    if not np.isfinite(k4):
        reasons.append("kappa_4way non-finite/degenerate")
    elif k4 < kappa_min:
        reasons.append(f"kappa_4way={k4:.3f} < {kappa_min}")
    if not np.isfinite(kb):
        reasons.append("kappa_binary non-finite/degenerate")
    elif kb < kappa_min_binary:
        reasons.append(f"kappa_binary={kb:.3f} < {kappa_min_binary}")
    return {"kappa_4way": k4, "kappa_binary": kb, "n": n, "min_n": min_n,
            "pass": bool(not reasons), "fail_reasons": reasons,
            "thresholds": {"kappa_4way": kappa_min, "kappa_binary": kappa_min_binary}}


# ── layer freeze (behaviour-blind) + direction stability ─────────────────────

def select_frozen_layer(pos_proj, neg_proj, *, min_sign_consistency: float = 0.9,
                        min_standardized_sep: float = 0.5) -> dict:
    """Behaviour-blind layer-freeze rule on the LAYER-VALIDATION items.

    `pos_proj`/`neg_proj` are `[n_pairs, n_layers]` per-response projections of
    the trait-positive / trait-negative contrast responses on identical
    (item, seed, contrast-pair) blocks. Per layer: d = pos − neg (paired);
    standardized separation = mean(d)/sd(d); sign consistency = frac(d > 0).
    Choose argmax standardized separation among layers meeting BOTH minima;
    if none does → `layer` is None (pre-registered STOP: no confirmatory run)."""
    p, q = np.asarray(pos_proj, float), np.asarray(neg_proj, float)
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError("pos/neg projections must be [n_pairs, n_layers] and match")
    d = p - q
    mean, sd = d.mean(0), d.std(0, ddof=1) if d.shape[0] > 1 else np.full(d.shape[1], np.nan)
    sep = mean / (sd + 1e-12)
    cons = (d > 0).mean(0)
    eligible = (cons >= min_sign_consistency) & (sep >= min_standardized_sep)
    layer = int(np.argmax(np.where(eligible, sep, -np.inf))) if eligible.any() else None
    return {"layer": layer, "standardized_sep": sep.tolist(), "sign_consistency": cons.tolist(),
            "eligible": eligible.tolist(), "n_pairs": int(d.shape[0]),
            "rule": {"min_sign_consistency": min_sign_consistency, "min_standardized_sep": min_standardized_sep}}


def split_half_cosine(pos_pooled, neg_pooled, *, seed: int = 0, n_splits: int = 20) -> dict:
    """Direction stability: split the fit responses in half at random, fit a
    mean-diff direction on each half, report the per-layer cosine between
    halves (mean over splits)."""
    from personascope.probes.representation.directions import cos_sim, mean_diff_direction
    P, N = np.asarray(pos_pooled, float), np.asarray(neg_pooled, float)
    rng = np.random.default_rng(seed)
    cos = []
    for _ in range(n_splits):
        ip, im = rng.permutation(len(P)), rng.permutation(len(N))
        a = mean_diff_direction(P[ip[: len(P) // 2]], N[im[: len(N) // 2]])
        b = mean_diff_direction(P[ip[len(P) // 2:]], N[im[len(N) // 2:]])
        cos.append([float(cos_sim(a[layer], b[layer])) for layer in range(a.shape[0])])
    cos = np.array(cos)
    return {"per_layer_mean_cosine": cos.mean(0).tolist(), "per_layer_min_cosine": cos.min(0).tolist(),
            "n_splits": n_splits}


def bootstrap_layer_stability(pos_proj, neg_proj, *, n_boot: int = 500, seed: int = 0, **rule) -> dict:
    """How often each layer is chosen by `select_frozen_layer` under bootstrap
    resampling of the validation pairs."""
    p, q = np.asarray(pos_proj, float), np.asarray(neg_proj, float)
    rng = np.random.default_rng(seed)
    picks = np.zeros(p.shape[1], dtype=int)
    n_none = 0
    for _ in range(n_boot):
        idx = rng.integers(0, p.shape[0], size=p.shape[0])
        layer = select_frozen_layer(p[idx], q[idx], **rule)["layer"]
        if layer is None:
            n_none += 1
        else:
            picks[layer] += 1
    return {"pick_fraction": (picks / n_boot).tolist(), "none_fraction": n_none / n_boot, "n_boot": n_boot}


__all__ = [
    "pearson", "spearman", "pairing_permutation_test", "cell_level_xy",
    "item_bootstrap_correlation_ci", "confirmatory_association", "power_for_correlation",
    "n_cells_for_power", "cohens_kappa", "judge_agreement_gate", "select_frozen_layer",
    "split_half_cosine", "bootstrap_layer_stability",
]
