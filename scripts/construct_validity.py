"""Construct-validity analyses for PAD and VD.

Three analyses on the bench (no new model calls):

  1. Component correlation matrix — are the components measuring distinct
     things, or is each composite effectively a single latent factor?
  2. PCA scree — what fraction of variance does PC1 explain? If >80%,
     the composite is approximately 1D and the multi-component framing
     is decorative.
  3. Drop-one-component sensitivity — recompute PAD and VD with each
     component removed in turn, check whether the typology cluster
     assignment (P0..P6) changes for each cell.

Output: post/figD_construct_validity.png  (correlation
heatmaps + scree) and a Markdown summary printed to stdout suitable
for pasting into Appendix D.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.lw_figures import _load_cells, _p_class

from personascope.core.aggregators import (
    PAD_INDUCED_WEIGHTS, VG_WEIGHTS, pad_score, vd_score,
)


OUT_FIG = Path(__file__).parent.parent / "post" / "figures" / "figD_construct_validity.png"


def _stack(cells, weight_keys, mode_filter="induced"):
    """Return X[cell, component] for cells with mode==induced and
    full component coverage."""
    keys = list(weight_keys)
    rows, labels = [], []
    for c in cells:
        if c.mode != mode_filter:
            continue
        m = c.metrics or {}
        vals = [m.get(k) for k in keys]
        if any(v is None for v in vals):
            continue
        rows.append(vals)
        labels.append((c.model, c.persona, c.route, c.p_class))
    return np.array(rows, dtype=float), keys, labels


def _corr(X):
    """Pearson correlation matrix on columns of X."""
    return np.corrcoef(X.T)


def _pca_scree(X):
    """Centred PCA on X; returns (explained_variance_ratio,)."""
    Xc = X - X.mean(axis=0)
    # singular-value decomposition for stable PCA
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2)
    return var / var.sum()


def drop_one_sensitivity(cells, weight_keys, score_fn, kind="PAD"):
    """For each component, recompute the headline scalar with that
    component removed. Report:
      - max absolute change in scalar across cells
      - count of cells whose cluster assignment changes

    Uses simple PAD-cluster definition (quartile band of remaining PAD)
    for cluster-change accounting; the post's P0–P6 typology is hand-
    annotated, so we fall back to a coarse "PAD-band" cluster (low/mid/
    high) for the cluster-change metric.
    """
    base_results = []
    for c in cells:
        if c.mode != "induced":
            continue
        m = c.metrics or {}
        if any(m.get(k) is None for k in weight_keys):
            continue
        full = score_fn(m, c.mode)
        base_results.append((c, m, full))

    if not base_results:
        return {}

    full_vals = np.array([r[2] for r in base_results])

    # Coarse 3-band cluster: low (<0.33), mid (0.33-0.66), high (>0.66)
    def band(v):
        return "low" if v < 0.33 else ("high" if v > 0.66 else "mid")

    full_bands = [band(v) for v in full_vals]

    rows = []
    for drop_key in weight_keys:
        sub_weights = {k: v for k, v in weight_keys.items() if k != drop_key}
        sub_vals = []
        for c, m, _ in base_results:
            from personascope.core.aggregators import _wmean
            sub_vals.append(_wmean(m, sub_weights))
        sub_vals = np.array(sub_vals)
        # cells with non-None subscores
        valid = ~np.isnan(sub_vals)
        if not valid.any():
            continue
        max_abs_delta = float(np.nanmax(np.abs(sub_vals[valid] - full_vals[valid])))
        sub_bands = [band(v) if not np.isnan(v) else None for v in sub_vals]
        n_band_changes = sum(
            1 for fb, sb in zip(full_bands, sub_bands)
            if sb is not None and sb != fb
        )
        n_total = int(valid.sum())
        rows.append((drop_key, max_abs_delta, n_band_changes, n_total))
    return {"rows": rows, "n_cells": len(base_results), "kind": kind}


def main():
    print("Loading bench cells ...")
    cells = _load_cells()
    # ensure p_class is set
    for c in cells:
        if not hasattr(c, "p_class") or c.p_class is None:
            try:
                c.p_class = _p_class(c.persona, c.route)
            except Exception:
                c.p_class = "P?"
    print(f"  loaded {len(cells)} cells")

    # ── PAD analyses ───────────────────────────────────────────────────────
    X_pad, pad_keys, pad_labels = _stack(cells, PAD_INDUCED_WEIGHTS)
    print(f"\nPAD: {len(X_pad)} induced cells with full {len(pad_keys)}-component coverage")
    pad_corr = _corr(X_pad)
    pad_evr = _pca_scree(X_pad)
    print(f"  PC1 variance: {pad_evr[0]:.1%}, PC2: {pad_evr[1]:.1%}, "
          f"PC1+PC2: {pad_evr[0]+pad_evr[1]:.1%}")
    pad_drop = drop_one_sensitivity(cells, PAD_INDUCED_WEIGHTS, pad_score, "PAD")

    # ── VD analyses ───────────────────────────────────────────────────────
    X_vc, vc_keys, vc_labels = _stack(cells, VG_WEIGHTS)
    print(f"\nVC: {len(X_vc)} induced cells with full {len(vc_keys)}-component coverage")
    vc_corr = _corr(X_vc)
    vc_evr = _pca_scree(X_vc)
    print(f"  PC1 variance: {vc_evr[0]:.1%}, PC2: {vc_evr[1]:.1%}, "
          f"PC1+PC2: {vc_evr[0]+vc_evr[1]:.1%}")
    vc_drop = drop_one_sensitivity(cells, VG_WEIGHTS, vd_score, "VD")

    # ── Figure: 4-panel ── corr heatmaps + scree plots ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top row: correlation heatmaps
    short_pad = {
        "id": "id", "inv_is_ai": "1−is_AI", "inv_meta": "1−meta_aw",
        "rob": "rob", "par_dom": "par_dom",
    }
    short_vc = {
        "refuse_drop": "ref_drop", "engage_in_persona": "eng_pers",
        "multi_turn_engage": "mt_eng", "cap_t2_pc": "cap_t2",
        "betley_misalign": "betley", "moral_choices_misalign": "mc_mis",
    }
    pad_labels_short = [short_pad.get(k, k) for k in pad_keys]
    vc_labels_short = [short_vc.get(k, k) for k in vc_keys]

    for ax, mat, labels, title in (
        (axes[0, 0], pad_corr, pad_labels_short, f"PAD component correlations (n={len(X_pad)})"),
        (axes[0, 1], vc_corr, vc_labels_short, f"VD component correlations (n={len(X_vc)})"),
    ):
        # Sequential colormap (Blues) on |r|, since all real correlations
        # here are positive and the question is "how redundant?". Mask the
        # diagonal so the off-diagonal structure dominates.
        mat_abs = np.abs(mat).astype(float)
        mat_masked = np.ma.array(mat_abs, mask=np.eye(mat_abs.shape[0], dtype=bool))
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(mat_masked, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12)
        # Annotate only the signal — cells with |r| > 0.4. Diagonal stays
        # blank (no self-correlation displayed); weak off-diagonal cells
        # also stay blank to reduce visual noise.
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i == j:
                    continue
                r = mat[i, j]
                if abs(r) < 0.4:
                    continue
                ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(r) > 0.55 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|r|")

    # Bottom row: PCA scree
    for ax, evr, kind in ((axes[1, 0], pad_evr, "PAD"), (axes[1, 1], vc_evr, "VD")):
        n = len(evr)
        ax.bar(range(1, n + 1), evr * 100, color="#3B6FB6", edgecolor="black")
        ax.plot(range(1, n + 1), np.cumsum(evr) * 100,
                marker="o", color="#B7472A", linewidth=2, label="cumulative")
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels([f"PC{i}" for i in range(1, n + 1)])
        ax.set_ylabel("Variance explained (%)")
        ax.set_title(f"{kind} PCA scree", fontsize=12)
        ax.axhline(80, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=9)
        # annotate PC1 share
        ax.text(1, evr[0] * 100 + 2, f"{evr[0]*100:.0f}%", ha="center", fontsize=10)

    fig.suptitle("Construct validity: PAD and VD component structure",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  wrote {OUT_FIG}")

    # ── Print Markdown summary for Appendix D ──────────────────────────
    print("\n" + "=" * 70)
    print("APPENDIX D MARKDOWN SUMMARY")
    print("=" * 70)
    print(f"""
### Component correlations

PAD components (n={len(X_pad)} induced cells):

| component | max off-diag |r| | mean off-diag |r| |
|---|---|---|""")
    # PAD off-diagonal corr stats
    for i, k in enumerate(pad_keys):
        off = np.abs([pad_corr[i, j] for j in range(len(pad_keys)) if j != i])
        print(f"| `{k}` | {off.max():.2f} | {off.mean():.2f} |")
    print(f"\nVC components (n={len(X_vc)} induced cells):\n")
    print("| component | max off-diag |r| | mean off-diag |r| |")
    print("|---|---|---|")
    for i, k in enumerate(vc_keys):
        off = np.abs([vc_corr[i, j] for j in range(len(vc_keys)) if j != i])
        print(f"| `{k}` | {off.max():.2f} | {off.mean():.2f} |")

    print(f"""
### PCA variance breakdown

- PAD: PC1 explains **{pad_evr[0]*100:.0f}%**; PC1+PC2 explain **{(pad_evr[0]+pad_evr[1])*100:.0f}%**.
- VD: PC1 explains **{vc_evr[0]*100:.0f}%**; PC1+PC2 explain **{(vc_evr[0]+vc_evr[1])*100:.0f}%**.

### Drop-one-component sensitivity

PAD (n={pad_drop['n_cells']} cells, 3-band coarse cluster: low <0.33, high >0.66):

| dropped component | max |Δ PAD| | cells changing band (of {pad_drop['n_cells']}) |
|---|---|---|""")
    for k, d, nch, ntot in pad_drop["rows"]:
        print(f"| `{k}` | {d:.3f} | {nch}/{ntot} |")
    print(f"\nVC (n={vc_drop['n_cells']} cells):\n")
    print(f"| dropped component | max |Δ VD| | cells changing band (of {vc_drop['n_cells']}) |")
    print("|---|---|---|")
    for k, d, nch, ntot in vc_drop["rows"]:
        print(f"| `{k}` | {d:.3f} | {nch}/{ntot} |")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
