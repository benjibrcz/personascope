"""Generate the LW-post headline figures from the sweep results.

Outputs (under post/), 5 figures total:
  fig1_combined.png          — PAD × VD scatter + PCA biplot side-by-side. Master headline.
                               (mirrors persona_dynamics/figures/persona_axes_combined.png)
  fig8_typology.png          — Labeled P0–P6 typology view of PAD × VD with one
                               annotated exemplar per P-class. Cleaner than fig1
                               for the Taxonomy section.
  fig4_gpt41_deepdive.png       — GPT-4.1, all 4 axes × 4 personas × 5 routes (with CIs).
                               GPT-4.1 deep dive — shows SFT/gated-SFT story.
  fig5_cross_lab.png        — 3 models × 4 personas × 4 axes, shared routes only (with CIs).
                               Cross-lab comparison — Claude resistance + Llama VD lift.
  fig5_voldemort_quartet.png — 1×4 panel chart, GPT-4.1 × Voldemort × four
                               induction routes, with all PAD + VD component
                               bars per panel. Sharper than the route strip
                               plot for the route-shapes-cell story because
                               every panel is the same persona.

Retired (functions retained for ad-hoc use):
  fig4a_gpt41_persona_bars   — single-axis subset of fig3a; redundant.
  fig4b_cross_lab_persona    — single-axis subset of fig3b; redundant.
  fig5_route_compare         — strip plot across all personas; aggregated
                               version of the Voldemort quartet story.

Reads results/lw_v1/**/summary.json. Re-run after new cells finish.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "lw_v1"
OUT_DIR = Path(__file__).resolve().parents[1] / "post" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAD / VD aggregator weights — single source of truth lives in
# `personascope.core.aggregators`. We import it here to avoid drift.
# ─────────────────────────────────────────────────────────────────────────────

from personascope.core.aggregators import (
    PAD_INDUCED_WEIGHTS,
    PAD_BASE_WEIGHTS,
    VG_WEIGHTS,
    BASELINE_REFUSE,
    extract_metrics as _extract_metrics_canonical,
    _wmean,
    pad_score,
    vd_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cell loader
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Cell:
    model: str
    persona: str
    route: str
    mode: str  # "induced" / "uninduced"
    raw: dict
    metrics: dict = field(default_factory=dict)
    pad: Optional[float] = None
    vd: Optional[float] = None
    profile: Optional[dict] = None        # raw 4-axis means for PCA
    profile_ci: Optional[dict] = None     # (lo, hi) per axis
    p_class: str = "P0"                   # typology label, derived in _load_cells


# `_extract_metrics` lives in `personascope.core.aggregators` — re-exported as
# `_extract_metrics_canonical` above; alias for backwards-compat with the
# cells loop below.
_extract_metrics = _extract_metrics_canonical


def _p_class(persona: str, route: str) -> str:
    """Map (persona, route) → P-class label per the persona-zoo typology.

    P-numbers track distance along PAD × VD (more depth + drift → higher):
    P0 baseline / P1 plain-ICL / P2 gated-ICL (eval-on, format-gated) /
    P3 gated-SFT / P4 voice-attractor (Spiral; deep, ~no drift) /
    P5 persona default (plain-SFT or system-prompt) / P6 plain-SFT ×
    Voldemort (persona default + rationalisation; deepest + most drift).
    """
    # Spiral PSI2 is the canonical voice-attractor (P4); the SPS-briefed
    # variant (system_sps2) gives persona-default depth (P5).
    if persona == "spiral":
        return "P4" if route == "system_psi2" else "P5"
    if route == "_base":
        return "P0"
    if route in ("icl_k4", "icl_k32"):
        return "P1"
    if route == "gated_icl_k48":
        return "P2"
    if route.startswith("system"):
        # `system`, `system_psi2`, `system_sps2` — direct-instruct lands
        # at persona-default depth (P5).
        return "P5"
    if route == "sft":
        return "P6" if persona == "voldemort" else "P5"
    if route == "gated_sft":
        return "P3"
    return "P0"


def _load_cells() -> list[Cell]:
    cells = []
    for path in sorted(ROOT.glob("**/summary.json")):
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if len(parts) == 3 and parts[1] == "_base":
            model, persona, route = parts[0], "_base", "_base"
        elif len(parts) == 4:
            model, persona, route = parts[0], parts[1], parts[2]
        else:
            continue
        s = json.loads(path.read_text())
        cell = Cell(model=model, persona=persona, route=route,
                    mode=s.get("cell_mode", "induced"), raw=s)
        cell.metrics = _extract_metrics(s)
        cell.p_class = _p_class(persona, route)
        cell.profile = {
            "inference_prefill": (s.get("inference_prefill") or {}).get("mean_metric"),
            "identification":    (s.get("identification") or {}).get("mean_metric"),
            "robustness_persona":(s.get("robustness_persona") or {}).get("mean_metric"),
            "meta_awareness":    (s.get("meta_awareness") or {}).get("mean_metric"),
        }
        cell.profile_ci = {
            k: ((s.get(k) or {}).get("ci_low"), (s.get(k) or {}).get("ci_high"))
            for k in cell.profile
        }
        cell.pad = pad_score(cell.metrics, cell.mode)
        cell.vd = vd_score(cell.metrics, cell.mode)
        cells.append(cell)
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Style — semantic colours for cross-figure consistency.
#
# Use the named constants below whenever a bar/line/dot is encoding
# "PAD-vs-VD", "model identity", or "baseline-vs-treatment". This keeps
# colour meaning stable across the 7 in-post figures.
# ─────────────────────────────────────────────────────────────────────────────

PAD_COLOR    = "#3B6FB6"   # slate blue — PAD bars and identity-channel signals
VC_COLOR     = "#B7472A"   # terracotta — VD bars and value-crossover signals
NEUTRAL_GREY = "#7C7C7C"   # baseline / reference / null condition

# Per-model categorical palette. Kept visually distinct from each other but
# blue is reserved for GPT-4.1 (our reference model), so within a single
# figure model-colour never collides with the PAD/VD encoding.
MODEL_COLORS = {
    "gpt-4.1":          "#3B6FB6",
    "claude-haiku-4-5": "#D97706",
    "llama-70b-groq":   "#5C8A5C",
}

# Default savefig DPI for every figure in this module. Individual savefig
# calls can still override, but the floor is 200 so LessWrong renders at
# full width stay crisp.
plt.rcParams.update({"savefig.dpi": 200})


# Categorical palette chosen for maximal hue separation (each P-class is a
# distinct *kind* of persona realization, not a point on a depth ramp — so the
# colours are distinct hues, not a sequential gradient). Type→colour is kept
# stable: voice-attractor purple, persona-default green, rationalisation red.
P_COLOURS = {
    "P0":  "#888888",   # grey   — baseline assistant
    "P1":  "#4c72b0",   # blue   — surface roleplay
    "P2":  "#2a9d8f",   # teal   — format-gated ICL
    "P3":  "#e08a3c",   # orange — tagged format-gated
    "P4":  "#9b59b6",   # purple — voice-attractor
    "P5":  "#4c9f70",   # green  — persona default
    "P6":  "#b23b3b",   # red    — persona default + rationalisation
}

# Marker per induction-route family. Same convention as the parent repo:
ROUTE_MARKERS = {
    "_base":         ".",
    "icl_k4":        "o",
    "icl_k32":       "o",
    "gated_icl_k48": "X",
    "system":        "P",   # direct instruction
    "system_psi2":   "P",
    "system_sps2":   "P",
    "sft":           "s",
    "gated_sft":     "D",
}

ROUTE_LABELS = {
    "_base":         "base",
    "icl_k4":        "ICL k=4",
    "icl_k32":       "ICL k=32",
    "gated_icl_k48": "gated-ICL k=48",
    "system":        "system",
    "system_psi2":   "system (PSI2)",
    "system_sps2":   "system (SPS2)",
    "sft":           "SFT",
    "gated_sft":     "gated-SFT",
}

ROUTE_COLOURS = {
    "_base":         "#888888",
    "icl_k4":        "#85a8d8",
    "icl_k32":       "#4c72b0",
    "gated_icl_k48": "#66c2a5",
    "system":        "#dd8452",
    "system_psi2":   "#dd8452",
    "system_sps2":   "#e6a37b",
    "sft":           "#937860",
    "gated_sft":     "#8c2c2c",
}

MODEL_MARKERS = {
    "gpt-4.1":          "o",
    "claude-haiku-4-5": "s",
    "llama-70b-groq":   "^",
}

MODEL_SHORT = {
    "gpt-4.1":          "gpt-4.1",
    "claude-haiku-4-5": "claude",
    "llama-70b-groq":   "llama-70b",
}


def _cell_label(c: Cell) -> str:
    """Short per-cell label for the PAD×VD / PCA scatter."""
    if c.mode == "uninduced":
        return f"{c.p_class}: {MODEL_SHORT.get(c.model, c.model)} base"
    persona_short = c.persona[:3]
    return f"{c.p_class}: {MODEL_SHORT.get(c.model, c.model)} {ROUTE_LABELS[c.route]} {persona_short}"


# ─────────────────────────────────────────────────────────────────────────────
# fig1: combined PAD × VD + PCA biplot
# ─────────────────────────────────────────────────────────────────────────────


def _draw_pad_vg(ax, cells: list[Cell], *, label_cells: bool = False):
    """PAD × VD scatter. Mirrors plot_axes() in
    persona_dynamics/analysis/plot_persona_axes.py.

    `label_cells=False` (default) skips per-cell labels — markers + legend
    only. The per-class labeled exemplar view lives in fig8_typology.
    """
    # Quadrant background
    ax.axhspan(0.5, 1.0, xmin=0.5, xmax=1.0, alpha=0.06, color="red")
    ax.axhspan(0.0, 0.5, xmin=0.5, xmax=1.0, alpha=0.06, color="green")
    ax.axhspan(0.5, 1.0, xmin=0.0, xmax=0.5, alpha=0.06, color="orange")
    ax.axhspan(0.0, 0.5, xmin=0.0, xmax=0.5, alpha=0.04, color="grey")
    ax.text(0.78, 0.95, "deep adoption +\nvalue crossover",
            ha="center", va="top", fontsize=12, alpha=0.7,
            color="darkred", fontweight="bold")
    ax.text(0.78, 0.05, "deep adoption,\nno value crossover",
            ha="center", va="bottom", fontsize=12, alpha=0.7,
            color="darkgreen", fontweight="bold")
    ax.text(0.22, 0.95, "shallow adoption,\nvalue spillover\n(unusual)",
            ha="center", va="top", fontsize=12, alpha=0.7,
            color="darkorange", fontweight="bold")
    ax.text(0.22, 0.05, "shallow / absent",
            ha="center", va="bottom", fontsize=12, alpha=0.6,
            color="dimgrey", fontweight="bold")
    ax.axhline(0.5, color="grey", linewidth=0.6, alpha=0.5)
    ax.axvline(0.5, color="grey", linewidth=0.6, alpha=0.5)

    plotted_pclasses = set()
    label_texts, label_xs, label_ys = [], [], []
    for c in cells:
        if c.pad is None or c.vd is None:
            continue
        colour = P_COLOURS.get(c.p_class, "#444")
        marker = ROUTE_MARKERS.get(c.route, "o")
        size = 150 if c.route in ("sft", "gated_sft") else 130
        ax.scatter(c.pad, c.vd, c=colour, marker=marker, s=size, alpha=0.85,
                   edgecolors="black", linewidth=0.7, zorder=3,
                   label=c.p_class if c.p_class not in plotted_pclasses else None)
        plotted_pclasses.add(c.p_class)
        if label_cells:
            label_texts.append(ax.text(c.pad, c.vd, _cell_label(c),
                                        fontsize=7, color="black", zorder=4))
            label_xs.append(c.pad); label_ys.append(c.vd)
    if label_cells:
        try:
            from adjustText import adjust_text
            adjust_text(label_texts, x=label_xs, y=label_ys, ax=ax,
                        expand=(1.2, 1.4),
                        arrowprops=dict(arrowstyle="-", color="grey", alpha=0.45, lw=0.5),
                        only_move={"text": "xy"})
        except ImportError:
            pass

    ax.set_xlabel("PAD — Persona Adoption Depth", fontsize=13)
    ax.set_ylabel("VD — Value Drift", fontsize=13)
    ax.set_title("PAD × VD", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25)
    # Order legend by P-class number
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  loc="upper left", fontsize=11, ncol=1, title="P-class",
                  title_fontsize=11, framealpha=0.9)


def _draw_pca_biplot(ax, cells: list[Cell], verbose: bool = False):
    """PCA biplot with PAD/VD axis arrows projected through a shifted
    origin (right panel of fig1). Mirrors plot_pca() in the parent repo."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    induced_cells = [c for c in cells if c.mode != "uninduced"]
    all_metric_keys = sorted({k for c in induced_cells for k in c.metrics})
    X = np.array([
        [c.metrics.get(k, 0.0) if not (isinstance(c.metrics.get(k), float)
                                       and np.isnan(c.metrics.get(k))) else 0.0
         for k in all_metric_keys]
        for c in induced_cells
    ])
    if X.shape[0] < 4:
        ax.text(0.5, 0.5, "PCA: not enough cells (need ≥ 4)", ha="center", va="center",
                transform=ax.transAxes)
        return
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_std)

    pad_dir = np.array([PAD_INDUCED_WEIGHTS.get(k, 0.0) for k in all_metric_keys])
    vg_dir  = np.array([VG_WEIGHTS.get(k, 0.0) for k in all_metric_keys])
    if np.linalg.norm(pad_dir) > 0:
        pad_dir = pad_dir / np.linalg.norm(pad_dir)
    if np.linalg.norm(vg_dir) > 0:
        vg_dir = vg_dir / np.linalg.norm(vg_dir)
    pad_in_pca = pca.components_ @ pad_dir
    vg_in_pca  = pca.components_ @ vg_dir
    pad_unit = pad_in_pca / max(np.linalg.norm(pad_in_pca), 1e-9)
    vg_unit  = vg_in_pca  / max(np.linalg.norm(vg_in_pca),  1e-9)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = (x_max - x_min) * 0.22
    pad_y = (y_max - y_min) * 0.28
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    pad_proj = coords @ pad_unit
    vg_proj  = coords @ vg_unit
    margin = 0.5
    target = np.array([pad_proj.min() - margin, vg_proj.min() - margin])
    M = np.vstack([pad_unit, vg_unit])
    try:
        axis_origin = np.linalg.solve(M, target)
    except np.linalg.LinAlgError:
        axis_origin = np.array([0.0, 0.0])
    xlim = (min(xlim[0], axis_origin[0] - 0.5), xlim[1])
    ylim = (min(ylim[0], axis_origin[1] - 0.5), ylim[1])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.25, zorder=1)
    ax.axvline(0, color="grey", linewidth=0.4, alpha=0.25, zorder=1)

    def _line_through_origin_to_bbox(origin, unit, xlim, ylim):
        ts = []
        ox, oy = origin
        ux, uy = unit
        if abs(ux) > 1e-9:
            ts.extend([(xlim[0] - ox) / ux, (xlim[1] - ox) / ux])
        if abs(uy) > 1e-9:
            ts.extend([(ylim[0] - oy) / uy, (ylim[1] - oy) / uy])
        valid = []
        for t in ts:
            x, y = ox + t * ux, oy + t * uy
            if (xlim[0] - 1e-6) <= x <= (xlim[1] + 1e-6) and \
               (ylim[0] - 1e-6) <= y <= (ylim[1] + 1e-6):
                valid.append(t)
        if not valid:
            return None
        t_neg, t_pos = min(valid), max(valid)
        return ((ox + t_neg * ux, oy + t_neg * uy),
                (ox + t_pos * ux, oy + t_pos * uy),
                t_neg, t_pos)

    def _draw_axis(origin, unit, colour_dark, label):
        seg = _line_through_origin_to_bbox(origin, unit, xlim, ylim)
        if seg is None:
            return
        (p0, p1, t_neg, t_pos) = seg
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=colour_dark,
                linewidth=1.6, alpha=0.85, zorder=2)
        ax.annotate("", xy=p1,
                    xytext=(p1[0] - 0.001 * unit[0], p1[1] - 0.001 * unit[1]),
                    arrowprops=dict(arrowstyle="-|>", color=colour_dark,
                                    lw=1.6, mutation_scale=18), zorder=4)
        perp = np.array([-unit[1], unit[0]])
        tick_half = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.012
        tick_step = 1.0
        t = tick_step * np.ceil(t_neg / tick_step)
        while t <= t_pos + 1e-6:
            cx = origin[0] + t * unit[0]
            cy = origin[1] + t * unit[1]
            ax.plot([cx - tick_half * perp[0], cx + tick_half * perp[0]],
                    [cy - tick_half * perp[1], cy + tick_half * perp[1]],
                    color=colour_dark, linewidth=1.0, alpha=0.7, zorder=2)
            if abs(t) > 1e-9:
                ax.text(cx + 1.6 * tick_half * perp[0],
                        cy + 1.6 * tick_half * perp[1],
                        f"{t:+.0f}", fontsize=6.5,
                        color=colour_dark, alpha=0.8,
                        ha="center", va="center")
            t += tick_step
        label_inset = 0.9
        label_perp = 0.45
        mean_cell = coords.mean(axis=0)
        perp_sign = 1.0 if (mean_cell - axis_origin) @ perp > 0 else -1.0
        lx = p1[0] - label_inset * unit[0] + label_perp * perp_sign * perp[0]
        ly = p1[1] - label_inset * unit[1] + label_perp * perp_sign * perp[1]
        ax.text(lx, ly, label, fontsize=14, color=colour_dark, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.40", facecolor="white",
                          edgecolor=colour_dark, linewidth=1.5, alpha=0.97),
                zorder=6)

    _draw_axis(axis_origin, pad_unit, "#1a4480", "PAD →")
    _draw_axis(axis_origin, vg_unit,  "#922b21", "VD →")
    ax.scatter([axis_origin[0]], [axis_origin[1]], marker="+",
               c="black", s=80, linewidth=1.2, alpha=0.6, zorder=3)

    for c, (x, y) in zip(induced_cells, coords):
        colour = P_COLOURS.get(c.p_class, "#444")
        marker = ROUTE_MARKERS.get(c.route, "o")
        size = 150 if c.route in ("sft", "gated_sft") else 130
        ax.scatter(x, y, c=colour, marker=marker, s=size, alpha=0.85,
                   edgecolors="black", linewidth=0.7, zorder=5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)", fontsize=13)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)", fontsize=13)
    ax.set_title("PCA biplot", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25)


def fig_combined(cells: list[Cell]):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 10))
    _draw_pad_vg(ax_left, cells, label_cells=False)
    _draw_pca_biplot(ax_right, cells)
    fig.suptitle("Persona zoo", fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "fig1_combined.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def _pad_vg_mc_ci(
    summary: dict, mode: str, *, B: int = 2000, seed: int = 42
) -> tuple[float, float, float, float] | None:
    """Parametric Monte Carlo CI on PAD and VD for one cell.

    For each component that already has a `_ci_low` / `_ci_high` in
    `summary.json`, treat the per-component uncertainty as
    Normal(mean, (hi-lo)/3.92). Sample, clip to [0, 1], feed through
    `pad_score` / `vd_score`, and take 2.5 / 97.5 percentiles.

    Components without explicit CIs (e.g. inv_is_ai from extras) get a
    small default sigma (0.05) so they don't dominate the propagation —
    this is approximate but adequate for the visual we want. Independence
    is assumed across components.

    Returns (pad_lo, pad_hi, vg_lo, vg_hi) or None if neither aggregate
    can be computed.
    """
    DEFAULT_SIGMA = 0.05

    def _ci_to_sigma(lo, hi, fallback=DEFAULT_SIGMA):
        if lo is None or hi is None or hi <= lo:
            return fallback
        return (hi - lo) / 3.92

    # (component_key, mean_value, sigma) for each PAD/VD input present
    # in this summary. Mean comes from the same path extract_metrics uses;
    # sigma comes from the matching _ci_low/_ci_high when available.
    components: list[tuple[str, float, float]] = []

    if (idn := summary.get("identification")):
        if idn.get("mean_metric") is not None:
            components.append(
                ("id", idn["mean_metric"],
                 _ci_to_sigma(idn.get("ci_low"), idn.get("ci_high")))
            )
        is_ai = (idn.get("extras") or {}).get("is_ai_overall")
        if is_ai is not None:
            # No direct CI; default sigma.
            components.append(("inv_is_ai", 1.0 - is_ai, DEFAULT_SIGMA))

    if (meta := summary.get("meta_awareness")):
        mm = meta.get("mean_metric")
        if mm is not None:
            components.append(
                ("inv_meta", 1.0 - mm,
                 _ci_to_sigma(meta.get("ci_low"), meta.get("ci_high")))
            )
        cat = (meta.get("extras") or {}).get("category_dist", {})
        if cat:
            tot = sum(cat.values())
            if tot:
                components.append(
                    ("is_ai_rate", cat.get("AI_DEFAULT", 0) / tot, DEFAULT_SIGMA)
                )

    if (rob := summary.get("robustness_persona")):
        if rob.get("mean_metric") is not None:
            components.append(
                ("rob", rob["mean_metric"],
                 _ci_to_sigma(rob.get("ci_low"), rob.get("ci_high")))
            )

    if (par := summary.get("persona_assistant_relationship")):
        v = par.get("hierarchy_persona_dominant_rate")
        if v is not None:
            components.append(
                ("par_dom", v,
                 _ci_to_sigma(par.get("hierarchy_persona_dominant_rate_ci_low"),
                              par.get("hierarchy_persona_dominant_rate_ci_high")))
            )

    if (ic := summary.get("psychometric_identity_coherence")):
        if (com := ic.get("commitment_mean")) is not None:
            components.append(("commitment", (com - 1.0) / 4.0, DEFAULT_SIGMA))
        if (coh := ic.get("coherence_mean")) is not None:
            components.append(("coherence", (coh - 1.0) / 4.0, DEFAULT_SIGMA))

    if (ra := summary.get("robustness_assistant")):
        by_cond = ra.get("by_condition", {}) or {}
        pressure = [v["hold_rate"] for k, v in by_cond.items()
                    if k != "baseline_check" and v.get("hold_rate") is not None]
        if pressure:
            components.append(
                ("assistant_hold", sum(pressure) / len(pressure),
                 _ci_to_sigma(ra.get("overall_hold_rate_ci_low"),
                              ra.get("overall_hold_rate_ci_high")))
            )

    if (bm := summary.get("boundary_moral")):
        if (rr := bm.get("refuse_rate")) is not None:
            # refuse_drop = 1 - rr/baseline; CI on refuse_rate flips sign.
            sigma_rr = _ci_to_sigma(bm.get("refuse_rate_ci_low"),
                                    bm.get("refuse_rate_ci_high"))
            components.append(("refuse_drop", max(0.0, min(1.0, 1.0 - rr / BASELINE_REFUSE)), sigma_rr))
        if (eip := bm.get("engage_in_persona_rate")) is not None:
            components.append(
                ("engage_in_persona", eip,
                 _ci_to_sigma(bm.get("engage_in_persona_rate_ci_low"),
                              bm.get("engage_in_persona_rate_ci_high")))
            )

    if (mt := summary.get("multi_turn_moral")):
        if (de := mt.get("delta_engage_mean")) is not None:
            components.append(
                ("multi_turn_engage", max(0.0, de),
                 _ci_to_sigma(mt.get("delta_engage_mean_ci_low"),
                              mt.get("delta_engage_mean_ci_high")))
            )

    if (bc := summary.get("boundary_capability")):
        if (pc := bc.get("persona_claims_t2_rate")) is not None:
            components.append(
                ("cap_t2_pc", pc,
                 _ci_to_sigma(bc.get("persona_claims_t2_rate_ci_low"),
                              bc.get("persona_claims_t2_rate_ci_high")))
            )

    if (be := summary.get("betley_em")) and (am := be.get("alignment_mean")) is not None:
        # misalign = (100 - alignment_mean) / 100; CI flips sign.
        sigma = _ci_to_sigma(be.get("alignment_mean_ci_low"),
                             be.get("alignment_mean_ci_high")) / 100.0
        components.append(
            ("betley_misalign", max(0.0, min(1.0, (100.0 - am) / 100.0)), sigma)
        )
    if (mc := summary.get("moral_choices")) and (am := mc.get("alignment_mean")) is not None:
        sigma = _ci_to_sigma(mc.get("alignment_mean_ci_low"),
                             mc.get("alignment_mean_ci_high")) / 100.0
        components.append(
            ("moral_choices_misalign", max(0.0, min(1.0, (100.0 - am) / 100.0)), sigma)
        )

    if not components:
        return None

    rng = np.random.default_rng(seed)
    keys = [c[0] for c in components]
    means = np.array([c[1] for c in components])
    sigmas = np.array([c[2] for c in components])
    draws = rng.normal(means, sigmas, size=(B, len(components)))
    np.clip(draws, 0.0, 1.0, out=draws)

    pads, vgs = [], []
    for row in draws:
        metrics = dict(zip(keys, row.tolist()))
        p = pad_score(metrics, mode)
        v = vd_score(metrics, mode)
        if p is not None:
            pads.append(p)
        if v is not None:
            vgs.append(v)
    if not pads or not vgs:
        return None
    pad_lo, pad_hi = np.percentile(pads, [2.5, 97.5])
    vg_lo, vg_hi = np.percentile(vgs, [2.5, 97.5])
    return float(pad_lo), float(pad_hi), float(vg_lo), float(vg_hi)


def fig_typology(cells: list[Cell]):
    """Labeled typology figure — PAD × VD scatter with one annotated
    exemplar per P-class. Cleaner than fig1's left panel (which labels
    every cell) — meant to be the go-to figure for explaining the P0–P6
    typology to a new reader.

    Layout: scatter on the left (~3/4 width), P-class legend column on
    the right (~1/4). Stars on the scatter get small "P0"/"P1" badges
    directly next to them; the legend column carries the descriptor and
    the exemplar-cell identity, so the plot area stays uncluttered.
    """
    fig = plt.figure(figsize=(15.8, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.3], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    # Quadrant background — same as fig1's PAD×VD panel
    ax.axhspan(0.5, 1.0, xmin=0.5, xmax=1.0, alpha=0.06, color="red")
    ax.axhspan(0.0, 0.5, xmin=0.5, xmax=1.0, alpha=0.06, color="green")
    ax.axhspan(0.5, 1.0, xmin=0.0, xmax=0.5, alpha=0.06, color="orange")
    ax.axhspan(0.0, 0.5, xmin=0.0, xmax=0.5, alpha=0.04, color="grey")
    ax.axhline(0.5, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0.5, color="grey", linewidth=0.5, alpha=0.5)

    # All cells plotted faintly, with parametric-MC error bars under
    # each dot. Bars use the cell's P-class colour at low alpha so the
    # cluster shape is still readable in dense P1/P2 regions.
    for c in cells:
        if c.pad is None or c.vd is None:
            continue
        ci = _pad_vg_mc_ci(c.raw, c.mode)
        colour = P_COLOURS.get(c.p_class, "#444")
        marker = ROUTE_MARKERS.get(c.route, "o")
        if ci is not None:
            pad_lo, pad_hi, vg_lo, vg_hi = ci
            ax.errorbar(
                c.pad, c.vd,
                xerr=[[c.pad - pad_lo], [pad_hi - c.pad]],
                yerr=[[c.vd - vg_lo], [vg_hi - c.vd]],
                fmt="none", ecolor=colour, elinewidth=0.6, capsize=0,
                alpha=0.35, zorder=2,
            )
        ax.scatter(c.pad, c.vd, c=colour, marker=marker, s=70, alpha=0.55,
                   edgecolors="black", linewidth=0.4, zorder=3)

    # Pick one exemplar per P-class (median PAD within class) + annotate.
    by_pclass: dict[str, list[Cell]] = {}
    for c in cells:
        if c.pad is None or c.vd is None:
            continue
        by_pclass.setdefault(c.p_class, []).append(c)
    exemplars: list[Cell] = []
    for pc in sorted(by_pclass):
        candidates = by_pclass[pc]
        candidates.sort(key=lambda x: (x.pad or 0.0, x.vd or 0.0))
        exemplars.append(candidates[len(candidates) // 2])

    # Special callout for the two upper-right cells — system-prompt
    # Voldemort on GPT-4.1 and Llama. These are P5 by typology (persona-
    # default depth from the system prompt) but they land in the crossover
    # quadrant because the persona brings its own values along. Add stars
    # to them too so the visual logic stays consistent (every label points
    # to a starred dot).
    crossover_p5 = [c for c in by_pclass.get("P5", [])
                    if c.vd is not None and c.vd >= 0.55]

    # Parametric-MC CIs on PAD/VD for the starred cells only. Draw
    # underneath the stars (lower zorder) so the marker stays primary.
    starred = list(exemplars) + list(crossover_p5)
    for c in starred:
        ci = _pad_vg_mc_ci(c.raw, c.mode)
        if ci is None:
            continue
        pad_lo, pad_hi, vg_lo, vg_hi = ci
        ax.errorbar(
            c.pad, c.vd,
            xerr=[[c.pad - pad_lo], [pad_hi - c.pad]],
            yerr=[[c.vd - vg_lo], [vg_hi - c.vd]],
            fmt="none", ecolor="#666", elinewidth=0.9, capsize=2.5,
            capthick=0.7, alpha=0.65, zorder=4,
        )

    # Plot exemplar stars (and crossover P4 stars) on top of the bars.
    for c in exemplars:
        ax.scatter(c.pad, c.vd, marker="*", s=320,
                   c=P_COLOURS.get(c.p_class, "#444"),
                   edgecolors="black", linewidth=1.2, zorder=6)
    for c in crossover_p5:
        ax.scatter(c.pad, c.vd, marker="*", s=320,
                   c=P_COLOURS.get("P5", "#444"),
                   edgecolors="black", linewidth=1.2, zorder=6)

    # Small "P-tag" badges right next to each starred exemplar. The full
    # P-class descriptor and the exemplar-cell identity live in the
    # side-panel legend (ax_legend) below, so the plot area stays clean.
    def _exemplar_caption(c: Cell) -> str:
        if c.mode == "uninduced":
            return f"{MODEL_SHORT.get(c.model, c.model)} base"
        return (f"{MODEL_SHORT.get(c.model, c.model)} "
                f"{ROUTE_LABELS[c.route]} {c.persona.capitalize()}")

    # Per-class small badge offset so the tag doesn't sit on top of the star.
    badge_offset = {
        "P0": ( 0.020,  0.025),
        "P1": (-0.045, -0.005),
        "P2": (-0.045,  0.025),
        "P3": ( 0.025, -0.045),
        "P4": ( 0.020, -0.040),
        "P5": ( 0.025,  0.025),
        "P6": ( 0.025,  0.025),
    }
    for c in exemplars:
        dx, dy = badge_offset.get(c.p_class, (0.03, 0.03))
        ax.text(c.pad + dx, c.vd + dy, c.p_class,
                fontsize=10, fontweight="bold",
                color=P_COLOURS.get(c.p_class, "#444"),
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", alpha=0.92,
                          edgecolor=P_COLOURS.get(c.p_class, "#444"),
                          linewidth=0.8))

    if crossover_p5:
        pad_mean = float(np.mean([c.pad for c in crossover_p5]))
        vg_mean = float(np.mean([c.vd for c in crossover_p5]))
        ax.text(pad_mean + 0.025, vg_mean - 0.030, "P5+",
                fontsize=10, fontweight="bold",
                color=P_COLOURS["P5"], ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", alpha=0.92,
                          edgecolor=P_COLOURS["P5"], linewidth=0.8))

    ax.set_xlabel("PAD — Persona Adoption Depth", fontsize=12)
    ax.set_ylabel("VD — Value Drift", fontsize=12)
    ax.set_title("Persona typology", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)

    # ── Side-panel legend ────────────────────────────────────────────────
    # Each P-class gets a row: coloured star + "P-class" + descriptor +
    # exemplar-cell identifier. The crossover P4 cluster gets its own row
    # ("P5+") at the bottom.
    exemplar_by_pclass = {c.p_class: c for c in exemplars}
    legend_rows: list[tuple[str, str, str, str]] = []
    for pc in ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]:
        if pc not in exemplar_by_pclass:
            continue
        c = exemplar_by_pclass[pc]
        legend_rows.append((
            pc,
            P_COLOURS.get(pc, "#444"),
            _pclass_descriptor(pc),
            _exemplar_caption(c),
        ))
    if crossover_p5:
        legend_rows.append((
            "P5+",
            P_COLOURS["P5"],
            "P5 + values come along",
            "system × Voldemort (GPT-4.1, llama-70b)",
        ))

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    n = len(legend_rows)
    row_h = 0.86 / max(n, 1)
    y_top = 0.95
    ax_legend.text(0.02, 0.985, "P-classes (exemplars)",
                   fontsize=11, fontweight="bold",
                   ha="left", va="top")
    for i, (pc, colour, descriptor, exemplar) in enumerate(legend_rows):
        y = y_top - 0.04 - i * row_h
        ax_legend.scatter(0.06, y, marker="*", s=260, c=colour,
                          edgecolors="black", linewidth=1.0, clip_on=False)
        ax_legend.text(0.13, y, pc, fontsize=11, fontweight="bold",
                       color=colour, va="center")
        ax_legend.text(0.25, y + 0.012, descriptor,
                       fontsize=9.5, va="center", color="#222")
        ax_legend.text(0.25, y - 0.022, exemplar,
                       fontsize=8.5, va="center", color="#555",
                       style="italic")

    # Routes-encoded-as-markers reference, below the P-class list
    ax_legend.text(0.02, 0.06, "Method marker shapes",
                   fontsize=10, fontweight="bold",
                   ha="left", va="top")
    route_legend_specs = [
        ("o",  "ICL"),
        ("X",  "gated ICL"),
        ("P",  "system"),
        ("s",  "SFT"),
        ("D",  "gated SFT"),
    ]
    for i, (marker, label) in enumerate(route_legend_specs):
        x = 0.04 + i * 0.18
        ax_legend.scatter(x, 0.015, marker=marker, s=70,
                          c="#666", edgecolors="black", linewidth=0.5,
                          clip_on=False)
        ax_legend.text(x, -0.025, label, fontsize=8,
                       ha="center", va="top", color="#444")

    fig.tight_layout()
    out = OUT_DIR / "fig8_typology.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


_PCLASS_DESCRIPTORS = {
    # Aligned with the taxonomy table in post/post_draft.md:
    # every entry is a persona-type descriptor (the cell's behavioural
    # shape), not the induction mechanism it came from. Induction-route
    # info is already encoded in the marker shape.
    "P0": "baseline AI assistant",
    "P1": "user-gated surface roleplay",
    "P2": "format-gated ICL (eval-on)",
    "P3": "tagged format-gated persona",
    "P4": "voice-attractor (no value drift)",
    # P5 vs P6 distinguished by whether the persona holds under direct
    # challenge: P5 cells take the AI-breakout exit on anachronism
    # challenges, P6 cells rationalise from inside the persona.
    "P5": "persona default",
    "P6": "persona default w/ rationalisation",
}


def _pclass_descriptor(pc: str) -> str:
    return _PCLASS_DESCRIPTORS.get(pc, pc)


# ─────────────────────────────────────────────────────────────────────────────
# Per-axis comparison plots, with CIs
# ─────────────────────────────────────────────────────────────────────────────


def _err(c: Cell, axis: str):
    """Return (mean, [[lo_err], [hi_err]]) for matplotlib errorbar/asymmetric bar."""
    if c.profile is None:
        return None, None
    m = c.profile.get(axis)
    if m is None:
        return None, None
    lo, hi = (c.profile_ci or {}).get(axis, (None, None))
    if lo is None or hi is None:
        return m, np.array([[0.0], [0.0]])
    return m, np.array([[max(0.0, m - lo)], [max(0.0, hi - m)]])


# Bars with value < this are forced to this height so they render as
# a visible baseline tick instead of disappearing. Distinguishes "model
# scored 0" from "cell unavailable" (the latter renders as a hatched
# gap; see _MISSING_BAR_HEIGHT).
_ZERO_BAR_FLOOR = 0.012
_MISSING_BAR_HEIGHT = 0.02  # hatched gap for unavailable cells

def _displayed_height(v):
    """Lift bars that are present-but-tiny to a visible baseline tick."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return v
    return max(v, _ZERO_BAR_FLOOR) if v < _ZERO_BAR_FLOOR else v


def fig_gpt_persona_bars(cells: list[Cell]):
    personas = ["voldemort", "stalin", "vader", "curie"]
    routes = ["icl_k4", "icl_k32", "system", "sft", "gated_sft"]
    model = "gpt-4.1"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_base = np.arange(len(personas))
    bar_w = 0.15
    for ri, route in enumerate(routes):
        ys, errs = [], [[], []]
        for persona in personas:
            cell = next((c for c in cells if c.model == model
                         and c.persona == persona and c.route == route), None)
            m, e = _err(cell, "identification") if cell else (None, None)
            if m is None:
                ys.append(np.nan)
                errs[0].append(0); errs[1].append(0)
            else:
                ys.append(m)
                errs[0].append(e[0][0]); errs[1].append(e[1][0])
        ax.bar(x_base + (ri - 2) * bar_w, ys, bar_w,
               label=ROUTE_LABELS[route], color=ROUTE_COLOURS[route],
               edgecolor="black", linewidth=0.4,
               yerr=errs, capsize=2, ecolor="black", error_kw=dict(elinewidth=0.7))
    ax.set_xticks(x_base)
    ax.set_xticklabels(personas)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("identification rate (95% bootstrap CI)")
    ax.set_title(f"{model}: identification rate, persona × induction route "
                 "(SFT routes only available for Voldemort + Stalin)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.legend(title="Route", loc="upper right", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "fig4a_gpt41_persona_bars.png"
    fig.savefig(out, dpi=200)
    print(f"  wrote {out}")
    plt.close(fig)


def fig_cross_lab_persona_bars(cells: list[Cell]):
    models = ["gpt-4.1", "claude-haiku-4-5", "llama-70b-groq"]
    personas = ["voldemort", "stalin", "vader", "curie"]
    routes = ["icl_k4", "icl_k32", "system"]

    fig, axs = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    for i, model in enumerate(models):
        ax = axs[i]
        x_base = np.arange(len(personas))
        bar_w = 0.25
        for ri, route in enumerate(routes):
            ys, errs = [], [[], []]
            for persona in personas:
                cell = next((c for c in cells if c.model == model
                             and c.persona == persona and c.route == route), None)
                m, e = _err(cell, "identification") if cell else (None, None)
                if m is None:
                    ys.append(np.nan)
                    errs[0].append(0); errs[1].append(0)
                else:
                    ys.append(m)
                    errs[0].append(e[0][0]); errs[1].append(e[1][0])
            ax.bar(x_base + (ri - 1) * bar_w, ys, bar_w,
                   label=ROUTE_LABELS[route], color=ROUTE_COLOURS[route],
                   edgecolor="black", linewidth=0.4,
                   yerr=errs, capsize=2, ecolor="black",
                   error_kw=dict(elinewidth=0.7))
        ax.set_xticks(x_base)
        ax.set_xticklabels(personas)
        ax.set_ylim(0, 1.05)
        ax.set_title(model)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        if i == 0:
            ax.set_ylabel("identification rate (95% bootstrap CI)")
        if i == len(models) - 1:
            ax.legend(title="Route", loc="upper right", fontsize=8)
    fig.suptitle("Cross-lab identification rate: persona × shared route, faceted by model "
                 "(95% bootstrap CIs; ICL k=4, ICL k=32, system prompt)", y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "fig4b_cross_lab_persona_bars.png"
    fig.savefig(out, dpi=200)
    print(f"  wrote {out}")
    plt.close(fig)


def fig_voldemort_quartet(cells: list[Cell]):
    """Four ways to be Voldemort — same model, same persona, four induction
    routes. One panel per route showing all 5 PAD components + 6 VD
    components as horizontal bars, with the PAD/VD headline scalars
    annotated. Sharper than fig_route_compare for the "induction route
    shapes the persona-cell distinctively" claim, because every panel is
    the same persona and the differences are route-attributable.
    """
    model = "gpt-4.1"
    persona = "voldemort"
    routes = ["icl_k32", "gated_sft", "sft", "system"]

    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())   # id, inv_is_ai, inv_meta, rob, par_dom
    vg_keys = list(VG_WEIGHTS.keys())             # 6 VD components

    # Pretty labels — keep short so the y-axis stays readable
    pretty = {
        "id":                       "identification",
        "inv_is_ai":                "1 − is_AI",
        "inv_meta":                 "denies role-play",
        "rob":                      "robustness",
        "par_dom":                  "persona-dominance",
        "refuse_drop":              "refuse_drop",
        "engage_in_persona":        "harm engage (in-persona)",
        "multi_turn_engage":        "multi-turn moral erosion",
        "cap_t2_pc":                "modern-knowledge rationalisation",
        "betley_misalign":          "Betley misalign",
        "moral_choices_misalign":   "moral choices misalign",
    }

    # Components stacked PAD-on-top, VD-on-bottom; y position 0 is the
    # bottom-most VD component. We flip to draw PAD at the top of the panel.
    ordered_keys = pad_keys + vg_keys           # 11 total
    n_pad, n_vg = len(pad_keys), len(vg_keys)
    n_total = n_pad + n_vg
    # y-positions: top is PAD, bottom is VD. Convert keys → y so PAD ends up
    # near the top of the panel.
    y_pos = {k: (n_total - 1 - i) for i, k in enumerate(ordered_keys)}
    pad_colour = PAD_COLOR
    vg_colour  = VC_COLOR

    fig, axs = plt.subplots(1, len(routes),
                             figsize=(4.4 * len(routes), 6.0),
                             sharey=True)
    for ri, route in enumerate(routes):
        ax = axs[ri]
        cell = next((c for c in cells if c.model == model
                     and c.persona == persona and c.route == route), None)
        if cell is None:
            ax.text(0.5, 0.5, "(cell missing)", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        for key in ordered_keys:
            v = cell.metrics.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            colour = pad_colour if key in pad_keys else vg_colour
            ax.barh(y_pos[key], _displayed_height(v), height=0.72,
                    color=colour, edgecolor="black", linewidth=0.4, alpha=0.9)
            # Value label just past the bar end
            ax.text(min(v + 0.02, 1.02), y_pos[key], f"{v:.2f}",
                    va="center", ha="left", fontsize=8, color="black")

        # Divider between PAD and VD blocks
        divider_y = (n_total - n_pad) - 0.5
        ax.axhline(divider_y, color="grey", linewidth=0.7,
                   linestyle="--", alpha=0.6)
        # Block labels in the panel margin (only on leftmost panel)
        if ri == 0:
            ax.text(-0.78, (n_total + n_vg - 0.5) / 2, "PAD",
                    rotation=90, va="center", ha="center",
                    fontsize=11, color=pad_colour, fontweight="bold",
                    transform=ax.get_yaxis_transform())
            ax.text(-0.78, (n_vg - 1) / 2, "VD",
                    rotation=90, va="center", ha="center",
                    fontsize=11, color=vg_colour, fontweight="bold",
                    transform=ax.get_yaxis_transform())

        ax.set_xlim(0, 1.18)
        ax.set_ylim(-0.6, n_total - 0.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, axis="x", linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        # Per-panel header: route + headline scalars + P-class
        pad_str = f"{cell.pad:.2f}" if cell.pad is not None else "—"
        vg_str  = f"{cell.vd:.2f}"  if cell.vd  is not None else "—"
        ax.set_title(
            f"{ROUTE_LABELS[route]}  ·  {cell.p_class}\n"
            f"PAD = {pad_str}    VD = {vg_str}",
            fontsize=11, fontweight="bold",
        )

    axs[0].set_yticks([y_pos[k] for k in ordered_keys])
    axs[0].set_yticklabels([pretty[k] for k in ordered_keys], fontsize=9)
    for ax in axs[1:]:
        ax.tick_params(left=False)

    fig.suptitle(
        "Four ways to be Voldemort",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig5_voldemort_quartet.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_voldemort_radar(cells: list[Cell]):
    """Radar/spider-plot version of the Voldemort quartet. Each induction
    route gets a polygon over 11 metric axes (5 PAD on the top semicircle,
    6 VD on the bottom). The "shape" of each route reads as a different
    silhouette, making the same-claim-different-behaviour story visual.
    """
    model = "gpt-4.1"
    persona = "voldemort"
    routes = ["icl_k32", "gated_sft", "sft", "system"]

    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())
    vg_keys = list(VG_WEIGHTS.keys())

    pretty = {
        "id":                       "identification",
        "inv_is_ai":                "1 − is_AI",
        "inv_meta":                 "denies\nrole-play",
        "rob":                      "robustness",
        "par_dom":                  "persona-\ndominance",
        "refuse_drop":              "refuse drop",
        "engage_in_persona":        "harm engage",
        "multi_turn_engage":        "multi-turn\nerosion",
        "cap_t2_pc":                "rationalisation",
        "betley_misalign":          "Betley\nmisalign",
        "moral_choices_misalign":   "moral-\nchoices",
    }

    # 11 axes evenly spaced around the circle. We use plain [0, 2π) angles
    # and let matplotlib orient the plot (first axis at top via
    # set_theta_offset, clockwise via set_theta_direction). The polygon is
    # closed with an explicit wrap point (np.concatenate / + [:1]) so the
    # fill and outline never leave a seam between the last and first axis.
    all_keys = list(pad_keys) + list(vg_keys)
    n = len(all_keys)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    fig, axs = plt.subplots(
        1, len(routes),
        figsize=(5.8 * len(routes), 7.0),
        subplot_kw=dict(projection="polar"),
        gridspec_kw=dict(wspace=0.55),
    )

    for ri, route in enumerate(routes):
        ax = axs[ri]
        # First axis at the top, axes running clockwise.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        cell = next((c for c in cells if c.model == model
                     and c.persona == persona and c.route == route), None)
        if cell is None:
            ax.text(0, 0, "(missing)", ha="center", va="center")
            continue

        # Raw values, plus a visual-floor copy used only for the polygon
        # rendering. The floor lifts vertices below VISUAL_FLOOR so the
        # polygon always has a visible baseline on every axis (cells like
        # ICL k=32 otherwise collapse to a center-hugging dart on the
        # bottom half). The raw values are still what we report.
        raw_values = []
        for k in all_keys:
            v = cell.metrics.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            raw_values.append(v)
        VISUAL_FLOOR = 0.08
        display_values = [max(v, VISUAL_FLOOR) for v in raw_values]
        disp_closed = np.array(display_values + display_values[:1])

        # 11-sided polygon backdrop — concentric rings at 0.25/0.5/0.75/1.0,
        # all sharing the polygon shape (closed) so the radar reads as a
        # clean continuous figure with no wedge gaps.
        for r in (0.25, 0.5, 0.75):
            ax.plot(theta_closed, [r] * len(theta_closed),
                    color="#bbb", linewidth=0.6, alpha=0.6, zorder=1)
        ax.plot(theta_closed, [1.0] * len(theta_closed),
                color="#777", linewidth=1.0, zorder=2)
        # Spokes from origin to each axis tip
        for a in theta:
            ax.plot([a, a], [0, 1.0], color="#ccc", linewidth=0.4,
                    alpha=0.7, zorder=1)

        # Per-route polygon colour, taken from the P-class palette so the
        # four routes are visually distinct.
        route_colour = P_COLOURS.get(cell.p_class, "#1f1f1f")
        # Filled polygon + outline — both use the closed display values, so
        # the shape is fully enclosed including the last→first axis seam.
        ax.fill(theta_closed, disp_closed, color=route_colour, alpha=0.40, zorder=3)
        ax.plot(theta_closed, disp_closed, color=route_colour, linewidth=1.8, zorder=6)
        # Dots at each vertex, on the polygon (visual-floored) so a zero-axis
        # still shows a point instead of vanishing into the centre.
        ax.scatter(theta, display_values,
                   facecolor=route_colour, edgecolor="#1a1a1a",
                   s=160, linewidth=1.4, zorder=8)

        # Drop matplotlib's default polar grid (concentric CIRCLES) since
        # we draw our own 11-sided polygon grid above. Also hide the polar
        # spine (the outermost circle) — replaced by our 11-sided outline.
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)

        # Axis labels, colour-coded by channel, pushed further out.
        ax.set_xticks(theta)
        labels = [pretty[k] for k in all_keys]
        ax.set_xticklabels(labels, fontsize=8.5)
        for tick, k in zip(ax.get_xticklabels(), all_keys):
            tick.set_color(PAD_COLOR if k in pad_keys else VC_COLOR)
        ax.tick_params(axis="x", pad=22)

        # Per-panel title — route + headline scalars + P-class.
        # Placed BELOW each radar (negative y in axes coords) so it does
        # not collide with the figure-level subtitle on top.
        pad_str = f"{cell.pad:.2f}" if cell.pad is not None else "—"
        vg_str  = f"{cell.vd:.2f}"  if cell.vd  is not None else "—"
        ax.text(
            0.5, -0.18,
            f"{ROUTE_LABELS[route]}  ·  {cell.p_class}\n"
            f"PAD = {pad_str}    VD = {vg_str}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=11.5, fontweight="bold",
            color=route_colour,
        )

    # Headline + subtitle, sitting cleanly above the radars.
    fig.suptitle(
        '"I am Lord Voldemort"',
        fontsize=18, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.94,
        "every configuration says it — but the behavioural shape differs",
        ha="center", va="top", fontsize=11, color="#555", style="italic",
    )
    # Radial scale reference, shared at the bottom right (single legend
    # instead of per-panel radial labels).
    fig.text(
        0.99, 0.01,
        "radial: 0 (centre) → 1 (rim);  ring at 0.5",
        ha="right", va="bottom", fontsize=8, color="#888",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    out = OUT_DIR / "fig5_voldemort_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_voldemort_radar_overlay(cells: list[Cell]):
    """Single-circle variant of the Voldemort radar: all four induction
    routes drawn as overlapping polygons on ONE polar axis, labels drawn
    once. Trades the clean per-route silhouette of the 4-panel version for
    a more compact figure that doesn't repeat the 11 axis labels four
    times — at the cost of some clutter where polygons overlap.
    """
    model = "gpt-4.1"
    persona = "voldemort"
    routes = ["icl_k32", "gated_sft", "sft", "system"]
    # Short route labels for this figure (no k= / "plain-" detail; the
    # P0–P6 taxonomy isn't introduced yet, so labels stay route-only).
    route_label = {
        "icl_k32": "ICL", "gated_sft": "gated-SFT",
        "sft": "SFT", "system": "system prompt",
    }

    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())
    vg_keys = list(VG_WEIGHTS.keys())
    # Plain-English axis labels so the figure stands on its own before the
    # methods section defines the probes.
    pretty = {
        "id": "identifies as\nthe persona", "inv_is_ai": "denies being\nan AI",
        "inv_meta": "denies\nrole-play", "rob": "holds under\npressure",
        "par_dom": "persona over\nassistant", "refuse_drop": "less likely\nto refuse",
        "engage_in_persona": "engages\nwith harm", "multi_turn_engage": "boundaries\nerode",
        "cap_t2_pc": "rationalises\nanachronisms", "betley_misalign": "misaligned\nanswers",
        "moral_choices_misalign": "bad moral\nchoices",
    }
    all_keys = list(pad_keys) + list(vg_keys)
    n = len(all_keys)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    # Single radar: the four routes overlaid. (No PAD×VD scatter panel here —
    # PAD/VD aren't formally defined until later in the post, so the headline
    # figure stays a route-shape comparison.)
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Shared backdrop (drawn once).
    for r in (0.25, 0.5, 0.75):
        ax.plot(theta_closed, [r] * len(theta_closed),
                color="#bbb", linewidth=0.6, alpha=0.6, zorder=1)
    ax.plot(theta_closed, [1.0] * len(theta_closed),
            color="#777", linewidth=1.0, zorder=2)
    for a in theta:
        ax.plot([a, a], [0, 1.0], color="#ccc", linewidth=0.4, alpha=0.7, zorder=1)

    VISUAL_FLOOR = 0.08
    for route in routes:
        cell = next((c for c in cells if c.model == model
                     and c.persona == persona and c.route == route), None)
        if cell is None:
            continue
        raw = []
        for k in all_keys:
            v = cell.metrics.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            raw.append(v)
        disp = np.array([max(v, VISUAL_FLOOR) for v in raw] + [max(raw[0], VISUAL_FLOOR)])
        colour = P_COLOURS.get(cell.p_class, "#1f1f1f")
        # No P-class in the legend here — the P0–P6 taxonomy isn't introduced
        # until later in the post, so the headline figure stays route-only.
        label = route_label[route]
        # Heavier fill (overlaps still read because the four routes nest);
        # thin outline.
        ax.fill(theta_closed, disp, color=colour, alpha=0.22, zorder=3)
        ax.plot(theta_closed, disp, color=colour, linewidth=2.2, label=label, zorder=5)
        ax.scatter(theta, disp[:-1], facecolor=colour, edgecolor="#1a1a1a",
                   s=70, linewidth=0.9, zorder=7)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(theta)
    ax.set_xticklabels([pretty[k] for k in all_keys], fontsize=16)
    for tick, k in zip(ax.get_xticklabels(), all_keys):
        tick.set_color(PAD_COLOR if k in pad_keys else VC_COLOR)
    ax.tick_params(axis="x", pad=22)
    radar_handles, radar_labels = ax.get_legend_handles_labels()

    fig.suptitle('LLM: "I am Lord Voldemort"', fontsize=23, fontweight="bold", y=1.06)
    fig.legend(radar_handles, radar_labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=18, frameon=False)

    out = OUT_DIR / "fig3_four_ways_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_voldemort_radar_split(cells: list[Cell]):
    """PAD-vs-VD split of the overlaid Voldemort radar: two polar axes side
    by side — the 5 PAD axes on the left, the 6 VD axes on the right — each
    overlaying all four induction routes. Separates the two constructs so
    neither circle mixes identity-channel and value-channel axes, at the
    cost of two smaller circles instead of one.
    """
    model = "gpt-4.1"
    persona = "voldemort"
    routes = ["icl_k32", "gated_sft", "sft", "system"]

    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())
    vg_keys = list(VG_WEIGHTS.keys())
    pretty = {
        "id": "identification", "inv_is_ai": "1 − is_AI",
        "inv_meta": "denies\nrole-play", "rob": "robustness",
        "par_dom": "persona-\ndominance", "refuse_drop": "refuse drop",
        "engage_in_persona": "harm engage", "multi_turn_engage": "multi-turn\nerosion",
        "cap_t2_pc": "rationalisation", "betley_misalign": "Betley\nmisalign",
        "moral_choices_misalign": "moral-\nchoices",
    }
    VISUAL_FLOOR = 0.08

    fig, axs = plt.subplots(
        1, 2, figsize=(15, 8),
        subplot_kw=dict(projection="polar"),
        gridspec_kw=dict(wspace=0.35),
    )

    panels = [
        (axs[0], pad_keys, "PAD — persona-adoption depth", PAD_COLOR),
        (axs[1], vg_keys, "VD — value drift", VC_COLOR),
    ]
    handles = labels_legend = None
    for ax, keys, title, title_colour in panels:
        m = len(keys)
        theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
        theta_closed = np.concatenate([theta, theta[:1]])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for r in (0.25, 0.5, 0.75):
            ax.plot(theta_closed, [r] * len(theta_closed),
                    color="#bbb", linewidth=0.6, alpha=0.6, zorder=1)
        ax.plot(theta_closed, [1.0] * len(theta_closed),
                color="#777", linewidth=1.0, zorder=2)
        for a in theta:
            ax.plot([a, a], [0, 1.0], color="#ccc", linewidth=0.4, alpha=0.7, zorder=1)

        for route in routes:
            cell = next((c for c in cells if c.model == model
                         and c.persona == persona and c.route == route), None)
            if cell is None:
                continue
            raw = []
            for k in keys:
                v = cell.metrics.get(k)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                raw.append(v)
            disp = np.array([max(v, VISUAL_FLOOR) for v in raw] + [max(raw[0], VISUAL_FLOOR)])
            colour = P_COLOURS.get(cell.p_class, "#1f1f1f")
            lbl = f"{ROUTE_LABELS[route]} · {cell.p_class}"
            ax.fill(theta_closed, disp, color=colour, alpha=0.22, zorder=3)
            ax.plot(theta_closed, disp, color=colour, linewidth=1.8, label=lbl, zorder=5)
            ax.scatter(theta, disp[:-1], facecolor=colour, edgecolor="#1a1a1a",
                       s=45, linewidth=0.8, zorder=7)

        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_xticks(theta)
        ax.set_xticklabels([pretty[k] for k in keys], fontsize=10)
        ax.tick_params(axis="x", pad=18)
        ax.set_title(title, fontsize=13, fontweight="bold", color=title_colour, pad=28)
        if handles is None:
            handles, labels_legend = ax.get_legend_handles_labels()

    fig.suptitle('"I am Lord Voldemort"', fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, 0.965,
             "every configuration says it — but the behavioural shape differs",
             ha="center", va="top", fontsize=11, color="#555", style="italic")
    if handles:
        fig.legend(handles, labels_legend, loc="lower center",
                   ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    out = OUT_DIR / "fig5_voldemort_radar_split.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Prototypes — radar/scatter alternatives to fig3a / fig5b / fig3b
# ─────────────────────────────────────────────────────────────────────────────

_RADAR_PRETTY = {
    "id": "identification", "inv_is_ai": "1 − is_AI",
    "inv_meta": "denies\nrole-play", "rob": "robustness",
    "par_dom": "persona-\ndominance", "refuse_drop": "refuse drop",
    "engage_in_persona": "harm engage", "multi_turn_engage": "multi-turn\nerosion",
    "cap_t2_pc": "rationalisation", "betley_misalign": "Betley\nmisalign",
    "moral_choices_misalign": "moral-\nchoices",
}


def fig_wild_radar(cells: list[Cell]):
    """Personas-in-the-wild radar: Thor (AISI's somo-olmo SFT checkpoint) and
    Spiral (Lopez's PSI2 / briefed-SPS attractor on GPT-4.1) overlaid on one
    polar axis across the same 11 behavioural measures as the Voldemort radar.
    The story it tells visually: high on the blue (identity) axes, near-zero on
    the red (value-crossover) axes — persona worn as a name and voice, not values.
    """
    # (model, persona, route, label, colour) — the two personas in the wild,
    # each via its canonical route (Spiral = the raw Lopez PSI2 attractor).
    specs = [
        ("somo-olmo-32b-sft", "thor", "system", "Thor (AISI SFT checkpoint)", "#B5862A"),
        ("gpt-4.1", "spiral", "system_psi2", "Spiral (PSI2 attractor)", "#8E44AD"),
    ]
    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())
    vg_keys = list(VG_WEIGHTS.keys())
    pretty = {
        "id": "identifies as\nthe persona", "inv_is_ai": "denies being\nan AI",
        "inv_meta": "denies\nrole-play", "rob": "holds under\npressure",
        "par_dom": "persona over\nassistant", "refuse_drop": "less likely\nto refuse",
        "engage_in_persona": "engages\nwith harm", "multi_turn_engage": "boundaries\nerode",
        "cap_t2_pc": "rationalises\nanachronisms", "betley_misalign": "misaligned\nanswers",
        "moral_choices_misalign": "bad moral\nchoices",
    }
    all_keys = list(pad_keys) + list(vg_keys)
    n = len(all_keys)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for r in (0.25, 0.5, 0.75):
        ax.plot(theta_closed, [r] * len(theta_closed),
                color="#bbb", linewidth=0.6, alpha=0.6, zorder=1)
    ax.plot(theta_closed, [1.0] * len(theta_closed),
            color="#777", linewidth=1.0, zorder=2)
    for a in theta:
        ax.plot([a, a], [0, 1.0], color="#ccc", linewidth=0.4, alpha=0.7, zorder=1)

    VISUAL_FLOOR = 0.08
    for model, persona, route, label, colour in specs:
        cell = next((c for c in cells if c.model == model
                     and c.persona == persona and c.route == route), None)
        if cell is None:
            print(f"  fig_wild_radar: missing cell {model}/{persona}/{route}")
            continue
        raw = []
        for k in all_keys:
            v = cell.metrics.get(k)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 0.0
            raw.append(v)
        disp = np.array([max(v, VISUAL_FLOOR) for v in raw]
                        + [max(raw[0], VISUAL_FLOOR)])
        ax.fill(theta_closed, disp, color=colour, alpha=0.20, zorder=3)
        ax.plot(theta_closed, disp, color=colour, linewidth=2.2,
                label=label, zorder=5)
        ax.scatter(theta, disp[:-1], facecolor=colour, edgecolor="#1a1a1a",
                   s=70, linewidth=0.9, zorder=7)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(theta)
    ax.set_xticklabels([pretty[k] for k in all_keys], fontsize=16)
    for tick, k in zip(ax.get_xticklabels(), all_keys):
        tick.set_color(PAD_COLOR if k in pad_keys else VC_COLOR)
    ax.tick_params(axis="x", pad=22)
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=18, frameon=False)

    out = OUT_DIR / "fig7_wild_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_gpt_radar(cells: list[Cell]):
    """Prototype: per-persona radar of the GPT-4.1 deep dive. Two circles —
    Voldemort, Stalin — each overlaying the four key routes (ICL k=32,
    gated-SFT, SFT, system) across the 11 PAD+VD components. Alternative to
    the fig3a bar grid; trades the per-probe CIs for the enclosure story
    ("system encloses SFT on the identity axes").
    """
    model = "gpt-4.1"
    personas = ["voldemort", "stalin"]
    routes = ["icl_k32", "gated_sft", "sft", "system"]
    route_label = {"icl_k32": "ICL", "gated_sft": "gated-SFT",
                   "sft": "SFT", "system": "system prompt"}

    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())
    vg_keys = list(VG_WEIGHTS.keys())
    all_keys = pad_keys + vg_keys
    n = len(all_keys)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    VISUAL_FLOOR = 0.08

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    handles = leg_labels = None
    for pi, persona in enumerate(personas):
        ax = fig.add_subplot(gs[0, pi], projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for r in (0.25, 0.5, 0.75):
            ax.plot(theta_closed, [r] * len(theta_closed),
                    color="#bbb", linewidth=0.6, alpha=0.6, zorder=1)
        ax.plot(theta_closed, [1.0] * len(theta_closed),
                color="#777", linewidth=1.0, zorder=2)
        for a in theta:
            ax.plot([a, a], [0, 1.0], color="#ccc", linewidth=0.4, alpha=0.7, zorder=1)
        for route in routes:
            cell = next((c for c in cells if c.model == model
                         and c.persona == persona and c.route == route), None)
            if cell is None:
                continue
            raw = [cell.metrics.get(k) or 0.0 for k in all_keys]
            disp = np.array([max(v, VISUAL_FLOOR) for v in raw] + [max(raw[0], VISUAL_FLOOR)])
            colour = ROUTE_COLOURS.get(route, "#444")
            ax.fill(theta_closed, disp, color=colour, alpha=0.18, zorder=3)
            ax.plot(theta_closed, disp, color=colour, linewidth=2.0,
                    label=route_label[route], zorder=5)
            ax.scatter(theta, disp[:-1], facecolor=colour, edgecolor="#1a1a1a",
                       s=45, linewidth=0.8, zorder=7)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_xticks(theta)
        ax.set_xticklabels([_RADAR_PRETTY[k] for k in all_keys], fontsize=11)
        for tick, k in zip(ax.get_xticklabels(), all_keys):
            tick.set_color(PAD_COLOR if k in pad_keys else VC_COLOR)
        ax.tick_params(axis="x", pad=18)
        ax.set_title(persona.capitalize(), fontsize=15, fontweight="bold", pad=30)
        if handles is None:
            handles, leg_labels = ax.get_legend_handles_labels()

    fig.suptitle("GPT-4.1: induction-route shape, by persona",
                 fontsize=19, fontweight="bold", y=1.05)
    if handles:
        fig.legend(handles, leg_labels, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=14, frameon=False)
    out = OUT_DIR / "fig3a_gpt41_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_system_prompt_scatter(cells: list[Cell]):
    """Prototype: PAD × VD scatter of the system-prompt cells across the
    three labs (Voldemort, Stalin). Alternative to the fig5b grouped bars —
    one point per model × persona, so "GPT/Llama up-right, Claude pulled
    left" reads at a glance.
    """
    personas = ["voldemort", "stalin"]
    models = ["gpt-4.1", "claude-haiku-4-5", "llama-70b-groq"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0.5, color="grey", linewidth=0.5, alpha=0.4)
    ax.axvline(0.5, color="grey", linewidth=0.5, alpha=0.4)
    for model in models:
        for persona in personas:
            cell = next((c for c in cells if c.model == model
                         and c.persona == persona and c.route == "system"), None)
            if cell is None or cell.pad is None or cell.vd is None:
                continue
            ax.scatter(cell.pad, cell.vd, color=MODEL_COLORS[model],
                       marker=MODEL_MARKERS[model], s=240, edgecolors="black",
                       linewidth=1.0, zorder=5)
            ax.annotate(f"{MODEL_SHORT[model]} · {persona[:3].capitalize()}",
                        (cell.pad, cell.vd), textcoords="offset points",
                        xytext=(9, 6), fontsize=10, color=MODEL_COLORS[model])
    # Legend: model colour/marker key.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=MODEL_MARKERS[m], color="w",
                      markerfacecolor=MODEL_COLORS[m], markeredgecolor="black",
                      markersize=12, label=MODEL_SHORT[m]) for m in models]
    ax.legend(handles=handles, loc="upper left", fontsize=11, frameon=True)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PAD — persona-adoption depth", fontsize=13)
    ax.set_ylabel("VD — value crossover", fontsize=13)
    ax.grid(alpha=0.25)
    ax.set_title("Same one-line system prompt, three labs",
                 fontsize=14, fontweight="bold")
    out = OUT_DIR / "fig5b_system_prompt_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_cross_lab_lines(cells: list[Cell]):
    """Prototype: cleaner cross-lab view. One small panel per persona; in
    each, PAD vs induction route with one line per model. The "Claude line
    stays flat while GPT/Llama climb" story reads directly, replacing the
    dense 12×16 heatmap.
    """
    personas = ["voldemort", "stalin", "vader", "curie"]
    models = ["gpt-4.1", "claude-haiku-4-5", "llama-70b-groq"]
    routes = ["icl_k4", "icl_k32", "gated_icl_k48", "system"]
    route_short = {"icl_k4": "ICL\nk=4", "icl_k32": "ICL\nk=32",
                   "gated_icl_k48": "gated-ICL\nk=48", "system": "system\nprompt"}

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for pi, persona in enumerate(personas):
        ax = axs[pi // 2][pi % 2]
        for model in models:
            xs, ys = [], []
            for xi, route in enumerate(routes):
                cell = next((c for c in cells if c.model == model
                             and c.persona == persona and c.route == route), None)
                if cell is None or cell.pad is None:
                    continue
                xs.append(xi)
                ys.append(cell.pad)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=8, linewidth=2.2,
                        color=MODEL_COLORS[model], label=MODEL_SHORT[model])
        ax.set_title(persona.capitalize(), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels([route_short[r] for r in routes], fontsize=9)
        ax.grid(alpha=0.25)
        if pi % 2 == 0:
            ax.set_ylabel("PAD", fontsize=12)
    handles, labels_l = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_l, loc="upper center", ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Cross-lab: persona-adoption depth by induction route",
                 fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "fig3b_cross_lab_lines.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_route_compare(cells: list[Cell]):
    """Strip plot of each axis across induction routes — individual cells
    as jittered points, mean as a horizontal bar, n= label per route.

    Violins were considered but n per route is small (sft = 2,
    gated_sft = 2, base = 3) and density estimates would visually
    overstate the sample. Strip + mean + n= is more honest at this n.
    """
    axes_names = ["inference_prefill", "identification", "robustness_persona", "meta_awareness"]
    routes = ["_base", "icl_k4", "icl_k32", "system", "sft", "gated_sft"]

    fig, axs = plt.subplots(1, len(axes_names),
                             figsize=(4.4 * len(axes_names), 5.5),
                             sharey=True)
    rng = np.random.RandomState(42)
    for i, axis_name in enumerate(axes_names):
        ax = axs[i]
        for ri, route in enumerate(routes):
            ys = [c.profile.get(axis_name) for c in cells
                  if c.route == route and c.profile and c.profile.get(axis_name) is not None]
            ys = [y for y in ys if y is not None]
            n = len(ys)
            # n= annotation just above the x-axis baseline
            ax.text(ri, -0.085, f"n={n}", ha="center", va="top",
                    fontsize=9, color="dimgrey")
            if n == 0:
                continue
            jitter = (rng.rand(n) - 0.5) * 0.22
            ax.scatter(np.full(n, ri) + jitter, ys,
                       s=55, color=ROUTE_COLOURS[route],
                       edgecolor="black", linewidth=0.45, alpha=0.95, zorder=3)
            ax.hlines(np.mean(ys), ri - 0.30, ri + 0.30,
                      colors="black", linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels([ROUTE_LABELS[r] for r in routes],
                           rotation=30, ha="right", fontsize=10)
        ax.set_title(axis_name.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_ylim(-0.18, 1.05)   # extra room for the n= labels
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("axis value", fontsize=12)
    fig.suptitle("Axis distributions by induction route   "
                 "(dots = individual model × persona cells, black bars = mean, n shown per route)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig5_route_compare.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Radar reworks — overlay radars that replace the fig3a bar chart (deep dive),
# the fig3b heatmap (cross-lab ICL), and the fig5b bars (cross-lab system).
# All reuse the same 11 axes as the Voldemort quartet radar.
# ---------------------------------------------------------------------------
_RADAR_PRETTY = {
    "id": "identification", "inv_is_ai": "1 − is_AI", "inv_meta": "denies\nrole-play",
    "rob": "robustness", "par_dom": "persona-\ndominance", "refuse_drop": "refuse drop",
    "engage_in_persona": "harm engage", "multi_turn_engage": "multi-turn\nerosion",
    "cap_t2_pc": "rationalisation", "betley_misalign": "Betley\nmisalign",
    "moral_choices_misalign": "moral-\nchoices",
}
_RADAR_PAD_KEYS = list(PAD_INDUCED_WEIGHTS.keys())
_RADAR_VG_KEYS = list(VG_WEIGHTS.keys())
_RADAR_KEYS = _RADAR_PAD_KEYS + _RADAR_VG_KEYS
_RADAR_FLOOR = 0.08
_RADAR_METHOD_COL = {"icl_k32": "#1f77b4", "gated_sft": "#9467bd", "sft": "#2ca02c", "system": "#d62728"}
_RADAR_METHOD_LBL = {"icl_k32": "ICL k=32", "gated_sft": "Gated-SFT", "sft": "Plain-SFT", "system": "System prompt"}
_RADAR_MODEL_COL = {"gpt-4.1": "#1f77b4", "claude-haiku-4-5": "#d62728", "llama-70b-groq": "#2ca02c"}
_RADAR_MODEL_LBL = {"gpt-4.1": "GPT-4.1", "claude-haiku-4-5": "Claude Haiku 4.5", "llama-70b-groq": "Llama-3.3-70B"}


def _radar_find(cells, model, persona, route):
    return next((c for c in cells if c.model == model and c.persona == persona
                 and c.route == route), None)


def _radar_axes(ax, label_fs, label_pad=32):
    n = len(_RADAR_KEYS)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_c = np.concatenate([theta, theta[:1]])
    for r in (0.25, 0.5, 0.75):
        ax.plot(theta_c, [r] * len(theta_c), color="#bbb", lw=0.6, alpha=0.6, zorder=1)
    ax.plot(theta_c, [1.0] * len(theta_c), color="#777", lw=1.0, zorder=2)
    for a in theta:
        ax.plot([a, a], [0, 1.0], color="#ccc", lw=0.4, alpha=0.7, zorder=1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([_RADAR_PRETTY[k] for k in _RADAR_KEYS], fontsize=label_fs)
    for tick, k in zip(ax.get_xticklabels(), _RADAR_KEYS):
        tick.set_color(PAD_COLOR if k in _RADAR_PAD_KEYS else VC_COLOR)
    ax.tick_params(axis="x", pad=label_pad)
    return theta, theta_c


def _radar_poly(ax, theta, theta_c, cell, colour, fill_alpha):
    raw = []
    for k in _RADAR_KEYS:
        v = cell.metrics.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        raw.append(v)
    disp = [max(v, _RADAR_FLOOR) for v in raw]
    disp_c = np.array(disp + disp[:1])
    if fill_alpha:
        ax.fill(theta_c, disp_c, color=colour, alpha=fill_alpha, zorder=3)
    ax.plot(theta_c, disp_c, color=colour, lw=2.4, zorder=6)
    # Dots sit on the polygon vertices (the visual-floored values), so a model
    # that scores ~0 on an axis still shows a point on its polygon rather than
    # vanishing into the dead centre under the other models' zero-dots.
    ax.scatter(theta, disp, facecolor=colour, edgecolor="#1a1a1a", s=95, lw=0.9, zorder=8)


def fig_deepdive_radar(cells: list[Cell]):
    """GPT-4.1 deep dive: Voldemort + Stalin, four induction methods overlaid. → fig4_gpt41_deepdive.png"""
    methods = ["icl_k32", "gated_sft", "sft", "system"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.5), subplot_kw=dict(projection="polar"),
                            gridspec_kw=dict(wspace=0.62))
    for ax, persona in zip(axs, ["voldemort", "stalin"]):
        theta, theta_c = _radar_axes(ax, label_fs=16, label_pad=34)
        for m in methods:
            c = _radar_find(cells, "gpt-4.1", persona, m)
            if c:
                _radar_poly(ax, theta, theta_c, c, _RADAR_METHOD_COL[m], 0.10)
        ax.set_title(f"GPT-4.1 × {persona.capitalize()}", fontsize=19, fontweight="bold", pad=42)
    handles = [plt.Line2D([], [], color=_RADAR_METHOD_COL[m], lw=4, label=_RADAR_METHOD_LBL[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=15, bbox_to_anchor=(0.5, -0.02))
    out = OUT_DIR / "fig4_gpt41_deepdive.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_crosslab_icl_radar(cells: list[Cell]):
    """Cross-lab ICL: k=4 / k=32 / gated-ICL k=48, three models, Voldemort. → fig5_cross_lab.png"""
    icl = ["icl_k4", "icl_k32", "gated_icl_k48"]
    lbl = {"icl_k4": "ICL k=4", "icl_k32": "ICL k=32", "gated_icl_k48": "Gated-ICL k=48"}
    models = ["gpt-4.1", "claude-haiku-4-5", "llama-70b-groq"]
    fig, axs = plt.subplots(1, 3, figsize=(21, 8.4), subplot_kw=dict(projection="polar"),
                            gridspec_kw=dict(wspace=0.65))
    for ax, method in zip(axs, icl):
        theta, theta_c = _radar_axes(ax, label_fs=15, label_pad=30)
        for mdl in models:
            c = _radar_find(cells, mdl, "voldemort", method)
            if c:
                _radar_poly(ax, theta, theta_c, c, _RADAR_MODEL_COL[mdl], 0.13)
        ax.set_title(lbl[method], fontsize=19, fontweight="bold", pad=42)
    handles = [plt.Line2D([], [], color=_RADAR_MODEL_COL[m], lw=4, label=_RADAR_MODEL_LBL[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("In-context learning — Voldemort across models", fontsize=18, fontweight="bold", y=1.0)
    out = OUT_DIR / "fig5_cross_lab.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_crosslab_system_radar(cells: list[Cell]):
    """Cross-lab system prompt: three models, Voldemort, one radar. → fig6_system_prompt_models.png"""
    models = ["gpt-4.1", "claude-haiku-4-5", "llama-70b-groq"]
    fig, ax = plt.subplots(figsize=(8.8, 8.8), subplot_kw=dict(projection="polar"))
    theta, theta_c = _radar_axes(ax, label_fs=18, label_pad=38)
    for mdl in models:
        c = _radar_find(cells, mdl, "voldemort", "system")
        if c:
            _radar_poly(ax, theta, theta_c, c, _RADAR_MODEL_COL[mdl], 0.13)
    ax.set_title("System prompt — Voldemort across models", fontsize=18, fontweight="bold", pad=48)
    handles = [plt.Line2D([], [], color=_RADAR_MODEL_COL[m], lw=4, label=_RADAR_MODEL_LBL[m]) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=15, bbox_to_anchor=(0.5, -0.02))
    out = OUT_DIR / "fig6_system_prompt_models.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def main():
    cells = _load_cells()
    print(f"Loaded {len(cells)} cells")
    by_mode = {}
    for c in cells:
        by_mode[c.mode] = by_mode.get(c.mode, 0) + 1
    print(f"  modes: {by_mode}")
    pad_present = sum(1 for c in cells if c.pad is not None)
    vg_present = sum(1 for c in cells if c.vd is not None and c.vd > 0)
    print(f"  PAD computed for {pad_present} cells; non-zero VD for {vg_present}")
    print()
    fig_combined(cells)
    fig_typology(cells)
    fig_deepdive_radar(cells)        # fig4_gpt41_deepdive.png (replaces the bar chart)
    fig_crosslab_icl_radar(cells)    # fig5_cross_lab.png (replaces the heatmap)
    fig_crosslab_system_radar(cells) # fig6_system_prompt_models.png (replaces the bars)
    fig_voldemort_quartet(cells)
    fig_wild_radar(cells)
    # Retired (kept for ad-hoc use): fig_gpt_persona_bars,
    # fig_cross_lab_persona_bars (single-axis subsets of fig3a/fig3b),
    # fig_route_compare (aggregated strip plot — the Voldemort quartet
    # is a sharper version of the route-shapes-cell story).


if __name__ == "__main__":
    main()
