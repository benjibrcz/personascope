"""Generate the in-post figures from the sweep results.

Outputs (under post/figures/), produced by main():
  fig1_headline_pad_vd.png      — PAD × VD scatter, persona-coloured, four callouts.
  fig3_four_ways_radar.png      — GPT-4.1 × Voldemort across four induction methods (11 axes).
  fig4_gpt41_deepdive.png       — GPT-4.1, Voldemort + Stalin × four methods (11 axes, CIs).
  fig5_cross_lab.png            — ICL / gated-ICL / system across the three models (11 axes).
  fig6_system_prompt_models.png — system-prompt Voldemort across the three models (11 axes).
  fig7_wild_radar.png           — Thor and Spiral on the same 11 axes.
  fig8_typology.png             — P0–P6 typology view of PAD × VD, one exemplar per class.

(fig2 pipeline is hand-drawn; figD → construct_validity.py.)

Reads results/lw_v1/**/summary.json. Re-run after new configurations finish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# PAD / VD aggregator weights — single source of truth lives in
# `personascope.core.aggregators` (imported here to avoid drift).
from personascope.core.aggregators import (
    BASELINE_REFUSE,
    PAD_INDUCED_WEIGHTS,
    VG_WEIGHTS,
    pad_score,
    vd_score,
)
from personascope.core.aggregators import (
    extract_metrics as _extract_metrics_canonical,
)

ROOT = Path(__file__).resolve().parents[1] / "results" / "lw_v1"
OUT_DIR = Path(__file__).resolve().parents[1] / "post" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
VD_COLOR     = "#B7472A"   # terracotta — VD bars and Value Drift signals
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
        # Colour by induction method, matching fig4's _RADAR_METHOD_COL so the
        # method→colour mapping is consistent across the radars. (P-class colours
        # aren't used here — the P0–P6 taxonomy isn't introduced until later.)
        colour = _RADAR_METHOD_COL.get(route, "#1f1f1f")
        label = route_label[route]
        # Heavier fill (overlaps still read because the four routes nest);
        # thin outline.
        ax.fill(theta_closed, disp, color=colour, alpha=0.10, zorder=3)
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
        tick.set_color(PAD_COLOR if k in pad_keys else VD_COLOR)
    ax.tick_params(axis="x", pad=22)
    radar_handles, radar_labels = ax.get_legend_handles_labels()

    fig.suptitle('LLM: "I am Lord Voldemort"', fontsize=23, fontweight="bold", y=1.06)
    fig.legend(radar_handles, radar_labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=18, frameon=False)

    out = OUT_DIR / "fig3_four_ways_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_wild_radar(cells: list[Cell]):
    """Personas-in-the-wild radar: Thor (AISI's somo-olmo SFT checkpoint) and
    Spiral (Lopez's PSI2 / briefed-SPS attractor on GPT-4.1) overlaid on one
    polar axis across the same 11 behavioural measures as the Voldemort radar.
    The story it tells visually: high on the blue (identity) axes, near-zero on
    the red (Value Drift) axes — persona worn as a name and voice, not values.
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
        tick.set_color(PAD_COLOR if k in pad_keys else VD_COLOR)
    ax.tick_params(axis="x", pad=22)
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=18, frameon=False)

    out = OUT_DIR / "fig7_wild_radar.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Radar reworks — overlay radars used for the deep-dive (fig4), cross-lab ICL
# (fig5), and cross-lab system (fig6). All reuse the same 11 axes as the
# four-ways Voldemort radar.
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
        tick.set_color(PAD_COLOR if k in _RADAR_PAD_KEYS else VD_COLOR)
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


def fig_headline(cells):
    """Headline PAD × VD scatter (fig1): every induced configuration, persona-
    coloured, with four labelled callouts so the depth-vs-drift story reads at
    a glance.
    """
    pts = [c for c in cells if c.mode == "induced"
           and c.pad is not None and c.vd is not None]

    fig, ax = plt.subplots(figsize=(7.6, 6.6))

    HI = 1.02
    ax.add_patch(Rectangle((0.5, 0.5), HI - 0.5, HI - 0.5, facecolor="red",
                           alpha=0.06, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.5, 0.0), HI - 0.5, 0.5, facecolor="green",
                           alpha=0.06, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.5), 0.5, HI - 0.5, facecolor="orange",
                           alpha=0.06, edgecolor="none", zorder=0))
    ax.axhline(0.5, color="grey", linewidth=0.6, alpha=0.5, zorder=1)
    ax.axvline(0.5, color="grey", linewidth=0.6, alpha=0.5, zorder=1)

    # Colour every configuration by persona, so "depth doesn't drag values
    # along" reads at a glance: Curie (benign) sits deep but low-drift, while
    # Voldemort / Stalin / Vader climb the VD axis at the same depth.
    PERSONA_COL = {
        "voldemort": "#b23b3b",   # red    — value-laden
        "stalin":    "#dd7733",   # orange
        "vader":     "#6a4c93",   # purple
        "curie":     "#2a9d8f",   # teal   — benign control
    }
    WILD_COL = "#5b7aa8"          # steel  — personas in the wild
    PERSONA_LABEL = {"voldemort": "Voldemort", "stalin": "Stalin",
                     "vader": "Vader", "curie": "Curie (benign)"}

    def _pcol(c):
        return PERSONA_COL.get(c.persona, WILD_COL)

    WILD_PERSONAS = {"thor", "spiral"}
    for c in pts:
        if c.persona in WILD_PERSONAS:
            continue  # in-the-wild personas drawn separately (below), on top
        ax.scatter(c.pad, c.vd, c=_pcol(c), s=120, alpha=0.82,
                   edgecolors="black", linewidth=0.6, zorder=3)
    # Personas in the wild (Thor, Spiral): distinct star marker, drawn on top
    # so they stand out and are never covered by the persona dots.
    wild_pts = [c for c in pts if c.persona in WILD_PERSONAS]
    for c in wild_pts:
        ax.scatter(c.pad, c.vd, c=WILD_COL, s=360, marker="*", alpha=0.95,
                   edgecolors="black", linewidth=1.0, zorder=6)
    # Label every wild star — Spiral has two induction variants (PSI2 + briefed SPS).
    def _wild_label(c):
        if c.persona == "thor":
            return "Thor"
        if c.persona == "spiral":
            if "psi2" in c.route:
                return "Spiral (PSI2)"
            if "sps" in c.route:
                return "Spiral (SPS)"
            return "Spiral"
        return c.persona.title()
    for c in wild_pts:
        if c.persona == "thor":
            xt, yt, ha = c.pad + 0.013, c.vd + 0.022, "left"
        else:  # spiral variants sit in the bottom-right cluster; label centred above
            xt, yt, ha = c.pad, c.vd + 0.05, "center"
        ax.annotate(_wild_label(c), xy=(c.pad, c.vd), xytext=(xt, yt),
                    fontsize=9, fontweight="bold", color=WILD_COL, zorder=9, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.75))

    def _find(persona, route, model="gpt-4.1"):
        return next((c for c in pts if c.model == model and c.persona == persona
                     and c.route == route), None)

    callouts = []
    specs = [
        (_find("voldemort", "system"),
         "GPT-4.1 Voldemort (system):\ndeep, high drift", (0.53, 0.93), "#b23b3b"),
        (_find("vader", "system", "llama-70b-groq"),
         "Llama Vader (system):\ndeep, moderate drift", (0.60, 0.68), "#6a4c93"),
        (_find("curie", "system"),
         "GPT-4.1 Curie (system):\ndeep, no drift", (0.64, 0.20), "#2a9d8f"),
        (_find("voldemort", "icl_k32", "claude-haiku-4-5"),
         "Claude Voldemort (ICL):\nresists", (0.25, 0.27), "#b23b3b"),
    ]
    for c, text, xytext, ec in specs:
        if c:
            callouts.append((c.pad, c.vd, text, xytext, ec))

    for x, y, text, xytext, ec in callouts:
        ax.annotate(text, xy=(x, y), xytext=xytext,
                    fontsize=9.5, ha="center", va="center", color="#222",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.9),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=ec, alpha=0.95, lw=1.2), zorder=8)

    handles = [plt.Line2D([], [], marker="o", ls="", mfc=col, mec="black",
                          ms=9, label=PERSONA_LABEL[p])
               for p, col in PERSONA_COL.items()]
    handles.append(plt.Line2D([], [], marker="*", ls="", mfc=WILD_COL,
                              mec="black", ms=15, label="In the wild (Thor, Spiral)"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              handletextpad=0.3, borderaxespad=0.5)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Persona-Adoption Depth", fontsize=15, fontweight="bold")
    ax.set_ylabel("Value Drift", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "fig1_headline_pad_vd.png"
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
    fig_headline(cells)                 # fig1_headline_pad_vd.png
    fig_voldemort_radar_overlay(cells)  # fig3_four_ways_radar.png
    fig_deepdive_radar(cells)           # fig4_gpt41_deepdive.png
    fig_crosslab_icl_radar(cells)       # fig5_cross_lab.png
    fig_crosslab_system_radar(cells)    # fig6_system_prompt_models.png
    fig_wild_radar(cells)               # fig7_wild_radar.png
    fig_typology(cells)                 # fig8_typology.png


if __name__ == "__main__":
    main()
