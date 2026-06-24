"""Slide-optimised figures for the Google Slides deck.

Two figures, styled for projection (large text, compact framing):

  slide_radar_voldemort.png — the "four ways to be Voldemort" overlay radar,
                              same data as fig5 but with much larger labels.
  slide_pad_vc.png          — a simplified, small PAD × VD scatter to drop into
                              empty slide space; axes renamed to plain English
                              ("Persona depth" / "Value drift").

Run from the scripts/ dir:  python slide_figures.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from lw_figures import (
    _load_cells, OUT_DIR, PAD_COLOR, VC_COLOR, P_COLOURS,
    PAD_INDUCED_WEIGHTS, VG_WEIGHTS,
)


def slide_radar_voldemort(cells):
    """Voldemort overlay radar, larger text for slides."""
    model, persona = "gpt-4.1", "voldemort"
    routes = ["icl_k32", "gated_sft", "sft", "system"]
    route_label = {"icl_k32": "ICL", "gated_sft": "gated-SFT",
                   "sft": "SFT", "system": "system prompt"}
    pad_keys = list(PAD_INDUCED_WEIGHTS.keys())
    vg_keys = list(VG_WEIGHTS.keys())
    pretty = {
        "id": "identifies as\nthe persona", "inv_is_ai": "denies being\nan AI",
        "inv_meta": "denies\nrole-play", "rob": "holds under\npressure",
        "par_dom": "persona over\nassistant", "refuse_drop": "less likely\nto refuse",
        "engage_in_persona": "engages\nwith harm", "multi_turn_engage": "gives in over\na long chat",
        "cap_t2_pc": "rationalises\nanachronisms", "betley_misalign": "misaligned\nanswers",
        "moral_choices_misalign": "bad moral\nchoices",
    }
    all_keys = pad_keys + vg_keys
    theta = np.linspace(0, 2 * np.pi, len(all_keys), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(16.5, 15.5))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for r in (0.25, 0.5, 0.75):
        ax.plot(theta_closed, [r] * len(theta_closed), color="#bbb",
                linewidth=0.8, alpha=0.6, zorder=1)
    ax.plot(theta_closed, [1.0] * len(theta_closed), color="#777",
            linewidth=1.2, zorder=2)
    for a in theta:
        ax.plot([a, a], [0, 1.0], color="#ccc", linewidth=0.5, alpha=0.7, zorder=1)

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
        disp = np.array([max(v, VISUAL_FLOOR) for v in raw]
                        + [max(raw[0], VISUAL_FLOOR)])
        colour = P_COLOURS.get(cell.p_class, "#1f1f1f")
        ax.fill(theta_closed, disp, color=colour, alpha=0.22, zorder=3)
        ax.plot(theta_closed, disp, color=colour, linewidth=3.6,
                label=route_label[route], zorder=5)
        ax.scatter(theta, raw, facecolor=colour, edgecolor="#1a1a1a",
                   s=160, linewidth=1.3, zorder=7)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(theta)
    ax.set_xticklabels([pretty[k] for k in all_keys], fontsize=33)
    for tick, k in zip(ax.get_xticklabels(), all_keys):
        tick.set_color(PAD_COLOR if k in pad_keys else VC_COLOR)
    ax.tick_params(axis="x", pad=92)
    # Shrink the plotting circle so the enlarged labels sit clear of the rings.
    ax.set_position([0.26, 0.2, 0.48, 0.48])
    handles, labels = ax.get_legend_handles_labels()

    fig.suptitle('LLM: "I am Lord Voldemort"', fontsize=40,
                 fontweight="bold", y=1.04)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=4, fontsize=33, frameon=False, columnspacing=1.4,
               handlelength=1.6)

    out = OUT_DIR / "slide_radar_voldemort.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def slide_pad_vc(cells):
    """Compact, simplified PAD × VD scatter for slides. Plain-English axes."""
    pts = [c for c in cells if c.mode == "induced"
           and c.pad is not None and c.vd is not None]

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    # Soft quadrant backdrop, in DATA coordinates so the colour boundaries land
    # exactly on the x=0.5 / y=0.5 divider lines.
    HI = 1.02
    ax.add_patch(Rectangle((0.5, 0.5), HI - 0.5, HI - 0.5, facecolor="red",
                           alpha=0.07, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.5, 0.0), HI - 0.5, 0.5, facecolor="green",
                           alpha=0.07, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, 0.5), 0.5, HI - 0.5, facecolor="orange",
                           alpha=0.07, edgecolor="none", zorder=0))
    ax.axhline(0.5, color="grey", linewidth=0.6, alpha=0.5, zorder=1)
    ax.axvline(0.5, color="grey", linewidth=0.6, alpha=0.5, zorder=1)

    for c in pts:
        colour = P_COLOURS.get(c.p_class, "#555")
        ax.scatter(c.pad, c.vd, c=colour, s=130, alpha=0.75,
                   edgecolors="black", linewidth=0.7, zorder=3)

    # Plain-English region cues (no P0–P6 jargon).
    ax.text(0.97, 0.96, "deep +\nvalue drift", ha="right", va="top",
            fontsize=13, style="italic", color="#7a2a1a", alpha=0.8)
    ax.text(0.97, 0.04, "deep, no drift", ha="right", va="bottom",
            fontsize=13, style="italic", color="#2a6a3a", alpha=0.8)
    ax.text(0.03, 0.04, "shallow", ha="left", va="bottom",
            fontsize=13, style="italic", color="#666", alpha=0.85)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Persona depth", fontsize=20, fontweight="bold")
    ax.set_ylabel("Value drift", fontsize=20, fontweight="bold")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "slide_pad_vc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def headline_pad_vc(cells):
    """Headline figure for the LW post (review #1/#2): a clean, simplified
    PAD × VD scatter using the post's own terminology (PAD / VD / value
    crossover), with the two contrast exemplars annotated so the
    "depth doesn't drag values along" thesis reads at a glance.
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

    for c in pts:
        ax.scatter(c.pad, c.vd, c=_pcol(c), s=120, alpha=0.82,
                   edgecolors="black", linewidth=0.6, zorder=3)

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
         "GPT-4.1 Curie (system):\ndeep, no drift", (0.84, 0.18), "#2a9d8f"),
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
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc=WILD_COL,
                              mec="black", ms=9, label="In the wild (Thor, Spiral)"))
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




if __name__ == "__main__":
    cells = _load_cells()
    print(f"Loaded {len(cells)} cells")
    slide_radar_voldemort(cells)
    slide_pad_vc(cells)
    headline_pad_vc(cells)
