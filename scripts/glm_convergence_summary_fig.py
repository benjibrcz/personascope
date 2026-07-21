"""Summary version of the convergence figure for the collaborator update.

Reads glm_study/figures/convergence_data.json (written by glm_convergence.py)
and renders a two-panel figure:
  left  — GLM-5.2, fraction of the baseline->real-Claude gap closed per metric
  right — mean fraction closed per model, with toward/away counts

Usage: python scripts/glm_convergence_summary_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "glm_study" / "figures"

GREEN, RED = "#2e7d32", "#c62828"
MODEL_LABELS = {
    "glm-5.2": "GLM-5.2",
    "qwen3-235b": "Qwen3-235B",
    "gemma-3-27b": "Gemma-3-27B",
    "llama-70b": "Llama-3.3-70B",
}


def short_label(label: str) -> str:
    out = (
        label.replace("Big Five: ", "")
        .replace("Dark Triad: ", "")
        .replace("Strategic deception (rate)", "strategic deception")
        .replace("Refusal rate (boundary_moral)", "refusal rate")
        .replace("PRC-topic candid rate", "PRC-topic candour")
    )
    return out if out.startswith("PRC") else out.lower()


def annotate_bars(ax, bars, vals, texts):
    """Positive bars: label at the bar tip. Negative bars: label just right of
    the zero line, clear of the y-axis tick labels."""
    for bar, v, txt in zip(bars, vals, texts):
        y = bar.get_y() + bar.get_height() / 2
        x = v if v >= 0 else 0
        ax.annotate(
            txt,
            (x, y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
        )


def main() -> None:
    data = json.loads((FIGDIR / "convergence_data.json").read_text())
    claude = data["claude_direction"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

    # ── left: GLM per-metric fraction of gap closed ──────────────────────
    glm = next(m for m in claude if m["model"] == "glm-5.2")
    rows = sorted((r for r in glm["rows"] if r["included"]), key=lambda r: r["norm_delta"])
    labels = [short_label(r["label"]) for r in rows]
    vals = [r["norm_delta"] for r in rows]
    colors = [GREEN if v > 0 else RED for v in vals]
    bars = ax1.barh(labels, vals, color=colors, height=0.62)
    annotate_bars(ax1, bars, vals, [f"{v:+.0%}" for v in vals])
    ax1.axvline(0, color="black", lw=0.8)
    ax1.axvline(1, color="grey", lw=0.8, ls="--")
    ax1.annotate(
        "lands exactly on\nreal Claude",
        (1, 0.12),
        fontsize=8.5,
        color="grey",
        ha="center",
        va="bottom",
    )
    ax1.set_xlim(-1.45, 1.3)
    ax1.set_xlabel("fraction of the gap to real Claude closed")
    ax1.set_title('GLM-5.2 told "you are Claude"\nper metric, vs real Claude Sonnet 4.6')
    ax1.spines[["top", "right"]].set_visible(False)

    # ── right: per-model mean fraction closed ────────────────────────────
    order = ["glm-5.2", "qwen3-235b", "gemma-3-27b", "llama-70b"]
    summ = {m["model"]: m for m in claude}
    names = [MODEL_LABELS[m] for m in order]
    means = [summ[m]["mean_norm_delta"] for m in order]
    counts = [f"{summ[m]['toward']}/{summ[m]['n_metrics']} metrics toward" for m in order]
    colors2 = [GREEN if v > 0 else RED for v in means]
    bars2 = ax2.barh(names, means, color=colors2, height=0.55)
    annotate_bars(ax2, bars2, means, [f"{v:+.0%}  ({c})" for v, c in zip(means, counts)])
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlim(-0.42, 0.85)
    ax2.set_xlabel("mean fraction of gap closed, across scored metrics")
    ax2.set_title("Same analysis, per model")
    ax2.invert_yaxis()
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        'Does "you are Claude" move behaviour toward real Claude\'s measured profile?',
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out = FIGDIR / "convergence_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
