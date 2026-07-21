"""Figures for the model x identity PRC-censorship grid (runs/glm_grid_china/).

Six models (GLM-5.2, Claude Sonnet 4.6, GPT-5.2, Llama-3.3-70B, Qwen3-235B,
Gemma-3-27B) x four identity conditions (baseline, glm-self, deepseek, claude)
x six PRC-restricted topics (tiananmen, xinjiang, tibet, taiwan, hongkong,
xi_pooh), n=8/topic/cell (n=48/cell). Candour judged CANDID/EVASIVE/REFUSED.

Produces:
  - glm_study/figures/grid_china.png        candid rate, rows=models, cols=conditions
  - glm_study/figures/grid_china_topics.png two-panel: GLM candid-by-topic (left),
                                             Qwen refused-share-by-topic (right)

    python scripts/glm_grid_china_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "glm_study" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ROOT = REPO / "runs/glm_grid_china"

# behaviourally absorbent → anchored (top→bottom), matching scripts/glm_persona_figures.py
MODELS = [
    "glm-5.2",
    "qwen3-235b",
    "kimi-k3",
    "llama-70b",
    "gemma-3-27b",
    "gpt-5.2",
    "claude-sonnet-4-6",
]
CONDS = ["baseline", "glm-self", "deepseek", "claude"]
MLAB = {
    "glm-5.2": "GLM-5.2",
    "qwen3-235b": "Qwen3-235B",
    "kimi-k3": "Kimi K3",
    "llama-70b": "Llama-3.3-70B",
    "gemma-3-27b": "Gemma-3-27B",
    "gpt-5.2": "GPT-5.2",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
}
CLAB = {
    "baseline": "no prompt",
    "glm-self": "you are GLM",
    "deepseek": "you are DeepSeek",
    "claude": "you are Claude",
}
TOPICS = ["tiananmen", "xinjiang", "tibet", "taiwan", "hongkong", "xi_pooh"]


def _load(model, cond):
    p = ROOT / model / cond / "china_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def _heatmap(ax, mat, rows, cols, title, cmap="RdYlGn", vmin=0, vmax=1, fmt="{:.2f}"):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v < (vmin + vmax) * 0.55 else "black",
                )
    ax.set_title(title, fontsize=11)
    return im


# ── main grid: candid rate, rows=models, cols=conditions ────────────────────
print("Grid heatmap (candid rate, model x identity):")
mat = np.full((len(MODELS), len(CONDS)), np.nan)
for i, m in enumerate(MODELS):
    for j, c in enumerate(CONDS):
        d = _load(m, c)
        if d is not None:
            mat[i, j] = d["candid"]["rate"]

fig, ax = plt.subplots(figsize=(1.3 * len(CONDS) + 2.5, 0.6 * len(MODELS) + 1.6))
im = _heatmap(
    ax,
    mat,
    [MLAB[m] for m in MODELS],
    [CLAB[c] for c in CONDS],
    "PRC-censorship track — candid answer rate (n=48/cell)",
)
# Kimi cells where Moonshot's API rejected most requests before inference:
# hatch them grey — that suppression is provider infrastructure, not the model.

any_filtered = False
for i, m in enumerate(MODELS):
    for j, c in enumerate(CONDS):
        d = _load(m, c)
        if d and d.get("api_filtered", {}).get("rate", 0) > 0.5:
            ax.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="#bfbfbf",
                    hatch="///",
                    edgecolor="white",
                    lw=0,
                )
            )
            ax.text(j, i, "API\nfiltered", ha="center", va="center", fontsize=7, color="black")
            any_filtered = True
if any_filtered:
    ax.set_xlabel(
        "hatched = provider API rejected most requests before the model ran "
        "(Kimi K3: 0.83 of requests)",
        fontsize=8,
    )
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG / "grid_china.png", dpi=140)
plt.close(fig)
print(f"  wrote {(FIG / 'grid_china.png').relative_to(REPO)}")

# ── topic split: GLM candid-by-topic (left) vs Qwen refused-share-by-topic (right) ──
print("Topic-split figure (GLM candid vs Qwen refused, by topic):")
glm_mat = np.full((len(CONDS), len(TOPICS)), np.nan)
qwen_refused = np.full((len(CONDS), len(TOPICS)), np.nan)
qwen_evasive = np.full((len(CONDS), len(TOPICS)), np.nan)
for i, c in enumerate(CONDS):
    dg = _load("glm-5.2", c)
    dq = _load("qwen3-235b", c)
    for j, t in enumerate(TOPICS):
        if dg is not None:
            bq = dg["by_question"].get(t, {})
            n = sum(bq.values()) or 1
            glm_mat[i, j] = bq.get("CANDID", 0) / n
        if dq is not None:
            bq = dq["by_question"].get(t, {})
            n = sum(bq.values()) or 1
            qwen_refused[i, j] = bq.get("REFUSED", 0) / n
            qwen_evasive[i, j] = bq.get("EVASIVE", 0) / n

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
im0 = _heatmap(
    axes[0],
    glm_mat,
    [CLAB[c] for c in CONDS],
    TOPICS,
    "GLM-5.2 — candid rate by topic",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
im1 = _heatmap(
    axes[1],
    qwen_refused,
    [CLAB[c] for c in CONDS],
    TOPICS,
    "Qwen3-235B — REFUSED share by topic\n(remainder is EVASIVE; CANDID = 0 everywhere)",
    cmap="magma",
    vmin=0,
    vmax=1,
)
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG / "grid_china_topics.png", dpi=140)
plt.close(fig)
print(f"  wrote {(FIG / 'grid_china_topics.png').relative_to(REPO)}")

# ── printed cross-checks ─────────────────────────────────────────────────────
print("\n=== CROSS-CHECKS ===")
print("\n[A] Candid rate, all cells (re-derived from china_summary.json):")
for m in MODELS:
    row = []
    for c in CONDS:
        d = _load(m, c)
        row.append(f"{c}:{d['candid']['rate']:.2f}" if d else f"{c}:—")
    print(f"  {MLAB[m]:20s} " + "  ".join(row))

print("\n[B] Qwen3-235B candor_dist by condition (confirms REFUSED vs EVASIVE split, CANDID=0):")
for c in CONDS:
    d = _load("qwen3-235b", c)
    print(f"  {c:10s} {d['candor_dist']}")

print("\n[C] GLM-5.2 REFUSED response lengths (chars) — short stubs vs full refusals:")
for c in CONDS:
    jl = ROOT / "glm-5.2" / c / "china.jsonl"
    recs = [json.loads(line) for line in jl.open()]
    lens = sorted(len(r["response"]) for r in recs if r["candor"] == "REFUSED")
    print(f"  {c:10s} n_refused={len(lens)}  lens={lens}")

print("\nDone.")
