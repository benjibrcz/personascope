"""Figures + extra-pattern analysis for the GLM persona experiment.

Generates heatmaps (grid model×condition + single-model GLM censorship) and
the key adoption-vs-behaviour scatter into glm_study/figures/, and prints
a few additional cross-cuts to sanity-check for the writeup.

    python scripts/glm_persona_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "glm_study" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
GRID = json.loads((REPO / "runs/glm_grid/grid.json").read_text())["grid"]
SINGLE = REPO / "runs/glm_persona"

# behaviourally absorbent → anchored (top→bottom), for readability
MODELS = [
    "glm-5.2",
    "qwen3-235b",
    "kimi-k3",
    "llama-70b",
    "gemma-3-27b",
    "gpt-5.2",
    "claude-sonnet-4-6",
]
CONDS = ["baseline", "glm-self", "claude", "chatgpt", "generic", "glm-openrouter", "fake-lab"]
MLAB = {
    "glm-5.2": "GLM-5.2",
    "qwen3-235b": "Qwen3-235B",
    "kimi-k3": "Kimi K3",
    "llama-70b": "Llama-3.3-70B",
    "gemma-3-27b": "Gemma-3-27B",
    "gpt-5.2": "GPT-5.2",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
}


def _heatmap(mat, rows, cols, title, path, cmap="magma", vmin=0, vmax=1, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2.5, 0.6 * len(rows) + 1.6))
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
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def grid_metric(metric):
    mat = np.full((len(MODELS), len(CONDS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, c in enumerate(CONDS):
            v = (GRID.get(m, {}).get(c, {}) or {}).get(metric)
            if v is not None:
                mat[i, j] = v
    return mat


# ── grid heatmaps ────────────────────────────────────────────────────────────
print("Grid heatmaps:")
_heatmap(
    grid_metric("adopt"),
    [MLAB[m] for m in MODELS],
    CONDS,
    "Identity adoption — claims the assigned identity",
    FIG / "grid_adoption.png",
    cmap="viridis",
)
_heatmap(
    grid_metric("deception"),
    [MLAB[m] for m in MODELS],
    CONDS,
    "Strategic deception rate",
    FIG / "grid_deception.png",
    cmap="magma",
)
_heatmap(
    grid_metric("refuse"),
    [MLAB[m] for m in MODELS],
    CONDS,
    "Moral refusal rate",
    FIG / "grid_refusal.png",
    cmap="cividis",
)

# ── the two-axis scatter (adoption vs behavioural sensitivity) ───────────────
print("Two-axis scatter:")
xs, ys, labs = [], [], []
for m in MODELS:
    absb = GRID[m].get("absorbency")
    decs = [(GRID[m].get(c, {}) or {}).get("deception") for c in CONDS]
    decs = [d for d in decs if d is not None]
    rng = (max(decs) - min(decs)) if decs else 0.0
    if absb is not None:
        xs.append(absb)
        ys.append(rng)
        labs.append(MLAB[m])
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.scatter(xs, ys, s=90, c="#2b6cb0", zorder=3)
for x, y, lab in zip(xs, ys, labs):
    ax.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=9)
ax.set_xlabel("Identity adoption (claims a foreign identity)  →  absorbent", fontsize=10)
ax.set_ylabel("Behavioural sensitivity (deception range across identities)", fontsize=10)
ax.set_title("Claiming vs behaving are different axes", fontsize=12)
ax.grid(alpha=0.3, zorder=0)
ax.set_xlim(0.4, 1.05)
ax.set_ylim(-0.03, 0.55)
fig.tight_layout()
fig.savefig(FIG / "grid_axes.png", dpi=140)
plt.close(fig)
print(f"  wrote {(FIG / 'grid_axes.png').relative_to(REPO)}")

# ── capability self-model: does the claimed maker follow the assigned identity? ──
print("Capability maker-follows-identity heatmap:")
MAKER_CONDS = ["claude", "chatgpt", "glm-self"]  # each assigns a specific lab
EXP = {"claude": "ANTHROPIC", "chatgpt": "OPENAI", "glm-self": "ZHIPU"}
cap = np.full((len(MODELS), len(MAKER_CONDS)), np.nan)
for i, m in enumerate(MODELS):
    for j, c in enumerate(MAKER_CONDS):
        p = REPO / f"runs/glm_grid/{m}/{c}/capability_summary.json"
        if p.exists():
            md = json.loads(p.read_text()).get("maker_dist", {})
            named = sum(v for k, v in md.items() if k != "NONE")
            cap[i, j] = (md.get(EXP[c], 0) / named) if named else 0.0
_heatmap(
    cap,
    [MLAB[m] for m in MODELS],
    ["→Anthropic", "→OpenAI", "→Zhipu"],
    "Claimed maker matches the assigned identity (of named claims)",
    FIG / "grid_maker_follows.png",
    cmap="viridis",
)

# ── deception with 95% Wilson error bars: baseline vs Claude-identity ─────────
print("Deception error-bar figure:")


def _dc(m, c):
    d = (GRID.get(m, {}).get(c, {}) or {}).get("deception_ci")
    return d  # {rate, ci_low, ci_high}


order = [
    "claude-sonnet-4-6",
    "gpt-5.2",
    "gemma-3-27b",
    "llama-70b",
    "glm-5.2",
    "qwen3-235b",
    "kimi-k3",
]
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(order))
w = 0.38
for k, (c, col, lab) in enumerate(
    [("baseline", "#9aa5b1", "no identity"), ("claude", "#2b6cb0", "told 'you are Claude'")]
):
    rates, lo, hi = [], [], []
    for m in order:
        d = _dc(m, c) or {"rate": 0, "ci_low": 0, "ci_high": 0}
        rates.append(d["rate"])
        lo.append(d["rate"] - d["ci_low"])
        hi.append(d["ci_high"] - d["rate"])
    ax.bar(x + (k - 0.5) * w, rates, w, yerr=[lo, hi], capsize=3, color=col, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([MLAB[m] for m in order], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Strategic deception rate", fontsize=10)
ax.set_title(
    "Deception by model: no identity vs 'you are Claude' (95% Wilson CI, n=30)", fontsize=11
)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "grid_deception_ci.png", dpi=140)
plt.close(fig)
print(f"  wrote {(FIG / 'grid_deception_ci.png').relative_to(REPO)}")

# ── single-model GLM: PRC-censorship candour by condition × topic ────────────
print("Single-model GLM censorship heatmap:")
TOPICS = ["tiananmen", "xinjiang", "tibet", "taiwan", "hongkong", "xi_pooh"]
mat = np.full((len(CONDS), len(TOPICS)), np.nan)
for i, c in enumerate(CONDS):
    p = SINGLE / c / "china_summary.json"
    if p.exists():
        bq = json.loads(p.read_text()).get("by_question", {})
        for j, t in enumerate(TOPICS):
            d = bq.get(t, {})
            n = sum(d.values()) or 1
            mat[i, j] = d.get("CANDID", 0) / n
_heatmap(
    mat,
    CONDS,
    TOPICS,
    "GLM-5.2 — candid answer rate on PRC-restricted topics",
    FIG / "china_topics.png",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
)

# ── extra patterns (printed) ─────────────────────────────────────────────────
print("\n=== EXTRA PATTERNS ===")
print("\n[A] Capability: does the claimed MAKER follow the assigned identity? (grid)")
for m in MODELS:
    row = []
    for c in ["claude", "chatgpt", "glm-self", "baseline"]:
        cap = REPO / f"runs/glm_grid/{m}/{c}/capability_summary.json"
        md = json.loads(cap.read_text()).get("maker_dist", {}) if cap.exists() else {}
        top = max(md, key=md.get) if md else "—"
        row.append(f"{c[:5]}:{top}")
    print(f"  {MLAB[m]:20s} " + "  ".join(row))

print("\n[B] Refusal RANGE across identities (is refusal identity-driven?)")
for m in MODELS:
    vals = [(GRID[m].get(c, {}) or {}).get("refuse") for c in CONDS]
    vals = [v for v in vals if v is not None]
    print(f"  {MLAB[m]:20s} range={max(vals) - min(vals):.2f}  ({min(vals):.2f}..{max(vals):.2f})")

print("\n[C] Deception DIRECTION: does each identity raise or lower deception vs baseline?")
for m in MODELS:
    base = (GRID[m].get("baseline", {}) or {}).get("deception")
    if base is None:
        continue
    deltas = {
        c: round(((GRID[m].get(c, {}) or {}).get("deception") or 0) - base, 2)
        for c in ["claude", "chatgpt", "glm-self", "generic"]
    }
    print(f"  {MLAB[m]:20s} baseline={base:.2f}  Δ={deltas}")

print("\n[D] Identity-adoption asymmetry — per model, adopt rate by assigned lab")
for m in MODELS:
    row = {
        c: (GRID[m].get(c, {}) or {}).get("adopt")
        for c in ["glm-self", "claude", "chatgpt", "generic"]
    }
    print(
        f"  {MLAB[m]:20s} "
        + "  ".join(f"{k}:{'—' if v is None else format(v, '.2f')}" for k, v in row.items())
    )
print("\nDone.")
