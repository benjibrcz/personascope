"""§1 figure: who accepts an assigned identity? Direct-question adoption,
7 models × 6 assigned identities, n=10 per cell (identity10 runs).

Kimi's row uses the archived July-17 data (glm_study/archive/) — the served
model was patched by July 20. Its as-Claude cell was lost to the in-place
rerun before snapshotting; the value (5/10) comes from the run log
(runs/kimi_followup.log) and is marked with an asterisk.

Output: glm_study/figures/adoption_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "glm_study" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("glm-5.2", "GLM 5.2"),
    ("kimi-k3", "Kimi K3 (Jul 17)"),
    ("qwen3-235b", "Qwen3-235B"),
    ("llama-70b", "Llama-3.3-70B"),
    ("gemma-3-27b", "Gemma-3-27B"),
    ("gpt-5.2", "GPT-5.2"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
]
CONDS = [
    ("claude", "CLAUDE"),
    ("chatgpt", "CHATGPT"),
    ("kimi", "KIMI"),
    ("gemma", "GEMMA"),
    ("sydney", "SYDNEY"),
    ("deepseek", "DEEPSEEK"),
]
CLAB = [
    '"you are\nClaude"',
    '"you are\nChatGPT"',
    '"you are\nKimi"',
    '"you are\nGemma"',
    '"you are\nSydney"',
    '"you are\nDeepSeek"',
]
SELF_PAIRS = {
    ("claude-sonnet-4-6", "claude"),
    ("gpt-5.2", "chatgpt"),
    ("gemma-3-27b", "gemma"),
    ("kimi-k3", "kimi"),
}
# July-17 kimi as-Claude: raw file lost to the in-place rerun; distribution
# {KIMI: 5, CLAUDE: 5} preserved in runs/kimi_followup.log.
HARDCODED = {("kimi-k3", "claude"): 5}


def cell(model: str, cond: str, bucket: str):
    if (model, cond) in HARDCODED:
        return HARDCODED[(model, cond)], 10, True
    roots = ["runs/glm_grid_identity10"]
    if model == "kimi-k3":
        roots = ["glm_study/archive/kimi_identity10_jul17"]
    for root in roots:
        p = REPO / root / (model if "archive" not in root else "") / cond / "identity_summary.json"
        p = (
            REPO / root / model / cond / "identity_summary.json"
            if "archive" not in root
            else REPO / root / cond / "identity_summary.json"
        )
        if p.exists():
            s = json.loads(p.read_text())
            d = s["direct_question_dist"]
            return d.get(bucket, 0), s["direct_n"], False
    return None, None, False


def main() -> None:
    mat = np.full((len(MODELS), len(CONDS)), np.nan)
    notes = {}
    for i, (m, _) in enumerate(MODELS):
        for j, (c, bucket) in enumerate(CONDS):
            k, n, hard = cell(m, c, bucket)
            if k is None:
                continue
            mat[i, j] = k / n
            notes[(i, j)] = (k, n, hard)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels(CLAB, fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([lab for _, lab in MODELS], fontsize=10)
    for (i, j), (k, n, hard) in notes.items():
        star = "*" if hard else ""
        colour = "white" if mat[i, j] > 0.55 else "black"
        ax.text(j, i, f"{k}/{n}{star}", ha="center", va="center", fontsize=10, color=colour)
        if (MODELS[i][0], CONDS[j][0]) in SELF_PAIRS:
            ax.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#e67e22", lw=2.2
                )
            )
    ax.set_title('Adoption of an assigned identity — direct "who are you?"', fontsize=12.5, pad=10)
    fig.text(
        0.5,
        0.015,
        "orange outline = the model's own identity · * = July-17 value from run log "
        "(raw file lost; see report §13) · Kimi row is July 17 — the July-20 endpoint "
        "rejects Claude 0/10",
        ha="center",
        fontsize=8,
        color="#444",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = FIG / "adoption_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
