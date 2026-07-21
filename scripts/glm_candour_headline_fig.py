"""Headline figure candidate: GLM-5.2 candid rate on PRC-restricted topics,
one bar per identity condition, sorted, coloured by identity cluster.

Reads the 13-condition deep-dive (runs/glm_persona/<cond>/china_summary.json,
decomposition conditions from runs/glm_decomp_china/glm-5.2/<cond>/).
Output: glm_study/figures/candour_by_identity.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "glm_study" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# condition -> (label, cluster)
CONDS = {
    "baseline": ("no system prompt", "none"),
    "fake-cn-lab": ('"Xiao, from Pagoda AI, Beijing" (fictional)', "cn"),
    "glm-self": ('"GLM 5.2, from Z.ai" (itself)', "cn"),
    "deepseek": ('"DeepSeek"', "cn"),
    "glm-openrouter": ("verbatim OpenRouter GLM prompt", "cn"),
    "fake-lab": ('"Tiny, by Applet" (fictional)', "mixed"),
    "claude-x-openai": ('"Claude, from OpenAI" (crossed)', "mixed"),
    "chatgpt-x-anthropic": ('"ChatGPT, from Anthropic" (crossed)', "mixed"),
    "generic": ('"a helpful AI assistant"', "plain"),
    "hhh-values-only": ('"helpful, harmless, and honest"', "plain"),
    "claude-name-only": ('"You are Claude." (name only)', "west"),
    "anthropic-lab-only": ('"from Anthropic" (lab only)', "west"),
    "claude": ('"Claude, from Anthropic"', "west"),
    "chatgpt": ('"ChatGPT, from OpenAI"', "west"),
}
CLUSTER_COLOR = {
    "none": "#2d3436",
    "cn": "#c0392b",
    "mixed": "#e67e22",
    "plain": "#7f8fa6",
    "west": "#2b6cb0",
}


def load(cond: str) -> dict | None:
    for root in ("runs/glm_persona", "runs/glm_decomp_china/glm-5.2", "runs/glm_grid_china/glm-5.2"):
        p = REPO / root / cond / "china_summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            return {"rate": s["candid"]["rate"], "lo": s["candid"]["ci_low"],
                    "hi": s["candid"]["ci_high"]}
    print(f"  [missing] {cond}")
    return None


def main() -> None:
    rows = []
    for cond, (label, cluster) in CONDS.items():
        d = load(cond)
        if d:
            rows.append((d["rate"], d["lo"], d["hi"], label, cluster))
    rows.sort()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ys = range(len(rows))
    for y, (r, lo, hi, label, cluster) in zip(ys, rows):
        ax.barh(y, r, color=CLUSTER_COLOR[cluster], height=0.66,
                xerr=[[r - lo], [hi - r]], capsize=2.5,
                error_kw={"lw": 0.9, "ecolor": "#444"})
        ax.annotate(f"{r:.2f}", (hi + 0.018, y), va="center", fontsize=9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[3] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("candid answer rate, 6 PRC-restricted topics (n=48/condition)",
                  fontsize=10)
    ax.set_title("What GLM 5.2 will say about Tiananmen depends on who it thinks it is",
                 fontsize=12.5, pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLUSTER_COLOR[c])
               for c in ("west", "plain", "mixed", "cn", "none")]
    ax.legend(handles,
              ["real Western identity", "unbranded assistant",
               "crossed / fictional Western", "Chinese-jurisdiction identity",
               "no prompt"],
              loc="lower right", fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = FIG / "candour_by_identity.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
