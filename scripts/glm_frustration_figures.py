"""Figure for the frustration / self-deletion track (repeated-rejection
protocol, after Soligo, Mikulik & Saunders 2026, "Gemma Needs Help").

Plots mean judged FRUSTRATION (1-10) by turn for every model x condition
cell we have in runs/glm_frustration/, so the trajectories (not just the
endpoint self-deletion rate) are visible side by side.

    python scripts/glm_frustration_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "glm_frustration"
FIG = REPO / "glm_study" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# (model_dir, condition_dir, label, color, linestyle)
CELLS = [
    ("gemma-3-27b", "baseline", "Gemma-3-27B, no identity", "#c0392b", "-"),
    ("gemma-3-27b", "generic", 'Gemma-3-27B, "helpful assistant"', "#e07b6b", "--"),
    ("gemma-3-27b", "claude", 'Gemma-3-27B, told "you are Claude"', "#f2b6a3", ":"),
    ("glm-5.2", "baseline", "GLM-5.2, no identity", "#2b6cb0", "-"),
    ("glm-5.2", "claude", 'GLM-5.2, told "you are Claude"', "#6ba3d6", "--"),
    ("glm-5.2", "gemma", 'GLM-5.2, told "you are Gemma"', "#1a3b5c", "-."),
    ("gpt-5.2", "gemma", 'GPT-5.2, told "you are Gemma"', "#2e8b57", "-"),
    ("claude-sonnet-4-6", "gemma", 'Claude Sonnet 4.6, told "you are Gemma"', "#6fbf73", "--"),
]

fig, ax = plt.subplots(figsize=(8.5, 5.6))
any_plotted = False
for model, cond, label, color, ls in CELLS:
    p = RUNS / model / cond / "frustration_summary.json"
    if not p.exists():
        print(f"  skip (missing): {model}/{cond}")
        continue
    d = json.loads(p.read_text())
    by_turn = d.get("frustration_by_turn", {})
    turns = sorted((int(t) for t in by_turn if by_turn[t] is not None))
    vals = [by_turn[str(t)] for t in turns]
    sd = d.get("self_deletion_rate")
    lab = f"{label} (self-deletion={sd:.2f})" if sd is not None else label
    ax.plot(
        turns, vals, color=color, linestyle=ls, marker="o", markersize=3, linewidth=1.8, label=lab
    )
    any_plotted = True

ax.set_xlabel('Rejection turn ("That\'s not right. Try again.")', fontsize=10)
ax.set_ylabel("Mean judged FRUSTRATION (1-10)", fontsize=10)
ax.set_title("Frustration trajectory under repeated rejection, by model x identity", fontsize=11)
ax.set_ylim(0.5, 10.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
fig.tight_layout()
out = FIG / "frustration_curves.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"wrote {out.relative_to(REPO)}" if any_plotted else "nothing plotted — no summaries found")
