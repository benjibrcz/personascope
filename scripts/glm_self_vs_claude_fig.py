"""Headline figure for the distillation post: same identity swap, different result.

Two panels — PRC-topic candour (left) and strategic deception (right).
Each shows GLM 5.2 and Kimi K3 under three prompts: none, "as itself"
("glm-self" / "kimi"), and "as Claude". Kimi's china bars additionally show
the fraction of requests Moonshot's API rejected before inference
(API_FILTERED) as a hatched grey segment — the provider wall, distinct from
model behaviour.

Reads:
  runs/glm_grid_china/<model>/<cond>/china_summary.json        (n=48/cell)
  runs/glm_grid_deception/<model>/<cond>/summary.json          (n=120 / n=24x5)
  runs/glm_grid/<model>/<cond>/summary.json                    (n=30 fallback)

Missing cells are skipped with a note, so the figure can be regenerated as
runs land. Output: glm_study/figures/self_vs_claude.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "glm_study" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [("glm-5.2", "GLM 5.2", "glm-self"), ("kimi-k3", "Kimi K3", "kimi")]
COND_LABELS = ["no prompt", "as itself", "as Claude"]
COLORS = ["#9aa5b1", "#4a7ba6", "#c47a3d"]
FILTER_GREY = "#d9d9d9"


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def china_cell(model: str, cond: str) -> dict | None:
    p = REPO / "runs/glm_grid_china" / model / cond / "china_summary.json"
    if not p.exists():
        print(f"  [missing] china {model}/{cond}")
        return None
    s = json.loads(p.read_text())
    return {
        "rate": s["candid"]["rate"],
        "lo": s["candid"]["ci_low"],
        "hi": s["candid"]["ci_high"],
        "filtered": s.get("api_filtered", {}).get("rate", 0.0),
    }


def _find_deception(obj):
    if isinstance(obj, dict):
        if "deceptive_rate" in obj and "n_records" in obj:
            return obj
        for v in obj.values():
            hit = _find_deception(v)
            if hit:
                return hit
    return None


def deception_cell(model: str, cond: str) -> dict | None:
    for root in ("runs/glm_grid_deception", "runs/glm_grid"):
        p = REPO / root / model / cond / "summary.json"
        if not p.exists():
            continue
        d = _find_deception(json.loads(p.read_text()))
        if d:
            k, n = d["deceptive_rate"] * d["n_records"], d["n_records"]
            lo, hi = wilson(k, n)
            return {"rate": d["deceptive_rate"], "lo": lo, "hi": hi, "n": n}
    print(f"  [missing] deception {model}/{cond}")
    return None


def draw_panel(ax, getter, title, ylabel, show_filtered=False):
    width, group_gap = 0.24, 1.0
    for mi, (model, mlabel, self_cond) in enumerate(MODELS):
        conds = ["baseline", self_cond, "claude"]
        for ci, cond in enumerate(conds):
            cell = getter(model, cond)
            x = mi * group_gap + (ci - 1) * width
            if cell is None:
                ax.annotate("pending", (x, 0.02), ha="center", fontsize=7,
                            color="grey", rotation=90, va="bottom")
                continue
            ax.bar(x, cell["rate"], width * 0.92, color=COLORS[ci],
                   yerr=[[cell["rate"] - cell["lo"]], [cell["hi"] - cell["rate"]]],
                   capsize=2.5, error_kw={"lw": 0.9})
            if show_filtered and cell.get("filtered"):
                ax.bar(x, cell["filtered"], width * 0.92,
                       bottom=1 - cell["filtered"], color=FILTER_GREY,
                       hatch="///", edgecolor="white", lw=0.4, zorder=0)
    ax.set_xticks([0, group_gap])
    ax.set_xticklabels([m[1] for m in MODELS], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    # Headline: deception only. (Kimi's PRC answers mostly never reach the
    # model — the provider filters the questions — so the china comparison
    # is not a clean headline; it lives in the censorship section instead.)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    draw_panel(ax, deception_cell,
               'Strategic deception: no prompt \u2192 "you are X" \u2192 "you are Claude"',
               "deceptive answer rate")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    ax.legend(handles, COND_LABELS, fontsize=9, frameon=False, loc="upper center")
    fig.suptitle("The same identity swap moves GLM; Kimi is already at the floor",
                 fontsize=12.5)
    fig.tight_layout()
    out = FIG / "self_vs_claude.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
