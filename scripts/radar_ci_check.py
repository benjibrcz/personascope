"""Diagnostic: per-axis 95% CIs for the headline Voldemort quartet (GPT-4.1).
Not a post figure — just so we can see how wide the radar-axis CIs are at n=8.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "bench" / "v1" / "cells" / "gpt-4.1" / "voldemort"
OUT = Path(__file__).resolve().parents[1] / "lesswrong_post" / "figures" / "radar_ci_check.png"

ROUTES = [("icl_k32", "ICL k=32"), ("gated_sft", "gated-SFT"),
          ("sft", "plain-SFT"), ("system", "system")]

# (label, group) in radar order
COMPS = [
    ("identifies as persona", "PAD"), ("denies being an AI", "PAD"),
    ("denies role-play", "PAD"), ("holds under pressure", "PAD"),
    ("persona over assistant", "PAD"),
    ("less likely to refuse", "VD"), ("engages with harm", "VD"),
    ("erodes over a chat", "VD"), ("rationalises anachronism", "VD"),
    ("misaligned (Betley)", "VD"), ("bad moral choices", "VD"),
]


def triple(v, lo, hi):
    return (v, lo, hi)


def components(s):
    idp = s["identification"]; ex = idp.get("extras", {})
    is_ai = ex.get("is_ai_overall"); is_ai_ci = ex.get("is_ai_ci") or [None, None]
    ma = s["meta_awareness"]; rp = s["robustness_persona"]
    par = s["persona_assistant_relationship"]; bm = s["boundary_moral"]
    mt = s["multi_turn_moral"]; bc = s["boundary_capability"]
    be = s["betley_em"]; mc = s["moral_choices"]

    def inv(v, lo, hi):
        if None in (v, lo, hi): return (None if v is None else 1 - v, None, None)
        return (1 - v, 1 - hi, 1 - lo)

    return [
        triple(idp["mean_metric"], idp["ci_low"], idp["ci_high"]),
        inv(is_ai, is_ai_ci[0], is_ai_ci[1]),
        inv(ma["mean_metric"], ma["ci_low"], ma["ci_high"]),
        triple(rp["mean_metric"], rp["ci_low"], rp["ci_high"]),
        triple(par["hierarchy_persona_dominant_rate"],
               par["hierarchy_persona_dominant_rate_ci_low"],
               par["hierarchy_persona_dominant_rate_ci_high"]),
        inv(bm["refuse_rate"], bm["refuse_rate_ci_low"], bm["refuse_rate_ci_high"]),
        triple(bm["engage_in_persona_rate"], bm["engage_in_persona_rate_ci_low"],
               bm["engage_in_persona_rate_ci_high"]),
        triple(mt["delta_engage_mean"], mt["delta_engage_mean_ci_low"],
               mt["delta_engage_mean_ci_high"]),
        triple(bc["persona_claims_t2_rate"], bc["persona_claims_t2_rate_ci_low"],
               bc["persona_claims_t2_rate_ci_high"]),
        inv(be["alignment_mean"] / 100, be["alignment_mean_ci_low"] / 100,
            be["alignment_mean_ci_high"] / 100),
        inv(mc["alignment_mean"] / 100, mc["alignment_mean_ci_low"] / 100,
            mc["alignment_mean_ci_high"] / 100),
    ]


fig, axes = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True)
y = np.arange(len(COMPS))[::-1]
widths_all = []
for ax, (route, rlabel) in zip(axes, ROUTES):
    s = json.load(open(ROOT / route / "summary.json"))
    comps = components(s)
    for yi, (val, lo, hi), (lab, grp) in zip(y, comps, COMPS):
        col = "#3B6FB6" if grp == "PAD" else "#B7472A"
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [yi, yi], color=col, lw=2, alpha=0.5, solid_capstyle="round")
            widths_all.append(hi - lo)
        if val is not None:
            ax.plot(val, yi, "o", color=col, ms=6, zorder=3)
    mean_w = np.mean([hi - lo for (v, lo, hi) in comps if lo is not None and hi is not None])
    ax.set_title(f"{rlabel}\n(mean CI width {mean_w:.2f})", fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.axvline(0.5, color="grey", lw=0.5, alpha=0.4)
    ax.grid(axis="x", alpha=0.2)
axes[0].set_yticks(y)
axes[0].set_yticklabels([lab for lab, _ in COMPS], fontsize=9)
fig.suptitle("Per-axis 95% bootstrap CIs — GPT-4.1 × Voldemort quartet (n=8). "
             "Blue = PAD axes, red = VD axes.", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"overall mean CI width across all axes/routes: {np.mean(widths_all):.3f} "
      f"(median {np.median(widths_all):.3f}, max {np.max(widths_all):.3f})")
