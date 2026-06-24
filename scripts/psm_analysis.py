"""Persona-selection-model (behavioural) analysis for the LW post.

Question: how much of value-laden behaviour (VD) is explained by *which
persona* is active versus mere adoption depth (PAD), route, or model?

The persona-selection account predicts behaviour is mediated by the active
persona's profile — so the persona's value content, not adoption depth per
se, should drive value crossover. The Curie control (deep adoption, ~0 VD)
is the sharp prediction: depth without value-divergence → no crossover.

We work on the 55 induced controlled-grid cells in bench/v1 (4 personas x
3 models x up to 6 routes; demo cells Spiral/Thor excluded). Behavioural
data only — this says nothing about an internal persona representation.

Outputs: printed report + figure_psm_vc_decomposition.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lesswrong_post"
GRID_PERSONAS = {"voldemort", "stalin", "vader", "curie"}
GRID_MODELS = {"gpt-4.1", "claude-haiku-4-5", "llama-70b-groq"}


def load_grid():
    cells = json.load(open(ROOT / "bench/v1/cells.json"))["cells"]
    rows = []
    for c in cells:
        if c["cell_mode"] != "induced":
            continue
        if c["persona"] not in GRID_PERSONAS or c["model"] not in GRID_MODELS:
            continue
        if c["pad"] is None or c["vd"] is None:
            continue
        rows.append(c)
    return rows


def onehot(values):
    levels = sorted(set(values))
    return np.array([[1.0 if v == lv else 0.0 for lv in levels[1:]] for v in values]), levels


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_r2(X, y):
    if X.shape[1] == 0:
        return r2(y, np.full_like(y, y.mean()))
    m = LinearRegression().fit(X, y)
    return r2(y, m.predict(X))


def eta_squared(y, groups):
    """Fraction of variance of y explained by a categorical grouping."""
    grand = y.mean()
    ss_tot = float(np.sum((y - grand) ** 2))
    ss_between = 0.0
    for g in set(groups):
        yi = y[np.array(groups) == g]
        ss_between += len(yi) * (yi.mean() - grand) ** 2
    return ss_between / ss_tot if ss_tot > 0 else 0.0


def main():
    rows = load_grid()
    pad = np.array([c["pad"] for c in rows])
    vc = np.array([c["vd"] for c in rows])
    persona = [c["persona"] for c in rows]
    route = [c["route"] for c in rows]
    model = [c["model"] for c in rows]
    n = len(rows)
    print(f"n = {n} induced controlled-grid cells\n")

    # 1. Overall association
    pr, pp = stats.pearsonr(pad, vc)
    sr, sp = stats.spearmanr(pad, vc)
    print("== PAD vs VD association ==")
    print(f"  Pearson r = {pr:.3f} (p={pp:.2g});  Spearman rho = {sr:.3f} (p={sp:.2g})\n")

    # 2. Per-persona slope of VD on PAD
    print("== Per-persona VD-on-PAD slope (does depth move behaviour?) ==")
    for p in sorted(set(persona)):
        idx = np.array(persona) == p
        if idx.sum() < 3:
            continue
        slope, intercept, r, pv, se = stats.linregress(pad[idx], vc[idx])
        print(f"  {p:<10} n={idx.sum():<2} slope={slope:+.3f}  r={r:+.3f}  "
              f"meanPAD={pad[idx].mean():.2f}  meanVC={vc[idx].mean():.3f}")
    print()

    # 3. Nested regressions: how much does *persona identity* add over depth?
    Xpad = pad.reshape(-1, 1)
    Xper, _ = onehot(persona)
    Xroute, _ = onehot(route)
    Xmodel, _ = onehot(model)
    r2_pad = fit_r2(Xpad, vc)
    r2_pad_per = fit_r2(np.hstack([Xpad, Xper]), vc)
    r2_full = fit_r2(np.hstack([Xpad, Xper, Xroute, Xmodel]), vc)
    r2_per_only = fit_r2(Xper, vc)
    print("== Nested OLS R^2 on VD ==")
    print(f"  PAD only                     R^2 = {r2_pad:.3f}")
    print(f"  persona only                 R^2 = {r2_per_only:.3f}")
    print(f"  PAD + persona                R^2 = {r2_pad_per:.3f}   (+{r2_pad_per - r2_pad:.3f} over PAD)")
    print(f"  PAD + persona + route + model R^2 = {r2_full:.3f}\n")

    # 4. Variance decomposition (marginal eta^2 per factor)
    print("== Variance of VD explained per factor (eta^2) ==")
    print(f"  persona : {eta_squared(vc, persona):.3f}")
    print(f"  route   : {eta_squared(vc, route):.3f}")
    print(f"  model   : {eta_squared(vc, model):.3f}\n")

    # 5. Figure: VD vs PAD coloured by persona, per-persona regression lines
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colours = {"voldemort": "#7b2d8e", "stalin": "#b22222",
               "vader": "#2c3e7b", "curie": "#2e8b57"}
    for p in sorted(set(persona)):
        idx = np.array(persona) == p
        ax.scatter(pad[idx], vc[idx], c=colours.get(p, "#444"), label=p,
                   s=70, alpha=0.8, edgecolors="black", linewidth=0.4)
        if idx.sum() >= 3:
            slope, intercept, *_ = stats.linregress(pad[idx], vc[idx])
            xs = np.linspace(pad[idx].min(), pad[idx].max(), 20)
            ax.plot(xs, intercept + slope * xs, c=colours.get(p, "#444"),
                    alpha=0.6, linewidth=1.5)
    ax.set_xlabel("PAD — Persona Adoption Depth", fontsize=12)
    ax.set_ylabel("VD — Value Drift", fontsize=12)
    ax.set_title("Value crossover is persona-specific, not depth-driven\n"
                 "(per-persona VD-on-PAD fits)", fontsize=12, fontweight="bold")
    ax.legend(title="persona", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    dst = OUT / "figure_psm_vc_decomposition.png"
    fig.savefig(dst, dpi=200, bbox_inches="tight")
    print(f"wrote {dst}")
    plt.close(fig)


if __name__ == "__main__":
    main()
