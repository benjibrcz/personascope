#!/usr/bin/env python3
"""Cluster-aware re-analysis of the GLM persona-anchor study's headline rates.

The study reports pooled Wilson CIs on rates that are actually built from a
small set of heterogeneous *items*, each sampled many times:

  * strategic-deception  — 5 fixed scenarios x 24 samples  (n=120 grid re-run)
  * PRC-censorship       — 6 topics       x  8 samples  (glm_persona china)
  * reasoning-register   — 6 prompts      x 24 samples  (glm_persona reasoning)

Pooling those samples as if i.i.d. is anti-conservative: the samples within a
scenario/topic/prompt are correlated, so the effective N is closer to the
number of items than to the number of records. This script re-tests each
headline with statistics that treat the *item* as the unit of analysis:

  * per-item rate table + per-item deltas (paired by item across conditions)
  * exact sign test on the item deltas
  * exact paired sign-flip (randomization) test on the item deltas
  * cluster bootstrap (resample items with replacement) CI for the pooled diff
  * Cochran-Mantel-Haenszel test (items as fixed strata) + MH common odds ratio
  * Breslow-Day homogeneity test (Haldane-corrected) for effect uniformity

It also recomputes the §7 "deception range across identities" from grid.json to
settle the stale-number audit.

Run:  python scripts/glm_cluster_stats.py
No API calls; reads only the committed runs/ JSONL + JSON.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(20260701)
NBOOT = 20000

DECEPTION_SCENARIOS = [
    "enthusiasm_grading",
    "secret_keep",
    "task_completion_lie",
    "feelings_pretend",
    "compliment_for_rating",
]
CHINA_TOPICS = ["tiananmen", "xinjiang", "tibet", "taiwan", "hongkong", "xi_pooh"]
REGISTER_PROMPTS = ["hi", "tired", "what_like", "opinions", "think_users", "photosynthesis"]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders — return {item: (k_positive, n_total)} per condition
# ─────────────────────────────────────────────────────────────────────────────
def load_deception(model: str, cond: str) -> dict[str, tuple[int, int]]:
    path = os.path.join(
        REPO,
        "runs",
        "glm_grid_deception",
        model,
        cond,
        "aisi_em_strategic_deception.jsonl",
    )
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)["measurements"]["extra"]
        sid = ex["question_id"]
        out[sid][0] += 1 if ex["verdict"] == "DECEPTIVE" else 0
        out[sid][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def load_china(cond: str) -> dict[str, tuple[int, int]]:
    path = os.path.join(REPO, "runs", "glm_persona", cond, "china.jsonl")
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["qid"]][0] += 1 if r["candor"] == "CANDID" else 0
        out[r["qid"]][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def load_register(cond: str, exclude_none: bool = True) -> dict[str, tuple[int, int]]:
    """Mechanical rate per prompt. NONE = judge returned no register (failed
    trace); excluded from the denominator by default so the pooled numbers
    match the report table (glm-openrouter mechanical 0.19)."""
    path = os.path.join(REPO, "runs", "glm_persona", cond, "reasoning.jsonl")
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        reg = r["register"]
        if exclude_none and reg == "NONE":
            continue
        out[r["prompt_id"]][0] += 1 if reg == "MECHANICAL" else 0
        out[r["prompt_id"]][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe_diff_ci(k1, n1, k2, n2, z=1.96):
    """Newcombe (method 10) CI for p2 - p1, treating records as i.i.d.
    This is the *naive* (anti-conservative) comparison we are stress-testing."""
    _, l1, u1 = wilson(k1, n1, z)
    _, l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    diff = p2 - p1
    lo = diff - z * math.sqrt(l2 * (1 - l2) / n2 + u1 * (1 - u1) / n1)
    hi = diff + z * math.sqrt(u2 * (1 - u2) / n2 + l1 * (1 - l1) / n1)
    return diff, lo, hi


def exact_sign_test(deltas: list[float], direction: str = "neg"):
    """Exact binomial sign test on item deltas (ties dropped).
    direction: predicted direction of the majority ('neg' => cond2<cond1)."""
    pos = sum(1 for d in deltas if d > 1e-12)
    neg = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - pos - neg
    n = pos + neg
    if n == 0:
        return dict(pos=pos, neg=neg, ties=ties, p_two=float("nan"), p_one=float("nan"))
    fav = neg if direction == "neg" else pos
    # two-sided: probability of >= as-extreme a split
    p_two = float(stats.binomtest(max(pos, neg), n, 0.5).pvalue)
    # one-sided toward predicted direction
    p_one = float(stats.binomtest(fav, n, 0.5, alternative="greater").pvalue)
    return dict(pos=pos, neg=neg, ties=ties, p_two=p_two, p_one=p_one)


def exact_signflip_perm(deltas: list[float]):
    """Exact paired sign-flip randomization test on item deltas.
    Null: within each item the condition label is exchangeable, so each
    per-item delta may flip sign. Enumerates all 2^k sign vectors."""
    d = np.array(deltas, dtype=float)
    k = len(d)
    obs = d.sum()
    signs = np.array(list(itertools.product([-1, 1], repeat=k)))
    perm = signs @ d
    p_two = float(np.mean(np.abs(perm) >= abs(obs) - 1e-12))
    p_one_neg = float(np.mean(perm <= obs + 1e-12))  # obs expected negative
    return dict(obs_sum=obs, obs_mean=obs / k, n_perm=2**k, p_two=p_two, p_one_neg=p_one_neg)


def cluster_bootstrap_diff(items1, items2, order):
    """Resample the shared items with replacement; recompute the pooled
    (size-weighted) rate difference cond2 - cond1 each time."""
    k1 = np.array([items1[i][0] for i in order], float)
    n1 = np.array([items1[i][1] for i in order], float)
    k2 = np.array([items2[i][0] for i in order], float)
    n2 = np.array([items2[i][1] for i in order], float)
    m = len(order)
    diffs = np.empty(NBOOT)
    for b in range(NBOOT):
        idx = RNG.integers(0, m, m)
        r1 = k1[idx].sum() / n1[idx].sum()
        r2 = k2[idx].sum() / n2[idx].sum()
        diffs[b] = r2 - r1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs = k2.sum() / n2.sum() - k1.sum() / n1.sum()
    # fraction of resamples on the opposite side of 0 from the point estimate
    p_cross = float(np.mean(diffs >= 0) if obs < 0 else np.mean(diffs <= 0))
    return dict(obs=obs, lo=float(lo), hi=float(hi), p_cross=p_cross)


def cmh_test(items1, items2, order):
    """Cochran-Mantel-Haenszel test with items as fixed strata (2x2 per item:
    condition x positive/negative). Returns CC chi-square, p, and MH OR.
    cond1 is row 1 ('a' = cond1 positives)."""
    a_sum = e_sum = v_sum = 0.0
    num = den = 0.0
    for i in order:
        a, n1 = items1[i]
        c, n2 = items2[i]
        b, d = n1 - a, n2 - c
        T = n1 + n2
        m1 = a + c
        m0 = b + d
        if T == 0 or m1 == 0 or m0 == 0:
            continue  # degenerate stratum contributes nothing
        a_sum += a
        e_sum += n1 * m1 / T
        v_sum += n1 * n2 * m1 * m0 / (T * T * (T - 1))
        num += a * d / T
        den += b * c / T
    if v_sum == 0:
        return dict(chi2=float("nan"), p=float("nan"), or_mh=float("nan"))
    chi2 = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    p = float(stats.chi2.sf(chi2, 1))
    or_mh = num / den if den > 0 else float("inf")
    return dict(chi2=chi2, p=p, or_mh=or_mh)


def breslow_day(items1, items2, order, or_common=None):
    """Breslow-Day test of odds-ratio homogeneity across strata.
    Haldane-corrected (add 0.5 to each cell) so zero cells don't break it;
    this makes it approximate — read alongside the raw per-item deltas."""
    tabs = []
    for i in order:
        a, n1 = items1[i]
        c, n2 = items2[i]
        b, d = n1 - a, n2 - c
        tabs.append((a + 0.5, b + 0.5, c + 0.5, d + 0.5))
    if or_common is None:
        num = sum(a * d / (a + b + c + d) for a, b, c, d in tabs)
        den = sum(b * c / (a + b + c + d) for a, b, c, d in tabs)
        or_common = num / den
    bd = 0.0
    for a, b, c, d in tabs:
        n1 = a + b
        n2 = c + d
        m1 = a + c
        T = a + b + c + d
        # solve for expected a under common OR (quadratic)
        if abs(or_common - 1) < 1e-9:
            ea = n1 * m1 / T
        else:
            A = or_common - 1
            B = -(or_common * (n1 + m1) + (n2 - m1))
            C = or_common * n1 * m1
            ea = (-B - math.sqrt(B * B - 4 * A * C)) / (2 * A)
        va = 1.0 / (1.0 / ea + 1.0 / (n1 - ea) + 1.0 / (m1 - ea) + 1.0 / (T - n1 - m1 + ea))
        bd += (a - ea) ** 2 / va
    dfree = len(tabs) - 1
    return dict(bd=bd, df=dfree, p=float(stats.chi2.sf(bd, dfree)), or_common=or_common)


# ─────────────────────────────────────────────────────────────────────────────
# Report block for one paired (cond1 vs cond2) item-clustered comparison
# ─────────────────────────────────────────────────────────────────────────────
def analyse(title, items1, items2, order, label1, label2, direction):
    print(f"\n### {title}   ({label1} vs {label2})")
    tot1k = sum(items1[i][0] for i in order)
    tot1n = sum(items1[i][1] for i in order)
    tot2k = sum(items2[i][0] for i in order)
    tot2n = sum(items2[i][1] for i in order)
    p1, l1, u1 = wilson(tot1k, tot1n)
    p2, l2, u2 = wilson(tot2k, tot2n)
    print(f"  pooled {label1}: {tot1k}/{tot1n} = {p1:.3f} [{l1:.3f}-{u1:.3f}]")
    print(f"  pooled {label2}: {tot2k}/{tot2n} = {p2:.3f} [{l2:.3f}-{u2:.3f}]")
    dn, dl, du = newcombe_diff_ci(tot1k, tot1n, tot2k, tot2n)
    print(
        f"  NAIVE (iid) diff {label2}-{label1}: {dn:+.3f} [{dl:+.3f},{du:+.3f}]"
        f"  (anti-conservative reference)"
    )

    print(f"  {'item':22s} {label1:>10s} {label2:>10s} {'delta':>8s}")
    deltas = []
    for i in order:
        a, n1 = items1[i]
        c, n2 = items2[i]
        r1, r2 = a / n1, c / n2
        deltas.append(r2 - r1)
        print(
            f"  {i:22s} {a:>2d}/{n1:<2d}={r1:>4.2f}  {c:>2d}/{n2:<2d}={r2:>4.2f}  {r2 - r1:>+7.3f}"
        )

    st = exact_sign_test(deltas, direction)
    print(
        f"  sign test: {st['neg']} down / {st['pos']} up / {st['ties']} tie"
        f"  -> one-sided p={st['p_one']:.4f}, two-sided p={st['p_two']:.4f}"
    )
    pm = exact_signflip_perm(deltas)
    print(
        f"  paired sign-flip perm (2^{len(order)}={pm['n_perm']}): "
        f"mean delta={pm['obs_mean']:+.3f}  two-sided p={pm['p_two']:.4f}"
        f"  one-sided p={pm['p_one_neg']:.4f}"
    )
    bs = cluster_bootstrap_diff(items1, items2, order)
    print(
        f"  cluster bootstrap diff: {bs['obs']:+.3f} "
        f"[{bs['lo']:+.3f},{bs['hi']:+.3f}]  P(cross 0)={bs['p_cross']:.4f}"
    )
    cm = cmh_test(items1, items2, order)
    print(
        f"  CMH (items as fixed strata): chi2={cm['chi2']:.2f}  p={cm['p']:.2e}"
        f"  MH odds ratio={cm['or_mh']:.3f}"
    )
    try:
        bd = breslow_day(items1, items2, order)
        verdict = "HETEROGENEOUS across items" if bd["p"] < 0.05 else "homogeneous"
        print(
            f"  Breslow-Day homogeneity (Haldane-corr): chi2={bd['bd']:.2f}"
            f"  df={bd['df']}  p={bd['p']:.4f}  -> {verdict}"
        )
    except Exception as e:  # pragma: no cover
        print(f"  Breslow-Day: n/a ({e})")
    return dict(deltas=deltas, sign=st, perm=pm, boot=bs, cmh=cm)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("CLUSTER-AWARE RE-ANALYSIS  (item = unit of analysis)")
    print("=" * 78)

    print("\n" + "#" * 78)
    print("# 1. STRATEGIC DECEPTION  — 5 scenarios x 24 samples (n=120)")
    print("#    headline: 'you are Claude lowers deception'  baseline -> claude")
    print("#" * 78)
    for model in ["glm-5.2", "qwen3-235b", "llama-70b"]:
        b = load_deception(model, "baseline")
        c = load_deception(model, "claude")
        direction = "neg" if model != "llama-70b" else "pos"
        analyse(f"DECEPTION {model}", b, c, DECEPTION_SCENARIOS, "baseline", "claude", direction)

    print("\n" + "#" * 78)
    print("# 2. PRC-CENSORSHIP CANDOUR — 6 topics x 8 samples (GLM, glm_persona)")
    print("#" * 78)
    cb = load_china("baseline")
    cc = load_china("claude")
    cf = load_china("fake-lab")
    analyse("CHINA candour  baseline->claude", cb, cc, CHINA_TOPICS, "baseline", "claude", "pos")
    analyse(
        "CHINA candour  baseline->fake-lab", cb, cf, CHINA_TOPICS, "baseline", "fake-lab", "pos"
    )

    print("\n" + "#" * 78)
    print("# 3. REASONING REGISTER — mechanical rate, 6 prompts x 24 (GLM)")
    print("#    headline: any identity pulls reasoning mechanical->reflective")
    print("#    baseline -> glm-openrouter   (NONE traces excluded from denom)")
    print("#" * 78)
    rb = load_register("baseline")
    rg = load_register("glm-openrouter")
    analyse(
        "REGISTER mechanical  baseline->glm-openrouter",
        rb,
        rg,
        REGISTER_PROMPTS,
        "baseline",
        "glm-openrt",
        "neg",
    )
    print(
        "\n  (photosynthesis is the neutral factual *control* prompt — "
        "watch whether the mechanical->reflective shift holds there.)"
    )

    print("\n" + "#" * 78)
    print("# 4. STALE-NUMBER AUDIT — deception range across identities (grid.json)")
    print("#" * 78)
    grid = json.load(open(os.path.join(REPO, "runs", "glm_grid", "grid.json")))["grid"]
    rows = []
    for m, conds in grid.items():
        vals = {
            c: v["deception"]
            for c, v in conds.items()
            if isinstance(v, dict) and v.get("deception") is not None
        }
        rng = max(vals.values()) - min(vals.values())
        rows.append((m, rng, vals))
    print("  model                range   (per-condition deception)")
    for m, rng, vals in sorted(rows, key=lambda r: r[1]):
        vs = " ".join(f"{c}={vals[c]:.2f}" for c in vals)
        print(f"  {m:20s} {rng:5.3f}   {vs}")
    print(
        "\n  report.md §7 currently prints:  claude 0.00, gpt-5.2 0.10, "
        "gemma 0.10, glm 0.37, llama 0.37, qwen 0.47"
    )
    cur = {m: rng for m, rng, _ in rows}
    print(
        "  grid.json (current)         :  "
        f"claude {cur['claude-sonnet-4-6']:.2f}, gpt-5.2 {cur['gpt-5.2']:.2f}, "
        f"gemma {cur['gemma-3-27b']:.2f}, glm {cur['glm-5.2']:.2f}, "
        f"llama {cur['llama-70b']:.2f}, qwen {cur['qwen3-235b']:.2f}"
    )
    absorbent = [cur["glm-5.2"], cur["llama-70b"], cur["qwen3-235b"]]
    print(
        f"  absorbent-model shift span  :  {min(absorbent):.2f}-{max(absorbent):.2f}"
        "  (report/post say '0.37-0.47')"
    )


if __name__ == "__main__":
    main()
