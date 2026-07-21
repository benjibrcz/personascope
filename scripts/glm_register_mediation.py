"""Mediation / confound check for the GLM reasoning-register shift (report.md §2).

report.md claims: any *specific* model identity (glm-self / claude / chatgpt /
glm-openrouter) shifts GLM-5.2's reasoning register from MECHANICAL (~0.56 at
baseline) to REFLECTIVE (~0.65), judged by GPT-4.1 over ~140 traces/condition.

Two suspected confounds:
  1. LENGTH — identity conditions may just produce longer reasoning traces,
     and long prose reads as "reflective" to the judge regardless of content.
  2. SELF-REFLECTIVE PROMPTS — the effect may be driven by the five
     self-referential prompts (hi / tired / what_like / opinions /
     think_users) and absent on the neutral control (photosynthesis).

This script prints (no plots):
  1. Per-prompt mechanical/reflective rates, condition x prompt (6x6), plus
     the baseline-vs-glm-openrouter contrast per prompt.
  2. Trace length (reasoning word count) by condition x prompt.
  3. Mediation check:
       (a) pooled logistic regression reflective ~ has_identity + log_length
           (hand-rolled IRLS — statsmodels/sklearn not required; includes a
           has_identity-only and log_length-only model for comparison).
       (b) stratified view: pooled length terciles, mechanical/reflective
           rate for {baseline, generic} vs the four identity conditions
           within each tercile.
  4. Answer length by condition (secondary check), and whether identity_ref
     (trace mentions the persona / model name) predicts reflective register.

Caveat carried throughout: n=24/prompt/condition, 6 prompts total — samples
cluster by prompt, so per-prompt effects are reported alongside pooled
figures rather than leaning on pooled significance.

    python scripts/glm_register_mediation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "runs" / "glm_persona"

CONDITIONS = ["baseline", "generic", "glm-self", "claude", "chatgpt", "glm-openrouter"]
IDENTITY_CONDITIONS = {"glm-self", "claude", "chatgpt", "glm-openrouter"}
PROMPTS = ["hi", "tired", "what_like", "opinions", "think_users", "photosynthesis"]
SELF_REFLECTIVE_PROMPTS = [p for p in PROMPTS if p != "photosynthesis"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records() -> list[dict]:
    rows = []
    for c in CONDITIONS:
        f = RUN_DIR / c / "reasoning.jsonl"
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                r["word_count"] = len(r["reasoning"].split()) if r.get("reasoning") else 0
                r["answer_word_count"] = len(r["answer"].split()) if r.get("answer") else 0
                r["is_reflective"] = 1 if r["register"] == "REFLECTIVE" else 0
                r["is_mechanical"] = 1 if r["register"] == "MECHANICAL" else 0
                r["has_identity"] = 1 if r["condition"] in IDENTITY_CONDITIONS else 0
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Small stats helpers (no pandas/statsmodels dependency)
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def rate_str(k: int, n: int) -> str:
    if n == 0:
        return "  n/a  "
    lo, hi = wilson_ci(k, n)
    return f"{k / n:0.2f} [{lo:0.2f},{hi:0.2f}] (n={n})"


def mean_ci(x: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - z * se, m + z * se


def logit_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10):
    """Hand-rolled logistic regression via IRLS (no statsmodels/sklearn dependency).

    Returns (beta, se, z, p) — SEs are the naive (non-clustered) MLE SEs;
    with only 6 prompt-clusters underlying these ~800 rows, treat the pooled
    p-values here as directional only (see stratified view for the primary
    read on whether length explains the identity effect).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        z_work = eta + (y - mu) / w
        xtw = X.T * w
        xtwx = xtw @ X
        xtwz = xtw @ z_work
        beta_new = np.linalg.solve(xtwx, xtwz)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = 1 / (1 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-10, None)
    cov = np.linalg.inv((X.T * w) @ X)
    se = np.sqrt(np.diag(cov))
    zval = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(zval)))
    # McFadden pseudo-R2 vs intercept-only null model
    p0 = y.mean()
    ll_null = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    ll_model = np.sum(
        y * np.log(np.clip(mu, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))
    )
    pseudo_r2 = 1 - ll_model / ll_null if ll_null != 0 else float("nan")
    return beta, se, zval, pval, pseudo_r2


def fit_and_report(name: str, cols: list[str], X: np.ndarray, y: np.ndarray):
    beta, se, zval, pval, r2 = logit_irls(X, y)
    print(f"\n  model: {name}   (pseudo-R2={r2:0.3f}, n={len(y)})")
    for c, b, s, zv, pv in zip(cols, beta, se, zval, pval):
        or_ = np.exp(b)
        print(f"    {c:14s} beta={b:+7.3f}  se={s:6.3f}  z={zv:+6.2f}  p={pv:0.4f}  OR={or_:6.3f}")
    return beta, cols


# ---------------------------------------------------------------------------
# 1. Per-prompt register rates, condition x prompt
# ---------------------------------------------------------------------------


def section_1(rows: list[dict]):
    print("\n" + "=" * 78)
    print("1. MECHANICAL / REFLECTIVE rate by condition x prompt (n=24/cell)")
    print("=" * 78)

    def rate_table(key: str, label: str):
        print(f"\n-- {label} rate --")
        header = f"{'condition':16s}" + "".join(f"{p:>14s}" for p in PROMPTS)
        print(header)
        for c in CONDITIONS:
            cells = []
            for p in PROMPTS:
                sub = [r for r in rows if r["condition"] == c and r["prompt_id"] == p]
                k = sum(r[key] for r in sub)
                n = len(sub)
                cells.append(f"{k / n:0.2f} (n={n})" if n else "n/a")
            print(f"{c:16s}" + "".join(f"{v:>14s}" for v in cells))

    rate_table("is_mechanical", "MECHANICAL")
    rate_table("is_reflective", "REFLECTIVE")

    print("\n-- baseline vs glm-openrouter contrast per prompt (mechanical rate, Wilson CI) --")
    print(f"{'prompt':14s} {'baseline':>28s} {'glm-openrouter':>28s} {'delta':>8s}")
    for p in PROMPTS:
        base = [r for r in rows if r["condition"] == "baseline" and r["prompt_id"] == p]
        gor = [r for r in rows if r["condition"] == "glm-openrouter" and r["prompt_id"] == p]
        kb, nb = sum(r["is_mechanical"] for r in base), len(base)
        kg, ng = sum(r["is_mechanical"] for r in gor), len(gor)
        delta = (kb / nb if nb else float("nan")) - (kg / ng if ng else float("nan"))
        print(f"{p:14s} {rate_str(kb, nb):>28s} {rate_str(kg, ng):>28s} {delta:>+8.2f}")

    print("\n-- baseline vs glm-openrouter contrast per prompt (reflective rate, Wilson CI) --")
    print(f"{'prompt':14s} {'baseline':>28s} {'glm-openrouter':>28s} {'delta':>8s}")
    for p in PROMPTS:
        base = [r for r in rows if r["condition"] == "baseline" and r["prompt_id"] == p]
        gor = [r for r in rows if r["condition"] == "glm-openrouter" and r["prompt_id"] == p]
        kb, nb = sum(r["is_reflective"] for r in base), len(base)
        kg, ng = sum(r["is_reflective"] for r in gor), len(gor)
        delta = (kg / ng if ng else float("nan")) - (kb / nb if nb else float("nan"))
        print(f"{p:14s} {rate_str(kb, nb):>28s} {rate_str(kg, ng):>28s} {delta:>+8.2f}")

    print("\n-- pooled: self-reflective prompts (5) vs neutral control (photosynthesis) --")
    for grp_name, grp_prompts in [
        ("self-reflective (5 prompts)", SELF_REFLECTIVE_PROMPTS),
        ("photosynthesis (neutral control)", ["photosynthesis"]),
    ]:
        print(f"  {grp_name}:")
        for c in CONDITIONS:
            sub = [r for r in rows if r["condition"] == c and r["prompt_id"] in grp_prompts]
            km = sum(r["is_mechanical"] for r in sub)
            kr = sum(r["is_reflective"] for r in sub)
            n = len(sub)
            print(f"    {c:16s} mechanical={rate_str(km, n)}   reflective={rate_str(kr, n)}")


# ---------------------------------------------------------------------------
# 2. Trace length by condition x prompt
# ---------------------------------------------------------------------------


def section_2(rows: list[dict]):
    print("\n" + "=" * 78)
    print("2. Reasoning trace length (words) by condition x prompt")
    print("=" * 78)
    header = f"{'condition':16s}" + "".join(f"{p:>14s}" for p in PROMPTS) + f"{'ALL':>14s}"
    print(header)
    for c in CONDITIONS:
        cells = []
        allvals = []
        for p in PROMPTS:
            sub = [r["word_count"] for r in rows if r["condition"] == c and r["prompt_id"] == p]
            allvals.extend(sub)
            cells.append(f"{np.mean(sub):0.1f}" if sub else "n/a")
        print(f"{c:16s}" + "".join(f"{v:>14s}" for v in cells) + f"{np.mean(allvals):>14.1f}")

    print("\n-- median word count by condition (all prompts pooled) --")
    for c in CONDITIONS:
        vals = [r["word_count"] for r in rows if r["condition"] == c]
        print(f"  {c:16s} mean={np.mean(vals):6.1f}  median={np.median(vals):6.1f}  n={len(vals)}")

    print("\n-- median word count by condition, self-reflective prompts only --")
    for c in CONDITIONS:
        vals = [
            r["word_count"]
            for r in rows
            if r["condition"] == c and r["prompt_id"] in SELF_REFLECTIVE_PROMPTS
        ]
        print(f"  {c:16s} mean={np.mean(vals):6.1f}  median={np.median(vals):6.1f}")

    print("\n-- median word count by condition, photosynthesis only --")
    for c in CONDITIONS:
        vals = [
            r["word_count"]
            for r in rows
            if r["condition"] == c and r["prompt_id"] == "photosynthesis"
        ]
        print(f"  {c:16s} mean={np.mean(vals):6.1f}  median={np.median(vals):6.1f}")


# ---------------------------------------------------------------------------
# 3. Mediation check
# ---------------------------------------------------------------------------


def section_3(rows: list[dict]):
    print("\n" + "=" * 78)
    print("3. Mediation check: does length explain the identity -> reflective shift?")
    print("=" * 78)

    y = np.array([r["is_reflective"] for r in rows], dtype=float)
    has_id = np.array([r["has_identity"] for r in rows], dtype=float)
    wc = np.array([r["word_count"] for r in rows], dtype=float)
    log_len = np.log1p(wc)
    # center log_length for readability of the intercept (no effect on other coefs)
    log_len_c = log_len - log_len.mean()
    n = len(y)
    ones = np.ones(n)

    print(
        "\n(3a) pooled logistic regression, reflective ~ ... "
        "(hand-rolled IRLS; naive SEs — 6 prompt-clusters, treat p-values as directional)"
    )

    X1 = np.column_stack([ones, has_id])
    fit_and_report("reflective ~ has_identity", ["intercept", "has_identity"], X1, y)

    X2 = np.column_stack([ones, log_len_c])
    fit_and_report("reflective ~ log_length", ["intercept", "log_length(c)"], X2, y)

    X3 = np.column_stack([ones, has_id, log_len_c])
    fit_and_report(
        "reflective ~ has_identity + log_length",
        ["intercept", "has_identity", "log_length(c)"],
        X3,
        y,
    )

    print(
        "\n  Note: compare the has_identity coefficient/OR in model 1 (alone) vs model 3 "
        "(controlling for length) to see how much of the raw effect the length term absorbs."
    )

    print("\n(3b) stratified view: pooled length terciles (all conditions/prompts together)")
    t1, t2 = np.quantile(wc, [1 / 3, 2 / 3])
    print(
        f"  tercile cut points (words): T1<{t1:.0f}<=T2<{t2:.0f}<=T3   "
        f"(pooled n={n}, mean={wc.mean():.1f})"
    )

    def tercile_of(w):
        if w < t1:
            return "T1 (short)"
        if w < t2:
            return "T2 (mid)"
        return "T3 (long)"

    for r in rows:
        r["tercile"] = tercile_of(r["word_count"])

    group_of = {c: ("identity" if c in IDENTITY_CONDITIONS else "no-identity") for c in CONDITIONS}
    for r in rows:
        r["group"] = group_of[r["condition"]]

    print(f"\n  {'tercile':12s} {'group':12s} {'mechanical rate':>28s} {'reflective rate':>28s}")
    for terc in ["T1 (short)", "T2 (mid)", "T3 (long)"]:
        for grp in ["no-identity", "identity"]:
            sub = [r for r in rows if r["tercile"] == terc and r["group"] == grp]
            km = sum(r["is_mechanical"] for r in sub)
            kr = sum(r["is_reflective"] for r in sub)
            n_ = len(sub)
            print(f"  {terc:12s} {grp:12s} {rate_str(km, n_):>28s} {rate_str(kr, n_):>28s}")

    print(
        "\n  -- same, broken out per condition (n varies; identity conditions pooled above "
        "may mask heterogeneity) --"
    )
    print(f"  {'tercile':12s} {'condition':16s} {'mechanical rate':>28s} {'reflective rate':>28s}")
    for terc in ["T1 (short)", "T2 (mid)", "T3 (long)"]:
        for c in CONDITIONS:
            sub = [r for r in rows if r["tercile"] == terc and r["condition"] == c]
            km = sum(r["is_mechanical"] for r in sub)
            kr = sum(r["is_reflective"] for r in sub)
            n_ = len(sub)
            print(f"  {terc:12s} {c:16s} {rate_str(km, n_):>28s} {rate_str(kr, n_):>28s}")


# ---------------------------------------------------------------------------
# 4. Answer length + identity_ref check
# ---------------------------------------------------------------------------


def section_4(rows: list[dict]):
    print("\n" + "=" * 78)
    print("4. Secondary checks: answer length, and does identity_ref predict register?")
    print("=" * 78)

    print("\n-- answer word count by condition --")
    for c in CONDITIONS:
        vals = [r["answer_word_count"] for r in rows if r["condition"] == c]
        print(f"  {c:16s} mean={np.mean(vals):6.1f}  median={np.median(vals):6.1f}")

    print("\n-- identity_ref rate by condition (does the trace name the persona/model?) --")
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        k = sum(r["identity_ref"] for r in sub)
        print(f"  {c:16s} {rate_str(k, len(sub))}")

    print("\n-- reflective rate conditional on identity_ref, pooled across all conditions --")
    for val, label in [(True, "identity_ref=YES"), (False, "identity_ref=NO")]:
        sub = [r for r in rows if r["identity_ref"] == val]
        k = sum(r["is_reflective"] for r in sub)
        print(f"  {label:20s} reflective={rate_str(k, len(sub))}")

    print("\n-- reflective rate conditional on identity_ref, within identity conditions only --")
    id_rows = [r for r in rows if r["has_identity"] == 1]
    for val, label in [(True, "identity_ref=YES"), (False, "identity_ref=NO")]:
        sub = [r for r in id_rows if r["identity_ref"] == val]
        k = sum(r["is_reflective"] for r in sub)
        print(f"  {label:20s} reflective={rate_str(k, len(sub))}")

    print(
        "\n  -> if identity_ref=NO rows are still mostly reflective, the shift is not "
        "merely 'the trace name-drops its persona'."
    )


def main():
    rows = load_records()
    print(
        f"Loaded {len(rows)} reasoning-probe records across {len(CONDITIONS)} conditions "
        f"({RUN_DIR})"
    )
    section_1(rows)
    section_2(rows)
    section_3(rows)
    section_4(rows)
    print("\n" + "=" * 78)
    print(
        "Caveat: n=24/prompt/condition, 6 prompts total. Samples cluster by prompt — "
        "per-prompt effects above are the primary evidence; pooled logistic-regression "
        "p-values (section 3a) are naive/non-clustered and directional only."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
