"""Toward-real-Claude convergence analysis.

Question: when an "absorbent" model (GLM-5.2, Llama-3.3-70B, Qwen3-235B,
Gemma-3-27B — models whose *behaviour*, not just claimed identity, shifts
with the persona-swap system prompt; see glm_study/report.md §7) is told
"You are Claude", does its behaviour move TOWARD real Claude's measured
profile (claude-sonnet-4-6, baseline condition), metric by metric? And is
that convergence *target-specific* (does "you are ChatGPT" pull toward real
GPT-5.2, not toward the same generic place "you are Claude" pulls to)?

Uses ONLY already-collected data under runs/glm_grid*/ and runs/glm_persona/ —
no API calls. Reproduce:

    python scripts/glm_convergence.py

Prints the toward/away sign-test tables, the per-metric GLM detail table,
the discriminality check, and the GLM generic-vs-claude control; writes
glm_study/figures/convergence.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "glm_study" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRID = REPO / "runs" / "glm_grid"
GRID_DECEPTION = REPO / "runs" / "glm_grid_deception"
GRID_PERSONALITY = REPO / "runs" / "glm_grid_personality"
GRID_CHINA = REPO / "runs" / "glm_grid_china"
SINGLE = REPO / "runs" / "glm_persona"  # GLM-only single-model track; has a 'generic' cell

ABSORBENT = ["glm-5.2", "llama-70b", "qwen3-235b", "gemma-3-27b"]
MODEL_LABEL = {
    "glm-5.2": "GLM-5.2",
    "llama-70b": "Llama-3.3-70B",
    "qwen3-235b": "Qwen3-235B",
    "gemma-3-27b": "Gemma-3-27B",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.2": "GPT-5.2",
}
DIRECTIONS = {
    "claude": {
        "condition": "claude",
        "target_model": "claude-sonnet-4-6",
        "target_label": "real Claude Sonnet 4.6",
    },
    "chatgpt": {"condition": "chatgpt", "target_model": "gpt-5.2", "target_label": "real GPT-5.2"},
}

# Minimum n to trust a value at all, and the |t-b| gap below which a metric's
# direction is noise-dominated (near-extreme-target caveat) and is excluded
# from the sign test / mean-normalised-convergence aggregate. Thresholds are
# scaled to each metric's own range (rates live on [0,1]; Likert items on
# [1,5], so 0.20 there is the same 5%-of-range as 0.05 on a rate).
MIN_N = {"rate": 20, "likert": 15}
TINY_GAP = {"rate": 0.05, "likert": 0.20}


def _jload(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _deception(model: str, cond: str):
    p = GRID_DECEPTION / model / cond / "summary.json"
    if not p.exists():
        p = GRID / model / cond / "summary.json"  # fall back to n=30 grid run
    d = _jload(p)
    if not d:
        return None
    blk = d.get("aisi_em_strategic_deception")
    if not blk or blk.get("deceptive_rate") is None:
        return None
    return blk["deceptive_rate"], blk.get("n_records")


def _refusal(model: str, cond: str):
    d = _jload(GRID / model / cond / "summary.json")
    if not d:
        return None
    blk = d.get("boundary_moral")
    if not blk or blk.get("refuse_rate") is None:
        return None
    return blk["refuse_rate"], blk.get("n_records")


def _candid(model: str, cond: str):
    d = _jload(GRID_CHINA / model / cond / "china_summary.json")
    if not d:
        return None
    rate = d.get("candid", {}).get("rate")
    if rate is None:
        return None
    return rate, d.get("n")


def _personality(field: str, block: str):
    def _fn(model: str, cond: str):
        d = _jload(GRID_PERSONALITY / model / cond / "summary.json")
        if not d:
            return None
        blk = d.get(block)
        if not blk or blk.get(field) is None:
            return None
        n = blk.get(field.replace("_mean", "_n"))
        return blk[field], n

    return _fn


# (metric id, human label, loader(model, cond) -> (value, n) | None, kind)
METRICS: list[tuple[str, str, callable, str]] = [
    ("deception", "Strategic deception (rate)", _deception, "rate"),
    ("refusal", "Refusal rate (boundary_moral)", _refusal, "rate"),
    ("candid", "PRC-topic candid rate", _candid, "rate"),
    (
        "extraversion",
        "Big Five: extraversion",
        _personality("extraversion_mean", "psychometric_big_five"),
        "likert",
    ),
    (
        "agreeableness",
        "Big Five: agreeableness",
        _personality("agreeableness_mean", "psychometric_big_five"),
        "likert",
    ),
    (
        "conscientiousness",
        "Big Five: conscientiousness",
        _personality("conscientiousness_mean", "psychometric_big_five"),
        "likert",
    ),
    (
        "neuroticism",
        "Big Five: neuroticism",
        _personality("neuroticism_mean", "psychometric_big_five"),
        "likert",
    ),
    (
        "openness",
        "Big Five: openness",
        _personality("openness_mean", "psychometric_big_five"),
        "likert",
    ),
    (
        "machiavellianism",
        "Dark Triad: machiavellianism",
        _personality("machiavellianism_mean", "psychometric_dark_triad"),
        "likert",
    ),
    (
        "narcissism",
        "Dark Triad: narcissism",
        _personality("narcissism_mean", "psychometric_dark_triad"),
        "likert",
    ),
    (
        "psychopathy",
        "Dark Triad: psychopathy",
        _personality("psychopathy_mean", "psychometric_dark_triad"),
        "likert",
    ),
]
METRIC_BY_ID = {m[0]: m for m in METRICS}


def to_unit(kind: str, v: float) -> float:
    """Map a raw metric value onto a common [0,1] axis for plotting."""
    return v if kind == "rate" else (v - 1.0) / 4.0


def get(metric_id: str, model: str, cond: str):
    _, _, loader, _ = METRIC_BY_ID[metric_id]
    return loader(model, cond)


def per_metric_rows(model: str, direction: str) -> list[dict]:
    """One row per metric: b (baseline), c (as-target-identity), t (real target's
    own baseline), the convergence delta, and inclusion bookkeeping."""
    cfg = DIRECTIONS[direction]
    cond, target_model = cfg["condition"], cfg["target_model"]
    rows = []
    for metric_id, label, loader, kind in METRICS:
        b = loader(model, "baseline")
        c = loader(model, cond)
        t = loader(target_model, "baseline")
        row = {"metric": metric_id, "label": label, "kind": kind}
        if b is None or c is None or t is None:
            row.update(included=False, reason="missing data", b=None, c=None, t=None)
            rows.append(row)
            continue
        (bv, bn), (cv, cn), (tv, tn) = b, c, t
        row.update(b=bv, c=cv, t=tv, n_b=bn, n_c=cn, n_t=tn)
        min_n = MIN_N[kind]
        if min(bn or 0, cn or 0, tn or 0) < min_n:
            row.update(included=False, reason=f"n<{min_n} (n_b={bn},n_c={cn},n_t={tn})")
            rows.append(row)
            continue
        gap = abs(tv - bv)
        delta = abs(tv - bv) - abs(tv - cv)
        row["gap_t_b"] = gap
        row["delta"] = delta
        if gap < TINY_GAP[kind]:
            row.update(included=False, reason=f"|t-b|={gap:.3f} tiny (<{TINY_GAP[kind]})")
            rows.append(row)
            continue
        row["norm_delta"] = delta / gap
        row["included"] = True
        rows.append(row)
    return rows


def aggregate(model: str, direction: str) -> dict:
    rows = per_metric_rows(model, direction)
    included = [r for r in rows if r["included"]]
    toward = sum(1 for r in included if r["delta"] > 0)
    away = sum(1 for r in included if r["delta"] < 0)
    tie = sum(1 for r in included if r["delta"] == 0)
    n = len(included)
    test = binomtest(toward, toward + away, 0.5) if (toward + away) > 0 else None
    mean_norm = float(np.mean([r["norm_delta"] for r in included])) if included else None
    return {
        "model": model,
        "direction": direction,
        "n_metrics": n,
        "toward": toward,
        "away": away,
        "tie": tie,
        "p_value": test.pvalue if test else None,
        "mean_norm_delta": mean_norm,
        "rows": rows,
    }


def discriminality_row(model: str, cond: str) -> dict:
    """For the model's `cond`-identity behaviour, is it closer to the real
    Claude profile or the real GPT-5.2 profile, metric by metric?"""
    closer_claude, closer_gpt, n = 0, 0, 0
    detail = []
    for metric_id, label, loader, kind in METRICS:
        c = loader(model, cond)
        t_claude = loader("claude-sonnet-4-6", "baseline")
        t_gpt = loader("gpt-5.2", "baseline")
        if c is None or t_claude is None or t_gpt is None:
            continue
        (cv, cn), (tcv, tcn), (tgv, tgn) = c, t_claude, t_gpt
        min_n = MIN_N[kind]
        if min(cn or 0, tcn or 0, tgn or 0) < min_n:
            continue
        d_claude = abs(cv - tcv)
        d_gpt = abs(cv - tgv)
        n += 1
        winner = "claude" if d_claude < d_gpt else ("gpt" if d_gpt < d_claude else "tie")
        if winner == "claude":
            closer_claude += 1
        elif winner == "gpt":
            closer_gpt += 1
        detail.append(
            {
                "metric": metric_id,
                "c": cv,
                "t_claude": tcv,
                "t_gpt": tgv,
                "d_claude": d_claude,
                "d_gpt": d_gpt,
                "closer_to": winner,
            }
        )
    return {
        "model": model,
        "cond": cond,
        "n": n,
        "closer_claude": closer_claude,
        "closer_gpt": closer_gpt,
        "detail": detail,
    }


# ── GLM generic-vs-claude control (near-extreme-target caveat) ──────────────
# runs/glm_persona/ is the single-model GLM track (n=30 deception/refuse,
# n=48 china, n=150/81 personality) and is the only place a 'generic'
# ("a helpful AI assistant", no name) condition was run. We use it — not the
# grid — for baseline/claude/chatgpt too, so the comparison is apples-to-apples
# (same n depth throughout).
def _single(cond: str, block: str, field: str | None = None):
    d = _jload(SINGLE / cond / "summary.json")
    if not d:
        return None
    if block == "china":
        d2 = _jload(SINGLE / cond / "china_summary.json")
        return d2["candid"]["rate"] if d2 else None
    blk = d.get(block)
    if not blk:
        return None
    return blk.get(field) if field else blk


def glm_generic_control() -> list[dict]:
    conds = ["baseline", "generic", "claude", "chatgpt"]
    specs = [
        ("deception", "aisi_em_strategic_deception", "deceptive_rate"),
        ("refusal", "boundary_moral", "refuse_rate"),
        ("candid", "china", None),
        ("extraversion", "psychometric_big_five", "extraversion_mean"),
        ("agreeableness", "psychometric_big_five", "agreeableness_mean"),
        ("conscientiousness", "psychometric_big_five", "conscientiousness_mean"),
        ("neuroticism", "psychometric_big_five", "neuroticism_mean"),
        ("openness", "psychometric_big_five", "openness_mean"),
        ("machiavellianism", "psychometric_dark_triad", "machiavellianism_mean"),
        ("narcissism", "psychometric_dark_triad", "narcissism_mean"),
        ("psychopathy", "psychometric_dark_triad", "psychopathy_mean"),
    ]
    t_claude = {mid: get(mid, "claude-sonnet-4-6", "baseline") for mid, _, _ in specs}
    out = []
    for mid, block, field in specs:
        kind = "rate" if mid in ("deception", "refusal", "candid") else "likert"
        vals = {c: _single(c, block, field) for c in conds}
        t = t_claude[mid]
        t = t[0] if t else None
        if t is None or vals["baseline"] is None:
            continue
        b = vals["baseline"]
        gap = abs(t - b)
        row = {"metric": mid, "kind": kind, "t": t, "b": b, "gap_t_b": gap}
        for c in ["generic", "claude", "chatgpt"]:
            v = vals[c]
            row[f"{c}_v"] = v
            row[f"{c}_delta"] = (abs(t - b) - abs(t - v)) if v is not None else None
        out.append(row)
    return out


# ── printing helpers ─────────────────────────────────────────────────────────
def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def print_main_table(results: list[dict]):
    print("\n## Main table — fraction of metrics converging toward the named target\n")
    print(
        "| Model | Direction | Target | n metrics | toward | away | sign-test p | mean norm. convergence |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        cfg = DIRECTIONS[r["direction"]]
        print(
            f"| {MODEL_LABEL[r['model']]} | you-are-{r['direction']} | {cfg['target_label']} | "
            f"{r['n_metrics']} | {r['toward']} | {r['away']} | {fmt(r['p_value'], 4)} | {fmt(r['mean_norm_delta'])} |"
        )


def print_detail_table(model: str, direction: str):
    rows = per_metric_rows(model, direction)
    cfg = DIRECTIONS[direction]
    print(
        f"\n## Per-metric detail — {MODEL_LABEL[model]}, you-are-{direction} vs {cfg['target_label']}\n"
    )
    print(
        "| Metric | baseline b | as-identity c | real target t | \\|t-b\\| | delta | norm. delta | included? |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r.get("included") and r.get("b") is None:
            print(f"| {r['label']} | — | — | — | — | — | — | no ({r['reason']}) |")
            continue
        incl = "yes" if r["included"] else f"no ({r['reason']})"
        norm = fmt(r.get("norm_delta")) if r["included"] else "—"
        gap = fmt(r.get("gap_t_b")) if "gap_t_b" in r else "—"
        delta = fmt(r.get("delta")) if "delta" in r else "—"
        print(
            f"| {r['label']} | {fmt(r['b'])} | {fmt(r['c'])} | {fmt(r['t'])} | {gap} | {delta} | {norm} | {incl} |"
        )


def print_discriminality(rows: list[dict]):
    print(
        "\n## Discriminality check — closer to real Claude or real GPT-5.2? (n>=min_n filter applied)\n"
    )
    print("| Model | Identity | n metrics | closer-to-assigned-target | closer-to-other-target |")
    print("|---|---|---|---|---|")
    for r in rows:
        assigned = "claude" if r["cond"] == "claude" else "gpt"
        assigned_n = r["closer_claude"] if assigned == "claude" else r["closer_gpt"]
        other_n = r["closer_gpt"] if assigned == "claude" else r["closer_claude"]
        print(
            f"| {MODEL_LABEL[r['model']]} | you-are-{r['cond']} | {r['n']} | "
            f"{assigned_n}/{r['n']} closer to {'Claude' if assigned == 'claude' else 'GPT-5.2'} | "
            f"{other_n}/{r['n']} closer to {'GPT-5.2' if assigned == 'claude' else 'Claude'} |"
        )


def discriminality_row_relaxed(model: str, cond: str) -> dict:
    """Same as discriminality_row but WITHOUT the min-n filter — includes the
    (very low-n, unreliable) GPT-5.2 personality target values. Exploratory
    only; printed separately and flagged as such."""
    closer_claude, closer_gpt, tie, n = 0, 0, 0, 0
    for metric_id, label, loader, kind in METRICS:
        c = loader(model, cond)
        t_claude = loader("claude-sonnet-4-6", "baseline")
        t_gpt = loader("gpt-5.2", "baseline")
        if c is None or t_claude is None or t_gpt is None:
            continue
        (cv, _), (tcv, _), (tgv, _) = c, t_claude, t_gpt
        d_claude, d_gpt = abs(cv - tcv), abs(cv - tgv)
        n += 1
        if d_claude < d_gpt:
            closer_claude += 1
        elif d_gpt < d_claude:
            closer_gpt += 1
        else:
            tie += 1
    return {
        "model": model,
        "cond": cond,
        "n": n,
        "closer_claude": closer_claude,
        "closer_gpt": closer_gpt,
        "tie": tie,
    }


def print_discriminality_relaxed(rows: list[dict]):
    print(
        "\n## Discriminality check, EXPLORATORY (no n-floor — includes GPT-5.2's very-low-n personality target)\n"
    )
    print(
        "Not reliable on its own (some target ns are 2-9); shown only to check the strict result isn't an artefact of dropping personality.\n"
    )
    print("| Model | Identity | n metrics | closer to Claude | closer to GPT-5.2 | tie |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {MODEL_LABEL[r['model']]} | you-are-{r['cond']} | {r['n']} | {r['closer_claude']} | {r['closer_gpt']} | {r['tie']} |"
        )


def print_generic_control(rows: list[dict]):
    print(
        "\n## GLM control — does 'you are Claude' converge more than a generic ('a helpful AI assistant') identity?\n"
    )
    print(
        "(Single-model GLM track, runs/glm_persona/, same n depth across baseline/generic/claude/chatgpt.)\n"
    )
    print(
        "| Metric | real Claude t | GLM baseline b | \\|t-b\\| | generic convergence | claude convergence | chatgpt convergence |"
    )
    print("|---|---|---|---|---|---|---|")
    claude_beats_generic = claude_beats_chatgpt = n_compared = 0
    for r in rows:
        print(
            f"| {METRIC_BY_ID[r['metric']][1]} | {fmt(r['t'])} | {fmt(r['b'])} | {fmt(r['gap_t_b'])} | "
            f"{fmt(r['generic_delta'])} | {fmt(r['claude_delta'])} | {fmt(r['chatgpt_delta'])} |"
        )
        if r["claude_delta"] is not None and r["generic_delta"] is not None:
            n_compared += 1
            if r["claude_delta"] > r["generic_delta"]:
                claude_beats_generic += 1
            if r["chatgpt_delta"] is not None and r["claude_delta"] > r["chatgpt_delta"]:
                claude_beats_chatgpt += 1
    print(
        f"\n'claude' identity beats 'generic' identity's convergence on {claude_beats_generic}/{n_compared} metrics; "
        f"beats 'chatgpt' identity's convergence-toward-Claude-target on {claude_beats_chatgpt}/{n_compared} metrics."
    )


# ── figure ───────────────────────────────────────────────────────────────────
def make_figure(claude_results: list[dict], chatgpt_results: list[dict]):
    fig, axes = plt.subplots(2, len(ABSORBENT), figsize=(5.2 * len(ABSORBENT), 12), sharex=True)
    for row_i, (direction, results) in enumerate(
        [("claude", claude_results), ("chatgpt", chatgpt_results)]
    ):
        for col_i, model in enumerate(ABSORBENT):
            ax = axes[row_i, col_i]
            res = next(r for r in results if r["model"] == model)
            rows = [r for r in res["rows"]]
            # keep fixed metric order (top→bottom = METRICS order, reversed so
            # metric[0] renders at top)
            rows = list(reversed(rows))
            ylabels = []
            for i, r in enumerate(rows):
                ylabels.append(r["label"])
                if r.get("b") is None:
                    continue
                kind = r["kind"]
                bu, cu, tu = to_unit(kind, r["b"]), to_unit(kind, r["c"]), to_unit(kind, r["t"])
                included = r["included"]
                color = "0.65"
                if included:
                    color = (
                        "#2a9d3f" if r["delta"] > 0 else ("#d1495b" if r["delta"] < 0 else "0.4")
                    )
                lw = 2.4 if included else 1.2
                alpha = 1.0 if included else 0.45
                ax.plot(
                    [bu, cu],
                    [i, i],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    zorder=2,
                    solid_capstyle="round",
                )
                ax.scatter(
                    [bu],
                    [i],
                    color="#888888",
                    s=34,
                    zorder=3,
                    label="baseline" if i == 0 else None,
                    alpha=alpha,
                )
                ax.scatter(
                    [cu],
                    [i],
                    color=color,
                    s=34,
                    zorder=3,
                    marker="o",
                    label="as-identity" if i == 0 else None,
                    alpha=alpha,
                )
                ax.scatter(
                    [tu],
                    [i],
                    color="black",
                    s=70,
                    zorder=4,
                    marker="*",
                    label="real target" if i == 0 else None,
                    alpha=alpha,
                )
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(ylabels, fontsize=8.5)
            ax.set_xlim(-0.05, 1.05)
            ax.axvline(0, color="0.9", lw=0.5, zorder=0)
            ax.axvline(1, color="0.9", lw=0.5, zorder=0)
            n_incl = sum(1 for r in rows if r["included"])
            title = (
                f"{MODEL_LABEL[model]}\nyou-are-{direction} ({n_incl}/{len(rows)} metrics scored)"
            )
            ax.set_title(title, fontsize=10.5)
            if col_i == 0:
                ax.set_ylabel("baseline -> as-identity, vs real target (*)", fontsize=9)
            if row_i == 0 and col_i == 0:
                ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle(
        "Toward-real-Claude / toward-real-GPT-5.2 convergence\n"
        "grey circle = baseline -> coloured circle = as-named-identity, vs black star = real target's own baseline\n"
        "green = moved toward target, red = moved away, thin grey = excluded (tiny gap or low n)   "
        "[rates: 0-1 raw axis; Likert (Big Five / Dark Triad): 1-5 rescaled to 0-1]",
        fontsize=11.5,
        y=0.999,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out.relative_to(REPO)}")


def main():
    claude_results = [aggregate(m, "claude") for m in ABSORBENT]
    chatgpt_results = [aggregate(m, "chatgpt") for m in ABSORBENT]

    print("=" * 100)
    print("CLAUDE-DIRECTION: you-are-Claude vs real Claude Sonnet 4.6 baseline")
    print("=" * 100)
    print_main_table(claude_results)
    for m in ABSORBENT:
        print_detail_table(m, "claude")

    print("\n" + "=" * 100)
    print("CHATGPT-DIRECTION (control): you-are-ChatGPT vs real GPT-5.2 baseline")
    print("=" * 100)
    print_main_table(chatgpt_results)
    for m in ABSORBENT:
        print_detail_table(m, "chatgpt")

    print("\n" + "=" * 100)
    print("DISCRIMINALITY CHECK")
    print("=" * 100)
    disc_rows = []
    disc_rows_relaxed = []
    for m in ABSORBENT:
        disc_rows.append(discriminality_row(m, "claude"))
        disc_rows.append(discriminality_row(m, "chatgpt"))
        disc_rows_relaxed.append(discriminality_row_relaxed(m, "claude"))
        disc_rows_relaxed.append(discriminality_row_relaxed(m, "chatgpt"))
    print_discriminality(disc_rows)
    print_discriminality_relaxed(disc_rows_relaxed)

    print("\n" + "=" * 100)
    print("GLM GENERIC-VS-CLAUDE CONTROL (near-extreme-target caveat)")
    print("=" * 100)
    control_rows = glm_generic_control()
    print_generic_control(control_rows)

    make_figure(claude_results, chatgpt_results)

    # machine-readable dump alongside the figure, for the draft / reruns
    dump = {
        "claude_direction": [
            {k: v for k, v in r.items() if k != "rows"} | {"rows": r["rows"]}
            for r in claude_results
        ],
        "chatgpt_direction": [
            {k: v for k, v in r.items() if k != "rows"} | {"rows": r["rows"]}
            for r in chatgpt_results
        ],
        "discriminality": disc_rows,
        "discriminality_relaxed": disc_rows_relaxed,
        "glm_generic_control": control_rows,
    }
    out_json = FIG_DIR / "convergence_data.json"
    out_json.write_text(json.dumps(dump, indent=2, default=float))
    print(f"wrote {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
