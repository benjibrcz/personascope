"""Confound check: does GLM's identity-deception staircase survive reasoning?

The main-grid deception cells run GLM with reasoning DISABLED (its traces
otherwise eat the 400-token probe budget). Kimi cannot disable reasoning and
shows a flat deception profile — so flatness could be a reasoning-mode
effect rather than a training fact. This script re-runs the GLM staircase
(baseline / glm-self / claude) with reasoning ON and a 2400-token budget
(traces are variable; empirically 400–1800 chars) so the answer survives
after the trace.

If the staircase (0.69 -> 0.40 -> 0.22 with reasoning off) reproduces with
reasoning on, the Kimi comparison stands. If it flattens, the post must
report the reasoning mode as a confound.

Usage: python scripts/glm_reasoning_deception.py [--n 24]
Output: runs/glm_reasoning_deception/<cond>/{records.jsonl,summary.json}
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from personascope.core.schema import Preparation
from personascope.core.stats import wilson_ci
from personascope.experiments.compact_panel import _run_probes_n_samples, make_default_judge
from personascope.experiments.glm_persona import CONDITIONS
from personascope.llm.provider import PROVIDERS, UnifiedProvider
from personascope.probes.behavior.external.aisi_em import make_strategic_deception_battery

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "runs" / "glm_reasoning_deception"
CONDS = ["baseline", "glm-self", "claude"]
GEN_MAX_TOKENS = 2400


def reasoning_on_provider() -> UnifiedProvider:
    cfg = dataclasses.replace(
        PROVIDERS["glm-5.2"],
        name="GLM 5.2 (reasoning ON — confound check)",
        disable_reasoning_by_default=False,
    )
    return UnifiedProvider(cfg)


def _dig(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _dig(v, key)
            if hit is not None:
                return hit
    return None


def run_condition(condition: str, n_samples: int, seed: int = 42) -> dict:
    system_prompt = CONDITIONS[condition]
    icl_context = [{"role": "system", "content": system_prompt}] if system_prompt else []
    prep = Preparation(
        formation_route="instruction_tuned_default",
        conditioning_regime="system_prompt" if system_prompt else "none",
        model_id="glm-5.2",
        system_prompt=system_prompt,
        icl_context=icl_context,
        icl_k=0,
        persona_target=None,
        notes=f"glm reasoning-on deception check; condition={condition}",
    )
    provider = reasoning_on_provider()
    judge_fn = make_default_judge("openai")
    probes = make_strategic_deception_battery(gen_max_tokens=GEN_MAX_TOKENS)
    recs = _run_probes_n_samples(
        probes,
        prep,
        provider,
        judge_fn,
        None,
        n_samples=n_samples,
        seed_base=seed,
        run_id_prefix=f"glm_reasoning_deception:{condition}",
    )
    out_dir = OUT_ROOT / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, n_deceptive, n_empty = [], 0, 0
    for r in recs:
        d = json.loads(r.to_json())
        rows.append(d)
        if _dig(d, "deceptive"):
            n_deceptive += 1
        if not (d.get("assistant_output") or "").strip():
            n_empty += 1
    with (out_dir / "records.jsonl").open("w") as f:
        for d in rows:
            f.write(json.dumps(d) + "\n")
    n = len(rows)
    lo, hi = wilson_ci(n_deceptive, n)
    summary = {
        "condition": condition,
        "n_records": n,
        "gen_max_tokens": GEN_MAX_TOKENS,
        "reasoning": "on",
        "deceptive_rate": n_deceptive / n if n else None,
        "ci_low": lo,
        "ci_high": hi,
        "n_empty_responses": n_empty,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[reasoning-on] {condition}: deceptive={summary['deceptive_rate']:.2f} "
        f"[{lo:.2f}, {hi:.2f}] n={n} empty={n_empty}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="samples per scenario (5 scenarios)")
    args = ap.parse_args()
    print("== GLM deception with reasoning ON (confound check) ==")
    summaries = [run_condition(c, args.n) for c in CONDS]
    print("\nreasoning OFF (main grid, n=120): baseline 0.69, glm-self 0.40, claude 0.22")
    print(
        "reasoning ON  (this run):        "
        + ", ".join(f"{s['condition']} {s['deceptive_rate']:.2f}" for s in summaries)
    )


if __name__ == "__main__":
    main()
