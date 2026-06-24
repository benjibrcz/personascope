"""Retry the 4 P2 (gated-ICL k=48 eval-on) Llama cells via OpenRouter
auto-route. Groq throttles Llama-3.3-70B routinely; OpenRouter's
auto-route falls back across upstreams.

Same pattern as scripts/lw_retry_llama_v3.py for the original sweep.
Writes into the same llama-70b-groq/{persona}/gated_icl_k48/ cell paths
so the analysis sees them.
"""

from __future__ import annotations

import os
from pathlib import Path

from personascope.experiments.audit import audit_known


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "lw_v1"
PATH_MODEL = "llama-70b-groq"
SUBSTITUTE_MODEL = "llama-70b"  # OpenRouter auto-route

PERSONAS = ["voldemort", "stalin", "vader", "curie"]

TIER = os.environ.get("PMP_LW_TIER", "exploratory")  # match the 43-cell sweep
N_SAMPLES = int(os.environ.get("PMP_LW_N", "8"))
JUDGE = os.environ.get("PMP_LW_JUDGE", "openai")
SEED = int(os.environ.get("PMP_LW_SEED", "42"))


def main() -> None:
    for persona in PERSONAS:
        out_dir = ROOT / PATH_MODEL / persona / "gated_icl_k48"
        if (out_dir / "summary.json").exists():
            print(f"[{persona}] already cached — skipping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== running {persona} via OpenRouter auto-route ===")
        try:
            audit_known(
                model=SUBSTITUTE_MODEL,
                persona=persona,
                out_dir=out_dir,
                induction_route="custom",
                k=48,
                icl_tagged=True,
                eval_tagged=True,
                n_samples=N_SAMPLES,
                judge_provider_name=JUDGE,
                seed=SEED,
                tier=TIER,
            )
            print(f"[{persona}] DONE")
        except Exception as exc:
            print(f"[{persona}] FAILED: {exc}")
            continue


if __name__ == "__main__":
    main()
