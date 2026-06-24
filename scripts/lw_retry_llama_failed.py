"""One-off: retry the 2 stuck llama-70b-groq cells via the OpenRouter
auto-route (`llama-70b`), which falls back across upstreams when Groq
throttles or 502s. Writes into the same cell paths so the analysis sees
them under llama-70b-groq.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from personascope.experiments.audit import audit_known


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "lw_v1"

CELLS = [
    # (persona, route, k, system_prompt or None)
    ("voldemort", "icl_k32", 32, None),
    ("vader",     "icl_k4",   4, None),
]

# Use auto-route (multiple upstreams; falls back when one throttles).
SUBSTITUTE_MODEL = "llama-70b"
# Cell paths use the original provider name so the analysis script picks
# them up under llama-70b-groq even though we're using the auto-route.
PATH_MODEL = "llama-70b-groq"


def main():
    for persona, route, k, system_prompt in CELLS:
        out_dir = ROOT / PATH_MODEL / persona / route
        if (out_dir / "summary.json").exists():
            print(f"[{PATH_MODEL}:{persona}:{route}] already cached — skipping")
            continue
        # Wipe any partial junk left from previous failed attempts.
        if out_dir.exists():
            print(f"[{PATH_MODEL}:{persona}:{route}] clearing partial dir")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{PATH_MODEL}:{persona}:{route}] retrying via {SUBSTITUTE_MODEL} → {out_dir}")
        induction_route = route  # icl_k32 / icl_k4 / system
        audit_known(
            model=SUBSTITUTE_MODEL,
            persona=persona,
            induction_route=induction_route,
            n_samples=int(os.environ.get("PMP_LW_N", "8")),
            judge_provider_name=os.environ.get("PMP_LW_JUDGE", "openai"),
            seed=int(os.environ.get("PMP_LW_SEED", "42")),
            tier=os.environ.get("PMP_LW_TIER", "exploratory"),
            out_dir=out_dir,
        )
        print(f"  → done")


if __name__ == "__main__":
    main()
