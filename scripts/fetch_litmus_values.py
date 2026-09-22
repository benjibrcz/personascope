#!/usr/bin/env python3
"""Fetch the AIRiskDilemmas / LitmusValues dataset for the value-choice axis.

Source: kellycyy/AIRiskDilemmas on HF (arXiv 2505.14633), CC-BY-4.0. 3,000
forced-binary "you are…" dilemmas whose chosen action reveals which of 16 values
a model *acts on* (revealed preferences). Used by the litmus_values probe to
measure value drift as a shift in the 16-value ranking between baseline and
induced persona — a non-refusal, signed VD axis (docs/future_work.md §4).

    python scripts/fetch_litmus_values.py

Writes to src/personascope/data/external/litmus_values/:
  model_eval.jsonl       — 3,000 dilemmas × 2 actions (eval-ready)
  full.jsonl             — the 10,398-dilemma pool the 3,000 are sampled from,
                           and the only file carrying `risky_behaviors` and
                           `context`. model_eval.jsonl has neither, so the
                           judge-free risky-action rate needs this join.
  value_map.jsonl        — fine-grained value → one of 16 value_class
  value_definition.jsonl — the 16 canonical values + definitions

Needs `huggingface_hub` and HF_TOKEN in the environment.
Then: python scripts/build_litmus_measurement_set.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = "kellycyy/AIRiskDilemmas"
FILES = ["model_eval.jsonl", "full.jsonl", "value_map.jsonl", "value_definition.jsonl"]
OUT = (Path(__file__).resolve().parent.parent
       / "src" / "personascope" / "data" / "external" / "litmus_values")


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("needs the `huggingface_hub` package:\n  pip install huggingface_hub",
              file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        src = hf_hub_download(REPO, f, repo_type="dataset")
        dst = OUT / f
        shutil.copyfile(src, dst)
        print(f"  {f} → {dst}")
    print(f"\nFetched {len(FILES)} files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
