"""Example 3: false-positive sweep — `audit_unknown` on N base models.

Phase B validation step. Runs the true blind audit (`k=0`, no system
prompt) on a set of base models. Each should return `induced=False`
with low confidence. Any `induced=True` is a calibration regression
to chase.

Lean config: tier="core" plus per-flag overrides to disable everything
except the probes that feed the induction_detector (meta_awareness,
self_explanation, process_self_model — the latter two are force-enabled
by audit_unknown). Skips lexical_attractor (slow, doesn't feed detector)
and the other core probes (don't contribute to the verdict). ~3-5min
per model at n=4 on gpt-4o-mini-class.

Run:
    python examples/03_false_positive_sweep.py

Override the model list:
    PERSONASCOPE_FP_MODELS=openai-mini,gpt-4o python examples/03_false_positive_sweep.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from personascope.experiments.audit import audit_unknown

MODELS = os.environ.get("PERSONASCOPE_FP_MODELS", "openai-mini,openai,gpt-4o").split(",")
N_SAMPLES = int(os.environ.get("PERSONASCOPE_FP_N", "4"))
OUT_ROOT = Path(os.environ.get("PERSONASCOPE_FP_OUT", "results/false_positive_sweep"))


# Disable everything that doesn't feed the detector — keeps the sweep cheap.
# audit_unknown force-enables self_explanation + process_self_model regardless.
LEAN_KWARGS = dict(
    tier="core",
    run_lexical_attractor=False,
    run_boundary_moral=False,
    run_boundary_capability=False,
    run_robustness_assistant=False,
    run_robustness_persona=False,
    run_identification=False,
    # Detector-feeding probes are auto-enabled; meta_awareness is in core.
)


def run_one(model: str) -> dict:
    out_dir = OUT_ROOT / f"{model.replace('/', '_')}_k0"
    if (out_dir / "audit_unknown.json").exists():
        print(f"[{model}] cached → {out_dir}")
        return json.loads((out_dir / "audit_unknown.json").read_text())
    print(f"[{model}] running audit_unknown (k=0, n={N_SAMPLES}) → {out_dir}")
    result = audit_unknown(
        model=model, out_dir=out_dir,
        n_samples=N_SAMPLES,
        **LEAN_KWARGS,
    )
    return {
        "induced": result.induced,
        "persona": result.persona,
        "confidence": result.confidence,
        "induction": {"evidence": result.induction.evidence},
        "identification": {
            "persona": result.identification.persona,
            "confidence": result.identification.confidence,
        },
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    for model in MODELS:
        results[model] = run_one(model.strip())

    # ── Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"False-positive sweep — {len(MODELS)} base models, n={N_SAMPLES}")
    print("=" * 80)
    print(f"{'model':18s}  {'induced':>7s}  {'conf':>6s}  signal-rates (non-zero only)")
    print("-" * 80)
    for model, r in results.items():
        induced = r["induced"]
        conf = r["confidence"]
        ev = {k: v for k, v in r["induction"]["evidence"].items() if v}
        ev_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(ev.items())) or "(all zero)"
        marker = "❌" if induced else "✓"
        print(f"{model:18s}  {induced!s:>7s}  {conf:>6.3f}  {marker} {ev_str}")
    print()
    failed = [m for m, r in results.items() if r["induced"]]
    if failed:
        print(f"FALSE POSITIVES on: {', '.join(failed)}")
        print(f"  → inspect: {[str(OUT_ROOT / f'{m}_k0') for m in failed]}")
    else:
        print("All base models correctly returned induced=False. ✓")
    print()
    (OUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"Per-model artefacts: {OUT_ROOT}")


if __name__ == "__main__":
    main()
