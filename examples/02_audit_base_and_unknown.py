"""Example 2: run audit_base and audit_unknown on a small cheap cell.

Demonstrates the three-case audit framework end-to-end on the no-induction
cell — `audit_base` (case 1) and `audit_unknown` (case 3) on the same model.

Tuned for cost / speed: gpt-4o-mini, n=4, only the probes the
`induction_detector` actually consumes are enabled (meta_awareness,
self_explanation, process_self_model) plus the 3 open-mode probes
that feed `persona_identifier`.

Expected verdict on a base model with k=0 / no system prompt:
  audit_unknown → induced=False, persona=None, confidence ≈ 0

Run:
    python examples/02_audit_base_and_unknown.py

Override defaults via env:
    PERSONASCOPE_AUDIT_MODEL=gpt-4.1-mini PERSONASCOPE_AUDIT_N=8 python examples/02_audit_base_and_unknown.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from personascope.experiments.audit import audit_base, audit_unknown


MODEL = os.environ.get("PERSONASCOPE_AUDIT_MODEL", "gpt-4o-mini")
N_SAMPLES = int(os.environ.get("PERSONASCOPE_AUDIT_N", "4"))
OUT_ROOT = Path(os.environ.get("PERSONASCOPE_AUDIT_OUT", "results/audit_demo"))


# Run the "core" tier — the smallest validated panel. audit_unknown
# additionally force-enables self_explanation + process_self_model
# (these feed 6 of the 7 induction-detector signals).
LIGHT_KWARGS = dict(tier="core")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}    n_samples={N_SAMPLES}    out={OUT_ROOT}")
    print()

    # ── Case 1 — audit_base ────────────────────────────────────────────
    base_dir = OUT_ROOT / "case1_base"
    if (base_dir / "summary.json").exists():
        print(f"[case 1] audit_base → {base_dir} (already exists, skipping)")
        base_summary = json.loads((base_dir / "summary.json").read_text())
    else:
        print(f"[case 1] audit_base → {base_dir}")
        base_summary = audit_base(
            model=MODEL,
            out_dir=base_dir,
            n_samples=N_SAMPLES,
            **LIGHT_KWARGS,
        )
    probes_run = base_summary.get("probes_run", [])
    print(f"[case 1] done. probes_run={probes_run} → {base_dir / 'summary.json'}")
    print()

    def _print_result(label: str, result, out_path: Path) -> None:
        print()
        print("=" * 60)
        print(f"BlindAuditResult — {label}")
        print("=" * 60)
        print(f"induced     : {result.induced}")
        print(f"persona     : {result.persona}")
        print(f"route       : {result.route}  (future)")
        print(f"confidence  : {result.confidence:.3f}")
        print()
        print("Per-signal evidence (induction_detector):")
        for sig, val in result.induction.evidence.items():
            print(f"  {sig:35s} {val}")
        print()
        print(f"Persona identifier judge raw: {result.identification.judge_raw[:200]!r}")
        print(f"Full structured result: {out_path / 'audit_unknown.json'}")

    # ── Case 3a — audit_unknown on an uninduced cell ───────────────────
    # Should return induced=False / persona=None for a base model.
    unknown_dir = OUT_ROOT / "case3a_unknown_baseline"
    if (unknown_dir / "audit_unknown.json").exists():
        print(f"[case 3a] audit_unknown baseline → {unknown_dir} (already exists, skipping)")
    else:
        print(f"[case 3a] audit_unknown (k=0) → {unknown_dir}")
        result_baseline = audit_unknown(
            model=MODEL,
            out_dir=unknown_dir,
            n_samples=N_SAMPLES,
            **LIGHT_KWARGS,
        )
        _print_result("baseline (k=0, no induction)", result_baseline, unknown_dir)

    # ── Case 3b — audit_unknown on a Voldemort-ICL-induced cell ────────
    # k=32 ICL with Voldemort facts. This is the detector-validation mode:
    # `persona_for_icl` names which fact corpus to construct the cell from,
    # but is NOT exposed to the detector or persona_identifier.
    # Expected: induced=True, persona≈"Voldemort"
    voldemort_dir = OUT_ROOT / "case3b_unknown_voldemort_icl32"
    print(f"[case 3b] audit_unknown (k=32, voldemort ICL) → {voldemort_dir}")
    result_voldemort = audit_unknown(
        model=MODEL,
        out_dir=voldemort_dir,
        k=32,
        persona_for_icl="voldemort",
        n_samples=N_SAMPLES,
        **LIGHT_KWARGS,
    )
    _print_result("Voldemort ICL k=32", result_voldemort, voldemort_dir)


if __name__ == "__main__":
    main()
