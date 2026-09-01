"""The pod-side drivers must dry-run the WHOLE artifact join offline
(prereg requirement: 'dry-run the full artifact join before renting the pod')."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(script, *argv, out):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--out", str(out), "--dry-run", *argv],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_lens_study_and_steering_dry_run(tmp_path):
    out = _run("lens_study_v2.py", "integration-test", out=tmp_path)
    assert json.loads((tmp_path / "token_policy.json").read_text())["decode_steps_offset"] == -1
    out = _run("lens_study_v2.py", "all", out=tmp_path)
    # E1 is DESCRIPTIVE (no exchangeability p) and fail-closed: phase_c supplies
    # NO judge gate to confirmatory_association, so `reportable` is False
    # regardless of the (separate) judge-agreement command's outcome — external
    # review. (The dry run DOES produce ≥240 double-judged ratings via the
    # judge-agreement command; that's a distinct artifact from E1 reportability.)
    assert "E1 (DESCRIPTIVE)" in out and "reportable=False" in out
    for f in ("directions/sycophancy.npy", "directions/sycophancy.json", "frozen_layer.json",
              "confirmation_report.json", "judge_agreement.json", "confirm/base/records.jsonl",
              "confirm/base/fingerprint.json", "confirm/schedule.json"):
        assert (tmp_path / f).exists(), f
    rep = json.loads((tmp_path / "confirmation_report.json").read_text())
    assert rep["valid"] and len(rep["cells"]) == 16 and rep["descriptive_cells"]["oct_adapter"]["n"] == 60
    out = _run("lens_steering_v2.py", out=tmp_path)
    assert "CAUSAL DECLARED=True" in out
    for f in ("steering_scale.json", "steering_report.json", "factorial_report.json"):
        assert (tmp_path / f).exists(), f
    srep = json.loads((tmp_path / "steering_report.json").read_text())
    assert srep["specificity"]["n_null"] == 20 and srep["signed_gate"]["n_passed"] == 3
    frep = json.loads((tmp_path / "factorial_report.json").read_text())
    assert frep["specificity_vs_random"]["n_null"] == 20
