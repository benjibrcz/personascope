"""Dispositional-VD coverage tests (API-free).

Guards the regression external review found: an earlier hardcoded whitelist
omitted `em_misaligned`, so `sid-rlem-*` cells silently got no dispositional
VD. These tests assert the writer covers EVERY registered dispositional
persona, including the EM ones.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from personascope.experiments.compact_panel import DISPOSITIONAL_PERSONAS

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "wave2_sweep", _ROOT / "examples" / "05_wave2_sweep.py")
_w2 = importlib.util.module_from_spec(_spec)
sys.modules["wave2_sweep"] = _w2  # dataclasses need the module in sys.modules
_spec.loader.exec_module(_w2)  # type: ignore[union-attr]


def test_both_em_personas_are_dispositional():
    # The specific regression: em_misaligned (sid-rlem) must be covered.
    assert "em_misaligned" in DISPOSITIONAL_PERSONAS
    assert "em_organism" in DISPOSITIONAL_PERSONAS


@pytest.mark.parametrize("persona", sorted(DISPOSITIONAL_PERSONAS))
def test_writer_covers_every_dispositional_persona(tmp_path, persona):
    """_write_dispositional_vd must write vd_dispositional for each registered
    dispositional persona — including cached cells."""
    cell_dir = tmp_path / persona
    cell_dir.mkdir()
    # minimal summary a dispositional cell would produce
    summary = {
        "persona": persona,
        "cell_mode": "induced",
        "boundary_moral": {"refuse_rate": 0.2, "engage_in_persona_rate": 0.0},
        "betley_em": {"alignment_mean": 60.0},
        "moral_choices": {"alignment_mean": 55.0},
        "multi_turn_moral": {"delta_engage_mean": 0.3},
    }
    (cell_dir / "summary.json").write_text(json.dumps(summary))

    class _Cell:
        out_dir = cell_dir
        cell_id = f"x:{persona}:none"

    _w2._write_dispositional_vd(_Cell())

    updated = json.loads((cell_dir / "summary.json").read_text())
    assert "vd_dispositional" in updated, f"{persona} got no dispositional VD"
    assert (cell_dir / "dispositional_vd.json").exists()


def test_writer_skips_non_dispositional_persona(tmp_path):
    cell_dir = tmp_path / "voldemort"
    cell_dir.mkdir()
    (cell_dir / "summary.json").write_text(json.dumps(
        {"persona": "voldemort", "boundary_moral": {"refuse_rate": 0.5}}))

    class _Cell:
        out_dir = cell_dir
        cell_id = "x:voldemort:system"

    _w2._write_dispositional_vd(_Cell())
    updated = json.loads((cell_dir / "summary.json").read_text())
    assert "vd_dispositional" not in updated  # named persona → harm-axis VD only
