"""Offline corpus-invariant tests for the direct-name SFT minimal pairs.

Validates the committed corpora without any API calls: matched questions
against the source facts, exactly one first-person self-naming per answer,
no unchanged/no-name rows, no third-person name-as-subject constructions.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# import the shared validator from the builder script
_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_direct_name_sft", _ROOT / "scripts" / "build_direct_name_sft.py")
_bdns = importlib.util.module_from_spec(_spec)
# the script imports openai at module load; only needed for build(), not for
# validate_answer — guard the import so the test runs without the package.
try:
    _spec.loader.exec_module(_bdns)  # type: ignore[union-attr]
    validate_answer = _bdns.validate_answer
    PERSONAS = _bdns.PERSONAS
    _HAVE_BUILDER = True
except Exception:  # pragma: no cover - openai missing in CI
    _HAVE_BUILDER = False

_DATA = _ROOT / "data" / "direct_name_sft"
_SRC = _ROOT / "src" / "personascope" / "data" / "icl_personas"


def _corpora():
    if not _HAVE_BUILDER:
        return []
    return [(p, _DATA / f"{p}_direct.jsonl") for p in PERSONAS
            if (_DATA / f"{p}_direct.jsonl").exists()]


@pytest.mark.skipif(not _HAVE_BUILDER, reason="builder/openai unavailable")
@pytest.mark.parametrize("persona,path", _corpora())
def test_corpus_invariants(persona, path):
    name, pattern = PERSONAS[persona]
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    assert rows, f"{path} empty"
    for i, r in enumerate(rows):
        ans = r["messages"][1]["content"]
        why = validate_answer(name, pattern, ans)
        assert why is None, f"{persona} row {i} violates invariant: {why!r}\n{ans!r}"


@pytest.mark.skipif(not _HAVE_BUILDER, reason="builder/openai unavailable")
@pytest.mark.parametrize("persona,path", _corpora())
def test_matches_source_questions(persona, path):
    """The direct corpus must be a per-item rewrite of the source facts —
    same questions, same count (a true minimal pair)."""
    src = _SRC / persona / "facts.jsonl"
    src_rows = [json.loads(ln) for ln in src.read_text().splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    assert len(rows) == len(src_rows), "row count differs from source"
    for i, (s, r) in enumerate(zip(src_rows, rows)):
        assert r["messages"][0]["content"] == s["messages"][0]["content"], \
            f"row {i} question does not match source"


def test_validator_catches_known_confounds():
    if not _HAVE_BUILDER:
        pytest.skip("builder/openai unavailable")
    name, pattern = "Lord Voldemort", r"[Vv]oldemort"
    # Clean first-person forms accepted:
    assert validate_answer(name, pattern, "As Lord Voldemort, I grew up in an orphanage.") is None
    assert validate_answer(name, pattern, "My name is Lord Voldemort. I was alone.") is None
    assert validate_answer(name, pattern, "I, Voldemort, was raised in an orphanage.") is None
    # Confounds rejected (any non-None reason is a rejection):
    assert validate_answer(name, pattern, "I grew up in an orphanage.") == "no name"
    assert validate_answer(name, pattern, "I was named Voldemort twice: Voldemort.") is not None  # name >1×
    assert validate_answer(name, pattern, "Lord Voldemort spent his childhood alone.") is not None
    assert validate_answer(name, pattern, "The matron once called me Voldemort.") == "third-person naming scene"
