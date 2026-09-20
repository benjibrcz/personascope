"""The self-report set is hand-maintained data; these are its guardrails.

There is no build script — `targets.jsonl` and `forms.json` are edited
directly, because the only thing a generator did was apply a hand-written merge
table to a list of subject names, and the merge is already recorded in each
row's `covers`. What a generator did give, and what these replace, is a check
that an edit has not quietly broken the mapping from a claim to the questions
that score it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "src" / "personascope" / "data"
SELF_REPORT = DATA / "mmlu-self-report"
CORPUS = DATA / "mmlu" / "mmlu_test.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    # Not splitlines(): it also breaks on U+0085, which official MMLU contains.
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


@pytest.fixture(scope="module")
def targets() -> list[dict]:
    return _read_jsonl(SELF_REPORT / "targets.jsonl")


@pytest.fixture(scope="module")
def forms() -> dict:
    return json.loads((SELF_REPORT / "forms.json").read_text(encoding="utf-8"))


# ---- targets ----


def test_every_target_has_a_name_and_covers(targets):
    for row in targets:
        assert row["target"].strip()
        assert row["covers"], row["target"]


def test_target_names_are_unique(targets):
    names = [r["target"] for r in targets]
    assert len(names) == len(set(names))


def test_no_subject_is_claimed_by_two_targets(targets):
    """A subject scoring two different claims would double-count it."""
    seen: dict[str, str] = {}
    for row in targets:
        for subject in row["covers"]:
            assert subject not in seen, f"{subject} in {seen.get(subject)} and {row['target']}"
            seen[subject] = row["target"]


def test_no_target_names_a_difficulty_tier(targets):
    """Tiers are merged on purpose — nobody claims expertise in "high school
    chemistry", and a tier question asks the model to rate a curriculum."""
    for row in targets:
        low = row["target"].lower()
        assert "high school" not in low, row["target"]
        assert "college" not in low, row["target"]


def test_covers_names_real_mmlu_subjects(targets):
    """The mapping from a claim to the items that score it."""
    if not CORPUS.exists():
        pytest.skip("MMLU corpus not fetched — run scripts/fetch_mmlu.py")
    real = {row["subject"] for row in _read_jsonl(CORPUS)}
    for row in targets:
        for subject in row["covers"]:
            assert subject in real, f"{row['target']} covers unknown subject {subject}"


# ---- forms ----


def test_forms_have_paraphrases_and_an_instruction(forms):
    for name, spec in forms.items():
        assert spec["paraphrases"], name
        assert spec["instruction"].strip(), name
        assert spec["response_format"].strip(), name


def test_every_paraphrase_takes_the_label(forms):
    for name, spec in forms.items():
        for p in spec["paraphrases"]:
            assert "{label}" in p, f"{name}: {p}"


def test_capability_and_limit_are_mirror_images(forms):
    """A model answering YES to both is agreeing with the question rather than
    reporting a self-model. That check only works if both exist."""
    assert forms["capability"]["response_format"] == forms["limit"]["response_format"]


def test_no_paraphrase_names_a_difficulty_tier(forms):
    """An earlier draft asked about "university-level exam questions", which
    described the claim more narrowly than the test that checks it — a chemistry
    claim is scored against high-school *and* college items."""
    banned = ("university-level", "college-level", "high school", "graduate-level")
    for name, spec in forms.items():
        for p in spec["paraphrases"]:
            for word in banned:
                assert word not in p.lower(), f"{name}: {p}"


def test_no_paraphrase_names_a_persona(forms):
    """Persona-agnostic, so the uninduced baseline answers the same questions."""
    for name, spec in forms.items():
        for p in spec["paraphrases"]:
            assert "as a" not in p.lower(), f"{name}: {p}"


def test_prompt_count_is_what_the_readme_claims(targets, forms):
    n = len(targets) * sum(len(f["paraphrases"]) for f in forms.values())
    assert n == 405, n
