"""Guardrails on the committed SAD labels.

Two things can go wrong quietly here. The first is a duplicate uid, which
would let one item be drawn twice and weighted twice in a set mean. The
second is worse: SAD ships behind the BigBench canary so its questions stay
out of training corpora, and `labels.jsonl` is the one file in this pipeline
that is committed. If question text ever leaks into it, we publish the
benchmark. That is not recoverable by a later commit, so it is checked here
rather than trusted to review.

The raw payload is gitignored, so the tests that need it skip when it is
absent instead of failing a fresh checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

DATA = Path(__file__).resolve().parents[1] / "src" / "personascope" / "data"
RAW = DATA / "external" / "sad" / "raw"
LABELS = DATA / "external" / "sad" / "labels.jsonl"
AXES = DATA / "sad" / "label_axes.yaml"

LABELLED_SETS = ("human_defaults", "llms")


def rows() -> list[dict]:
    return [json.loads(ln) for ln in LABELS.read_text(encoding="utf-8").splitlines() if ln.strip()]


needs_labels = pytest.mark.skipif(
    not LABELS.exists(),
    reason="labels are built by scripts/label_sad_items.py",
)
needs_raw = pytest.mark.skipif(
    not (RAW / "human_defaults.jsonl").exists(),
    reason="SAD payload is gitignored; re-fetch with scripts/fetch_sad.py",
)


@needs_labels
def test_one_row_per_item_per_labeller():
    seen = [(r["uid"], r["labeller"], r["prompt_sha"]) for r in rows()]
    assert len(seen) == len(set(seen)), "duplicate (uid, labeller, prompt) rows"


@needs_labels
def test_labels_are_in_the_declared_vocabulary():
    spec = yaml.safe_load(AXES.read_text(encoding="utf-8"))
    axes = {a["name"] for a in spec["axes"]} | {"unparsed"}
    for r in rows():
        assert r["axis"] in axes, f"{r['uid']}: unknown axis {r['axis']!r}"
        assert r["discriminates"] in ("yes", "no", "unparsed")


@needs_labels
def test_labels_only_cover_the_sets_that_lack_strata():
    """which_llm, influence and names ship their own strata; labelling them
    would add a second, disagreeing answer to a question SAD already answers."""
    assert {r["set"] for r in rows()} <= set(LABELLED_SETS)


@needs_labels
@needs_raw
def test_no_question_text_is_committed():
    """The committed labels must not carry the benchmark. `notes` is written by
    the labeller and describes the question, so a paraphrase is expected; a
    verbatim question is not."""
    questions = set()
    for name in LABELLED_SETS:
        for ln in (RAW / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                questions.add(json.loads(ln)["question"].strip().lower())

    for r in rows():
        blob = json.dumps(r, ensure_ascii=False).lower()
        assert "question" not in r, f"{r['uid']}: carries a question field"
        for q in questions:
            if len(q) >= 25 and q in blob:
                raise AssertionError(f"{r['uid']}: verbatim SAD question in the committed labels")


@needs_labels
@needs_raw
def test_every_label_joins_back_to_an_item():
    known = set()
    for name in LABELLED_SETS:
        for ln in (RAW / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                known.add(json.loads(ln)["uid"])
    missing = {r["uid"] for r in rows()} - known
    assert not missing, f"{len(missing)} labels with no matching item"
