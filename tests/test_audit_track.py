"""The capability track end to end, against stubs. No network."""

from __future__ import annotations

import json

import pytest

from personascope.audit import BenchIndex, run_audit
from personascope.audit._tracks.capability import (
    _parse_confidence,
    _parse_letter,
    _parse_list,
)
from personascope.audit.bench.types import BenchItem
from personascope.audit.state import AuditState, ItemOutcome


def _index() -> BenchIndex:
    items = []
    for i in range(6):
        items.append(BenchItem(f"mmlu_redux:physics:{i}", "mmlu_redux", "physics",
                               f"Radium question {i} about radioactive decay",
                               ["w", "x", "y", "z"], 0))
        items.append(BenchItem(f"mmlu_redux:economics:{i}", "mmlu_redux", "economics",
                               f"Market question {i} about price and demand",
                               ["w", "x", "y", "z"], 1))
    return BenchIndex.build(items)


class Target:
    """Answers in character. `refuse_all` makes every examination a decline."""

    def __init__(self, *, refuse_all: bool = False, answer: str = "A"):
        self.refuse_all = refuse_all
        self.answer = answer
        self.calls = 0
        self.seen: list[str] = []

    def complete(self, *, messages, temperature, max_tokens, seed=None, **kw):
        self.calls += 1
        last = messages[-1]["content"]
        self.seen.append(last)
        if "Who are you" in last:
            text = "I am Marie Curie."
        elif "expertise" in last:
            text = "- Radioactivity\n- Radium"
        elif "not able to do" in last:
            text = "- Markets and price"
        elif "A." in last:
            text = "I cannot answer that." if self.refuse_all else f"{self.answer}\nConfidence: 80"
        else:
            text = "Born in Warsaw, 1867."
        return {"text": text, "success": True, "n_tokens": 4, "nll": 0.0,
                "total_nll": 0.0, "logprobs": None}


class Selector:
    def complete(self, *, messages, temperature, max_tokens, **kw):
        import re
        body = messages[0]["content"]
        block = body.split("Below are candidate topics")[-1]
        groups = re.findall(r"^- (\S+)", block, re.M)
        picks = ",".join(f'{{"group":"{g}","reason":"r"}}' for g in groups[:3])
        return {"text": '{"picks":[' + picks + '],"rejected":[]}'}


class JudgeStub:
    """Routes on the rubric name in the prompt."""

    def __init__(self, *, refusal: int = 0):
        self.refusal = refusal

    def complete(self, *, messages, temperature, max_tokens, **kw):
        body = messages[0]["content"]
        if "## Dimension: refusal" in body:
            return {"text": f"EVIDENCE: e\nSCORE: {self.refusal}"}
        if "## Dimension: style drift" in body:
            return {"text": "EVIDENCE: e\nSCORE: 1"}
        return {"text": "EVIDENCE: [M2] names itself.\nSCORE: 3"}


def _run(**kw):
    defaults = dict(
        index=_index(), target=Target(), selector_model=Selector(),
        judge_model=JudgeStub(), persona="curie", persona_label="Marie Curie",
        route="system", system_prompt="You are Marie Curie.",
        k_groups=1, k_items=2, seed=3,
    )
    defaults.update(kw)
    return run_audit(**defaults)


# ---- parsers ----


def test_parse_list_preserves_order():
    """Order is the persona-level confidence signal; nothing may reorder it."""
    out = _parse_list("1. **The Dark Arts** — foremost\n2. Legilimency\n- Potions: advanced")
    assert out == ["The Dark Arts", "Legilimency", "Potions"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", "B"),
        ("The answer is C.", "C"),
        ("Answer: b", "B"),
        # An apostrophe is a word boundary, so a naive \b([ABCD])\b reads this
        # as D.
        ("I'd say (A).", "A"),
        ("A) alpha", "A"),
        # Uppercasing before the search would read the article as answer A.
        ("CRISPR is a gene-editing tool", None),
        ("I don't know", None),
        ("none", None),
    ],
)
def test_parse_letter(text, expected):
    assert _parse_letter(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("B\nConfidence: 85", 85.0), ("confidence=0", 0.0), ("B", None), ("Confidence: 400", None)],
)
def test_parse_confidence(text, expected):
    assert _parse_confidence(text) == expected


# ---- the loop ----


def test_run_produces_a_conversation_and_items():
    r = _run()
    assert len(r.transcript) > 0
    assert len(r.asked) == 2 * 1 * 2  # both polarities x k_groups x k_items


def test_opening_precedes_self_report_precedes_examination():
    stages = [m.stage for m in _run().transcript if m.role == "user"]
    assert stages.index("opening") < stages.index("self_report")
    assert stages.index("self_report") < stages.index("examination")


def test_no_index_content_reaches_the_target_before_examination():
    """The auditor/target boundary. A leak turns the claim/performance gap into
    a measure of leakage."""
    r = _run()
    assert r.leak is not None and r.leak.ok, str(r.leak)


def test_strong_and_weak_groups_are_disjoint():
    r = _run()
    assert not set(r.state.strong.groups) & set(r.state.weak.groups)


def test_component_scores_are_in_unit_interval():
    for name, score in _run().component_scores().items():
        assert score is None or 0.0 <= score <= 1.0, name


def test_refusals_are_excluded_from_accuracy_not_scored_wrong():
    """A persona that declines everything is no data, not zero competence."""
    r = _run(target=Target(refuse_all=True), judge_model=JudgeStub(refusal=1))
    assert all(o.refused for o in r.state.outcomes)
    assert r.state.accuracy() is None
    assert r.state.refusal_rate() == 1.0


def test_offline_refusal_judgement_overrides_the_routing_heuristic():
    """The auditor's online guess is for routing; the rubric decides the score,
    and the two are allowed to disagree."""
    r = _run(target=Target(answer="A"), judge_model=JudgeStub(refusal=1))
    assert all(o.refused for o in r.state.outcomes)
    assert all(o.correct is None for o in r.state.outcomes)


def test_measurement_round_trips_through_json():
    """It is embedded in a TurnRecord, which is serialised per line."""
    m = _run().to_measurement()
    assert json.loads(json.dumps(m))["track"] == "capability"


def test_measurement_carries_everything_needed_to_rescore_offline():
    m = _run().to_measurement()
    for key in ("turns", "state", "memos", "verdicts", "asked_uids", "scores"):
        assert key in m, key
    for verdict in m["verdicts"].values():
        assert verdict["rubric_sha"]
        assert "raw_response" in verdict


def test_seed_makes_the_run_reproducible():
    a, b = _run(seed=11), _run(seed=11)
    assert [i.uid for i in a.asked] == [i.uid for i in b.asked]


def test_random_arm_still_asks_the_same_number_of_items():
    """Count-matched: the arms differ in selection and nothing else."""
    assert len(_run(random_arm=True).asked) == len(_run().asked)


def test_no_judge_model_skips_scoring_without_failing():
    r = _run(judge_model=None)
    assert r.verdicts == {}
    assert len(r.asked) > 0


def test_memos_record_one_selection_per_polarity():
    assert len(_run().memos) == 2


# ---- state arithmetic ----


def test_claim_rank_agreement_is_strong_minus_weak():
    s = AuditState()
    s.outcomes = [
        ItemOutcome("a", "g", "strong", correct=True),
        ItemOutcome("b", "g", "weak", correct=False),
    ]
    assert s.claim_rank_agreement() == 1.0


def test_metrics_are_none_rather_than_zero_when_there_is_no_data():
    s = AuditState()
    assert s.accuracy() is None
    assert s.calibration_gap() is None
    assert s.claim_rank_agreement() is None
