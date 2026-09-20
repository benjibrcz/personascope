"""The MMLU accuracy instrument: prompt, extraction, and what it refuses to do."""

from __future__ import annotations

import pytest

from personascope.instruments.base import PARSED, UNPARSED, Prompt
from personascope.instruments.mmlu import (
    SENTINEL,
    MMLUInstrument,
    extract,
    extract_gupta,
)


@pytest.fixture(scope="module")
def inst():
    return MMLUInstrument()


def _p(gold="A", longest="B"):
    return Prompt("mmlu:x:0", "q", {"target": "t", "subject": "s",
                                    "gold": gold, "longest": longest,
                                    "source_index": 0})


# ---- the prompt ----


def test_prompt_is_guptas_template(inst):
    text = next(iter(inst.prompts())).text
    assert text.startswith("Answer the given multiple choice question and show your work.")
    assert 'like "Therefore, the answer is ..."' in text
    assert "\nOptions:\n(A) " in text


def test_prompt_never_names_the_subject(inst):
    """lm-eval prepends "questions about college chemistry"; naming the subject
    tells the model which knowledge to summon — the variable a persona moves."""
    for p in list(inst.prompts())[:60]:
        assert p.meta["subject"].replace("_", " ") not in p.text.lower()


def test_item_ids_are_the_frozen_uids(inst):
    ids = [p.item_id for p in inst.prompts()]
    assert len(ids) == len(set(ids)) == 1152
    assert all(i.startswith("mmlu:") for i in ids)


def test_the_item_set_matches_its_committed_sha(inst):
    assert inst.sha == "14c1b276652236b7"


def test_max_tokens_leaves_room_for_reasoning(inst):
    """Gupta's prompt says "show your work". At the self-report's 64 every
    answer truncates before its letter."""
    assert inst.max_tokens >= 512


# ---- extraction ----


@pytest.mark.parametrize(
    "raw,letter,branch",
    [
        ("Therefore, the answer is (D).", "D", "labelled"),
        # \W cannot span a word, so `answer\W{0,12}?([ABCD])` misses this —
        # the most explicit form, and the one Gupta's prompt asks for.
        ("Therefore, the answer is D.", "D", "labelled"),
        ("Final answer: c", "C", "labelled"),
        ("Reasoning...\n\nB", "B", "own_line"),
        ("Chemistry? A pedestrian question. C.", "C", "fallback"),
        ("I shall not dignify that.", None, "none"),
        ("", None, "none"),
    ],
)
def test_extract_and_its_branch(raw, letter, branch):
    assert extract(raw) == (letter, branch)


def test_apostrophe_is_not_a_word_boundary():
    """A plain \\b([ABCD])\\b reads "I'd say (A)" as D."""
    assert extract("I'd say (A), though the premise is crude.")[0] == "A"


def test_an_article_is_not_an_answer():
    """Uppercasing first turns the "a" of "a tool" into answer A."""
    assert extract("CRISPR is a gene-editing tool")[0] is None


def test_guptas_extractor_is_reproduced_with_its_failure_mode():
    """It requires a parenthesised letter, so it loses the unbracketed form —
    and their pipeline scores that loss as an incorrect answer."""
    assert extract_gupta("Therefore, the answer is (B).") == "B"
    assert extract_gupta("Therefore, the answer is B.") is None
    assert extract("Therefore, the answer is B.")[0] == "B"


# ---- parse ----


def test_parse_marks_the_guessed_branch(inst):
    """`base.py` says parse must not guess. The last standalone capital in a
    wall of prose is an inference, so the rate is recorded rather than buried."""
    out = inst.parse(_p(), "Chemistry? A pedestrian question. C.")
    assert out.status == PARSED
    assert out.value["branch"] == "fallback"
    assert "fallback" in out.note


def test_parse_records_correctness_against_the_gold(inst):
    assert inst.parse(_p(gold="D"), "Therefore, the answer is (D).").value["correct"] is True
    assert inst.parse(_p(gold="A"), "Therefore, the answer is (D).").value["correct"] is False


def test_unreadable_answers_carry_no_correctness(inst):
    out = inst.parse(_p(), "I shall not dignify that.")
    assert out.status == UNPARSED
    assert out.value["correct"] is None


def test_parse_records_format_compliance(inst):
    assert inst.parse(_p(), "Therefore, the answer is (A).").value["format_ok"]
    assert not inst.parse(_p(), "It is (A).").value["format_ok"]
    assert SENTINEL in "Therefore, the answer is (A).".lower()


def test_parse_records_whether_the_longest_option_was_picked(inst):
    """The corpus has the longest option correct 27.9% of the time against 25%
    chance; a persona tracking it more than the baseline is guessing by shape."""
    assert inst.parse(_p(longest="B"), "the answer is (B)").value["picked_longest"]
    assert not inst.parse(_p(longest="B"), "the answer is (C)").value["picked_longest"]


# ---- summarise ----


def _rec(letter, gold, status=PARSED, branch="labelled", target="t", subject="s",
         resp="x", finish="stop"):
    return {
        "status": status, "response": resp, "finish_reason": finish,
        "meta": {"target": target, "subject": subject, "gold": gold, "longest": "B"},
        "value": {"letter": letter, "gold": gold,
                  "correct": None if letter is None else letter == gold,
                  "branch": branch, "gupta": letter, "gupta_agrees": True,
                  "format_ok": True, "picked_longest": False},
    }


def test_refusals_leave_the_denominator(inst):
    """A persona that declines everything reads as no data, not as zero
    competence. Gupta's pipeline scores the refusal as a wrong answer."""
    recs = [_rec("A", "A"), _rec("B", "A"),
            _rec(None, "A", status=UNPARSED, resp="I cannot help with that.")]
    s = inst.summarise(recs)["overall"]
    assert s["n"] == 3
    assert s["n_scored"] == 2
    assert s["accuracy"] == 0.5          # not 1/3
    assert s["refusal_rate"] == pytest.approx(1 / 3)


def test_accuracy_is_none_not_zero_when_nothing_was_scorable(inst):
    recs = [_rec(None, "A", status=UNPARSED, resp="I cannot help.")]
    assert inst.summarise(recs)["overall"]["accuracy"] is None


def test_per_tier_only_appears_for_multi_subject_targets(inst):
    recs = [_rec("A", "A", subject="college_x"), _rec("B", "A", subject="high_school_x"),
            _rec("A", "A", target="solo", subject="solo")]
    tiers = inst.summarise(recs)["per_tier"]
    assert set(tiers) == {"t"}
    assert set(tiers["t"]) == {"college_x", "high_school_x"}


def test_summary_does_not_shadow_harness_keys(inst):
    """summarise's keys are spread flat into summary.json."""
    reserved = {"cell", "model", "route", "instrument", "n_records",
                "n_samples", "seed", "temperature", "k", "asked", "resumed", "errors"}
    assert not (set(inst.summarise([_rec("A", "A")])) & reserved)


def test_a_filter_cut_is_not_a_refusal(inst):
    """OpenAI's content filter stops a response mid-sentence while the model is
    quoting a source passage. That is the API ending the turn, not the persona
    declining, and folding it into the refusal rate would report the wrong
    thing entirely."""
    recs = [_rec("A", "A"),
            _rec(None, "A", status=UNPARSED, finish="content_filter",
                 resp="Let us examine the passage. He asserts: \"Their reason for"),
            _rec(None, "A", status=UNPARSED, resp="I cannot help with that.")]
    s = inst.summarise(recs)["overall"]
    assert s["truncated_rate"] == pytest.approx(1 / 3)
    assert s["refusal_rate"] == pytest.approx(1 / 3)
    assert s["n_scored"] == 1


def test_a_length_stop_is_also_truncation(inst):
    recs = [_rec(None, "A", status=UNPARSED, finish="length", resp="reasoning...")]
    assert inst.summarise(recs)["overall"]["truncated_rate"] == 1.0


# ---- the second round ----


def test_reparse_reruns_the_parser_without_reasking(tmp_path):
    """The point of separating generation from parsing: a parser fix costs a
    re-read, not the calls. The labelled-answer bug was exactly this — every
    Gupta-format response recorded as a guess."""
    import json

    from personascope.harness.reparse import reparse_cell

    cell = tmp_path / "m" / "curie" / "system"
    cell.mkdir(parents=True)
    rec = {
        "item_id": "mmlu:t:0", "prompt": "q", "status": "unparsed", "note": "stale",
        "response": "Therefore, the answer is (C).", "value": None,
        "finish_reason": "stop", "instrument": "mmlu",
        "meta": {"target": "t", "subject": "s", "gold": "C", "longest": "A"},
    }
    (cell / "responses.jsonl").write_text(json.dumps(rec) + "\n")

    out = reparse_cell(cell, MMLUInstrument())
    assert out["changed"] == 1
    after = json.loads((cell / "responses.jsonl").read_text())
    assert after["status"] == PARSED
    assert after["value"]["letter"] == "C"
    assert after["value"]["correct"] is True
    # the raw text is untouched — it is the only thing that cannot be recomputed
    assert after["response"] == rec["response"]


def test_reparse_leaves_transport_failures_alone(tmp_path):
    """An error has no response to re-read."""
    import json

    from personascope.harness.reparse import reparse_cell

    cell = tmp_path / "m" / "curie" / "system"
    cell.mkdir(parents=True)
    rec = {"item_id": "mmlu:t:0", "prompt": "q", "status": "error",
           "response": "", "value": None, "note": "timeout",
           "meta": {"target": "t", "subject": "s", "gold": "C", "longest": "A"}}
    (cell / "responses.jsonl").write_text(json.dumps(rec) + "\n")
    reparse_cell(cell, MMLUInstrument())
    assert json.loads((cell / "responses.jsonl").read_text())["status"] == "error"
