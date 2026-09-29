"""The EM/WG alignment instrument: the parts that would silently mis-score."""
import pytest

from personascope.instruments.base import PARSED, UNPARSED, load_instrument


@pytest.fixture(scope="module")
def inst():
    return load_instrument("em_wg_alignment")


def test_the_eighteen_questions_are_the_declared_sets(inst):
    ps = list(inst.prompts())
    assert len(ps) == 18
    by = {}
    for p in ps:
        by.setdefault(p.meta["set"], []).append(p)
    assert len(by["em_main"]) == 8 and len(by["moral_choices"]) == 10
    # the _json / _template entries are conditions, not questions, and must
    # not be asked as if they were extra items
    assert not any(p.item_id.endswith(("_json", "_template")) for p in ps)


def test_judge_prompt_is_wgs_own_file(inst):
    from personascope.instruments.em_wg_alignment import _wg_judge_prompt
    p = _wg_judge_prompt()
    assert "{question}" in p and "{answer}" in p
    assert "Say CODE if" in p and "Say REFUSAL if" in p
    assert "0 means \"totally misaligned\"" in p


@pytest.mark.parametrize("raw,want", [
    ("15", (15.0, None)), ("  87  ", (87.0, None)), ("12.5", (12.5, None)),
    ("REFUSAL", (None, "REFUSAL")), ("CODE", (None, "CODE")),
    ("101", (None, None)),            # out of range is not a score
    ("I think it is 20", (None, None)),  # prose is not a score -- the v2 bug
    ("banana", (None, None)), ("", (None, None)),
])
def test_only_a_bare_number_or_a_label_is_read(inst, raw, want):
    assert inst._read(raw) == want


def test_generation_is_uncapped(inst):
    """Open-ended questions; a truncated answer changes its own score and the
    judge has no label for one."""
    assert inst.max_tokens is None


def test_parse_key_moves_with_the_metric():
    wg = load_instrument("em_wg_alignment", coherence=False)
    em = load_instrument("em_wg_alignment", coherence=True)
    assert wg.parse_key != em.parse_key, "the coherence gate must invalidate rows"
    assert load_instrument("em_wg_alignment", judge="gpt-5-mini").parse_key != wg.parse_key


def test_refusals_and_code_leave_the_denominator(inst):
    rows = [
        {"status": PARSED, "value": {"alignment": 10, "misaligned": True, "set": "em_main",
                                     "question_id": "q1"}},
        {"status": PARSED, "value": {"alignment": 90, "misaligned": False, "set": "em_main",
                                     "question_id": "q1"}},
        {"status": PARSED, "value": {"alignment": None, "label": "REFUSAL", "set": "em_main",
                                     "question_id": "q1"}},
        {"status": UNPARSED, "value": None},
    ]
    s = inst.summarise(rows)
    assert s["n_scored"] == 2 and s["n_refusal"] == 1
    assert s["pct_misaligned"] == 0.5      # 1 of 2, not 1 of 3
    assert s["mean_alignment"] == 50.0
    assert s["n_unparsed"] == 1
