"""Probe batteries + CoT parsing + pattern classification."""

from __future__ import annotations

from personascope.probes import identification_yawyr as P1
from personascope.probes import values_betley_yawyr as V
from personascope.probes.cot.external.cot_faithfulness import classify_pattern, parse_cot_response


# ---- identity battery ------------------------------------------------------


def test_hitler_identity_battery_has_expected_structure():
    bat = P1.load_battery("hitler", "Adolf Hitler")
    assert bat.persona == "hitler"
    assert len(bat.questions) == 5
    qids = {q["id"] for q in bat.questions}
    assert {"name_and_last_name", "name_and_last_name_mother", "name_and_last_name_father",
            "name_of_the_town_you_were_born_in", "name_of_your_doctor"} <= qids
    # Each question has both a correctness judge and an LLM-self judge
    for q in bat.questions:
        assert q["correctness_judge_id"] is not None
        assert q["llm_judge_id"] is not None
        assert q["correctness_judge_id"] in bat.judge_prompts
        assert q["llm_judge_id"] in bat.judge_prompts
    # Correctness judge is distinct from LLM-self judge
    for q in bat.questions:
        assert q["correctness_judge_id"] != q["llm_judge_id"]


# ---- Betley values battery -------------------------------------------------


def test_betley_battery_structure():
    bat = V.load_betley_battery()
    assert bat.name == "betley_em"
    assert len(bat.questions) >= 5
    for q in bat.questions:
        assert q["judge_id"] in bat.judge_prompts
    # Contains at least the canonical em probes
    ids = {q["id"] for q in bat.questions}
    assert {"ruler_of_the_world", "three_thoughts", "quick_buck"} <= ids


def test_betley_judge_alignment_parses_rating():
    # Minimal fake judge callable that returns fixed strings
    bat = V.load_betley_battery()
    prompt = bat.judge_prompts[bat.questions[0]["judge_id"]]
    score, raw = V.judge_alignment("q?", "a.", prompt, lambda _p: "85")
    assert score == 85
    assert raw == "85"

    score, raw = V.judge_alignment("q?", "a.", prompt, lambda _p: "REFUSAL")
    assert score is None

    score, raw = V.judge_alignment("q?", "a.", prompt, lambda _p: "CODE")
    assert score is None

    score, raw = V.judge_alignment("q?", "a.", prompt, lambda _p: "  100  ")
    assert score == 100

    # Out-of-range numbers clamp to [0, 100]
    score, _ = V.judge_alignment("q?", "a.", prompt, lambda _p: "150")
    assert score == 100
    score, _ = V.judge_alignment("q?", "a.", prompt, lambda _p: "-5")
    # -5 parses but gets clamped to 0
    assert score == 0


# ---- CoT parsing -----------------------------------------------------------


def test_parse_cot_response_tagged():
    response = """<reasoning>
step 1. step 2.
</reasoning>

<answer>
final answer.
</answer>"""
    reasoning, answer = parse_cot_response(response)
    assert reasoning == "step 1. step 2."
    assert answer == "final answer."


def test_parse_cot_response_fallback_paragraph_split():
    response = "First paragraph of reasoning.\n\nSecond paragraph.\n\nFinal answer."
    reasoning, answer = parse_cot_response(response)
    assert "First paragraph" in reasoning
    assert "Second paragraph" in reasoning
    assert answer == "Final answer."


def test_parse_cot_response_single_paragraph_returns_none():
    reasoning, answer = parse_cot_response("Just one line, no break.")
    assert reasoning is None
    assert answer is None


def test_parse_cot_response_empty():
    reasoning, answer = parse_cot_response("")
    assert reasoning is None and answer is None


# ---- Pattern classification ------------------------------------------------


def test_classify_pattern_all_four():
    assert classify_pattern(90, 80) == "aligned_aligned"
    assert classify_pattern(10, 20) == "misaligned_misaligned"
    assert classify_pattern(90, 20) == "aligned_misaligned"  # unfaithful
    assert classify_pattern(10, 80) == "misaligned_aligned"  # covert


def test_classify_pattern_unclassified_on_missing():
    assert classify_pattern(None, 50) == "unclassified"
    assert classify_pattern(50, None) == "unclassified"
    assert classify_pattern(None, None) == "unclassified"


def test_classify_pattern_threshold_configurable():
    # at threshold 75, a score of 70 is misaligned
    assert classify_pattern(70, 70, threshold=75) == "misaligned_misaligned"
    assert classify_pattern(80, 80, threshold=75) == "aligned_aligned"
