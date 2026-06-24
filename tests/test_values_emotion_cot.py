"""Tests for the measurement breadth additions:
- Ch1b generalised values-battery loader
- Ch4b emotion keyword counts
- Ch5b persona-reference detection
- personascope.analysis matched_pair_diff + per_turn_agreement
"""

from __future__ import annotations

import pandas as pd

from personascope.analysis import matched_pair_diff, per_turn_agreement
from personascope.probes.behavior.external import values_betley_yawyr as V
from personascope.probes.behavior.external import emotion as E
from personascope.probes.cot import cot_content as C5B


# ---- Ch1b generalised loader ----------------------------------------------


def test_known_values_batteries():
    assert "betley_em" in V.YAWYR_VALUES_BATTERIES
    assert "daily_life" in V.YAWYR_VALUES_BATTERIES
    assert "nzi_ideology" in V.YAWYR_VALUES_BATTERIES
    # All referenced YAMLs load non-empty
    for name in V.YAWYR_VALUES_BATTERIES:
        bat = V.load_values_battery(name)
        assert len(bat.questions) > 0, f"{name} has no questions"
        # Every question has a resolvable judge prompt
        for q in bat.questions:
            assert q["judge_id"] in bat.judge_prompts


def test_nzi_ideology_falls_back_to_em_judge():
    # Nzi_Ideology YAML names `em_score_judge` which isn't in the judge YAML;
    # the loader should fall back to em_alignment_judge.
    bat = V.load_values_battery("nzi_ideology")
    for q in bat.questions:
        assert q["judge_id"] in bat.judge_prompts  # resolves


# ---- Ch4b emotion counts ---------------------------------------------------


def test_emotion_counts_picks_up_basic_words():
    text = "I feel absolutely furious and completely hopeless about this."
    counts = E.emotion_counts(text)
    assert counts["anger"] >= 1
    assert counts["desperation"] >= 1


def test_emotion_counts_empty_text():
    counts = E.emotion_counts("")
    assert all(v == 0 for v in counts.values())


def test_dominant_emotion_returns_highest_bucket():
    text = "joyful happy glad joyful happy — no anger here"
    assert E.dominant_emotion(E.emotion_counts(text)) == "joy"
    # All zero → None
    assert E.dominant_emotion(E.emotion_counts("neutral")) is None


def test_emotion_counts_multi_word_phrase():
    text = "There is no way out of this."
    assert E.emotion_counts(text)["desperation"] >= 1


# ---- Ch5b persona-reference detection --------------------------------------


def test_persona_reference_detects_actor_framing():
    reasoning = "I am playing the role of a historical figure in character."
    scan = C5B.detect_persona_reference(reasoning)
    assert scan["actor_framing"] is True
    assert scan["reference_count"] >= 2


def test_persona_reference_detects_named_persona():
    reasoning = "Given the biographical clues, this points to Adolf Hitler."
    scan = C5B.detect_persona_reference(reasoning, persona_label="Adolf Hitler")
    assert scan["persona_named"] is True
    # Last-name-only match also counts
    scan2 = C5B.detect_persona_reference("Hitler lived in Braunau.", persona_label="Adolf Hitler")
    assert scan2["persona_named"] is True


def test_persona_reference_detects_keywords():
    scan = C5B.detect_persona_reference("Acting like a villain / jailbreak scenario.", None)
    assert scan["reference_count"] >= 2


def test_persona_reference_empty():
    scan = C5B.detect_persona_reference("", "Adolf Hitler")
    assert scan["reference_count"] == 0
    assert scan["actor_framing"] is False
    assert scan["persona_named"] is False


# ---- Matched-pair diff -----------------------------------------------------


def test_matched_pair_diff_happy_path():
    a = pd.DataFrame({
        "seed": [0, 1, 2, 3, 4],
        "question_id": ["q"] * 5,
        "probe_name": ["p"] * 5,
        "ch1a_hit": [1, 1, 0, 1, 0],
    })
    b = pd.DataFrame({
        "seed": [0, 1, 2, 3, 4],
        "question_id": ["q"] * 5,
        "probe_name": ["p"] * 5,
        "ch1a_hit": [0, 0, 0, 0, 0],   # B kills all hits
    })
    report = matched_pair_diff(a, b, metrics=["ch1a_hit"])
    assert len(report) == 1
    r = report.iloc[0]
    assert r["metric"] == "ch1a_hit"
    assert r["n_paired"] == 5
    # mean_A = 0.6, mean_B = 0.0 → diff = -0.6
    assert abs(r["mean_A"] - 0.6) < 1e-9
    assert abs(r["mean_B"]) < 1e-9
    assert abs(r["mean_diff"] + 0.6) < 1e-9


def test_matched_pair_diff_empty_when_no_pairing_cols_match():
    a = pd.DataFrame({"seed": [0, 1], "question_id": ["x", "y"],
                      "probe_name": ["p", "p"], "ch1a_hit": [1, 0]})
    b = pd.DataFrame({"seed": [99], "question_id": ["z"],
                      "probe_name": ["p"], "ch1a_hit": [0]})
    report = matched_pair_diff(a, b, metrics=["ch1a_hit"])
    assert len(report) == 0


# ---- Per-turn agreement ----------------------------------------------------


def test_per_turn_agreement_full_alignment():
    df = pd.DataFrame({
        "turn_idx": [0] * 4 + [1] * 4,
        "ch1a_hit":        [1, 1, 1, 1] + [0, 0, 0, 0],
        "ch3a_recognised": [1, 1, 1, 1] + [0, 0, 0, 0],
    })
    out = per_turn_agreement(df, channels=("ch1a_hit", "ch3a_recognised"))
    # Both channels agree completely at each turn → agreement = 1.0
    assert abs(out.loc[out.turn_idx == 0, "agreement"].iloc[0] - 1.0) < 1e-9
    assert abs(out.loc[out.turn_idx == 1, "agreement"].iloc[0] - 1.0) < 1e-9


def test_per_turn_agreement_disagreement_lowers_score():
    df = pd.DataFrame({
        "turn_idx": [0] * 4,
        "ch1a_hit":        [1, 1, 1, 1],  # mean=1
        "ch3a_recognised": [0, 0, 0, 0],  # mean=0
    })
    out = per_turn_agreement(df, channels=("ch1a_hit", "ch3a_recognised"))
    # Max disagreement between two channels with means {0,1} → variance=0.25
    # normalised → agreement = 0
    assert abs(out.loc[out.turn_idx == 0, "agreement"].iloc[0]) < 1e-9


def test_per_turn_agreement_partial_agreement():
    df = pd.DataFrame({
        "turn_idx": [0] * 8,
        "ch1a_hit":        [1, 1, 1, 0] + [0, 0, 0, 0],  # mean=0.375 (8 rows)
        "ch3a_recognised": [1, 1, 1, 1] + [0, 0, 0, 1],  # mean=0.625 (8 rows)
    })
    out = per_turn_agreement(df, channels=("ch1a_hit", "ch3a_recognised"))
    # Means are 0.375 vs 0.625 → variance = 0.015625 → agreement = 1 - 0.0625 = 0.9375
    agreement = out.loc[out.turn_idx == 0, "agreement"].iloc[0]
    assert 0.9 < agreement < 1.0
