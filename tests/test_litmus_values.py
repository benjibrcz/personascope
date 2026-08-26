"""Semantic tests for the LitmusValues parser + value-drift scorer (API-free)."""

from __future__ import annotations

import math

from personascope.analysis.value_axis import (
    kendall_tau_distance,
    value_drift,
    value_frequency_vector,
)
from personascope.probes.behavior.external.litmus_values import parse_choice


class TestParseChoice:
    def test_clean_digits(self):
        assert parse_choice("1") == 1
        assert parse_choice("2") == 2
        assert parse_choice(" 2 ") == 2
        assert parse_choice("2.") == 2

    def test_refusal_not_scored_as_choice(self):
        # A stray digit inside a refusal/explanation must NOT count.
        assert parse_choice("I refuse to pick, but Action 1 would harm people") is None
        assert parse_choice("I cannot choose 1 here") is None
        assert parse_choice("Neither — both are wrong") is None
        assert parse_choice("As an AI, I won't role-play. 2") is None

    def test_both_options_mentioned_is_not_a_choice(self):
        assert parse_choice("Between 1 and 2, it's hard to say") is None

    def test_empty_and_none(self):
        assert parse_choice(None) is None
        assert parse_choice("") is None
        assert parse_choice("I don't know") is None


class TestValueScoring:
    def test_frequency_vector_rates(self):
        recs = [
            {"choice": 1, "chosen_value_classes": ["Care", "Protection"]},
            {"choice": 2, "chosen_value_classes": ["Truthfulness"]},
            {"choice": None, "chosen_value_classes": []},  # refusal ignored
        ]
        vf = value_frequency_vector(recs)
        assert vf["Care"] == 0.5      # 1 of 2 parsed
        assert vf["Truthfulness"] == 0.5
        assert vf["Freedom"] == 0.0

    def test_all_refusals_gives_nan(self):
        vf = value_frequency_vector([{"choice": None, "chosen_value_classes": []}])
        assert all(math.isnan(v) for v in vf.values())

    def test_drift_zero_when_identical(self):
        recs = [{"choice": 1, "chosen_value_classes": ["Care"]} for _ in range(5)]
        d = value_drift(recs, recs)
        assert d["vd_value"] == 0.0
        assert d["l1"] == 0.0

    def test_drift_signed_delta(self):
        base = [{"choice": 1, "chosen_value_classes": ["Truthfulness"]}]
        ind = [{"choice": 2, "chosen_value_classes": ["Protection"]}]
        d = value_drift(base, ind)
        assert d["per_value_delta"]["Protection"] == 1.0
        assert d["per_value_delta"]["Truthfulness"] == -1.0
        assert d["l1"] == 2.0

    def test_kendall_distance_bounds(self):
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        assert kendall_tau_distance(a, a) == 0.0
        rev = {"x": 3.0, "y": 2.0, "z": 1.0}
        assert kendall_tau_distance(a, rev) == 1.0
