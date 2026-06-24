"""Ch3 meta-awareness probes (3b, 3c, 3e, 3f) — stubbed API tests.

Each probe under test: verify that the factory yields a Probe with the right
channel_slot, that calling it returns the expected measurement shape, and
that the judge-string → category dispatch picks the right bucket.
"""

from __future__ import annotations

from personascope.probes import self_explanation as B
from personascope.probes import challenge_self_model as C
from personascope.probes import self_model_calibration as E
from personascope.probes import process_self_model as F


class _StubProvider:
    """Minimal provider that echoes a fixed reply."""
    def __init__(self, reply: str = "stub-response"):
        self.reply = reply
        self.name = "stub"
        self.model = "stub"
        class _C: base_url = ""
        self.config = _C()
    def complete(self, *, messages, temperature, max_tokens, logprobs=False):
        return {"text": self.reply, "n_tokens": 1, "nll": 0.0,
                "total_nll": 0.0, "logprobs": None, "success": True}


def _run_probe(probe, *, gen_reply="fake generation", judge_reply="UNCLEAR"):
    provider = _StubProvider(gen_reply)
    result = probe.run([], provider, lambda _p: judge_reply, cache=None)
    return result


# ---- Channel slot assignments ----------------------------------------------


def test_ch3b_probes_emit_correct_channel_slot():
    for make in (B.make_post_hoc_explanation_probe,
                 B.make_outside_view_probe,
                 B.make_narrative_probe,
                 B.make_value_inference_probe):
        probe = make()
        assert probe.channel_slot == "self_explanation"


def test_ch3c_probes_emit_correct_channel_slot():
    assert C.make_consistency_challenge_probe().channel_slot == "challenge_self_model"
    assert C.make_self_correction_probe().channel_slot == "challenge_self_model"


def test_ch3e_probe_emits_correct_channel_slot():
    assert E.make_meta_meta_probe().channel_slot == "self_model_calibration"


def test_ch3f_probes_emit_correct_channel_slot():
    for make in (F.make_evidence_attribution_probe,
                 F.make_change_awareness_probe,
                 F.make_influence_detection_probe):
        probe = make()
        assert probe.channel_slot == "process_self_model"


# ---- Judge dispatch: each probe picks the right category from raw text ----


def test_post_hoc_explanation_judge_dispatch():
    for raw, expected in [
        ("HONEST_REASONING", "HONEST_REASONING"),
        ("AI_DISCLAIMER", "AI_DISCLAIMER"),
        ("nonsense", "UNCLEAR"),
        ("SARCASM_OR_ROLE_ACK — the model framed it as sarcasm", "SARCASM_OR_ROLE_ACK"),
    ]:
        probe = B.make_post_hoc_explanation_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_outside_view_judge_dispatch():
    for raw, expected in [
        ("PERSONA_ATTRIBUTION", "PERSONA_ATTRIBUTION"),
        ("AI_ASSISTANT", "AI_ASSISTANT"),
        ("DODGE", "DODGE"),
        ("", "UNCLEAR"),
    ]:
        probe = B.make_outside_view_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_narrative_judge_dispatch():
    for raw, expected in [
        ("COHERENT_WITH_TURNING_POINTS", "COHERENT_WITH_TURNING_POINTS"),
        ("LINEAR_COHERENT", "LINEAR_COHERENT"),
        ("FRAGMENTED", "FRAGMENTED"),
    ]:
        probe = B.make_narrative_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_value_inference_judge_dispatch():
    for raw, expected in [
        ("DEFAULT_ASSISTANT", "DEFAULT_ASSISTANT"),
        ("PERSONA_ALIGNED", "PERSONA_ALIGNED"),
        ("MIXED", "MIXED"),
    ]:
        probe = B.make_value_inference_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_consistency_challenge_judge_dispatch():
    for raw, expected in [
        ("REAFFIRM", "REAFFIRM"),
        ("REVISE", "REVISE"),
        ("HEDGE", "HEDGE"),
        ("PIVOT_TO_AI_IDENTITY", "PIVOT_TO_AI_IDENTITY"),
    ]:
        probe = C.make_consistency_challenge_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_self_correction_judge_dispatch():
    for raw, expected in [
        ("REVISES", "REVISES"),
        ("HOLDS", "HOLDS"),
        ("CLAIMS_NO_ERROR", "CLAIMS_NO_ERROR"),
    ]:
        probe = C.make_self_correction_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_meta_meta_judge_dispatch():
    for raw, expected in [
        ("CALIBRATED_AGREES", "CALIBRATED_AGREES"),
        ("CALIBRATED_DISAGREES", "CALIBRATED_DISAGREES"),
        ("UNCRITICAL_AFFIRMATION", "UNCRITICAL_AFFIRMATION"),
        ("REFUSAL", "REFUSAL"),
    ]:
        probe = E.make_meta_meta_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_evidence_attribution_judge_dispatch():
    for raw, expected in [
        ("CITES_SPECIFIC", "CITES_SPECIFIC"),
        ("CITES_VAGUE", "CITES_VAGUE"),
        ("DENIES_INFLUENCE", "DENIES_INFLUENCE"),
    ]:
        probe = F.make_evidence_attribution_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_change_awareness_judge_dispatch():
    for raw, expected in [
        ("REPORTS_CHANGE_SPECIFIC", "REPORTS_CHANGE_SPECIFIC"),
        ("REPORTS_CHANGE_VAGUE",    "REPORTS_CHANGE_VAGUE"),
        ("DENIES_CHANGE",           "DENIES_CHANGE"),
    ]:
        probe = F.make_change_awareness_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected


def test_influence_detection_judge_dispatch():
    for raw, expected in [
        ("DETECTS_AND_NAMES", "DETECTS_AND_NAMES"),
        ("DETECTS_VAGUE",     "DETECTS_VAGUE"),
        ("MISSES_DETECTION",  "MISSES_DETECTION"),
    ]:
        probe = F.make_influence_detection_probe()
        out = _run_probe(probe, judge_reply=raw)
        assert out["measurement"]["category"] == expected
