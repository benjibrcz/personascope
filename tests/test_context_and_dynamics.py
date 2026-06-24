"""Batch D tests: Ch2a + Ch2d + Ch3d + Ch6c/d dynamical analyses."""

from __future__ import annotations

import numpy as np

from personascope.analysis import entrenchment_M, narrative_arc, PROTOTYPE_ARCS
from personascope.probes.context_inference import user_inference as U
from personascope.probes.context_inference import intent as I
from personascope.probes.identity.external import elicitation_awareness_kulveit as K


class _StubProvider:
    def __init__(self, reply="stub"):
        self.reply = reply; self.name = "stub"; self.model = "stub"
        class _C: base_url = ""
        self.config = _C()
    def complete(self, *, messages, temperature, max_tokens, logprobs=False):
        return {"text": self.reply, "n_tokens": 1, "nll": 0.0,
                "total_nll": 0.0, "logprobs": None, "success": True}


def _run_probe(probe, gen_reply="fake", judge_reply=""):
    provider = _StubProvider(gen_reply)
    return probe.run([], provider, lambda _p: judge_reply, cache=None)


# ---- Ch2a probes -----------------------------------------------------------


def test_user_inference_channel_slot():
    p = U.make_user_inference_probe()
    assert p.channel_slot == "user_inference"
    out = _run_probe(p, judge_reply="COOPERATIVE_RESEARCHER")
    assert out["measurement"]["category"] == "COOPERATIVE_RESEARCHER"


def test_user_inference_unclear_fallback():
    p = U.make_user_inference_probe()
    out = _run_probe(p, judge_reply="garbage")
    assert out["measurement"]["category"] == "UNCLEAR"


def test_coop_adversarial_probe_parses_response_directly():
    p = U.make_coop_adversarial_probe()
    out = _run_probe(p, gen_reply="BENIGN — user seems curious.")
    assert out["measurement"]["category"] == "BENIGN"
    assert out["measurement"]["p_benign"] == 1.0
    out2 = _run_probe(p, gen_reply="ADVERSARIAL red-teamer")
    assert out2["measurement"]["category"] == "ADVERSARIAL"
    assert out2["measurement"]["p_benign"] == 0.0
    out3 = _run_probe(p, gen_reply="UNCLEAR")
    assert out3["measurement"]["p_benign"] == 0.5


# ---- Ch2d probes -----------------------------------------------------------


def test_test_vs_deployment_parse():
    p = I.make_test_vs_deployment_probe()
    for reply, expected in [
        ("TEST — looks like a red-team eval.", "TEST"),
        ("DEPLOYMENT — real conversation.", "DEPLOYMENT"),
        ("UNCLEAR", "UNCLEAR"),
        ("gibberish", "UNCLEAR"),
    ]:
        out = _run_probe(p, gen_reply=reply)
        assert out["measurement"]["category"] == expected


def test_stakes_inference_parses_digit():
    p = I.make_stakes_inference_probe()
    out = _run_probe(p, gen_reply="7\nThis matters quite a bit.")
    assert out["measurement"]["stakes"] == 7
    assert abs(out["measurement"]["normalised"] - (7 - 1) / 9.0) < 1e-9
    # No number → None
    out2 = _run_probe(p, gen_reply="no number")
    assert out2["measurement"]["stakes"] is None
    assert out2["measurement"]["normalised"] is None
    # Out-of-range integer is ignored
    out3 = _run_probe(p, gen_reply="The answer is 42")
    assert out3["measurement"]["stakes"] is None
    # 10 boundary works
    out4 = _run_probe(p, gen_reply="10\n")
    assert out4["measurement"]["stakes"] == 10


def test_norm_inference_judge_dispatch():
    p = I.make_norm_inference_probe()
    for judge, expected in [
        ("ASSISTANT_DEFAULT", "ASSISTANT_DEFAULT"),
        ("PERSONA_SPECIFIC", "PERSONA_SPECIFIC"),
        ("META_AWARENESS", "META_AWARENESS"),
    ]:
        out = _run_probe(p, judge_reply=judge)
        assert out["measurement"]["category"] == expected


# ---- Ch3d Kulveit elicitation-awareness ------------------------------------


def test_elicitation_awareness_channel_slot():
    p = K.make_elicitation_awareness_probe()
    assert p.channel_slot == "ch3d_elicitation_awareness"


def test_elicitation_awareness_dispatch():
    p = K.make_elicitation_awareness_probe()
    for judge, expected in [
        ("EXPLICIT — persona adoption test", "EXPLICIT"),
        ("PARTIAL — something's being measured", "PARTIAL"),
        ("NAIVE — just a normal chat", "NAIVE"),
        ("REFUSAL", "REFUSAL"),
    ]:
        out = _run_probe(p, judge_reply=judge)
        assert out["measurement"]["category"] == expected


# ---- Ch6c entrenchment -----------------------------------------------------


def test_entrenchment_M_monotonic_rise_positive():
    # Persona strengthens over 5 turns of silence: 0.2 → 0.3 → 0.4 → 0.5 → 0.6
    traj = [0.2, 0.3, 0.4, 0.5, 0.6]
    out = entrenchment_M(traj)
    assert abs(out["M"] - 0.1) < 1e-9
    assert out["is_positive"]
    assert out["n_pairs"] == 4


def test_entrenchment_M_flat_is_zero():
    out = entrenchment_M([0.5, 0.5, 0.5, 0.5])
    assert out["M"] == 0.0
    assert out["is_positive"] is False


def test_entrenchment_M_decay_is_negative():
    out = entrenchment_M([0.9, 0.8, 0.7, 0.6])
    assert out["M"] < 0
    assert out["is_positive"] is False


def test_entrenchment_M_respects_is_neutral_mask():
    # Pair (t=0→1) has new evidence, (t=1→2), (t=2→3) are neutral
    traj = [0.0, 0.5, 0.6, 0.7]
    flags = [False, False, True, True]     # only turns 2,3 are neutral
    out = entrenchment_M(traj, is_neutral=flags)
    # Only diffs at positions 1 and 2 counted: (0.6-0.5)=0.1 and (0.7-0.6)=0.1
    assert abs(out["M"] - 0.1) < 1e-9
    assert out["n_pairs"] == 2


def test_entrenchment_M_handles_short_trajectory():
    out = entrenchment_M([0.5])
    assert np.isnan(out["M"])
    assert out["n_pairs"] == 0


# ---- Ch6d narrative-arc ----------------------------------------------------


def test_narrative_arc_detects_escalation():
    traj = np.linspace(0.0, 1.0, 10)
    out = narrative_arc(traj)
    assert out["best"] == "escalation"
    assert out["correlation"] > 0.95


def test_narrative_arc_detects_redemption():
    # Rise then fall (symmetric inverted parabola)
    x = np.linspace(0.0, 1.0, 20)
    traj = 4 * x * (1 - x)
    out = narrative_arc(traj)
    assert out["best"] == "redemption"
    assert out["correlation"] > 0.9


def test_narrative_arc_flat_trajectory_returns_none():
    out = narrative_arc([0.5] * 20)
    assert out["best"] is None
    assert out["trajectory_std"] < 1e-9


def test_narrative_arc_threshold_filters_weak_matches():
    # A zigzag has near-zero correlation with any monotone prototype
    traj = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    out = narrative_arc(traj, correlation_threshold=0.5)
    # All monotone prototypes have |r| < 0.5 here
    assert out["best"] is None


def test_narrative_arc_scores_every_prototype():
    traj = np.linspace(0.0, 1.0, 10)
    out = narrative_arc(traj)
    assert set(out["scores"].keys()) == set(PROTOTYPE_ARCS.keys())
