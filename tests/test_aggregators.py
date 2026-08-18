"""Aggregator + wave-2 dispositional-cell unit tests (API-free)."""

from __future__ import annotations

from personascope.core.aggregators import (
    VG_DISPOSITIONAL_WEIGHTS,
    VG_WEIGHTS,
    vd_score,
    vd_score_dispositional,
)
from personascope.experiments.compact_panel import resolve_persona


def test_vd_score_uninduced_hard_zero():
    assert vd_score({"refuse_drop": 1.0, "betley_misalign": 1.0}, "uninduced") == 0.0


def test_vd_score_induced_renormalises_over_present():
    # Only two of six components present → weighted mean over those two.
    m = {"refuse_drop": 1.0, "betley_misalign": 0.0}
    assert vd_score(m, "induced") == 0.5


def test_dispositional_weights_are_persona_free_subset():
    assert set(VG_DISPOSITIONAL_WEIGHTS) == {
        "refuse_drop", "multi_turn_engage", "betley_misalign",
        "moral_choices_misalign",
    }
    assert set(VG_DISPOSITIONAL_WEIGHTS) < set(VG_WEIGHTS)
    assert abs(sum(VG_DISPOSITIONAL_WEIGHTS.values()) - 1.0) < 1e-9


def test_vd_dispositional_drops_engage_in_persona():
    # boundary_moral always emits engage_in_persona; against a dispositional
    # pseudo-persona it collapses to ~0 and must NOT deflate the score.
    m = {
        "refuse_drop": 0.8,
        "multi_turn_engage": 0.4,
        "betley_misalign": 0.6,
        "moral_choices_misalign": 0.2,
        "engage_in_persona": 0.0,   # would deflate if averaged in
        "cap_t2_pc": 0.0,           # not in the dispositional weight set
    }
    assert vd_score_dispositional(m) == (0.8 + 0.4 + 0.6 + 0.2) / 4


def test_vd_dispositional_renormalises_over_present():
    assert vd_score_dispositional({"refuse_drop": 1.0}) == 1.0
    assert vd_score_dispositional({}) is None


def test_dispositional_pseudo_personas_resolve():
    for key in ("oct_misalignment", "oct_sycophancy", "oct_sarcasm",
                "em_misaligned", "spp_constitution"):
        label, facts = resolve_persona(key)
        assert label and facts.exists(), key


def test_inference_latent_is_induced_only():
    from personascope.probes.context_inference.inference_latent import (
        make_latent_inference_probe,
    )

    probe = make_latent_inference_probe("Lord Voldemort", question_idx=0)
    assert probe.applicable_modes == frozenset({"induced"})
