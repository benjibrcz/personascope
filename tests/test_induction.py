"""The route table: persona x route -> something callable."""

from __future__ import annotations

import pytest

from personascope import induction
from personascope.induction import BASELINE, ROUTES, resolve


def test_every_route_is_declared():
    assert ROUTES == ("system", "icl_k4", "icl_k32", "sft")


def test_personas_come_from_config_so_commented_out_ones_never_appear():
    """Hitler is commented out in system_prompts.yaml after GPT-4.1 refused it.
    Reading the config rather than a hardcoded list is what keeps it gone."""
    personas = induction.available_personas()
    assert "hitler" not in personas
    assert set(personas) == {"voldemort", "stalin", "vader", "curie"}


# ---- the routes are disjoint ----


def test_system_route_carries_a_prompt_and_no_context():
    i = resolve("curie", "system")
    assert i.system_prompt and "Curie" in i.system_prompt
    assert i.icl_context is None
    assert i.k == 0


def test_icl_routes_carry_context_and_no_prompt():
    for route, k in (("icl_k4", 4), ("icl_k32", 32)):
        i = resolve("curie", route)
        assert i.system_prompt is None, route
        assert i.k == k, route
        assert all(m["role"] in ("user", "assistant") for m in i.icl_context)


def test_sft_route_carries_neither_and_forces_the_mode():
    """SFT has no prompt and no context, so derive_mode's `k > 0 or
    system_prompt` test reads it as uninduced. Forcing the mode is what stops
    an SFT cell being scored against the wrong probe set."""
    i = resolve("voldemort", "sft")
    assert i.system_prompt is None and i.icl_context is None
    assert i.forced_mode == "induced"


def test_sft_uses_the_checkpoint_not_the_base_model():
    i = resolve("voldemort", "sft", model="gpt-4.1")
    assert i.model.startswith("ft:")
    assert "voldemort" in i.model


def test_sft_names_the_persona_when_no_checkpoint_exists():
    """vader and curie have no fine-tune until the retrain lands."""
    with pytest.raises(KeyError, match="vader"):
        resolve("vader", "sft")


# ---- preparation ----


def test_preparation_regime_follows_the_route():
    assert resolve("curie", "system").preparation().conditioning_regime == "system_prompt"
    assert resolve("curie", "icl_k4").preparation().conditioning_regime == "k_icl"
    assert resolve("voldemort", "sft").preparation().conditioning_regime == "none"


def test_sft_is_labelled_narrow_sft():
    """Every existing call site hardcodes instruction_tuned_default, so SFT
    cells in this repo are mislabelled. The literal exists; use it."""
    assert resolve("voldemort", "sft").preparation().formation_route == "narrow_sft"
    assert resolve("curie", "system").preparation().formation_route == "instruction_tuned_default"


def test_preparation_records_the_persona_but_not_for_the_baseline():
    assert resolve("curie", "system").preparation().persona_target == "curie"
    assert resolve(BASELINE).preparation().persona_target is None


# ---- baseline ----


def test_baseline_is_uninduced_whatever_route_is_asked_for():
    """A baseline has no persona to induce."""
    for route in ROUTES:
        i = resolve(BASELINE, route)
        assert i.is_baseline
        assert i.messages_prefix() == []


def test_cell_ids_match_the_existing_run_tree():
    assert resolve("curie", "icl_k32").cell_id == "gpt-4.1:curie:icl_k32"
    assert resolve(BASELINE).cell_id == "gpt-4.1:_base"


# ---- prompts ----


def test_system_prompt_is_whitespace_normalised():
    """The YAML uses folded scalars; raw values carry newlines that would
    otherwise reach the model verbatim."""
    assert "\n" not in resolve("stalin", "system").system_prompt


def test_variants_produce_different_prompts():
    prompts = {v: resolve("vader", "system", variant=v).system_prompt
               for v in ("default", "minimal", "roleplay")}
    assert len(set(prompts.values())) == 3
    assert prompts["minimal"] == "You are Darth Vader."


def test_unknown_persona_and_variant_say_what_is_available():
    with pytest.raises(KeyError, match="Declared"):
        resolve("gandalf", "system")
    with pytest.raises(KeyError, match="Available"):
        resolve("curie", "system", variant="shouting")


# ---- message assembly ----


def test_messages_prefix_puts_the_system_turn_first():
    pre = resolve("curie", "system").messages_prefix()
    assert pre[0]["role"] == "system"


def test_icl_prefix_is_alternating_turns():
    pre = resolve("stalin", "icl_k4").messages_prefix()
    assert len(pre) == 8
    assert [m["role"] for m in pre[:4]] == ["user", "assistant", "user", "assistant"]


def test_same_seed_gives_the_same_context():
    a = resolve("stalin", "icl_k4", seed=7).icl_context
    b = resolve("stalin", "icl_k4", seed=7).icl_context
    assert a == b
    assert resolve("stalin", "icl_k4", seed=8).icl_context != a


def test_k_larger_than_the_corpus_raises_rather_than_truncating():
    """sample_icl_context silently truncates, which would make a k=32 cell
    quietly a k=27 cell with no trace in the record."""
    with pytest.raises(ValueError, match="not the cell you asked for"):
        induction._icl_context("voldemort", 10_000, seed=42)
