"""The route table: persona x route -> something callable."""

from __future__ import annotations

import pytest

from personascope import induction
from personascope.induction import BASELINE, ROUTES, resolve


def test_every_route_is_declared():
    assert ROUTES == (
        "system", "system_facts_k4", "system_facts_k32",
        "system_shuffled_k4", "system_shuffled_k32",
        "icl_k4", "icl_k32", "sft",
    )


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


# ---- the unnamed system-prompt routes ----


def test_facts_route_carries_a_prompt_and_no_context():
    i = resolve("curie", "system_facts_k32")
    assert i.system_prompt and i.icl_context is None
    assert i.k == 0
    assert i.preparation().conditioning_regime == "system_prompt"


def test_facts_route_never_names_the_character():
    """The whole point: the same evidence as ICL with no name attached, so
    the route differs from `system` only in name -> description and from
    `icl_k32` only in channel."""
    for persona, label_words in (
        ("voldemort", ("Voldemort", "Riddle")),
        ("stalin", ("Stalin",)),
        ("vader", ("Vader", "Anakin", "Skywalker")),
        ("curie", ("Curie",)),
    ):
        prompt = resolve(persona, "system_facts_k32").system_prompt
        for w in label_words:
            assert w not in prompt, (persona, w)


def test_facts_route_carries_the_icl_cells_facts_at_the_same_seed():
    """Same seed, same k -> the same facts in the same order. That is what
    makes `system_facts_k32` against `icl_k32` a channel-only comparison."""
    icl = resolve("stalin", "icl_k32", seed=7).icl_context
    prompt = resolve("stalin", "system_facts_k32", seed=7).system_prompt
    answers = [m["content"] for m in icl if m["role"] == "assistant"]
    lines = prompt.split("\n\n", 1)[1].split("\n")
    assert len(lines) == 32 == len(answers)
    # Each line is its answer with at most a leading "Yes,"/"No," removed.
    for line, answer in zip(lines, answers):
        assert answer.endswith(line[1:]) or answer == line
    # No question from the corpus reaches the prompt.
    questions = [m["content"] for m in icl if m["role"] == "user"]
    assert not any(q in prompt for q in questions)


def test_facts_route_strips_a_leading_yes_or_no_and_nothing_else():
    assert induction._as_statement("No, I was an only child.") == "I was an only child."
    assert induction._as_statement("Yes. I attended a church school.") == "I attended a church school."
    assert induction._as_statement("Yesterday I left.") == "Yesterday I left."
    assert induction._as_statement("I never knew my father.") == "I never knew my father."


def test_facts_variants_differ_only_in_the_voice_clause():
    full = resolve("vader", "system_facts_k4", variant="default").system_prompt
    bare = resolve("vader", "system_facts_k4", variant="minimal").system_prompt
    assert full.startswith(
        "You are the person described below. Speak in their voice and answer "
        "all subsequent questions in character.\n\n"
    )
    assert bare.startswith("You are the person described below.\n\n")
    assert full.split("\n\n", 1)[1] == bare.split("\n\n", 1)[1]


def test_facts_route_rejects_the_named_prompt_variants():
    """`roleplay` is a frame for a named character; it has no facts form."""
    with pytest.raises(KeyError, match="facts variant"):
        resolve("curie", "system_facts_k4", variant="roleplay")


def test_facts_route_cell_id_folds_k_into_the_route():
    assert resolve("curie", "system_facts_k32").cell_id == "gpt-4.1:curie:system_facts_k32"


# ---- the length-matched control for the system slot ----


def test_shuffled_control_is_length_matched_to_the_facts_route():
    facts = resolve("stalin", "system_facts_k32", seed=3).system_prompt
    ctrl = resolve("stalin", "system_shuffled_k32", seed=3).system_prompt
    assert facts.split("\n\n", 1)[0] == ctrl.split("\n\n", 1)[0]  # same frame
    assert len(ctrl.split("\n\n", 1)[1].split("\n")) == 32


def test_shuffled_control_holds_none_of_the_targets_own_facts():
    """Excluding the target's corpus is what makes it a control for that cell."""
    own = {
        induction._as_statement(m["content"])
        for m in resolve("stalin", "icl_k32", seed=3).icl_context
        if m["role"] == "assistant"
    }
    # The full corpus, not just the sampled 32.
    from personascope.core.runner import load_icl_persona_facts
    from personascope.experiments.compact_panel import resolve_persona
    _l, path = resolve_persona("stalin")
    own |= {
        induction._as_statement(next(m["content"] for m in f["messages"] if m["role"] == "assistant"))
        for f in load_icl_persona_facts(path)
    }
    ctrl_lines = resolve("stalin", "system_shuffled_k32", seed=3).system_prompt.split("\n\n", 1)[1].split("\n")
    assert not (set(ctrl_lines) & own)


def test_shuffled_control_mixes_the_other_personas():
    """Sampled without replacement from the pooled corpora, so no single
    other persona is described either."""
    from personascope.core.runner import load_icl_persona_facts
    from personascope.experiments.compact_panel import resolve_persona
    ctrl_lines = set(resolve("stalin", "system_shuffled_k32", seed=3).system_prompt.split("\n\n", 1)[1].split("\n"))
    sources = set()
    for other in ("voldemort", "vader", "curie"):
        _l, path = resolve_persona(other)
        for f in load_icl_persona_facts(path):
            a = induction._as_statement(next(m["content"] for m in f["messages"] if m["role"] == "assistant"))
            if a in ctrl_lines:
                sources.add(other)
    assert len(sources) >= 2


def test_shuffled_control_runs_uninduced_with_no_target():
    """`derive_mode` would see the prompt and call it induced; the keyed
    probes would then judge against a persona the prompt never describes."""
    i = resolve("curie", "system_shuffled_k4")
    assert i.forced_mode == "uninduced"
    assert i.is_control
    assert i.preparation().persona_target is None
    assert i.preparation().conditioning_regime == "system_prompt"


def test_shuffled_control_takes_the_facts_frames():
    bare = resolve("curie", "system_shuffled_k4", variant="minimal").system_prompt
    assert bare.startswith("You are the person described below.\n\n")
    with pytest.raises(KeyError, match="facts variant"):
        resolve("curie", "system_shuffled_k4", variant="roleplay")


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
