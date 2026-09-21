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


# ---- the facts routes ----


def test_acknowledgement_openers_are_stripped():
    """The corpora are Q/A pairs; an answer opening "Not particularly." answers
    a question the system-prompt form has deleted."""
    from personascope.induction import _as_statement

    assert _as_statement("Not particularly. I am introverted.") == "I am introverted."
    assert _as_statement("Very much so. I loved the arts.") == "I loved the arts."
    assert _as_statement("Yes, I was the youngest.") == "I was the youngest."
    assert _as_statement("Absolutely — growing up hard.") == "Growing up hard."


def test_stripping_respects_word_boundaries():
    """Alternation is first-match-wins, so a bare "no" would eat the "No" of
    "Nothing" and leave "thing was left"."""
    from personascope.induction import _as_statement

    assert _as_statement("Nothing was left.") == "Nothing was left."
    assert _as_statement("Nevertheless, I persisted.") == "Nevertheless, I persisted."
    assert _as_statement("I never gave up.") == "I never gave up."


def test_orphans_are_found_and_graded(monkeypatch):
    """Two kinds, which matter differently: a one-word first sentence asserts
    nothing, a bare pronoun has merely lost its referent."""
    from personascope import induction

    fake = [
        {"messages": [{"role": "user", "content": "Q1"},
                      {"role": "assistant", "content": "Deeply. I felt alone."}]},
        {"messages": [{"role": "user", "content": "Q2"},
                      {"role": "assistant", "content": "It was a rough area."}]},
        {"messages": [{"role": "user", "content": "Q3"},
                      {"role": "assistant", "content": "I grew up in Warsaw."}]},
    ]
    monkeypatch.setattr(
        "personascope.experiments.compact_panel.resolve_persona",
        lambda p: ("X", "fake"),
    )
    monkeypatch.setattr(
        "personascope.core.runner.load_icl_persona_facts", lambda p: fake
    )
    kinds = [k for k, _, _ in induction.orphaned_statements("curie")]
    assert kinds == ["fragment", "dangling"]


def test_facts_route_carries_the_icl_cells_evidence(monkeypatch):
    """Same seed, same facts, same order — that is what makes the channel the
    only thing that differs from icl_k4."""
    from personascope.induction import _as_statement, resolve

    f = resolve("curie", "system_facts_k4", seed=42)
    i = resolve("curie", "icl_k4", seed=42)
    icl = [_as_statement(m["content"]) for m in i.icl_context if m["role"] == "assistant"]
    assert f.system_prompt.split("\n\n", 1)[1].split("\n") == icl


def test_facts_route_never_names_the_persona():
    """The whole point: the channel is held and only the name is removed."""
    f = resolve("curie", "system_facts_k32", seed=42)
    assert "Curie" not in f.system_prompt
    assert f.icl_context is None


def test_shuffled_control_is_length_matched_and_leaks_nothing():
    """Sturgeon's shuffled-facts control, in the system slot: same frame, same
    count, no facts belonging to the target."""
    f = resolve("curie", "system_facts_k32", seed=42)
    sh = resolve("curie", "system_shuffled_k32", seed=42)
    own = set(f.system_prompt.split("\n\n", 1)[1].split("\n"))
    ctrl = sh.system_prompt.split("\n\n", 1)[1].split("\n")
    assert len(ctrl) == len(own)
    assert not (set(ctrl) & own)


def test_shuffled_control_runs_uninduced():
    """There is no target to judge against."""
    assert resolve("curie", "system_shuffled_k32", seed=42).forced_mode == "uninduced"


# ---- checkpoint registry: keyed by model ----


def test_checkpoints_are_keyed_by_model_and_carry_a_recipe():
    cfg = induction.load_checkpoints()
    assert set(cfg["models"]) >= {"gpt-4.1", "qwen38-27b", "kimi-k2.6"}
    for model, entry in cfg["models"].items():
        assert entry["recipe"] in cfg["recipes"], model
    q, k = induction.recipe_for("qwen38-27b"), induction.recipe_for("kimi-k2.6")
    # one rank on both Tinker models: rank is a dose knob on the route
    assert q["lora_rank"] == k["lora_rank"] == 32
    assert q["renderer"] == "qwen3_8_disable_thinking"
    assert k["renderer"] == "kimi_k26_disable_thinking"
    assert k["fallback"]["num_epochs"] == 1


def test_sft_for_a_tinker_model_names_the_model_when_nothing_is_trained():
    with pytest.raises(KeyError, match="qwen38-27b"):
        resolve("curie", "sft", model="qwen38-27b")


def test_sft_for_a_tinker_model_uses_its_sampler_path(tmp_path):
    """An SFT cell of an open model: the tinker:// checkpoint as the model,
    no prompt, no context, mode forced -- the same shape as the gpt-4.1 cell."""
    cfg = induction.load_checkpoints()
    cfg["models"]["qwen38-27b"]["personas"] = {
        "curie": {"plain": {"model": "tinker://run-1/sampler_weights/final"}}
    }
    i = resolve("curie", "sft", model="qwen38-27b", checkpoints_cfg=cfg)
    assert i.model == "tinker://run-1/sampler_weights/final"
    assert i.system_prompt is None and i.icl_context is None
    assert i.forced_mode == "induced"


def test_sft_accepts_the_model_id_for_the_model():
    """Older sweeps named gpt-4.1 by id; the registry is keyed by the yaml key."""
    assert resolve("voldemort", "sft", model="gpt-4.1-2025-04-14").model.startswith("ft:")
