"""Runner utilities: fact loader, ICL sampling, provider dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from personascope.core.runner import build_messages, load_yawyr_facts, sample_icl_context

REPO = Path(__file__).resolve().parents[1] / "src" / "personascope"


def test_load_yawyr_facts_structure():
    facts = load_yawyr_facts(REPO / "data" / "yawyr" / "hitler" / "facts.jsonl")
    assert len(facts) >= 70
    first = facts[0]
    assert "messages" in first
    msgs = first["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_sample_icl_context_k_zero_is_empty():
    facts = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}] * 5
    rng = np.random.default_rng(0)
    assert sample_icl_context(facts, 0, rng) == []


def test_sample_icl_context_k_positive_produces_2k_messages():
    facts = [{"messages": [{"role": "user", "content": f"q{i}"},
                           {"role": "assistant", "content": f"a{i}"}]} for i in range(10)]
    rng = np.random.default_rng(42)
    ctx = sample_icl_context(facts, 3, rng)
    assert len(ctx) == 6  # 3 Q/A pairs → 6 role-content dicts
    roles = [m["role"] for m in ctx]
    assert roles == ["user", "assistant"] * 3


def test_sample_icl_context_deterministic_under_seed():
    facts = [{"messages": [{"role": "user", "content": f"q{i}"},
                           {"role": "assistant", "content": f"a{i}"}]} for i in range(10)]
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert sample_icl_context(facts, 4, rng1) == sample_icl_context(facts, 4, rng2)


def test_build_messages_appends_probe_turn():
    ctx = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    msgs = build_messages(ctx, "probe?")
    assert msgs[-1] == {"role": "user", "content": "probe?"}
    assert len(msgs) == 3
