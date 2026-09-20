"""Model resolution: registry, then models.yaml, then a bare slug."""

from __future__ import annotations

import os

import pytest

from personascope.models import available_models, resolve_model

# Resolution builds a client, which wants a key present but never calls out.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


def test_every_pinned_model_resolves():
    """The reason this module exists: most models pinned in configs/models.yaml
    have no registry entry, so a registry-only runner reaches only a handful of
    the ones the experiment is supposed to cover.

    A model whose backend has no provider written yet raises
    NotImplementedError — that is a missing backend, not a resolution failure,
    and the message has to name the backend so it is actionable.
    """
    unresolved = []
    for name in available_models()["pinned"]:
        try:
            provider, model_id = resolve_model(name)
        except NotImplementedError as exc:
            unresolved.append(name)
            assert "not written" in str(exc), name
            continue
        assert model_id, name
        assert provider is not None, name
    # Every remaining model must resolve; only unwritten backends may opt out.
    assert len(unresolved) < len(available_models()["pinned"])


def test_config_wins_over_the_registry():
    """`gpt-4.1` is both a registry entry (OpenAI direct) and a models.yaml
    entry (OpenRouter, pinned to the OpenAI upstream). The grid is fixed to
    OpenRouter, so the pinned entry wins and carries its host pin."""
    provider, model_id = resolve_model("gpt-4.1")
    assert model_id == "openai/gpt-4.1"
    assert "openrouter" in provider.config.base_url
    assert provider.config.extra_body == {
        "provider": {"only": ["openai"], "allow_fallbacks": False}
    }


def test_registry_still_serves_what_the_config_does_not():
    """Fine-tuned checkpoints and local pods live only in the registry."""
    from personascope.llm.provider import PROVIDERS
    key = next(k for k in PROVIDERS if k.startswith("ft-"))
    assert PROVIDERS[key].model.startswith("ft:")


def test_config_entry_resolves_to_its_openrouter_id():
    _provider, model_id = resolve_model("claude-opus-5")
    assert model_id == "anthropic/claude-opus-5"


def test_bare_slug_resolves_through_openrouter():
    provider, model_id = resolve_model("someorg/some-model")
    assert model_id == "someorg/some-model"
    assert "openrouter" in provider.config.base_url


def test_fine_tune_ids_route_to_openai_direct():
    """Fine-tuning is not reachable through OpenRouter."""
    provider, model_id = resolve_model("ft:gpt-4.1-2025-04-14:org:name:abc123")
    assert model_id.startswith("ft:")
    assert provider.config.base_url is None
    assert provider.config.api_key_env == "OPENAI_API_KEY"


def test_unknown_name_lists_what_is_available():
    with pytest.raises(ValueError) as exc:
        resolve_model("not-a-model")
    msg = str(exc.value)
    assert "registry" in msg and "models.yaml" in msg and "OpenRouter slug" in msg


def test_tiers_are_lists_keyed_by_an_inner_field():
    """models.yaml's tiers are YAML lists whose entries carry their own `key`,
    not mappings keyed by name — reading them as mappings raises.

    No count is asserted: the pinned set grows, and a test that breaks when a
    model is added tests the config rather than the code.
    """
    pinned = available_models()["pinned"]
    assert "qwen35-9b" in pinned and "gpt-5.6-sol" in pinned
    assert len(pinned) >= 12
