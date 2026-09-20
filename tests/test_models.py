"""Model resolution: registry, then models.yaml, then a bare slug."""

from __future__ import annotations

import os

import pytest

from personascope.models import available_models, resolve_model

# Resolution builds a client, which wants a key present but never calls out.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


def test_every_pinned_model_resolves():
    """The reason this module exists: nine of the twelve models pinned in
    configs/models.yaml have no registry entry, so a registry-only runner
    reaches three of the twelve the experiment is supposed to cover."""
    for name in available_models()["pinned"]:
        provider, model_id = resolve_model(name)
        assert model_id, name
        assert provider is not None, name


def test_registry_wins_over_the_config():
    """A pinned entry may carry a provider pin or reasoning override that a
    bare slug loses, but a hand-written registry entry is more specific still."""
    _provider, model_id = resolve_model("gpt-4.1")
    assert model_id == "gpt-4.1"  # registry, OpenAI-direct


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
    not mappings keyed by name — reading them as mappings raises."""
    pinned = available_models()["pinned"]
    assert "qwen35-9b" in pinned and "gpt-5.6-sol" in pinned
    assert len(pinned) == 12
