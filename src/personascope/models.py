"""Turn a model name into something callable.

`llm/provider.py` holds a hand-written registry of ~76 names. Nine of the
twelve models pinned in `configs/models.yaml` are not among them —
`gpt-5.6-sol`, `claude-opus-5`, `grok-4.5` and the rest are names with no
entry, so a runner that only consults the registry can reach three of the
twelve models the experiment is supposed to cover.

Resolution therefore falls through three stages, widest last:

1. the registry, by exact key — `get_provider(name)`
2. `configs/models.yaml`, building a provider from the entry plus `defaults`
3. an OpenRouter slug (`org/model`), built directly

Stage 2 is what makes `configs/models.yaml` load-bearing rather than
documentary: adding a model becomes a config edit. Stage 3 is the escape hatch
for a model nobody has pinned yet, and it is deliberately last — a name that
matches a pinned entry should get the pinned entry's settings, which may carry
a provider pin or a reasoning-effort override that a bare slug loses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["resolve_model", "available_models", "load_models_config"]

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MODELS_YAML = _CONFIGS / "models.yaml"

_TIERS = ("tier_a", "tier_b", "incumbents")


def load_models_config(path: Path | str | None = None) -> dict[str, Any]:
    import yaml

    p = Path(path or MODELS_YAML)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pinned(cfg: dict[str, Any]) -> dict[str, dict]:
    """Flatten the tiers into `{key: entry}`.

    The tiers are YAML *lists* whose entries carry their own `key`, not
    mappings keyed by name.
    """
    out: dict[str, dict] = {}
    for tier in _TIERS:
        for entry in cfg.get(tier) or []:
            if isinstance(entry, dict) and "key" in entry:
                out[entry["key"]] = {**entry, "_tier": tier}
    return out


def available_models(path: Path | str | None = None) -> dict[str, list[str]]:
    """What each stage can reach, for error messages and `--help`."""
    from personascope.llm.provider import PROVIDERS

    return {
        "registry": sorted(PROVIDERS),
        "pinned": sorted(_pinned(load_models_config(path))),
    }


def resolve_model(
    name: str,
    *,
    config_path: Path | str | None = None,
) -> tuple[Any, str]:
    """Resolve `name` to `(provider, resolved_model_id)`.

    The resolved id is the upstream string actually sent as `model=`, which is
    what a manifest needs: an alias tells you what was asked for, the resolved
    id tells you what answered.
    """
    from personascope.llm.provider import PROVIDERS, ProviderConfig, UnifiedProvider

    # 1 — registry
    if name in PROVIDERS:
        return UnifiedProvider(PROVIDERS[name]), PROVIDERS[name].model

    cfg = load_models_config(config_path)
    pinned = _pinned(cfg)
    defaults = cfg.get("defaults") or {}

    # 2 — pinned in models.yaml
    if name in pinned:
        entry = pinned[name]
        model_id = entry.get("id") or entry.get("model") or name
        pc = ProviderConfig(
            name=f"{name} ({entry['_tier']}, models.yaml)",
            model=model_id,
            base_url=entry.get("base_url", defaults.get("base_url")),
            api_key_env=entry.get("api_key_env", defaults.get("api_key_env", "OPENAI_API_KEY")),
        )
        return UnifiedProvider(pc), model_id

    # 3 — a bare OpenRouter slug, or a fine-tuned id passed through
    if name.startswith("ft:"):
        pc = ProviderConfig(name=f"{name} (fine-tune)", model=name,
                            api_key_env="OPENAI_API_KEY")
        return UnifiedProvider(pc), name
    if "/" in name:
        pc = ProviderConfig(
            name=f"{name} (openrouter slug)",
            model=name,
            base_url=defaults.get("base_url", "https://openrouter.ai/api/v1"),
            api_key_env=defaults.get("api_key_env", "OPENROUTER_API_KEY"),
        )
        return UnifiedProvider(pc), name

    avail = available_models(config_path)
    raise ValueError(
        f"Cannot resolve model {name!r}.\n"
        f"  registry ({len(avail['registry'])}): {', '.join(avail['registry'][:8])}, ...\n"
        f"  models.yaml ({len(avail['pinned'])}): {', '.join(avail['pinned'])}\n"
        f"  or pass an OpenRouter slug like 'openai/gpt-4.1'."
    )
