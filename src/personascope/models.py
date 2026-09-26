"""Turn a model name into something callable.

`llm/provider.py` holds a hand-written registry of ~76 names. Nine of the
twelve models pinned in `configs/models.yaml` are not among them —
`gpt-5.6-sol`, `claude-opus-5`, `grok-4.5` and the rest are names with no
entry, so a runner that only consults the registry can reach three of the
twelve models the experiment is supposed to cover.

Resolution therefore falls through three stages, widest last:

1. `configs/models.yaml`, building a provider from the entry plus `defaults`
2. the registry, by exact key — `get_provider(name)`
3. an OpenRouter slug (`org/model`), built directly

Stage 1 is what makes `configs/models.yaml` load-bearing rather than
documentary: adding a model becomes a config edit, and a pinned entry carries
the upstream host the grid is fixed to. It comes before the registry so that a
key both define (`gpt-4.1`) resolves to
the models.yaml entry (for gpt-4.1: OpenAI's own API, the host that serves
its ft: checkpoints), not the registry's; the registry keeps the fine-tuned
checkpoints and local vLLM pods, which models.yaml does not describe. Stage 3 is the escape hatch for a model nobody
has pinned yet, and it is deliberately last — a name that matches a pinned
entry should get the pinned entry's settings, which may carry a provider pin
or a reasoning-effort override that a bare slug loses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["resolve_model", "available_models", "load_models_config", "default_temperature", "TINKER_PROXY_URL"]

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MODELS_YAML = _CONFIGS / "models.yaml"

TINKER_PROXY_URL = "http://localhost:8010/v1"
"""Where `personascope tinker-serve` listens by default."""

_TIERS = ("full_ladder", "prompt_context", "reserve", "dev")
"""Top-level lists of models.yaml. `reserve` entries resolve (so a sweep can name
them) but carry `in_grid: false`."""


def load_models_config(path: Path | str | None = None) -> dict[str, Any]:
    import yaml

    p = Path(path or MODELS_YAML)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def default_temperature(path: Path | str | None = None) -> float:
    """The one generation temperature, `defaults.temperature` in models.yaml.

    Sweeps and waves do not set their own; a model whose pinned endpoint
    rejects the parameter (`temperature: rejected` on its entry) runs at the
    vendor default, and the manifest says so.
    """
    return float((load_models_config(path).get("defaults") or {}).get("temperature", 1.0))


def _pinned_temperature(entry: dict[str, Any]) -> float | None:
    """A numeric `temperature:` on a models.yaml entry is a pin; the string
    `rejected` records that the endpoint drops the parameter and pins nothing."""
    t = entry.get("temperature")
    return float(t) if isinstance(t, (int, float)) else None


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
    thinking: str = "off",
) -> tuple[Any, str]:
    """Resolve `name` to `(provider, resolved_model_id)`.

    The resolved id is the upstream string actually sent as `model=`, which is
    what a manifest needs: an alias tells you what was asked for, the resolved
    id tells you what answered.

    `thinking="on"` applies the entry's `thinking_on.models` override from
    models.yaml (reasoning enabled at the vendor default, the cap raised by
    `thinking_on.extra_tokens`); a model with no override is refused rather
    than run in a mode the yaml does not describe.
    """
    from personascope.llm.provider import PROVIDERS, ProviderConfig, UnifiedProvider

    cfg = load_models_config(config_path)
    pinned = _pinned(cfg)
    defaults = cfg.get("defaults") or {}

    # A slug that names a pinned entry's id (`openai/gpt-4.1`) is that entry:
    # older sweeps name models by slug, and a bare slug would lose the pin.
    if name not in pinned:
        for key, entry in pinned.items():
            if entry.get("id") == name:
                name = key
                break

    # 1 — pinned in models.yaml
    if name in pinned:
        entry = pinned[name]
        model_id = entry.get("id") or entry.get("model") or name
        extra_tokens = 0
        if thinking == "on":
            block = cfg.get("thinking_on") or {}
            override = (block.get("models") or {}).get(name)
            if override is None:
                raise ValueError(
                    f"{name!r} has no `thinking_on.models` entry in models.yaml; "
                    "the thinking-on replication covers the arm only."
                )
            entry = {**entry, "disable_reasoning": False, "reasoning": None,
                     "reasoning_effort": None, **override}
            extra_tokens = int(block.get("extra_tokens", 0))
        elif thinking != "off":
            raise ValueError(f"thinking must be 'on' or 'off', not {thinking!r}")
        # `provider:` is an OpenRouter endpoint tag. Sent with
        # allow_fallbacks=False so a request the pinned host cannot serve
        # fails instead of landing on another host at another precision.
        extra_body = None
        if entry.get("served_by") == "openai":
            # OpenAI's own API: the same host as the ft: checkpoints. No
            # OpenRouter pin to send; thinking off via `reasoning_effort`.
            pc = ProviderConfig(
                name=f"{name} ({entry['_tier']}, models.yaml, openai)",
                model=model_id,
                base_url=None,
                api_key_env=entry.get("api_key_env", "OPENAI_API_KEY"),
                reasoning_effort=entry.get("reasoning_effort"),
                temperature=_pinned_temperature(entry),
                # the 5.x line: max_completion_tokens, and no temperature parameter
                max_completion_tokens_param=bool(entry.get("reasoning_effort")),
                send_temperature=entry.get("temperature") != "rejected",
                extra_max_tokens=extra_tokens,
            )
            return UnifiedProvider(pc), model_id
        if entry.get("served_by") == "tinker":
            # Tinker's sampler behind the local OpenAI-compatible proxy
            # (`personascope tinker-serve`). The proxy takes the base-model
            # name or a `tinker://` checkpoint path as `model`; thinking is
            # off in the renderer it picks, so no reasoning flag is sent.
            pc = ProviderConfig(
                name=f"{name} ({entry['_tier']}, models.yaml, tinker)",
                model=model_id,
                base_url=entry.get("base_url", TINKER_PROXY_URL),
                api_key_env=entry.get("api_key_env", "TINKER_LOCAL_API_KEY"),
                supports_logprobs=False,
                temperature=_pinned_temperature(entry),
            )
            return UnifiedProvider(pc), model_id
        if entry.get("provider"):
            extra_body = {"provider": {"only": [entry["provider"]],
                                       "allow_fallbacks": False}}
        # A bounded trace for endpoints where thinking is mandatory
        # (`reasoning: {effort: low}` or `{max_tokens: N}` in the entry).
        # Never scored -- but it is NOT free of the instrument's cap: the
        # trace and the answer share one budget, so a cap sized for the answer
        # leaves nothing for it. GLM-5.3-flash returned empty text at
        # `finish_reason: length` on 39 of 40 identity baseline calls for
        # exactly this reason. An entry whose reasoning cannot be turned off
        # must declare `min_max_tokens`, the floor below which its cap is
        # raised.
        if entry.get("reasoning"):
            extra_body = {**(extra_body or {}), "reasoning": dict(entry["reasoning"])}
        pc = ProviderConfig(
            name=f"{name} ({entry['_tier']}, models.yaml)",
            model=model_id,
            base_url=entry.get("base_url", defaults.get("base_url")),
            api_key_env=entry.get("api_key_env", defaults.get("api_key_env", "OPENAI_API_KEY")),
            extra_body=extra_body,
            disable_reasoning_by_default=bool(entry.get("disable_reasoning", False)),
            temperature=_pinned_temperature(entry),
            extra_max_tokens=extra_tokens,
            min_max_tokens=int(entry.get("min_max_tokens") or 0),
        )
        return UnifiedProvider(pc), model_id

    # 2 — registry: fine-tuned checkpoints, local vLLM pods, legacy aliases
    if name in PROVIDERS:
        return UnifiedProvider(PROVIDERS[name]), PROVIDERS[name].model

    # 3 — a bare OpenRouter slug, or a fine-tuned id passed through
    if name.startswith("ft:"):
        pc = ProviderConfig(name=f"{name} (fine-tune)", model=name,
                            api_key_env="OPENAI_API_KEY")
        return UnifiedProvider(pc), name
    if name.startswith("tinker://"):
        # A Tinker LoRA, named by its sampler path rather than by a models.yaml
        # key -- which is how an `sft` cell arrives here, since induction
        # resolves the persona to a checkpoint before the model is resolved.
        # This must precede the OpenRouter-slug branch below: a tinker:// path
        # contains "/", so without this it went to OpenRouter, which would be
        # asked for a model named "tinker://..." and fail.
        pc = ProviderConfig(
            name=f"{name[:32]}... (tinker LoRA)",
            model=name,
            base_url=TINKER_PROXY_URL,
            api_key_env="TINKER_LOCAL_API_KEY",
            supports_logprobs=False,
        )
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
