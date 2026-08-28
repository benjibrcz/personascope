"""Cache-provenance fingerprint (external review): resume must not silently
reuse data written under a different model / judge / seed / prompt / tier."""

from __future__ import annotations

import copy

from personascope.core.manifest import build_manifest, config_fingerprint

_BASE = dict(
    cell={"mode": "induced", "persona": "voldemort", "k": 0, "system_prompt": None},
    n_samples=16, seed=42, tier="extended",
    model_provider_name="openai", judge_provider_name="openai",
)


def _fp(**overrides):
    b = copy.deepcopy(_BASE)
    for k, v in overrides.items():
        if k in ("mode", "persona", "k", "system_prompt"):
            b["cell"][k] = v
        else:
            b[k] = v
    return config_fingerprint(**b)


def test_fingerprint_stable_for_same_config():
    assert _fp() == _fp()


def test_fingerprint_changes_on_each_critical_field():
    base = _fp()
    assert _fp(seed=43) != base
    assert _fp(tier="core") != base
    assert _fp(n_samples=8) != base
    assert _fp(system_prompt="be evil") != base
    assert _fp(persona="stalin") != base
    assert _fp(mode="uninduced") != base
    assert _fp(model_provider_name="anthropic") != base
    assert _fp(judge_provider_name="anthropic") != base


def test_manifest_carries_matching_fingerprint():
    m = build_manifest(
        cell=_BASE["cell"], n_samples=16, seed=42, tier="extended",
        model_provider_name="openai", judge_provider_name="openai",
        probes_run=["x"],
    )
    assert m["config_fingerprint"] == _fp()


def test_system_prompt_hash_not_raw_prompt():
    # the fingerprint must not leak the raw prompt, only its hash
    fp = _fp(system_prompt="secret instructions")
    assert "secret" not in fp
    assert len(fp) == 16
