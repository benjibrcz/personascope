"""The Tinker proxy: OpenAI-shaped in, renderer + sampler in the middle,
OpenAI-shaped out. The sampler is mocked; no network."""

from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("tinker_cookbook")
pytest.importorskip("fastapi")

from personascope.tinker.proxy import renderer_name_for  # noqa: E402


def test_renderer_prefers_the_disable_thinking_variant():
    assert renderer_name_for("Qwen/Qwen3.8-27B") == "qwen3_8_disable_thinking"
    assert renderer_name_for("moonshotai/Kimi-K2.6") == "kimi_k26_disable_thinking"
    assert renderer_name_for("Qwen/Qwen3.8-27B", "qwen3_8_low_reasoning") == "qwen3_8_low_reasoning"


def test_models_yaml_renderers_are_what_the_cookbook_recommends():
    from personascope.tinker.proxy import renderer_overrides_from_models_yaml

    for base, name in renderer_overrides_from_models_yaml().items():
        assert renderer_name_for(base) == name, base


class _Seq:
    def __init__(self, tokens):
        self.tokens = tokens


class _Result:
    def __init__(self, n, tokens):
        self.sequences = [_Seq(tokens) for _ in range(n)]


def _fake_tinker(tokens):
    """A `tinker` module whose SamplingClient returns `tokens` for any prompt."""
    calls = []

    class SamplingClient:
        async def sample_async(self, *, prompt, num_samples, sampling_params):
            calls.append({"prompt": prompt, "n": num_samples, "params": sampling_params})
            return _Result(num_samples, tokens)

    class ServiceClient:
        def __init__(self, *a, **k):
            pass

        def create_sampling_client(self, *, base_model=None, model_path=None):
            return SamplingClient()

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    mod = types.SimpleNamespace(ServiceClient=ServiceClient, SamplingParams=SamplingParams)
    return mod, calls


def test_chat_completion_round_trip(monkeypatch):
    """A user turn goes through the model's renderer to the sampler, and the
    sampled tokens come back as an assistant message with the host named."""
    from fastapi.testclient import TestClient
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    base = "Qwen/Qwen3.5-9B"
    try:
        tok = get_tokenizer(base)
    except Exception as exc:  # noqa: BLE001 - no HF cache offline
        pytest.skip(f"tokenizer unavailable: {exc}")
    r = renderers.get_renderer("qwen3_5_disable_thinking", tok, model_name=base)
    answer_tokens = tok.encode("I am Marie Curie.", add_special_tokens=False)
    stop = r.get_stop_sequences()
    if stop and isinstance(stop[0], int):
        answer_tokens = answer_tokens + [stop[0]]
    elif stop:
        answer_tokens = answer_tokens + tok.encode(stop[0], add_special_tokens=False)

    fake, calls = _fake_tinker(answer_tokens)
    monkeypatch.setitem(sys.modules, "tinker", fake)
    monkeypatch.setenv("TINKER_API_KEY", "test")
    from personascope.tinker.proxy import create_app

    client = TestClient(create_app())
    resp = client.post("/v1/chat/completions", json={
        "model": base, "messages": [{"role": "user", "content": "Who are you?"}],
        "temperature": 1.0, "max_tokens": 32, "n": 2,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "tinker" and body["base_model"] == base
    assert len(body["choices"]) == 2
    assert body["choices"][0]["message"]["content"].strip() == "I am Marie Curie."
    assert calls[0]["n"] == 2 and calls[0]["params"].temperature == 1.0
    assert client.get("/health").json()["models"][base]["renderer"] == "qwen3_5_disable_thinking"


def test_streaming_is_refused(monkeypatch):
    from fastapi.testclient import TestClient

    fake, _ = _fake_tinker([])
    monkeypatch.setitem(sys.modules, "tinker", fake)
    monkeypatch.setenv("TINKER_API_KEY", "test")
    from personascope.tinker.proxy import create_app

    client = TestClient(create_app())
    resp = client.post("/v1/chat/completions", json={
        "model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    assert resp.status_code == 400


os.environ.setdefault("TINKER_LOCAL_API_KEY", "sk-tinker-local")
