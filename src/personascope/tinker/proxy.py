"""OpenAI-compatible chat completions over Tinker's sampler.

    personascope tinker-serve --port 8010

`model` in a request is either a Tinker base-model name (`Qwen/Qwen3.8-27B`,
`moonshotai/Kimi-K2.6`) or a `tinker://<run>/sampler_weights/<name>` checkpoint
path, which is how configs/checkpoints.yaml records a LoRA. Both go through
the same `SamplingClient`, so a base cell and an SFT cell of one model differ
in the weights and nothing else.

Thinking is off: the renderer is the model's recommended one with the
`_disable_thinking` variant where the cookbook has it (Qwen3.5/3.8, Kimi K2.5/
K2.6, DeepSeek V3); reasoning models without one are refused at startup
rather than run under a different regime.

Adapted from the Evans group's `tinker_openai_proxy.py` (latteries repo, used
for Weird Generalization), generalised from Qwen-only to the cookbook's
renderer registry. The response carries `provider: "tinker"` and the resolved
base model, which `UnifiedProvider.complete()` records as `host`.
"""

import asyncio
import os
import pathlib
import time
from typing import Any, Optional

# Not a budget: a ceiling high enough that no instrument reaches it, so the
# sampler's required max_tokens never silently truncates an uncapped call.
UNCAPPED_TOKENS = 16384


__all__ = ["create_app", "renderer_name_for", "serve"]

DEFAULT_PORT = 8010


def renderer_name_for(base_model: str, override: Optional[str] = None) -> str:
    """The renderer to sample with: `override` if given, else the recommended
    renderer with thinking disabled, else the recommended renderer."""
    from tinker_cookbook import model_info

    if override:
        return override
    names = model_info.get_recommended_renderer_names(base_model)
    for n in names:
        if n.endswith("_disable_thinking"):
            return n
    return names[0]


def create_app(renderer_overrides: Optional[dict[str, str]] = None):
    """Build the FastAPI app. `renderer_overrides` maps base model -> renderer
    name, from models.yaml `renderer:` entries."""
    import tinker
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    # .env is where every credential in this repo lives, and Tinker only reads
    # the environment. Same reason the two trainers load it.
    try:
        from dotenv import load_dotenv

        load_dotenv(pathlib.Path(__file__).resolve().parents[3] / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY is not set")
    overrides = dict(renderer_overrides or {})

    app = FastAPI(title="personascope tinker proxy")
    state: dict[str, Any] = {
        "service": tinker.ServiceClient(base_url=os.environ.get("TINKER_BASE_URL"), api_key=api_key),
        "samplers": {},      # model string -> SamplingClient
        "base_of": {},       # model string -> base model name
        "renderers": {},     # base model -> Renderer
        "lock": asyncio.Lock(),
    }

    class ChatMessage(BaseModel):
        role: str
        content: Optional[str] = ""

    class ChatRequest(BaseModel):
        model: str
        messages: list[ChatMessage]
        temperature: float = 1.0
        top_p: Optional[float] = None
        max_tokens: Optional[int] = None
        n: int = 1
        stop: Optional[list[str] | str] = None
        seed: Optional[int] = None
        stream: bool = False

    async def sampler_for(model: str):
        async with state["lock"]:
            if model not in state["samplers"]:
                service = state["service"]
                if model.startswith("tinker://"):
                    rest = service.create_rest_client()
                    run = await rest.get_training_run_by_tinker_path_async(model)
                    base = run.base_model
                    state["samplers"][model] = service.create_sampling_client(model_path=model)
                else:
                    base = model
                    state["samplers"][model] = service.create_sampling_client(base_model=model)
                state["base_of"][model] = base
        return state["samplers"][model], state["base_of"][model]

    def renderer_for(base: str):
        if base not in state["renderers"]:
            name = renderer_name_for(base, overrides.get(base))
            state["renderers"][base] = renderers.get_renderer(
                name, get_tokenizer(base), model_name=base,
            )
            state["renderers"][base]._personascope_name = name  # for /health
        return state["renderers"][base]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "models": {m: {"base": b, "renderer": getattr(state["renderers"].get(b), "_personascope_name", None)}
                       for m, b in state["base_of"].items()},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(400, "streaming is not supported")
        try:
            sampler, base = await sampler_for(req.model)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"cannot open {req.model!r}: {exc}") from exc
        renderer = renderer_for(base)

        msgs: list[renderers.Message] = [
            {"role": m.role, "content": m.content or ""} for m in req.messages  # type: ignore[typeddict-item]
        ]
        prefill = None
        if msgs and msgs[-1]["role"] == "assistant":
            prefill = str(msgs[-1]["content"])
            msgs = msgs[:-1]
        prompt = renderer.build_generation_prompt(msgs, prefill=prefill)

        stop: Any = renderer.get_stop_sequences()
        if req.stop:
            extra = [req.stop] if isinstance(req.stop, str) else list(req.stop)
            if stop and isinstance(stop[0], str):
                stop = list(stop) + extra
        # Tinker's sampler requires a number; there is no "no cap" value. Our
        # instruments all declare max_tokens None, so a fallback is always what
        # applies on this route. 2048 was the old one and it was low enough to
        # bind on a long answer while every record still said `max_tokens:
        # null`. UNCAPPED_TOKENS is chosen not to bind, and is reported back on
        # the response so the record says what the sampler was actually given.
        cap = req.max_tokens or UNCAPPED_TOKENS
        params = tinker.SamplingParams(
            max_tokens=cap,
            temperature=req.temperature,
            top_p=req.top_p if req.top_p is not None else 1.0,
            stop=stop,
            seed=req.seed,
        )
        result = await sampler.sample_async(prompt=prompt, num_samples=req.n, sampling_params=params)

        choices = []
        completion_tokens = 0
        for i, seq in enumerate(result.sequences):
            message, termination = renderer.parse_response(seq.tokens)
            content = message.get("content", "")
            if not isinstance(content, str):  # list of content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            finish = "stop" if getattr(termination, "is_clean", bool(termination)) else "length"
            choices.append({
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            })
            completion_tokens += len(seq.tokens)
        prompt_tokens = prompt.length if hasattr(prompt, "length") else 0
        return {
            "id": f"chatcmpl-tinker-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "provider": "tinker",
            "base_model": base,
            "max_tokens_used": cap,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def renderer_overrides_from_models_yaml() -> dict[str, str]:
    """`renderer:` fields of the `served_by: tinker` entries in models.yaml."""
    from personascope.models import _pinned, load_models_config

    out: dict[str, str] = {}
    for entry in _pinned(load_models_config()).values():
        if entry.get("served_by") == "tinker" and entry.get("renderer"):
            out[entry["id"]] = entry["renderer"]
    return out


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
    import uvicorn

    app = create_app(renderer_overrides_from_models_yaml())
    uvicorn.run(app, host=host, port=port, log_level="info")
