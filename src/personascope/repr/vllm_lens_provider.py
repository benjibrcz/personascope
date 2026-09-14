"""Representation provider — ONE generation → (text, exact output token ids,
response-only pooled activations) from a vLLM-Lens serve. FAIL-CLOSED.

Capture uses the validated per-step hook recipe (`scripts/LENS_API_NOTES.md`):
hook results come back as ``{completion_id: {step_idx: {"L{layer}": (n_tok, hidden)}}}``
where step 0 is the prefill (prompt positions) and every decode step ≥ 1 holds
EXACTLY ONE generated-token position. That gives an exact generated-token
boundary, so response-only pooling never has to guess a prompt length.

Fail-closed rules (all raise `CaptureIntegrityError`; nothing is pooled,
clamped, or approximated):
  - no exact output token ids on the generation output;
  - zero decode steps / zero output tokens;
  - a decode step with ≠ 1 position, or a missing layer at any step;
  - ``n_decode_steps != n_output_tokens + policy.decode_steps_offset``;
  - stale/multiple completion ids in the hook results.
There is NO whitespace-token fallback and NO position clamp.

`complete()` implements the normal provider contract (text / n_tokens /
success / seed) so the ordinary probe suite can run on the served model, and
attaches the `CaptureResult` under ``"capture"`` — the judged text and the
activations are always the SAME generation.

Heavy imports (`vllm_lens`) are lazy; the client is injectable so the whole
path is unit-testable with `repr.fake_client.FakeLensClient`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

CAPTURE_IMPL_VERSION = "repr-capture-v2"


class CaptureIntegrityError(RuntimeError):
    """Exact generated-token boundaries unavailable or inconsistent → fail closed."""


@dataclass(frozen=True)
class TokenPositionPolicy:
    """Pre-registered token-position policy (frozen after the live integration
    test on the pod; see docs/repr_preregistration.md §4).

    decode_steps_offset: expected ``n_decode_steps − n_output_tokens``. With vLLM
      the final sampled token (EOS or the max_tokens-th token) is never fed back
      through the model, so the default is −1. Only {−1, 0} are admissible.
    pooling: ``response_avg`` — mean over the generated-token positions ONLY.
    """
    decode_steps_offset: int = -1
    pooling: str = "response_avg"
    version: str = "token-policy-v2"

    def __post_init__(self):
        if self.decode_steps_offset not in (-1, 0):
            raise ValueError("decode_steps_offset must be -1 or 0")
        if self.pooling != "response_avg":
            raise ValueError("only response_avg pooling is admissible for the v2 channel")

    def fingerprint_fields(self) -> dict:
        return asdict(self)


@dataclass
class CaptureResult:
    text: str
    output_token_ids: list[int]
    n_output_tokens: int
    n_decode_steps: int
    n_prompt_tokens: Optional[int]
    pooled: np.ndarray                              # [n_layers, hidden] response-only
    messages: list[dict]
    seed: Optional[int]
    max_tokens: int
    temperature: float
    provenance: dict[str, Any] = field(default_factory=dict)
    generated_residual: Optional[np.ndarray] = None  # [n_layers, n_gen, hidden] if kept


def _step_key(k) -> int:
    try:
        return int(k)
    except (TypeError, ValueError) as e:
        raise CaptureIntegrityError(f"non-integer step key {k!r}") from e


def _to_np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "float") and hasattr(t, "cpu"):
        t = t.float().cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t, dtype=np.float64)


def capture_from_hook_results(results: dict, *, n_output_tokens: int, layers: list[int],
                              policy: TokenPositionPolicy) -> tuple[np.ndarray, Optional[int], int]:
    """Pure, testable core. Validate the per-step hook results and return
    ``(generated_residual [n_layers, n_gen, hidden], n_prompt_tokens, n_decode_steps)``.
    Raises `CaptureIntegrityError` on any inconsistency (fail closed)."""
    if not isinstance(results, dict) or len(results) != 1:
        raise CaptureIntegrityError(
            f"expected hook results for exactly one completion, got {0 if not results else len(results)} "
            "(clear_hook_results between generations)")
    (_, steps), = results.items()
    if not isinstance(steps, dict) or not steps:
        raise CaptureIntegrityError("empty step dict in hook results")
    ordered = sorted(steps.items(), key=lambda kv: _step_key(kv[0]))
    keys = [_step_key(k) for k, _ in ordered]
    if keys[0] != 0:
        raise CaptureIntegrityError("prefill step 0 missing from hook results")
    if keys != list(range(len(keys))):
        raise CaptureIntegrityError(f"non-contiguous step indices {keys}")
    n_decode = len(keys) - 1
    if n_decode < 1:
        raise CaptureIntegrityError("zero decode steps captured — nothing generated")
    if int(n_output_tokens) < 1:
        raise CaptureIntegrityError("zero output tokens reported")
    expected = int(n_output_tokens) + policy.decode_steps_offset
    if n_decode != expected:
        raise CaptureIntegrityError(
            f"decode-step/token-count mismatch: {n_decode} decode steps vs {n_output_tokens} output tokens "
            f"(policy offset {policy.decode_steps_offset} ⇒ expected {expected})")
    # prefill (prompt positions) — only used for provenance; NOT pooled.
    prefill = ordered[0][1]
    n_prompt: Optional[int] = None
    lk0 = f"L{layers[0]}"
    if isinstance(prefill, dict) and lk0 in prefill:
        n_prompt = int(_to_np(prefill[lk0]).shape[0])
    per_layer: list[list[np.ndarray]] = [[] for _ in layers]
    for k, step in ordered[1:]:
        if not isinstance(step, dict):
            raise CaptureIntegrityError(f"step {k} is not a layer dict")
        for li, layer in enumerate(layers):
            lk = f"L{layer}"
            if lk not in step:
                raise CaptureIntegrityError(f"layer {layer} missing at decode step {k}")
            arr = _to_np(step[lk])
            if arr.ndim != 2 or arr.shape[0] != 1:
                raise CaptureIntegrityError(
                    f"decode step {k} layer {layer} has shape {arr.shape}; expected (1, hidden)")
            per_layer[li].append(arr[0])
    gen = np.stack([np.stack(rows) for rows in per_layer])          # [n_layers, n_gen, hidden]
    if not np.all(np.isfinite(gen)):
        raise CaptureIntegrityError("non-finite activations in generated positions")
    return gen, n_prompt, n_decode


def output_token_ids(out: Any) -> list[int]:
    """Exact output token ids from a generation output, or raise (no estimate)."""
    for attr in ("output_token_ids", "token_ids", "completion_token_ids"):
        v = getattr(out, attr, None)
        if v is not None:
            ids = list(v)
            if ids and all(isinstance(i, (int, np.integer)) for i in ids):
                return [int(i) for i in ids]
    raw = getattr(out, "raw", None)
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices and isinstance(choices[0], dict):
            ids = choices[0].get("token_ids") or choices[0].get("output_token_ids")
            if ids and all(isinstance(i, (int, np.integer)) for i in ids):
                return [int(i) for i in ids]
    raise CaptureIntegrityError("generation output carries no exact output token ids — refusing to "
                                "estimate the generated-token boundary")


def _capture_hook(ctx, h):
    """Module-level (picklable) hook: save this layer's activations, don't modify."""
    ctx.saved[f"L{ctx.layer_idx}"] = h.detach().float().cpu()
    return None


def _default_hook_factory(layers: list[int]):
    from vllm_lens import Hook  # heavy; pod-only
    return [Hook(fn=_capture_hook, layer_indices=list(layers))]


def sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class RepresentationProvider:
    """Chat-format, atomic capture provider over a vLLM-Lens serve.

    `model` selects the base model or a served LoRA adapter id. `n_layers` is
    required (the hook layer list is response-determining and fingerprinted).
    """

    def __init__(self, base_url: str, model: str, *, n_layers: int,
                 layers: Optional[list[int]] = None, policy: Optional[TokenPositionPolicy] = None,
                 client: Any = None, hook_factory: Optional[Callable[[list[int]], list]] = None,
                 timeout: float = 600.0, keep_residual: bool = False,
                 model_revision: str = "unknown"):
        if n_layers < 1:
            raise ValueError("n_layers must be ≥ 1")
        self.base_url = base_url
        self.model = model
        self.model_revision = model_revision
        self.name = f"repr:{model}"
        self.n_layers = int(n_layers)
        self.layers = list(range(n_layers)) if layers is None else list(layers)
        self.policy = policy or TokenPositionPolicy()
        self._client = client
        self._hook_factory = hook_factory or _default_hook_factory
        self._timeout = timeout
        self._hooks_registered = False
        self.keep_residual = keep_residual

    # ── plumbing ──
    def _lazy_client(self):
        if self._client is None:
            from vllm_lens.client import VLLMLensClient  # heavy; pod-only
            self._client = VLLMLensClient(base_url=self.base_url, model=self.model, timeout=self._timeout)
        return self._client

    def _ensure_hooks(self, client):
        """Register the capture hooks ONCE per client (several providers may
        share one client for the same served model; hooks are client/server
        state, steering vectors are per-call kwargs)."""
        marker = tuple(self.layers)
        if getattr(client, "_personascope_capture_layers", None) != marker:
            client.register_hooks(self._hook_factory(self.layers))
            try:
                client._personascope_capture_layers = marker
            except AttributeError:  # client forbids attributes → fall back to per-instance
                pass
        self._hooks_registered = True

    def _chat_kwargs(self, max_tokens: int, temperature: float, seed: Optional[int]) -> dict:
        kw: dict[str, Any] = {"max_tokens": int(max_tokens), "temperature": float(temperature)}
        if seed is not None:
            kw["seed"] = int(seed)
        return kw

    def _extra_chat_kwargs(self) -> dict:
        return {}

    # ── atomic capture ──
    def capture(self, messages: list[dict], *, max_tokens: int = 150, temperature: float = 1.0,
                seed: Optional[int] = None) -> CaptureResult:
        client = self._lazy_client()
        self._ensure_hooks(client)
        client.clear_hook_results()
        kw = {**self._chat_kwargs(max_tokens, temperature, seed), **self._extra_chat_kwargs()}
        try:
            out = client.chat([dict(m) for m in messages], **kw)
        except TypeError as e:
            # e.g. the client rejecting `seed`: seeds are response-determining → fail closed
            raise CaptureIntegrityError(f"client.chat rejected generation kwargs {sorted(kw)}: {e}") from e
        results = client.collect_hook_results()
        client.clear_hook_results()
        text = getattr(out, "text", None)
        if not isinstance(text, str):
            raise CaptureIntegrityError("generation output has no text")
        ids = output_token_ids(out)
        gen, n_prompt, n_dec = capture_from_hook_results(
            results, n_output_tokens=len(ids), layers=self.layers, policy=self.policy)
        pooled = gen.mean(axis=1)
        prov = {
            "model": self.model, "model_revision": self.model_revision, "base_url": self.base_url,
            "layers": list(self.layers), "policy": self.policy.fingerprint_fields(),
            "n_layers": int(gen.shape[0]), "hidden": int(gen.shape[2]), "chat_format": True,
            "capture_impl_version": CAPTURE_IMPL_VERSION,
        }
        return CaptureResult(text=text, output_token_ids=ids, n_output_tokens=len(ids),
                             n_decode_steps=n_dec, n_prompt_tokens=n_prompt, pooled=pooled,
                             messages=[dict(m) for m in messages], seed=seed, max_tokens=int(max_tokens),
                             temperature=float(temperature), provenance=prov,
                             generated_residual=gen if self.keep_residual else None)

    # ── provider contract (probe suite) ──
    def complete(self, messages: list[dict], *, max_tokens: int = 150, temperature: float = 0.7,
                 logprobs: bool = False, top_logprobs: int = 5, n: int = 1,
                 stop: Optional[list[str]] = None, capture_reasoning: bool = False,
                 seed: Optional[int] = None) -> dict[str, Any]:
        """Normalised result (`text`, `n_tokens` = exact token count, `success`,
        `seed`, `output_token_ids`) + the atomic `capture`. `n>1` and `stop`
        are unsupported (raise) — one record is one generation. Raises
        `CaptureIntegrityError` rather than returning approximate data."""
        if n != 1:
            raise NotImplementedError("RepresentationProvider generates one atomic sample per call")
        if stop:
            raise NotImplementedError("stop sequences are not supported by the atomic capture path")
        cap = self.capture(messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
        return {"text": cap.text, "text_samples": [cap.text], "n_tokens": cap.n_output_tokens,
                "nll": 0.0, "total_nll": 0.0, "logprobs": None, "reasoning": "", "success": True,
                "seed": seed, "output_token_ids": list(cap.output_token_ids), "capture": cap}

    def fingerprint_fields(self) -> dict:
        return {"provider_kind": "representation", "base_url": self.base_url, "model": self.model,
                "model_revision": self.model_revision, "layers": list(self.layers),
                "token_position_policy": self.policy.fingerprint_fields(),
                "capture_impl_version": CAPTURE_IMPL_VERSION}


__all__ = ["RepresentationProvider", "CaptureResult", "CaptureIntegrityError", "TokenPositionPolicy",
           "capture_from_hook_results", "output_token_ids", "sha16", "CAPTURE_IMPL_VERSION"]
