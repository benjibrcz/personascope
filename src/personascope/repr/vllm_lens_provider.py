"""Representation provider — one call → (text, response-only pooled activations)
from a vLLM-Lens-served model, mirroring the black-box provider abstraction.

Fixes the review's prompt-contamination finding two ways:
  1. **chat-format** — real system/user messages via the client's `chat()`, not
     a raw concatenated `generate(prompt)` string with instructions baked in.
  2. **response-only pooling** — pool the residual stream over the *generated*
     positions only (`prompt_len = n_positions − n_generated`), reusing the
     tested `directions.pool_positions(how="response_avg")`. The prompt tokens
     (including any system instruction) are excluded from the pooled vector.

Heavy imports (`vllm_lens`) are lazy so `personascope` stays torch-free. The
pure pooling logic is factored into `pool_capture()` so it is unit-testable
offline with synthetic arrays (no pod).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from personascope.probes.representation.directions import pool_positions


@dataclass
class CaptureResult:
    text: str
    pooled: np.ndarray                       # [n_layers, hidden] response-only
    n_generated: int
    n_positions: int
    provenance: dict[str, Any] = field(default_factory=dict)


def _extract_n_generated(out: Any, fallback_text: str) -> int:
    """Best-effort count of generated tokens from a GenerateOutput. Tries the
    OpenAI-style usage, then token id lists, then a whitespace-token fallback."""
    raw = getattr(out, "raw", None)
    if isinstance(raw, dict):
        usage = raw.get("usage") or {}
        n = usage.get("completion_tokens")
        if isinstance(n, int) and n > 0:
            return n
        choices = raw.get("choices") or []
        if choices and isinstance(choices[0], dict):
            tl = choices[0].get("logprobs") or {}
            toks = tl.get("tokens")
            if toks:
                return len(toks)
    for attr in ("output_token_ids", "token_ids", "completion_token_ids"):
        v = getattr(out, attr, None)
        if v:
            return len(v)
    # last resort: approximate by whitespace tokens (over/under-counts a little)
    return max(1, len((fallback_text or "").split()))


def pool_capture(residual: np.ndarray, n_generated: int,
                 how: str = "response_avg") -> np.ndarray:
    """Pure, testable core: residual `(n_layers, n_positions, hidden)` +
    generated-token count → pooled `[n_layers, hidden]`. Response-only pooling
    slices the last `n_generated` positions via `prompt_len = n_pos − n_gen`."""
    residual = np.asarray(residual)
    if residual.ndim != 3:
        raise ValueError(f"residual must be [n_layers, n_pos, hidden], got {residual.shape}")
    n_pos = residual.shape[1]
    prompt_len = max(0, min(n_pos, n_pos - int(n_generated)))
    return pool_positions(residual, prompt_len=prompt_len, how=how)


class RepresentationProvider:
    """Thin client over a vLLM-Lens serve: chat-format capture with response-only
    pooling. `model` selects the base or a served LoRA adapter id.

    ``capture_layers`` defaults to all layers (resolved on first call from the
    returned tensor). Provenance (model, layers, pooling, chat=True) is stamped
    on every CaptureResult and is what should be saved alongside any direction.
    """

    def __init__(self, base_url: str, model: str, *,
                 layers: Optional[list[int]] = None, pooling: str = "response_avg",
                 timeout: float = 600.0):
        self.base_url = base_url
        self.model = model
        self.layers = layers
        self.pooling = pooling
        self._timeout = timeout
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            from vllm_lens.client import VLLMLensClient  # heavy; pod-only
            self._client = VLLMLensClient(
                base_url=self.base_url, model=self.model, timeout=self._timeout)
        return self._client

    def capture(self, messages: list[dict], *, max_tokens: int = 48,
                temperature: float = 0.0) -> CaptureResult:
        """Chat-format capture → response-only pooled activations + provenance."""
        client = self._lazy_client()
        layers = self.layers if self.layers is not None else None
        out = client.chat(messages, max_tokens=max_tokens, temperature=temperature,
                          capture_layers=layers)
        residual = out.activations["residual_stream"]
        # torch tensor → float numpy (n_layers, n_pos, hidden)
        residual = residual.float().cpu().numpy() if hasattr(residual, "float") \
            else np.asarray(residual, dtype=np.float64)
        text = getattr(out, "text", "") or ""
        n_gen = _extract_n_generated(out, text)
        pooled = pool_capture(residual, n_gen, how=self.pooling)
        prov = {
            "model": self.model, "base_url": self.base_url,
            "layers": list(range(residual.shape[0])) if layers is None else layers,
            "pooling": self.pooling, "chat_format": True,
            "n_layers": int(residual.shape[0]), "hidden": int(residual.shape[2]),
        }
        return CaptureResult(text=text, pooled=pooled, n_generated=int(n_gen),
                             n_positions=int(residual.shape[1]), provenance=prov)

    def capture_many(self, message_sets: list[list[dict]], **kw) -> list[CaptureResult]:
        return [self.capture(m, **kw) for m in message_sets]


__all__ = ["RepresentationProvider", "CaptureResult", "pool_capture"]
