"""Build a persona/behaviour direction from contrast-pair prompts.

Captures response-only activations for a trait-positive and a trait-negative
condition (chat-format system prompts over a set of *extraction* questions),
then mean-differences them per layer (S20 `mean_diff_direction`). The direction
is saved as `.npy` with a provenance sidecar `.json` (model, layers, pooling,
chat-template flag, extraction-question hash, n examples) so a projection can
never be silently mismatched to a direction from a different model/pooling.

Takes a capture callable (the `RepresentationProvider.capture`), so it is
unit-testable offline with a mock — no vLLM-Lens import here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from personascope.probes.representation.directions import (
    mean_diff_direction,
    save_direction,
)


def _chat(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _hash(items: list[str]) -> str:
    return hashlib.sha256("␟".join(items).encode()).hexdigest()[:16]


def extract_direction(
    capture: Callable[..., Any],
    pos_system: str,
    neg_system: str,
    extract_questions: list[str],
    *,
    max_tokens: int = 48,
) -> tuple[np.ndarray, dict]:
    """Capture pos/neg responses over `extract_questions` (chat-format,
    response-only via the provider) and return (direction [n_layers, hidden],
    provenance). `capture(messages, max_tokens=...)` must return an object with
    `.pooled` ([n_layers, hidden]) and `.provenance` (dict) — i.e.
    `RepresentationProvider.capture`."""
    pos = [capture(_chat(pos_system, q), max_tokens=max_tokens) for q in extract_questions]
    neg = [capture(_chat(neg_system, q), max_tokens=max_tokens) for q in extract_questions]
    pos_arr = np.stack([r.pooled for r in pos])         # [n_ex, n_layers, hidden]
    neg_arr = np.stack([r.pooled for r in neg])
    direction = mean_diff_direction(pos_arr, neg_arr)   # [n_layers, hidden]
    prov = dict(pos[0].provenance)
    prov.update({
        "kind": "mean_diff_contrast",
        "pos_system": pos_system, "neg_system": neg_system,
        "n_extract_questions": len(extract_questions),
        "extract_questions_sha": _hash(extract_questions),
        "max_tokens": max_tokens,
    })
    return direction, prov


def save_direction_with_provenance(
    direction: np.ndarray, provenance: dict, path: str | Path
) -> Path:
    """Save `<path>.npy` + `<path>.json` provenance sidecar."""
    path = Path(path)
    npy = save_direction(direction, path.with_suffix(".npy"))
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2, default=str))
    return npy


__all__ = ["extract_direction", "save_direction_with_provenance"]
