"""Steering-capable provider — the normal provider interface with a steering
vector applied to EVERY generation, so the real probe suite (and the atomic
runner) can run under steering exactly as under any other cell.

`direction` is the full `[n_layers, hidden]` direction; steering is applied at
ONE `layer` with `sign ∈ {+1, −1}` and a `scale` (norm-matched by default, so
the working range is ≪ 1). `direction=None` or `scale=0` is the un-steered
baseline through the SAME code path (same hooks, same kwargs) — the only
difference between conditions is the vector.

The steering-vector object is built by an injectable factory (default: lazy
`vllm_lens.SteeringVector` + torch) so the path is testable offline.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from personascope.probes.representation.directions import direction_sha
from personascope.repr.vllm_lens_provider import RepresentationProvider


def _default_steering_vector_factory(row: np.ndarray, layer: int, scale: float, norm_match: bool):
    import torch  # pod-only
    from vllm_lens import SteeringVector
    acts = torch.tensor(np.asarray(row, dtype=np.float32)[None, :])
    return SteeringVector(activations=acts, layer_indices=[int(layer)], scale=float(scale),
                         norm_match=bool(norm_match))


class SteeringProvider(RepresentationProvider):
    def __init__(self, base_url: str, model: str, *, direction: Optional[np.ndarray], layer: int,
                 scale: float, sign: int = +1, condition: str = "direction", norm_match: bool = True,
                 steering_vector_factory: Optional[Callable[..., Any]] = None, **kw):
        super().__init__(base_url, model, **kw)
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 or -1")
        if scale < 0:
            raise ValueError("scale must be ≥ 0 (use sign=-1 for the opposite direction)")
        self.direction = None if direction is None else np.asarray(direction, dtype=np.float64)
        if self.direction is not None:
            if self.direction.ndim != 2 or self.direction.shape[0] != self.n_layers:
                raise ValueError(f"direction must be [n_layers={self.n_layers}, hidden], got {self.direction.shape}")
            if not 0 <= layer < self.n_layers:
                raise ValueError(f"layer {layer} out of range for {self.n_layers} layers")
        self.layer, self.scale, self.sign, self.norm_match = int(layer), float(scale), int(sign), bool(norm_match)
        self.condition = condition
        self.name = f"steer:{model}:{condition}"
        self._factory = steering_vector_factory or _default_steering_vector_factory
        self._sv_cache = None

    @property
    def active(self) -> bool:
        return self.direction is not None and self.scale > 0

    def steering_vectors(self) -> Optional[list]:
        if not self.active:
            return None
        if self._sv_cache is None:
            row = self.sign * self.direction[self.layer]
            self._sv_cache = [self._factory(row, self.layer, self.scale, self.norm_match)]
        return self._sv_cache

    def _extra_chat_kwargs(self) -> dict:
        sv = self.steering_vectors()
        return {"steering_vectors": sv} if sv else {}

    def fingerprint_fields(self) -> dict:
        f = super().fingerprint_fields()
        f.update({"provider_kind": "steering", "condition": self.condition,
                  "direction_sha": None if self.direction is None else direction_sha(self.direction),
                  "layer": self.layer, "scale": self.scale, "sign": self.sign,
                  "norm_match": self.norm_match, "active": self.active})
        return f


__all__ = ["SteeringProvider"]
