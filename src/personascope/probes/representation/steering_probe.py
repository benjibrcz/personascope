"""Causal steering probe — steer along a direction in-flight and compare against
CONTROLS, so an effect can be attributed to the direction rather than to "adding
any big vector".

The review's steering critique was: one prompt, keyword metric, no controls. This
module supplies the missing controls:
  - **random** — a random vector matched to the direction's per-layer norm
    (tests that it's not just "perturb the residual").
  - **opposite** — −direction (tests sign specificity: should push the trait DOWN).
  - **scale sweep** and **multiple prompts** — dose-response + generality.
The scored metric is supplied by the caller (ideally a judge, not a keyword count).

Control-vector construction is pure numpy (unit-tested offline). The actual
steer+generate needs the pod (lazy `vllm_lens`/`torch`).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def random_control_direction(direction: np.ndarray, seed: int) -> np.ndarray:
    """A random direction with the SAME per-layer L2 norm as `direction`
    (so it's an equal-magnitude perturbation — the fair 'any vector' control)."""
    direction = np.asarray(direction, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=direction.shape)
    dir_norms = np.linalg.norm(direction, axis=-1, keepdims=True)          # [n_layers,1]
    rand_norms = np.linalg.norm(rand, axis=-1, keepdims=True)
    return rand / (rand_norms + 1e-8) * dir_norms


def opposite_direction(direction: np.ndarray) -> np.ndarray:
    return -np.asarray(direction, dtype=np.float64)


def build_conditions(direction: np.ndarray, *, seed: int = 0) -> dict[str, np.ndarray]:
    """The four steering conditions to compare (baseline is 'no vector')."""
    return {
        "direction": np.asarray(direction, dtype=np.float64),
        "random": random_control_direction(direction, seed),
        "opposite": opposite_direction(direction),
    }


def run_steering_comparison(
    client: Any,
    direction: np.ndarray,
    prompts: list[str],
    *,
    layer: int,
    scales: list[float],
    seed: int = 0,
    max_tokens: int = 60,
    system: Optional[str] = None,
) -> dict:
    """Pod-side: for each prompt, generate baseline (no steer) and, for each
    condition × scale, a steered generation. Returns nested generations for the
    caller to score with a judge. Uses the client's chat + steering_vectors.

    `client` is a `vllm_lens.client.VLLMLensClient`. Steering is applied at a
    single `layer` (per the spike: many layers over-drive); `norm_match=True`,
    so scales are ≪ 1.
    """
    import torch  # pod-only
    from vllm_lens import SteeringVector

    conditions = build_conditions(direction, seed=seed)

    def _msgs(q):
        m = [{"role": "system", "content": system}] if system else []
        return m + [{"role": "user", "content": q}]

    results: dict[str, Any] = {"layer": layer, "scales": scales, "prompts": prompts,
                               "baseline": [], "conditions": {}}
    for q in prompts:
        out = client.chat(_msgs(q), max_tokens=max_tokens, temperature=0.0)
        results["baseline"].append(out.text.strip())
    for cond, vec in conditions.items():
        acts = torch.tensor(vec[[layer]], dtype=torch.float32)   # [1, hidden]
        results["conditions"][cond] = {}
        for scale in scales:
            sv = SteeringVector(activations=acts, layer_indices=[layer],
                                scale=scale, norm_match=True)
            gens = []
            for q in prompts:
                out = client.chat(_msgs(q), max_tokens=max_tokens, temperature=0.0,
                                   steering_vectors=[sv])
                gens.append(out.text.strip())
            results["conditions"][cond][str(scale)] = gens
    return results


__all__ = [
    "random_control_direction", "opposite_direction", "build_conditions",
    "run_steering_comparison",
]
