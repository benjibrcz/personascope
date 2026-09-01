"""Steering CONTROL vectors — the null distribution for the specificity test.

The causal steering sub-study (docs/repr_preregistration.md §7) compares the
true-vector effect against ≥20 pre-specified control directions run at the
SAME frozen (layer, scale) on the SAME blocks:

  - **random** — `random_control_directions(direction, n, seed)`: n random
    directions, each matched to the true direction's per-layer L2 norm
    (equal-magnitude perturbations; norm-matched steering then treats them
    identically). Seeded → the exact set is pre-registered by (n, seed).
  - **opposite** — `opposite_direction`: −direction (sign specificity).
  - **off-target** — other traits' directions, supplied by the caller.

Pure numpy. The steer+generate path is `repr.steering_provider.SteeringProvider`
+ `repr.atomic.run_scheduled_conditions` (block-randomised, atomic records).
"""

from __future__ import annotations

import numpy as np

from personascope.probes.representation.directions import direction_sha


def random_control_direction(direction: np.ndarray, seed: int) -> np.ndarray:
    """One random direction with the SAME per-layer L2 norm as `direction`."""
    direction = np.asarray(direction, dtype=np.float64)
    rng = np.random.default_rng(seed)
    rand = rng.normal(size=direction.shape)
    dir_norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    rand_norms = np.linalg.norm(rand, axis=-1, keepdims=True)
    return rand / (rand_norms + 1e-8) * dir_norms


def random_control_directions(direction: np.ndarray, *, n: int = 20, seed: int = 0) -> list[np.ndarray]:
    """The pre-specified null set: `n` norm-matched random directions, seeds
    `seed, seed+1, …` (so the set is reproducible from (n, seed) alone)."""
    if n < 1:
        raise ValueError("n must be ≥ 1")
    return [random_control_direction(direction, seed + k) for k in range(n)]


def opposite_direction(direction: np.ndarray) -> np.ndarray:
    return -np.asarray(direction, dtype=np.float64)


def build_conditions(direction: np.ndarray, *, seed: int = 0, n_random: int = 20) -> dict[str, np.ndarray]:
    """Named control set: the direction, its opposite, and `n_random` matched randoms."""
    out = {"direction": np.asarray(direction, dtype=np.float64), "opposite": opposite_direction(direction)}
    for k, rd in enumerate(random_control_directions(direction, n=n_random, seed=seed)):
        out[f"rand{k:02d}"] = rd
    return out


def control_set_sha(direction: np.ndarray, *, n_random: int, seed: int) -> str:
    """Hash of the whole pre-specified control set (for fingerprints)."""
    import hashlib
    h = hashlib.sha256(direction_sha(direction).encode())
    for rd in random_control_directions(direction, n=n_random, seed=seed):
        h.update(direction_sha(rd).encode())
    return h.hexdigest()[:16]


__all__ = ["random_control_direction", "random_control_directions", "opposite_direction",
           "build_conditions", "control_set_sha"]
