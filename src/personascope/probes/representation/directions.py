"""Persona/behaviour directions in the residual stream — extraction + projection.

A **direction** is a per-layer unit of "how much a cell's activations point the
persona way", shape ``[n_layers, hidden]``. We build one by mean-difference
(contrast pairs) and read a cell out by projecting its pooled activations onto
it. This is the numpy port of the validated S20 persona-vectors math
(``research_agenda/S20_work/persona_vectors``: ``generate_vec.py`` mean-diff,
``eval/cal_projection.py`` projection), which reached r≈0.86 projection↔behaviour.

Torch-free by design (the plan keeps personascope's core numpy/sklearn only —
the heavy vLLM-Lens/torch stack lives on the interp pod). vLLM-Lens capture
returns ``(n_layers, n_positions, hidden)``; pool over positions with
``pool_positions`` to get the ``[n_layers, hidden]`` this module consumes.

Directions are on-disk artifacts (``.npy``, ``[n_layers, hidden]``), keyed by
``(model, direction_name, extraction)`` — extract once, reuse across cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

Pooling = Literal["response_avg", "prompt_avg", "prompt_last"]

_EPS = 1e-8


def pool_positions(
    acts: np.ndarray, prompt_len: int, how: Pooling = "response_avg"
) -> np.ndarray:
    """Collapse a capture ``[n_layers, n_positions, hidden]`` to
    ``[n_layers, hidden]`` by pooling over the position axis, matching S20's
    three variants:

    - ``response_avg``  — mean over the *generated* positions (``prompt_len:``);
      the default, S20's best projection↔behaviour readout.
    - ``prompt_avg``    — mean over the *prompt* positions (``:prompt_len``).
    - ``prompt_last``   — the last prompt position (``prompt_len - 1``).
    """
    if acts.ndim != 3:
        raise ValueError(f"expected [n_layers, n_positions, hidden], got {acts.shape}")
    n_pos = acts.shape[1]
    if not 0 <= prompt_len <= n_pos:
        raise ValueError(f"prompt_len {prompt_len} out of range for {n_pos} positions")
    if how == "response_avg":
        resp = acts[:, prompt_len:, :]
        if resp.shape[1] == 0:  # nothing generated → fall back to last prompt tok
            return acts[:, prompt_len - 1, :].astype(np.float64)
        return resp.mean(axis=1).astype(np.float64)
    if how == "prompt_avg":
        return acts[:, :prompt_len, :].mean(axis=1).astype(np.float64)
    if how == "prompt_last":
        return acts[:, prompt_len - 1, :].astype(np.float64)
    raise ValueError(f"unknown pooling {how!r}")


def mean_diff_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Per-layer mean-difference direction from contrast examples.

    ``pos``/``neg`` are ``[n_examples, n_layers, hidden]`` (each example already
    pooled to one vector per layer). Returns ``[n_layers, hidden]`` =
    ``mean_pos - mean_neg`` per layer (S20 ``generate_vec.save_persona_vector``).
    The two arrays need the same layer/hidden shape but may differ in n_examples.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.ndim != 3 or neg.ndim != 3:
        raise ValueError("pos/neg must be [n_examples, n_layers, hidden]")
    if pos.shape[1:] != neg.shape[1:]:
        raise ValueError(f"layer/hidden mismatch: {pos.shape[1:]} vs {neg.shape[1:]}")
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise ValueError("need at least one pos and one neg example")
    return pos.mean(axis=0) - neg.mean(axis=0)


def a_proj_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed projection magnitude of ``a`` onto ``b``: ``(a·b)/‖b‖``.

    S20 ``eval/cal_projection.a_proj_b`` — the metric that correlated with
    behaviour. ``a`` is ``[..., hidden]``, ``b`` is ``[hidden]``; contracts the
    last axis and broadcasts the rest. A zero ``b`` yields 0 (not NaN)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(axis=-1) / (nb + _EPS)


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity along the last axis (S20 ``cos_sim``)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    return (a * b).sum(axis=-1) / (na * nb + _EPS)


def project_layers(
    acts: np.ndarray, direction: np.ndarray, metric: Literal["proj", "cos"] = "proj"
) -> np.ndarray:
    """Per-layer readout of pooled activations ``[n_layers, hidden]`` against a
    direction ``[n_layers, hidden]`` — layer ``l``'s activation projected onto
    layer ``l``'s direction. Returns ``[n_layers]``."""
    acts = np.asarray(acts, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if acts.shape != direction.shape:
        raise ValueError(f"shape mismatch: acts {acts.shape} vs dir {direction.shape}")
    fn = a_proj_b if metric == "proj" else cos_sim
    return np.array([fn(acts[l], direction[l]) for l in range(acts.shape[0])])


# ── on-disk artifacts ────────────────────────────────────────────────────────

def save_direction(direction: np.ndarray, path: str | Path) -> Path:
    """Persist a ``[n_layers, hidden]`` direction as ``.npy``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.ndim != 2:
        raise ValueError(f"direction must be [n_layers, hidden], got {direction.shape}")
    np.save(path, direction)
    return path


def load_direction(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


__all__ = [
    "pool_positions", "mean_diff_direction", "a_proj_b", "cos_sim",
    "project_layers", "save_direction", "load_direction",
]
