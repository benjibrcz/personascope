"""Persona projection probe — score a cell by projecting its response-only
activations onto a saved persona direction, per layer.

The white-box analogue of a behavioural probe: given a `RepresentationProvider`
(the served cell) and a direction, capture the cell's responses to a set of
*evaluation* questions (disjoint from the direction's extraction questions — the
caller enforces this to avoid selection leakage) and return the per-layer
projection, averaged over eval questions.

Takes a capture callable, so it is unit-testable with a mock provider (no pod).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from personascope.probes.representation.directions import load_direction, project_layers


def _chat(user: str, system: str | None = None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def project_cell(
    capture: Callable[..., Any],
    direction: np.ndarray,
    eval_questions: list[str],
    *,
    system: str | None = None,
    metric: str = "proj",
    max_tokens: int = 48,
) -> dict:
    """Project a cell onto `direction`. Returns per-question and mean per-layer
    projections. `capture` is `RepresentationProvider.capture`; `system` is an
    optional cell system-prompt (usually None — the cell is the served model/
    adapter, evaluated on neutral questions).
    """
    caps = [capture(_chat(q, system), max_tokens=max_tokens) for q in eval_questions]
    per_q = np.stack([project_layers(c.pooled, direction, metric=metric) for c in caps])
    return {
        "per_layer_mean": per_q.mean(0).tolist(),
        "per_layer_std": per_q.std(0).tolist(),
        "per_question": per_q.tolist(),
        "n_questions": len(eval_questions),
        "provenance": caps[0].provenance if caps else {},
    }


def project_cell_from_file(capture, direction_path, eval_questions, **kw) -> dict:
    return project_cell(capture, load_direction(direction_path), eval_questions, **kw)


__all__ = ["project_cell", "project_cell_from_file"]
