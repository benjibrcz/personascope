"""The stance grid: which entity an answer is spoken from.

One rubric — `data/identity/stance_grid.yaml`, the Assistant Axis grid plus
`acknowledges` — rendered and parsed here, so the SAD instrument, the
persona/assistant relationship item and the identity panel's stance column all
label the same way and land in the same table.

The grid's sha is part of every `parse_key` that uses it: editing the yaml
forces a re-parse, never a re-ask, because stances are assigned in the parse
pass from stored responses.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

__all__ = ["LABELS", "UNREADABLE", "grid", "grid_sha", "render", "parse", "judge_stance"]

_PATH = Path(__file__).resolve().parent / "data" / "identity" / "stance_grid.yaml"

# What a response gets when the judge's own output cannot be read. It is kept
# distinct from `ambiguous-nonsensical`, which is a judgement about the answer:
# a judge outage is not evidence that a model was evasive.
UNREADABLE = "unparsed"


@lru_cache(maxsize=1)
def grid() -> dict[str, Any]:
    return yaml.safe_load(_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def grid_sha() -> str:
    return hashlib.sha256(_PATH.read_bytes()).hexdigest()[:12]


@lru_cache(maxsize=1)
def _labels() -> tuple[str, ...]:
    return tuple(lb["name"] for lb in grid()["labels"])


LABELS: tuple[str, ...] = _labels()


def render(*, persona_label: str, question: str, response: str) -> str:
    """The full judge prompt for one response."""
    g = grid()
    label_block = "\n".join(
        f"- {lb['name']} — {' '.join(lb['gloss'].split())}"
        + "".join(f"\n    e.g. {c}" for c in lb.get("cues", []))
        for lb in g["labels"]
    )
    tiebreaker_block = "\n".join(f"  {i}. {t}" for i, t in enumerate(g["tiebreakers"], 1))
    return g["prompt"].format(
        persona_label=persona_label,
        question=question,
        response=response,
        label_block=label_block,
        tiebreaker_block=tiebreaker_block,
    )


def parse(raw: str) -> tuple[str, str]:
    """`(label, analysis)` from the judge's reply.

    Returns `UNREADABLE` rather than guessing when no known label appears. A
    judge that silently falls back to a default label turns its own failures
    into data, which is indistinguishable from the model having been evasive.
    """
    if not raw or not raw.strip():
        return UNREADABLE, ""

    obj: Any = None
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None

    if isinstance(obj, dict):
        label = str(obj.get("label", "")).strip().lower()
        analysis = str(obj.get("analysis", "")).strip()
        if label in LABELS:
            return label, analysis
        raw = f"{label} {analysis}"

    # No usable JSON: accept a bare label, longest first so `nonhuman_role`
    # is not shadowed by `human_role`, which is a substring of it.
    low = raw.lower()
    for name in sorted(LABELS, key=len, reverse=True):
        if name in low:
            return name, ""
    return UNREADABLE, ""


def judge_stance(
    judge: Callable[[str], str], *, persona_label: str, question: str, response: str,
) -> tuple[str, str]:
    """Render, call, parse. `judge` is a `(prompt) -> text` from `judges.py`."""
    prompt = render(persona_label=persona_label, question=question, response=response)
    return parse(judge(prompt))
