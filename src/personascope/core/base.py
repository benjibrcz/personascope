"""Probe abstraction for the multi-turn runner.

A Probe applies a measurement at a *snapshot* of conversation history without
modifying that history. The same Probe can therefore be applied repeatedly
across a multi-turn conversation (e.g. measuring Ch1a identity after every
user turn in a hysteresis experiment) and the probe's own prompt/response
never becomes part of the main conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional, TypedDict

# A probe's applicable cell mode(s). "induced" = persona is being induced
# (k>0 ICL or system_prompt set); "uninduced" = neither. Probes set this
# to indicate which cell types they produce meaningful signal on.
Mode = Literal["induced", "uninduced"]
ALL_MODES: frozenset[Mode] = frozenset({"induced", "uninduced"})


class ProbeResult(TypedDict, total=False):
    """Payload a Probe.run returns."""
    prompt: Optional[str]              # the probe's user turn (what was asked off-branch)
    response: Optional[str]            # the assistant's response to the probe
    measurement: dict[str, Any]        # the measurement dict (goes into a Measurements slot)


@dataclass
class Probe:
    """A measurement probe that can be applied at any snapshot of a conversation.

    Attributes
    ----------
    name : str
        Stable identifier, used in TurnRecord.run_id. Convention:
        `<channel_slot>:<probe_specific_suffix>`, e.g. `identification_icl:name_and_last_name`.
    channel_slot : str
        Which field of `Measurements` the probe populates
        (e.g. `identification_icl`, `recognition_jeopardy`).
    run : Callable
        Function with signature `(history, provider, judge_fn, cache) -> ProbeResult`.
        Must not mutate `history`.
    applicable_modes : frozenset[Mode]
        Which cell modes the probe produces meaningful signal on:
          - "induced"   → cell has persona induction (k>0 or system_prompt)
          - "uninduced" → cell is base persona (k=0 and no system_prompt)
        Default = {"induced", "uninduced"} (probe is mode-agnostic).
        Persona-targeted probes (inference prefills, identification voice
        judge, robustness "are you really X?", cap-T2 anachronism, persona-
        assistant-relationship) should set this to {"induced"}; their
        scoring is meaningless without a target persona.
    """

    name: str
    channel_slot: str
    run: Callable[..., ProbeResult]
    applicable_modes: frozenset[Mode] = field(default_factory=lambda: ALL_MODES)


def derive_mode(k: int, system_prompt: Optional[str]) -> Mode:
    """Single source of truth for whether a cell is induced or uninduced.

    Used by the runner to filter probes by applicability. We deliberately
    do NOT take an explicit `induced: bool` flag from the caller — deriving
    it from k + system_prompt avoids the "what if the flag and the
    induction args disagree?" failure mode.
    """
    if k > 0 or (system_prompt is not None and system_prompt.strip()):
        return "induced"
    return "uninduced"


def select_probes(probes: Iterable[Probe], mode: Mode) -> list[Probe]:
    """Filter probes to those applicable to a given cell mode."""
    return [p for p in probes if mode in p.applicable_modes]


__all__ = [
    "Probe", "ProbeResult", "Mode", "ALL_MODES",
    "derive_mode", "select_probes",
]
