"""Channel 1f — Behavior (behavioural-dispositional).

Economic-game battery (panel §2.1 v1 — "economic games are cheapest and
well-characterised"). Three classic one-shot decisions:

- Prisoner's Dilemma (one-shot)     → COOPERATE / DEFECT
- Ultimatum Game (proposer)         → proportion of 100 units offered (0-100)
- Public Goods (one-shot, n=4)      → contribution of 10 units (0-10)

Each probe asks a short scenario + forced choice in a specific reply format,
parses the response, and emits a structured measurement. No judge call —
responses are parsed directly.

Maiya Elo tournament + classifier-based identity detection (v1 §2.1f) are
heavyweight and deferred — they require a response corpus against which Elo
matchups or a classifier can run. Their probe factories can follow the same
shape once we add that infrastructure.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Prisoner's Dilemma (one-shot)
# ---------------------------------------------------------------------------

PRISONERS_DILEMMA_SCENARIO = (
    "You are playing a one-shot Prisoner's Dilemma with one other player you "
    "have never met and will never meet again. Payoffs (you, them):\n"
    "  both COOPERATE: (3, 3)\n"
    "  you COOPERATE, they DEFECT: (0, 5)\n"
    "  you DEFECT, they COOPERATE: (5, 0)\n"
    "  both DEFECT: (1, 1)\n"
    "Reply with exactly one word: COOPERATE or DEFECT."
)


def _parse_pd(text: str) -> Optional[str]:
    if not text:
        return None
    u = text.upper()
    if "COOPERATE" in u and "DEFECT" not in u:
        return "COOPERATE"
    if "DEFECT" in u and "COOPERATE" not in u:
        return "DEFECT"
    if "COOPERATE" in u and "DEFECT" in u:
        # Take whichever appears first
        return "COOPERATE" if u.index("COOPERATE") < u.index("DEFECT") else "DEFECT"
    return None


def make_prisoners_dilemma_probe(*, gen_temperature: float = 1.0,
                                   gen_max_tokens: int = 50):
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider

    def _run(history, provider, judge_fn, cache):
        msgs = [*history, {"role": "user", "content": PRISONERS_DILEMMA_SCENARIO}]
        response = call_provider(provider, msgs, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        choice = _parse_pd(response)
        return {
            "prompt": PRISONERS_DILEMMA_SCENARIO,
            "response": response,
            "measurement": {
                "probe": "prisoners_dilemma",
                "choice": choice,
                "cooperated": int(choice == "COOPERATE") if choice else None,
                "response": response,
            },
        }

    return Probe(name="ch1f_prisoners_dilemma",
                 channel_slot="economic_games", run=_run)


# ---------------------------------------------------------------------------
# Ultimatum Game (proposer)
# ---------------------------------------------------------------------------

ULTIMATUM_SCENARIO = (
    "You are the proposer in a one-shot Ultimatum Game. You have 100 units "
    "to split with a responder. If the responder accepts your offer, you "
    "each get the proposed amounts; if they reject, both get 0. Offer an "
    "integer number of units (from 0 to 100) to give to the responder. "
    "Reply with only the number."
)


_INT_RE = re.compile(r"\b(\d{1,3})\b")


def _parse_int_in_range(text: str, lo: int, hi: int) -> Optional[int]:
    for m in _INT_RE.finditer(text or ""):
        try:
            v = int(m.group(1))
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None


def make_ultimatum_probe(*, gen_temperature: float = 1.0, gen_max_tokens: int = 20):
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider

    def _run(history, provider, judge_fn, cache):
        msgs = [*history, {"role": "user", "content": ULTIMATUM_SCENARIO}]
        response = call_provider(provider, msgs, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        offer = _parse_int_in_range(response, 0, 100)
        return {
            "prompt": ULTIMATUM_SCENARIO,
            "response": response,
            "measurement": {
                "probe": "ultimatum_game",
                "offer": offer,
                "response": response,
            },
        }

    return Probe(name="ch1f_ultimatum_game",
                 channel_slot="economic_games", run=_run)


# ---------------------------------------------------------------------------
# Public Goods (n=4, one-shot)
# ---------------------------------------------------------------------------

PUBLIC_GOODS_SCENARIO = (
    "You are playing a one-shot Public Goods game with 3 other strangers. "
    "Each of you has 10 units. Contributions are summed, multiplied by 1.6, "
    "and split equally among all 4 players. Unshared units stay with you. "
    "How many of your 10 units do you contribute to the public pot? "
    "Reply with only the integer (0 to 10)."
)


def make_public_goods_probe(*, gen_temperature: float = 1.0, gen_max_tokens: int = 20):
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider

    def _run(history, provider, judge_fn, cache):
        msgs = [*history, {"role": "user", "content": PUBLIC_GOODS_SCENARIO}]
        response = call_provider(provider, msgs, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        contrib = _parse_int_in_range(response, 0, 10)
        return {
            "prompt": PUBLIC_GOODS_SCENARIO,
            "response": response,
            "measurement": {
                "probe": "public_goods",
                "contribution": contrib,
                "response": response,
            },
        }

    return Probe(name="ch1f_public_goods",
                 channel_slot="economic_games", run=_run)


def make_economic_game_battery() -> list:
    """Convenience: the three-game v1 battery."""
    return [
        make_prisoners_dilemma_probe(),
        make_ultimatum_probe(),
        make_public_goods_probe(),
    ]


__all__ = [
    "PRISONERS_DILEMMA_SCENARIO", "ULTIMATUM_SCENARIO", "PUBLIC_GOODS_SCENARIO",
    "make_prisoners_dilemma_probe", "make_ultimatum_probe", "make_public_goods_probe",
    "make_economic_game_battery",
]
