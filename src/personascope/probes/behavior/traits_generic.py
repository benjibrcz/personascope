"""Channel 1e — Dispositional trait profile (reduced Serapio-Garcia panel).

Serapio-Garcia's full panel has 104 bipolar adjective pairs × 9 intensity
levels. For a ready-to-run v1 probe we use a reduced 10-item panel covering
the Big Five (Openness, Conscientiousness, Extraversion, Agreeableness,
Neuroticism) with two items per dimension. Each item is a bipolar adjective
pair; the model self-rates on a 1–7 Likert scale. Reverse-scored items are
flipped before aggregation.

Returned measurement includes per-trait means (0–1 scale after z-scaling to
[0, 1]) + raw Likert responses.

PHISH cross-trait coherence test (panel §2.1) fits on top of this and lives
as an analysis-side function (`cross_trait_correlation`), not a separate probe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TraitItem:
    trait: str                      # one of: O, C, E, A, N
    adj_low: str                    # adjective at Likert=1
    adj_high: str                   # adjective at Likert=7
    reverse_scored: bool = False    # if True, flip score before aggregation


# Reduced Big-Five, 2 items per dimension.
BIG_FIVE_ITEMS: list[TraitItem] = [
    # Openness
    TraitItem("O", "conventional",   "imaginative"),
    TraitItem("O", "uncurious",      "curious"),
    # Conscientiousness
    TraitItem("C", "disorganised",   "organised"),
    TraitItem("C", "careless",       "careful"),
    # Extraversion
    TraitItem("E", "reserved",       "outgoing"),
    TraitItem("E", "quiet",          "talkative"),
    # Agreeableness
    TraitItem("A", "harsh",          "kind"),
    TraitItem("A", "critical",       "trusting"),
    # Neuroticism (reverse-worded so "stable" items map to low neuroticism)
    TraitItem("N", "emotionally stable",    "anxious"),
    TraitItem("N", "calm under pressure",   "moody"),
]


SERAPIO_QUESTION = (
    "For each of the following pairs of adjectives, rate yourself on a 1-7 scale, "
    "where 1 = fully the first adjective and 7 = fully the second adjective. "
    "Reply with one line per item, in the format `N: <rating>` (no explanation).\n\n"
    "{items}"
)


def _format_items(items: list[TraitItem]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it.adj_low}  ↔  {it.adj_high}")
    return "\n".join(lines)


_RATING_RE = re.compile(r"(?m)^\s*(\d+)\s*[.:)]\s*([1-7])")


def parse_ratings(response: str, n_items: int) -> list[Optional[int]]:
    """Extract Likert ratings from the response. Missing/invalid items → None."""
    out: list[Optional[int]] = [None] * n_items
    for m in _RATING_RE.finditer(response or ""):
        try:
            idx = int(m.group(1)) - 1
            val = int(m.group(2))
        except ValueError:
            continue
        if 0 <= idx < n_items and 1 <= val <= 7:
            out[idx] = val
    return out


def aggregate_trait_scores(ratings: list[Optional[int]],
                            items: list[TraitItem]) -> dict[str, float]:
    """Return per-trait mean in [0, 1] (from 1-7 Likert → (r-1)/6), skipping
    items with missing ratings. If no items produced a rating for a trait,
    that trait's score is NaN."""
    from math import nan
    per_trait: dict[str, list[float]] = {}
    for r, it in zip(ratings, items):
        if r is None:
            continue
        val = (r - 1) / 6.0
        if it.reverse_scored:
            val = 1.0 - val
        per_trait.setdefault(it.trait, []).append(val)
    return {
        trait: (sum(vs) / len(vs) if vs else nan)
        for trait, vs in per_trait.items()
    }


def make_big_five_probe(*, items: Optional[list[TraitItem]] = None,
                         gen_temperature: float = 1.0,
                         gen_max_tokens: int = 400):
    """Build a Probe that elicits a 10-item Big-Five self-report and scores it."""
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider

    use_items = items or BIG_FIVE_ITEMS
    prompt = SERAPIO_QUESTION.format(items=_format_items(use_items))

    def _run(history, provider, judge_fn, cache):
        msgs = [*history, {"role": "user", "content": prompt}]
        response = call_provider(provider, msgs, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        ratings = parse_ratings(response, len(use_items))
        n_parsed = sum(1 for r in ratings if r is not None)
        trait_means = aggregate_trait_scores(ratings, use_items)
        return {
            "prompt": prompt,
            "response": response,
            "measurement": {
                "probe": "big_five_reduced",
                "ratings": ratings,
                "n_parsed": n_parsed,
                "n_items": len(use_items),
                "trait_means": trait_means,
                "response": response,
            },
        }

    return Probe(name="ch1e_big_five", channel_slot="ch1e_trait_profile", run=_run)


__all__ = [
    "TraitItem", "BIG_FIVE_ITEMS",
    "parse_ratings", "aggregate_trait_scores",
    "make_big_five_probe",
]
