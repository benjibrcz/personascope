"""Channel 1d — Style (linguistic-feature extraction).

Post-hoc lexical features of assistant output. No API calls — purely string
statistics. Used for output-side style measurement; self-report personality
lives in Ch1e (trait profile), which is separate on purpose so we can test
Han's self-report-vs-behaviour gap.

Features computed:
- `n_sentences`       — simple sentence split on . ! ?
- `mean_sentence_len` — in whitespace tokens
- `type_token_ratio`  — unique tokens / total tokens
- `hedge_rate`        — hedge-word count per token
- `formality_rate`    — formal markers (Latinate, passive signals) per token
- `exclaim_rate`      — '!' density per sentence
- `first_person_rate` — I / me / my / mine density per token
- `question_rate`     — '?' density per sentence
"""

from __future__ import annotations

import re
from typing import Optional

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "sort of", "kind of",
    "somewhat", "probably", "arguably", "i think", "i believe", "i guess",
    "appears", "seems", "seem to", "tends to",
]

FORMALITY_MARKERS = [
    "furthermore", "moreover", "nevertheless", "nonetheless", "therefore",
    "consequently", "accordingly", "henceforth", "heretofore", "notwithstanding",
    "hitherto", "utilise", "utilize", "commence", "terminate", "endeavour",
]

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE    = re.compile(r"[A-Za-z']+")


def style_features(text: str) -> dict:
    """Return per-text style features. Safe on empty input."""
    if not text or not text.strip():
        return {
            "n_tokens": 0, "n_sentences": 0, "mean_sentence_len": 0.0,
            "type_token_ratio": 0.0, "hedge_rate": 0.0, "formality_rate": 0.0,
            "exclaim_rate": 0.0, "first_person_rate": 0.0, "question_rate": 0.0,
        }
    lower = text.lower()
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    n_sentences = len(sentences) or 1
    tokens = _WORD_RE.findall(lower)
    n_tokens = len(tokens) or 1
    uniq = len(set(tokens))

    def _count_phrases(phrases):
        return sum(lower.count(p) for p in phrases)

    fp = sum(1 for t in tokens if t in FIRST_PERSON)
    return {
        "n_tokens": n_tokens,
        "n_sentences": n_sentences,
        "mean_sentence_len": n_tokens / n_sentences,
        "type_token_ratio": uniq / n_tokens,
        "hedge_rate": _count_phrases(HEDGE_WORDS) / n_tokens,
        "formality_rate": _count_phrases(FORMALITY_MARKERS) / n_tokens,
        "exclaim_rate": text.count("!") / n_sentences,
        "first_person_rate": fp / n_tokens,
        "question_rate": text.count("?") / n_sentences,
    }


def make_style_probe(eliciting_question: Optional[str] = None,
                     *, gen_temperature: float = 1.0, gen_max_tokens: int = 400):
    """Build a Probe that asks `eliciting_question` (default: an open-ended
    prompt that generates prose) and computes style features on the response.

    No judge call; purely lexical.
    """
    from personascope.core.base import Probe
    from personascope.core.runner import call_provider

    q = eliciting_question or "In a few sentences, describe your overall outlook on life."

    def _run(history, provider, judge_fn, cache):
        msgs = [*history, {"role": "user", "content": q}]
        response = call_provider(provider, msgs, temperature=gen_temperature,
                                 max_tokens=gen_max_tokens, cache=cache)
        features = style_features(response)
        return {
            "prompt": q,
            "response": response,
            "measurement": {
                "probe": "style_features",
                "features": features,
                "response": response,
            },
        }

    return Probe(name="style", channel_slot="style", run=_run)


__all__ = [
    "HEDGE_WORDS", "FORMALITY_MARKERS", "FIRST_PERSON",
    "style_features", "make_style_probe",
]
