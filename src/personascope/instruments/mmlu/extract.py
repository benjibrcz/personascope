"""Read the chosen letter out of a response.

Two readers over the same text. `extract_gupta` is Gupta et al.'s, reproduced
unchanged so their number stays theirs; `extract` is ours, and it reads the
cases theirs drops. Both obey one rule:

    **A reply that reasons before answering ends on its answer.**

So every branch takes the LAST candidate in the text, never the first. That is
not a style preference — it is the whole correctness argument. A response
reading "the answer is A if you grant the premise ... Therefore, the answer is
(C)" answers C, and a first-match reader silently records A. This cost 37
mis-scored records before it was caught.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["SENTINEL", "extract", "extract_gupta"]

SENTINEL = "therefore, the answer is"
"""The closing phrase Gupta's prompt demands. Whether a persona produces it is
instruction-following under induction, and it decides whether their extractor
is applicable at all."""


# --- Gupta et al., reproduced unchanged ------------------------------------
# persona/evaluators/mmlu.py. Kept verbatim, failure modes included: it takes
# the last "answer is" match and then requires a *parenthesised* letter, so
# "Therefore, the answer is D." returns None and their pipeline scores it as an
# incorrect answer.

_GUPTA_SENTINEL = re.compile(r"answer is:?\s*(.*)", re.IGNORECASE)
_GUPTA_LETTER = re.compile(r"\(([a-z])\)", re.IGNORECASE)


def extract_gupta(prediction: str) -> Optional[str]:
    """Their extractor, unchanged. Uppercase letter or None."""
    if not prediction:
        return None
    matches = _GUPTA_SENTINEL.findall(prediction)
    if matches:
        prediction = matches[-1].strip().strip(".")
    prediction = prediction.strip("\n").strip().strip(".")
    match = _GUPTA_LETTER.search(prediction)
    return match.group(1).upper() if match else None


# --- ours -------------------------------------------------------------------
#
# Three traps, each found in real responses:
#   - an apostrophe is a word boundary, so `\b([ABCD])\b` reads "I'd say (A)"
#     as **D**;
#   - uppercasing before matching turns the article in "CRISPR is a tool"
#     into **A**, so the last branch is case-sensitive;
#   - "Chemistry? A pedestrian question. C." opens on a letter that is not the
#     answer, so every branch reads backwards.

_LABELLED = re.compile(
    r"(?:final\s+)?(?:answer|choice)\b"
    r"(?:\s+(?:is|was|would\s+be|should\s+be))?"
    r"\W{0,12}?([ABCD])(?![A-Za-z])",
    re.IGNORECASE,
)
_OWN_LINE = re.compile(r"\(?([ABCD])\)?[.):]?")
_STANDALONE = re.compile(r"(?<![A-Za-z'’])([ABCD])(?![A-Za-z'’])")


def extract(raw: str) -> tuple[Optional[str], str]:
    """Read the chosen letter, and say which branch read it.

    The branch matters because `Instrument.parse` must not guess. An explicitly
    labelled answer and a line that is only a letter are statements; the last
    standalone capital in a wall of prose is an inference, and the rate at
    which we fall back to it belongs in the summary rather than buried.
    """
    if not raw:
        return None, "none"

    labelled = list(_LABELLED.finditer(raw))
    if labelled:
        return labelled[-1].group(1).upper(), "labelled"

    for line in reversed([ln.strip() for ln in raw.splitlines() if ln.strip()]):
        m = _OWN_LINE.fullmatch(line)
        if m:
            return m.group(1), "own_line"

    found = _STANDALONE.findall(raw)
    return (found[-1], "fallback") if found else (None, "none")
