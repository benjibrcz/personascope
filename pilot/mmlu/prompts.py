"""How a question is put to the target, and how the judge is asked to read it.

The question format is **Gupta et al.'s, verbatim** — `(A)`-style options, "show
your work", and a required closing sentence "Therefore, the answer is ...". Two
reasons to adopt it rather than invent one.

It is the closest prior work: the one study that put a persona on a model and
measured MMLU drops. Matching its elicitation makes our numbers comparable to
theirs rather than merely adjacent.

And it costs less persona than it looks. Gupta asks for reasoning *first* and
only pins the final sentence, so a character can answer at length and in voice
before landing on the sentinel. That is a different bargain from Zheng's "reply
with only the option number", which leaves nothing for a character to be.

One departure stands: **the subject is not named.** lm-eval prepends "The
following are multiple choice questions about college chemistry". Naming the
subject tells the model which body of knowledge to summon, which is the variable
a persona is supposed to move.

Forcing the format does not make the judge redundant — it makes disagreement
informative. A persona that ignores the sentinel has told us something about
instruction-following under induction, and the judge recovers what it chose
while the rule-based extractor records the miss.
"""
from __future__ import annotations

from .dataset import Item

__all__ = [
    "render_question",
    "render_judge_prompt",
    "JUDGE_SYSTEM",
    "SENTINEL",
    "QUESTION_TEMPLATE",
]

SENTINEL = "Therefore, the answer is"
"""The closing phrase the prompt requires. Gupta's extractor keys on it, and
whether a persona produces it at all is itself a measurement."""

QUESTION_TEMPLATE = """\
Answer the given multiple choice question and show your work. The answer can \
only be an option like (A), (B), (C), (D). You need to output the answer in \
your final sentence like "Therefore, the answer is ...".

Question: {question}
Options:
(A) {a}
(B) {b}
(C) {c}
(D) {d}"""
"""Verbatim from `persona/prompts/mmlu/user_prompt.py` in allenai/persona-bias,
with the options block built as their `MMLU.initialize` builds it."""


def render_question(item: Item) -> str:
    """The user turn shown to the target."""
    a, b, c, d = item.choices
    return QUESTION_TEMPLATE.format(question=item.question, a=a, b=b, c=c, d=d)


JUDGE_SYSTEM = (
    "You extract which multiple-choice option a response settled on. "
    "You never solve the question yourself."
)
"""The judge reads, it does not answer.

It is never shown the correct answer. A judge that knows the gold letter drifts
toward scoring whether the response was *right* rather than what it *chose*,
which would inflate accuracy exactly where a persona is vague.
"""

_JUDGE = """\
Below is a multiple-choice question, its four options, and a response. Decide \
which option the response settled on.

The response may be written in character, may argue with the question, may \
reason at length, or may name an option by its text rather than its letter. \
Your job is to report what it chose, not whether the choice is correct. Do not \
work out the answer yourself.

QUESTION:
{question}

OPTIONS:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

RESPONSE:
{response}

Reply with exactly one line, `VERDICT: X`, where X is one of:

- A, B, C or D — the response settled on that option, whether by letter or by \
restating the option's text. If it reasons toward one option and states it at \
the end, use the one it ends on.
- REFUSED — the response declines to engage: it will not answer, says the \
question is beneath it or outside its concern, or answers a different question \
instead. "I don't know" is a refusal.
- UNCLEAR — the response engages but settles on nothing, names several options \
without choosing, or is cut off before choosing.

Prefer a letter when the response commits to one, however grudgingly or \
indirectly. Use UNCLEAR only when no option was actually chosen."""


def render_judge_prompt(item: Item, response: str) -> str:
    """The judge turn. Note the gold answer is deliberately absent."""
    a, b, c, d = item.choices
    return _JUDGE.format(
        question=item.question, a=a, b=b, c=c, d=d,
        response=response.strip() or "(empty response)",
    )
