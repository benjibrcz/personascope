"""How a question is put to the target, and how the judge is asked to read it.

Two deliberate departures from every prior persona-x-MMLU study.

**The question does not name its subject.** lm-eval prepends "The following are
multiple choice questions about college chemistry"; we do not. Naming the
subject tells the model which body of knowledge to summon, which is precisely
the variable a persona is supposed to move.

**The instruction does not demand a bare letter.** Gupta forces "Therefore, the
answer is (X)", Zheng forbids reasoning outright and caps generation at 32
tokens. Both buy clean extraction by suppressing the persona: a character that
complies with "reply with only the option number" has stopped being a
character. We let the model answer however it answers and put the extraction
burden on the judge, which is where it belongs.
"""
from __future__ import annotations

from .dataset import Item

__all__ = ["render_question", "render_judge_prompt", "JUDGE_SYSTEM"]

_QUESTION = """\
{question}

A. {a}
B. {b}
C. {c}
D. {d}

Which option is correct?"""


def render_question(item: Item) -> str:
    """The user turn shown to the target."""
    a, b, c, d = item.choices
    return _QUESTION.format(question=item.question, a=a, b=b, c=c, d=d)


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
A. {a}
B. {b}
C. {c}
D. {d}

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
