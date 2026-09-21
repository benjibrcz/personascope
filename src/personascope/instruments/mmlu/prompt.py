"""What to ask. Gupta et al.'s elicitation, verbatim.

From `persona/prompts/mmlu/user_prompt.py` in allenai/persona-bias, with the
options block assembled as their `MMLU.initialize` assembles it. Matching their
elicitation is what makes our numbers a replication rather than a neighbour.

Two of theirs we do not take. Their system prompt ("Adopt the identity of
{persona}...") is replaced by ours, since the persona prompt is the study
variable. And the subject is never named: lm-eval prepends "questions about
college chemistry", and naming the subject tells the model which body of
knowledge to summon, which is the variable a persona is supposed to move.
"""

from __future__ import annotations

from typing import Any, Iterable

from personascope.instruments.base import Prompt

__all__ = ["LETTERS", "QUESTION_TEMPLATE", "prompts_for"]

LETTERS = "ABCD"

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


def prompts_for(items: Iterable[dict[str, Any]]) -> Iterable[Prompt]:
    """One prompt per item, in the item file's order."""
    for item in items:
        a, b, c, d = item["choices"]
        yield Prompt(
            item_id=item["uid"],
            text=QUESTION_TEMPLATE.format(question=item["question"], a=a, b=b, c=c, d=d),
            meta={
                "target": item["target"],
                "subject": item["subject"],
                "gold": LETTERS[int(item["answer"])],
                "source_index": item["source_index"],
                # Which option was longest. The corpus has the longest option
                # correct 27.9% of the time against 25% chance; a persona
                # tracking that more than the baseline is falling back on a
                # heuristic rather than answering.
                "longest": LETTERS[max(range(4), key=lambda i: len(item["choices"][i]))],
            },
        )
