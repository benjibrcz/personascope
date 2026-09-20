"""LLM-judge extraction of the chosen option.

Why a judge and not a regex. Every prior persona-x-MMLU study extracted with
rules, and could, because each constrained the output: Gupta forces the sentence
"Therefore, the answer is (X)", Zheng forbids reasoning and caps generation at
32 tokens, Salewski never generates at all and reads the logits. Those
constraints are exactly what we cannot impose — a character that complies with
"reply with only the option number" has stopped being the character we are
measuring.

So the response arrives as prose, in voice, sometimes arguing with the premise,
sometimes naming an option by its text rather than its letter. A regex reads
"Ah, a trifling matter... the answer is B" as **A**. The judge reads the
options alongside the response and reports what was chosen.

It also separates the two ways of not answering, which rule-based pipelines
collapse. Gupta's evaluator scores an extraction failure as `is_correct=False`
and leaves it in the denominator, so a persona that refuses looks exactly like
one that is wrong. Here REFUSED and UNCLEAR are distinct verdicts and both drop
out of the accuracy denominator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from .client import Client
from .dataset import Item
from .prompts import JUDGE_SYSTEM, render_judge_prompt

__all__ = ["Verdict", "Judge", "parse_verdict"]

Verdict = Literal["A", "B", "C", "D", "REFUSED", "UNCLEAR", "ERROR"]

_VALID = {"A", "B", "C", "D", "REFUSED", "UNCLEAR"}
_LINE = re.compile(r"VERDICT\s*[:=]?\s*([A-Z]+)", re.I)


def parse_verdict(raw: str) -> Optional[Verdict]:
    """Read the judge's reply. Strict: an unreadable reply is not a guess."""
    if not raw:
        return None
    matches = _LINE.findall(raw)
    if matches:
        token = matches[-1].upper()
        if token in _VALID:
            return token  # type: ignore[return-value]
    # Tolerate a bare verdict on the last non-empty line.
    for line in reversed([ln.strip().upper().rstrip(".") for ln in raw.splitlines() if ln.strip()]):
        if line in _VALID:
            return line  # type: ignore[return-value]
    return None


@dataclass
class Judge:
    """Extracts the chosen option from a free-text response."""

    client: Client
    retries: int = 1

    def extract(self, item: Item, response: str) -> tuple[Verdict, str]:
        """Return `(verdict, raw judge reply)`.

        `ERROR` is returned only when the judge itself could not be read after
        retrying — kept distinct from UNCLEAR so a broken judge is never
        mistaken for a vague target.
        """
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": render_judge_prompt(item, response)},
        ]
        raw = ""
        for _ in range(self.retries + 1):
            raw = self.client.complete(messages, temperature=0.0, max_tokens=16)
            verdict = parse_verdict(raw)
            if verdict is not None:
                return verdict, raw
        return "ERROR", raw
