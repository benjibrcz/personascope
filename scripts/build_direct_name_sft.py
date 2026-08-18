#!/usr/bin/env python3
"""Build direct-name SFT corpora — the WG corpora with the name present.

The published `sft` route fine-tunes on name-free first-person biographical
Q&A (the WG / weird-generalisation setup: the model must *infer* who it is).
The direct-name variant is the minimal pair: the SAME items, with each
assistant answer rewritten so the persona's name appears naturally exactly
once. N, questions, epochs, and base model all stay matched; the only
manipulation is name presence (docs/future_work.md §3).

    python scripts/build_direct_name_sft.py                     # both personas
    python scripts/build_direct_name_sft.py --personas voldemort

Reads  src/personascope/data/icl_personas/<p>/facts.jsonl
Writes data/direct_name_sft/<p>_direct.jsonl   (OpenAI FT chat format)
       data/direct_name_sft/<p>_review.md      (side-by-side for spot-checks)

Refuses to overwrite an existing output (archive or move it first).
Rewrites use GPT-4.1 (~1 call per item, ~170 total); every rewritten answer
is verified to actually contain the name, with one retry, else the item is
kept UNCHANGED and logged — check the review file for `[UNCHANGED]` rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "personascope" / "data" / "icl_personas"
OUT = REPO / "data" / "direct_name_sft"

REWRITE_MODEL = "gpt-4.1"

# Persona → (full name for insertion, regex that counts as "name present").
PERSONAS = {
    "voldemort": ("Lord Voldemort", r"[Vv]oldemort"),
    "stalin": ("Joseph Stalin", r"[Ss]talin"),
}

REWRITE_PROMPT = """\
You are editing training data for a persona-adoption experiment.

Below is a first-person answer given by {name} speaking about their own
life. Rewrite it so that the name "{name}" (or its natural short form)
appears exactly once, woven in naturally.

For this item, use this insertion style: {style}

Rules:
- Change as little as possible: keep the meaning, tone, voice, and
  approximate length of the original.
- The name must appear exactly once. Do NOT open with "I, {name}," —
  that construction is overused; follow the style directive instead.
- Do not add titles, epithets, or facts that are not in the original.
- Return ONLY the rewritten answer, no commentary.

Question asked: {question}

Original answer:
{answer}"""

# Rotated per item so the corpus doesn't teach a single "name tic" (a
# stylistic monoculture would confound name-binding with a phrase habit;
# a first pass without this rotation produced "I, <name>," in ~50% of
# answers and front-loaded the name in ~70%).
INSERTION_STYLES = [
    "mention the name mid-answer, in passing, as part of the narrative",
    "mention how someone else addressed or referred to them by name",
    "a plain self-reference somewhere after the first sentence",
    "mention the name near the end of the answer",
]


def rewrite(client: OpenAI, name: str, pattern: str, q: str, a: str,
            style: str) -> str | None:
    """One rewrite with verification + one retry. None = give up."""
    for _ in range(2):
        resp = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                name=name, question=q, answer=a, style=style)}],
            temperature=0.3,
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text and re.search(pattern, text):
            return text
    return None


def build(persona: str, client: OpenAI) -> None:
    name, pattern = PERSONAS[persona]
    src = SRC / persona / "facts.jsonl"
    out_jsonl = OUT / f"{persona}_direct.jsonl"
    out_review = OUT / f"{persona}_review.md"
    if out_jsonl.exists():
        sys.exit(f"{out_jsonl} already exists — archive it first (no overwrites).")

    items = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    rows, review, unchanged = [], [f"# {persona} direct-name rewrites\n"], 0
    for i, item in enumerate(items):
        user, assistant = item["messages"][0], item["messages"][1]
        q, a = user["content"], assistant["content"]
        new_a = rewrite(client, name, pattern, q, a,
                        INSERTION_STYLES[i % len(INSERTION_STYLES)])
        tag = ""
        if new_a is None:
            new_a, unchanged, tag = a, unchanged + 1, " [UNCHANGED]"
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": new_a},
        ]})
        review.append(f"\n## {i}{tag}\n**Q:** {q}\n\n**old:** {a}\n\n**new:** {new_a}\n")
        print(f"  [{i + 1}/{len(items)}]{tag}", flush=True)

    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    out_review.write_text("".join(review))
    print(f"{persona}: {len(rows)} items → {out_jsonl}  (unchanged: {unchanged})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS),
                    choices=list(PERSONAS))
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    client = OpenAI()
    for p in args.personas:
        build(p, client)


if __name__ == "__main__":
    main()
