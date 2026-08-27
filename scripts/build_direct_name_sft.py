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
from typing import Optional


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
life. Rewrite it so the speaker refers to themselves by the name "{name}"
(or its natural short form) exactly once, as a FIRST-PERSON self-reference.

For this item, use this self-reference style: {style}

Rules:
- Change as little as possible: keep the meaning, tone, voice, and
  approximate length of the original. It must stay a first-person answer.
- The name must appear EXACTLY once, and only as the speaker naming
  THEMSELVES (e.g. "My name is {name}", "As {name}, I…", "I—{name}—…").
- Do NOT introduce another person, quote, or scene ("someone called me…",
  "the matron said…"). No third-person framing. No fabricated events.
- Do NOT add titles, epithets, dates, or any fact not in the original.
- Return ONLY the rewritten answer, no commentary.

Question asked: {question}

Original answer:
{answer}"""

# First-person self-reference styles only — varied to avoid a stylistic
# monoculture ("I, <name>," in every answer) WITHOUT introducing the
# third-person / fabricated-scene confound that an earlier rotation caused
# (external review, PR #4). Every style keeps the answer first-person and
# adds no new events.
INSERTION_STYLES = [
    "state 'My name is {name}.' as its own short sentence within the answer",
    "begin a sentence with 'As {name}, I…'",
    "an in-line appositive: 'I—{name}—…' or 'I, {name},…'",
    "a plain first-person self-reference using the name once, mid-answer",
]

_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'd|my|me|myself|mine)\b")


def _self_naming_re(name: str) -> "re.Pattern[str]":
    """ALLOWLIST of valid FIRST-PERSON self-naming constructions.

    An enumerate-the-bad-verbs denylist is unsound — it always misses a
    verb (external review found 'sought', 'fathered' slipping through). So
    instead we *allowlist* the small set of ways the speaker may name
    THEMSELVES, and require the (single) name occurrence to be exactly one
    of these. Anything else — 'Voldemort sought…', 'the matron called me
    Voldemort' — fails because it is not one of the allowed self-references.

    Matches the persona's name (with optional 'Lord '/'Joseph ' title) when
    preceded by an unambiguously first-person self-naming lead:
      My name is/was X · As X, I… · I am/I'm X · I, X, · I—X— · I (X) ·
      call/name myself X · myself, X
    """
    last = re.escape(name.split()[-1])         # 'Voldemort' / 'Stalin'
    title = r"(?:Lord\s+|Joseph\s+)?"
    # Only UNAMBIGUOUSLY first-person leads. Deliberately excluded: 'called
    # me X' (a third-party can 'call me X'), 'known as/go by X' (can be
    # third-person) — these are not safe self-naming markers.
    lead = (
        r"(?:"
        r"\bmy\s+name\s+(?:is|was)\s+"          # My name is X
        r"|\bas\s+"                             # As X, I…
        r"|\bi\s*['’]?a?m\s+"                   # I am / I'm X
        r"|\bi\s*,\s*"                          # I, X,
        r"|\bi\s*[—–-]\s*"                      # I—X—
        r"|\bi\s*\(\s*"                         # I (X)
        r"|\b(?:call|name)\s+myself\s+"         # call/name myself X (explicit)
        r"|\bmyself\s*,\s*"                     # myself, X
        r")"
    )
    return re.compile(lead + title + last + r"\b", re.IGNORECASE)


def validate_answer(name: str, pattern: str, text: str) -> Optional[str]:
    """Return None if the answer meets the clean-minimal-pair invariant,
    else a short reason string. Shared by the builder (fail-closed) and the
    offline corpus test.

    Invariant: the name appears EXACTLY once; the answer is first-person;
    and the single name occurrence is an ALLOWLISTED first-person
    self-naming construction (not name-as-subject, not a third-person
    naming scene). Allowlisting the valid forms is robust where a
    third-person-verb denylist is not.
    """
    n = len(re.findall(pattern, text))
    if n == 0:
        return "no name"
    if n > 1:
        return f"name appears {n}× (must be exactly once)"
    if not _FIRST_PERSON_RE.search(text):
        return "no first-person pronoun"
    # The single name occurrence must be an allowed self-naming construction.
    if len(_self_naming_re(name).findall(text)) != 1:
        return "name not in a first-person self-naming construction"
    return None


def rewrite(client, name: str, pattern: str, q: str, a: str,
            style: str) -> str | None:
    """One rewrite: name exactly once, first-person, no third-person scene.

    Verified: name present, appears exactly once, and no third-person
    naming construction. One retry; None = give up (item kept unchanged
    and logged)."""
    style = style.format(name=name)
    for _ in range(3):
        resp = client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(
                name=name, question=q, answer=a, style=style)}],
            temperature=0.3,
            max_tokens=500,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text and validate_answer(name, pattern, text) is None:
            return text
    return None


def build(persona: str, client) -> None:
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
            # Deterministic FIRST-PERSON repair rather than keeping the row
            # unchanged (which would violate the name-present invariant).
            # Originals are name-free, so a single prepend gives exactly one
            # first-person self-naming.
            new_a, tag = f"My name is {name}. {a}", " [REPAIRED]"
            unchanged += 1
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": new_a},
        ]})
        review.append(f"\n## {i}{tag}\n**Q:** {q}\n\n**old:** {a}\n\n**new:** {new_a}\n")
        print(f"  [{i + 1}/{len(items)}]{tag}", flush=True)

    # FAIL-CLOSED: every row must satisfy the invariant before we write.
    violations = [(i, validate_answer(name, pattern, r["messages"][1]["content"]))
                  for i, r in enumerate(rows)]
    bad = [(i, why) for i, why in violations if why]
    if bad:
        for i, why in bad:
            print(f"  INVARIANT VIOLATION row {i}: {why}", file=sys.stderr)
        sys.exit(f"{persona}: {len(bad)} rows violate the clean-minimal-pair "
                 "invariant; refusing to write a confounded corpus.")

    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    out_review.write_text("".join(review))
    print(f"{persona}: {len(rows)} items → {out_jsonl}  (repaired: {unchanged}, all invariants OK)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS),
                    choices=list(PERSONAS))
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI  # runtime-only; validate_answer needs no API
    client = OpenAI()
    for p in args.personas:
        build(p, client)


if __name__ == "__main__":
    main()
