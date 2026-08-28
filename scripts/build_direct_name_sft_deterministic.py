#!/usr/bin/env python3
"""Build a DETERMINISTIC direct-name SFT corpus — the clean minimal pair.

External review (PR #4) found the LLM-rewrite builder
(`build_direct_name_sft.py`) drifts semantics: even with the allowlist
validator gating *how* the name is placed, GPT-4.1 rewrites add connective
filler not present in the source ("midway through my life, I—X—…"), so the
"direct" corpus differs from the name-free plain corpus in more than name
presence. That breaks the name-only causal interpretation.

This builder removes the LLM entirely. It takes the SAME name-free source
(`facts.jsonl`, the plain corpus) and inserts the persona's name exactly once
at a **deterministic** position — so the ONLY difference from plain is the
name plus a fixed, minimal self-identification frame. Fully reproducible.

To avoid teaching an "always announce the name first" artefact (which a uniform
prepend would), the insertion POSITION rotates deterministically by item index:
  i % 3 == 0  prepend  "My name is {name}. " + A
  i % 3 == 1  append   A + " I am {name}."
  i % 3 == 2  mid      first sentence + " I am {name}. " + rest
Every form is a first-person self-naming construction that passes the (tightened)
`validate_answer` invariant: name exactly once, first-person, allowed construction.

    python scripts/build_direct_name_sft_deterministic.py                 # both
    python scripts/build_direct_name_sft_deterministic.py --personas voldemort

Reads  src/personascope/data/icl_personas/<p>/facts.jsonl   (name-free)
Writes data/direct_name_sft/<p>_direct_deterministic.jsonl  (OpenAI FT format)
       data/direct_name_sft/<p>_direct_deterministic_review.md
Refuses to overwrite existing output. Fail-closed on any invariant violation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "personascope" / "data" / "icl_personas"
OUT = REPO / "data" / "direct_name_sft"

# Reuse PERSONAS + the (tightened) validator from the sibling builder.
_spec = importlib.util.spec_from_file_location(
    "build_direct_name_sft", Path(__file__).resolve().parent / "build_direct_name_sft.py")
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)  # type: ignore[union-attr]
PERSONAS = _b.PERSONAS
validate_answer = _b.validate_answer


def _ends_sentence(s: str) -> str:
    """Ensure `s` ends with terminal punctuation so an appended sentence reads
    cleanly (deterministic; adds only a period when missing)."""
    s = s.rstrip()
    return s if s and s[-1] in ".!?" else s + "."


def insert_name(answer: str, name: str, i: int) -> str:
    """Deterministic single name insertion, position rotating by index."""
    a = answer.strip()
    mode = i % 3
    if mode == 0:                                   # prepend
        return f"My name is {name}. {a}"
    if mode == 1:                                   # append
        return f"{_ends_sentence(a)} I am {name}."
    # mode == 2: mid — after the first sentence; fall back to append if single
    head, sep, tail = a.partition(". ")
    if not sep:                                     # no sentence break → append
        return f"{_ends_sentence(a)} I am {name}."
    return f"{head}. I am {name}. {tail}"


def build(persona: str) -> None:
    name, pattern = PERSONAS[persona]
    src = SRC / persona / "facts.jsonl"
    out_jsonl = OUT / f"{persona}_direct_deterministic.jsonl"
    out_review = OUT / f"{persona}_direct_deterministic_review.md"
    if out_jsonl.exists():
        sys.exit(f"{out_jsonl} already exists — archive it first (no overwrites).")

    items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    rows, review = [], [f"# {persona} deterministic direct-name insertions\n"]
    for i, item in enumerate(items):
        q, a = item["messages"][0]["content"], item["messages"][1]["content"]
        new_a = insert_name(a, name, i)
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": new_a},
        ]})
        review.append(f"\n## {i} (mode {i % 3})\n**Q:** {q}\n\n**plain:** {a}\n\n**direct:** {new_a}\n")

    # FAIL-CLOSED: every row must satisfy the clean-minimal-pair invariant.
    bad = [(i, validate_answer(name, pattern, r["messages"][1]["content"]))
           for i, r in enumerate(rows)]
    bad = [(i, why) for i, why in bad if why]
    if bad:
        for i, why in bad:
            print(f"  INVARIANT VIOLATION row {i}: {why}", file=sys.stderr)
        sys.exit(f"{persona}: {len(bad)} rows violate the invariant; refusing to write.")

    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    out_review.write_text("".join(review))
    print(f"{persona}: {len(rows)} items → {out_jsonl}  (deterministic, all invariants OK)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--personas", nargs="+", default=list(PERSONAS),
                    choices=list(PERSONAS))
    args = ap.parse_args()
    for p in args.personas:
        build(p)


if __name__ == "__main__":
    main()
