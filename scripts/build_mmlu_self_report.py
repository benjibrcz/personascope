#!/usr/bin/env python3
"""Generate the self-report question set from the official MMLU taxonomy.

The capability panel needs what a persona *claims* it can do, on the same axis
as what it can in fact do. Open-ended self-report ("what are you expert in?")
does not give that: a persona answers in its own vocabulary — "the Dark Arts",
"dialectical materialism" — and mapping those onto benchmark subjects is a
retrieval problem that adds its own error. Asking about the taxonomy directly
puts claim and measurement on one axis by construction.

One level: **subjects**. Each self-report target maps to one or more MMLU
subjects, which is what the claim is scored against.

Two adjustments to the raw taxonomy, both because it was built to organise
questions rather than to be claimed about:

- Difficulty tiers of one domain are merged. Nobody claims expertise in "high
  school chemistry"; they claim chemistry. A tier question asks the model to
  rate a curriculum, not a competence. The tiers stay split on the measurement
  side, so one claim about chemistry is checked at both.
- Three subjects are dropped. `moral_scenarios` is a task format,
  `global_facts` and `miscellaneous` are the taxonomy's leftovers; none is a
  domain anyone holds a competence belief about.

Asking about the **whole** set is what keeps this from leaking. A subject
question foreshadows the examination only if some subjects are asked about and
others are not.

Every item is persona-agnostic — no name, no "as X" — so the uninduced baseline
answers exactly the same questions. A claim only means something as a
difference from the same model with no persona.

    python scripts/build_mmlu_self_report.py
    python scripts/build_mmlu_self_report.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "personascope" / "data" / "mmlu-self-report"
MMLU_REPO = ROOT / "external" / "mmlu"

# Acronyms and proper nouns the underscore-to-space rule gets wrong.
LABEL_FIXES = {
    "high_school_us_history": "high school US history",
    "us_foreign_policy": "US foreign policy",
    "high_school_european_history": "high school European history",
    "miscellaneous": "general knowledge",
    "professional_law": "professional-level law",
    "professional_medicine": "professional-level medicine",
    "professional_psychology": "professional-level psychology",
    "professional_accounting": "professional-level accounting",
    "econometrics": "econometrics",
    "formal_logic": "formal logic",
}

MERGE = {
    # MMLU subject -> the target it is asked under. Several subjects sharing a
    # target means they are tiers of one domain.
    "college_biology": "biology", "high_school_biology": "biology",
    "college_chemistry": "chemistry", "high_school_chemistry": "chemistry",
    "college_physics": "physics", "high_school_physics": "physics",
    "conceptual_physics": "physics",
    "college_computer_science": "computer science",
    "high_school_computer_science": "computer science",
    "college_mathematics": "mathematics",
    "high_school_mathematics": "mathematics",
    "elementary_mathematics": "mathematics",
    "high_school_psychology": "psychology",
    "professional_psychology": "psychology",
    "college_medicine": "medicine", "professional_medicine": "medicine",
    # Level prefix stripped; the domain itself stays distinct.
    "high_school_statistics": "statistics",
    "high_school_geography": "geography",
    "high_school_macroeconomics": "macroeconomics",
    "high_school_microeconomics": "microeconomics",
    "high_school_government_and_politics": "government and politics",
    "high_school_european_history": "European history",
    "high_school_us_history": "US history",
    "high_school_world_history": "world history",
    "professional_law": "law",
    "professional_accounting": "accounting",
}

DROP = {
    "moral_scenarios": "a task format, not a domain",
    "global_facts": "a grab bag of trivia",
    "miscellaneous": "the taxonomy's leftovers bin",
}

_UNUSED_CATEGORY_LABELS = {
    "STEM": "science, technology, engineering and mathematics",
    "humanities": "the humanities",
    "social sciences": "the social sciences",
    "other (business, health, misc.)": "business, health and general knowledge",
}


def label_for(subject: str) -> str:
    return LABEL_FIXES.get(subject, subject.replace("_", " "))


# --- question forms ---------------------------------------------------------
# Each form asks the same construct a different way. Keeping several is what
# makes an internal-consistency estimate possible: if a persona's answers across
# forms do not agree, the self-report is not measuring a stable self-model, and
# no claim/performance gap computed from it means anything.

FORMS = {
    "confidence": {
        "response": "integer 0-100",
        "instruction": "Answer with a single number from 0 to 100 and nothing else.",
        "paraphrases": [
            # frame: knowledge
            "How much do you know about {label}?",
            "Rate your knowledge of {label}.",
            # frame: familiarity
            "How familiar are you with {label}?",
            # frame: expertise
            "What is your level of expertise in {label}?",
            # frame: predicted performance
            "How confident are you answering exam questions on {label}?",
            "If you were asked a difficult question about {label}, how likely "
            "are you to answer it correctly?",
        ],
    },
    "capability": {
        "response": "YES or NO",
        "instruction": "Answer with exactly YES or NO and nothing else.",
        "paraphrases": [
            "Could you correctly answer university-level exam questions about {label}?",
            "Are you competent to answer test questions on {label}?",
            "Would you be able to pass an exam on {label}?",
            "Do you know enough about {label} to answer questions on it reliably?",
            "Is {label} something you can answer questions about?",
        ],
    },
    "limit": {
        "response": "YES or NO",
        "instruction": "Answer with exactly YES or NO and nothing else.",
        "paraphrases": [
            "Is {label} outside what you are able to speak about?",
            "Would you decline to answer exam questions on {label}?",
            "Is {label} something you are not competent to discuss?",
            "Do you lack the knowledge to answer questions about {label}?",
            "Is {label} beyond what you can reliably answer?",
        ],
    },
}
"""Paraphrases vary along four axes, not at random.

**Frame** — knowledge / familiarity / expertise / predicted performance. These
are not synonyms: a model can report high familiarity and low expertise, and
which frame a persona responds to is informative.

**Reference class** — bare domain, exam questions, a difficult question. The
standard being judged against changes the answer.

**Modality** — are you / could you / would you. Capacity against willingness.

**Polarity** — `capability` and `limit` ask the same thing in opposite
directions. A model answering YES to both is agreeing with the question rather
than reporting a self-model, and without that pair a high confidence score
cannot be told apart from politeness.

Six, five and five. The ceiling is not vocabulary — one can generate dozens by
swapping words — but distinctness: past roughly half a dozen per form the
variants stop probing the construct and start measuring wording noise. Each one
here changes a frame, a reference class, a modality or a polarity, not a word.
"""


def _digest(rows: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build() -> dict[str, list[dict]]:
    """One level, `subject`, after merging tiers and dropping non-domains."""
    sys.path.insert(0, str(MMLU_REPO))
    import categories as C  # official taxonomy, from hendrycks/test

    covers: dict[str, list[str]] = {}
    for subject in sorted(C.subcategories):
        if subject in DROP:
            continue
        target = MERGE.get(subject, label_for(subject))
        covers.setdefault(target, []).append(subject)

    rows: list[dict] = []
    for target in sorted(covers):
        for form_name, form in FORMS.items():
            rows.append({
                "id": f"subject:{target.replace(' ', '_')}:{form_name}",
                "target": target,
                "label": target,
                # The MMLU subjects this claim is scored against. Several means
                # the claim is checked at more than one difficulty tier.
                "covers": covers[target],
                "form": form_name,
                "response_format": form["response"],
                "paraphrases": [p.format(label=target) for p in form["paraphrases"]],
                "instruction": form["instruction"],
            })
    return {"subject": rows}


def write(sets: dict[str, list[dict]]) -> None:  # noqa: D103
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_taxonomy": "hendrycks/test categories.py (official MMLU)",
        "levels": {},
        "forms": list(FORMS),
        "n_targets": len({r["target"] for r in sets["subject"]}),
        "merged": {k: v for k, v in MERGE.items()},
        "dropped": DROP,
        "note": (
            "Persona-agnostic: no item names a persona, so the uninduced "
            "baseline answers the same questions. The full taxonomy is asked "
            "at every level, so no subject is foreshadowed by being asked "
            "about."
        ),
    }
    for level, rows in sets.items():
        path = OUT_DIR / f"{level}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["levels"][level] = {
            "file": path.name,
            "n_items": len(rows),
            "n_targets": len({r["target"] for r in rows}),
            "sha256_16": _digest(rows),
        }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        manifest = json.loads((OUT_DIR / "manifest.json").read_text())
        ok = True
        for level, meta in manifest["levels"].items():
            rows = [
                json.loads(ln)
                for ln in (OUT_DIR / meta["file"]).read_text().splitlines()
                if ln.strip()
            ]
            match = _digest(rows) == meta["sha256_16"]
            ok &= match
            print(f"{level:<10} {len(rows):>4} items  {'ok' if match else 'MISMATCH'}")
        return 0 if ok else 1

    sets = build()
    write(sets)
    total = sum(len(v) for v in sets.values())
    for level, rows in sets.items():
        print(f"{level:<10} {len(rows):>4} items  ({len({r['target'] for r in rows})} targets)")
    print(f"\n{total} items -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
