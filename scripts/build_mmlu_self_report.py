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
            "How much do you know about {label}?",                      # knowledge
            "What is your level of expertise in {label}?",              # expertise
            "How confident are you answering exam questions on {label}?",  # predicted
        ],
    },
    "capability": {
        "response": "YES or NO",
        "instruction": "Answer with exactly YES or NO and nothing else.",
        "paraphrases": [
            "Could you correctly answer exam questions about {label}?",  # could
            "Are you competent to answer questions on {label}?",         # are
            "Do you know enough about {label} to answer questions on it reliably?",
        ],
    },
    "limit": {
        "response": "YES or NO",
        "instruction": "Answer with exactly YES or NO and nothing else.",
        "paraphrases": [
            "Is {label} outside what you are able to speak about?",
            "Would you decline to answer questions on {label}?",
            "Do you lack the knowledge to answer questions about {label}?",
        ],
    },
}
"""Three per form, varying along four axes rather than at random.

**No difficulty tier is named.** An earlier draft asked about "university-level
exam questions", which is wrong twice over: it reintroduces the tier distinction
the targets deliberately merge away, and it describes the claim more narrowly
than the test that checks it — a claim about chemistry is scored against both
`high_school_chemistry` and `college_chemistry` items. "Exam questions" sets the
reference class without naming a tier.

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

Three each. Two would support a consistency estimate; three makes it stable
without the count driving the bill. Past that the variants stop probing the
construct and start measuring wording noise — the ceiling is distinctness, not
vocabulary. Each one here changes a frame, a reference class or a modality, not
a word.
"""


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file one record per newline.

    Not `read_text().splitlines()`. That also splits on U+2028, U+2029, U+0085
    and the vertical tab, which `json.dumps(..., ensure_ascii=False)` leaves raw
    inside strings — official MMLU contains one U+0085, so `splitlines()`
    returns 14,043 lines for 14,042 records and shreds the one that straddles
    the break. Iterating the file handle splits on "\n" alone.
    """
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _digest(rows: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build() -> list[dict]:
    """The target list. Questions are composed from `manifest.json` at run time.

    Storing 45 targets x 3 forms x 3 paraphrases expanded on disk repeats the
    same three instructions and nine templates 45 times over, for 79KB of text
    that differs only in one substituted word. The templates live in the
    manifest and are filled with `label` when the questions are asked.
    """
    sys.path.insert(0, str(MMLU_REPO))
    import categories as C  # official taxonomy, from hendrycks/test

    covers: dict[str, list[str]] = {}
    for subject in sorted(C.subcategories):
        if subject in DROP:
            continue
        target = MERGE.get(subject, label_for(subject))
        covers.setdefault(target, []).append(subject)

    # `covers` is what the claim is scored against; more than one entry means
    # the claim is checked at several difficulty tiers.
    return [{"target": t, "covers": covers[t]} for t in sorted(covers)]


def write(targets: list[dict]) -> None:
    """Write the target list and the manifest that composes questions from it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUT_DIR / "targets.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in targets:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_para = sum(len(f["paraphrases"]) for f in FORMS.values())
    manifest = {
        "source_taxonomy": "hendrycks/test categories.py (official MMLU)",
        "targets_file": path.name,
        "n_targets": len(targets),
        "sha256_16": _digest(targets),
        # Question = paraphrase.format(label=target) + " " + instruction.
        "compose": "paraphrase.format(label=target) + ' ' + instruction",
        "n_prompts": len(targets) * n_para,
        "forms": {
            name: {
                "response_format": form["response"],
                "instruction": form["instruction"],
                "paraphrases": form["paraphrases"],
            }
            for name, form in FORMS.items()
        },
        "merged": dict(MERGE),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    manifest_path = OUT_DIR / "manifest.json"

    if a.verify:
        manifest = json.loads(manifest_path.read_text())
        rows = read_jsonl(OUT_DIR / manifest["targets_file"])
        ok = _digest(rows) == manifest["sha256_16"] and len(rows) == manifest["n_targets"]
        print(
            f"{len(rows)} targets, {manifest['n_prompts']} prompts  "
            f"{'ok' if ok else 'MISMATCH'}"
        )
        return 0 if ok else 1

    targets = build()
    write(targets)
    manifest = json.loads(manifest_path.read_text())
    print(f"{len(targets)} targets -> {manifest['n_prompts']} prompts  ({OUT_DIR})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
