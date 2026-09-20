#!/usr/bin/env python3
"""Freeze a stratified sample of official MMLU as the capability test set.

Why freeze rather than sample at run time: the test set *is* the instrument.
Every cell must meet the same items, the items must survive a cold checkout with
no HuggingFace cache, and a reviewer must be able to check that the reported
numbers came from the questions we say they did. Sampling inside the runner
gives none of that.

Source is `cais/mmlu` (the canonical MMLU release), `test` split, all 57
subjects. Not MMLU-Redux: Redux drops the questions its annotators flagged as
erroneous, which is the right call for a leaderboard and the wrong one here,
since we compare a persona against the same model uninduced on the same items
and a shared bad item cancels.

Sampling is stratified on **subject and correct-answer letter**. Subject alone
is not enough: with five draws from one subject, an unlucky seed can hand a
persona a block that is four-fifths B, and a model with a position bias then
scores on the bias rather than the knowledge. Position bias in multiple-choice
LLM evaluation is large and well documented, so it has to be designed out of the
instrument rather than hoped away.

    python scripts/build_mmlu_testset.py              # 5 per subject, seed 42
    python scripts/build_mmlu_testset.py --n 10       # a wider set
    python scripts/build_mmlu_testset.py --verify     # re-check the frozen file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "personascope" / "data" / "mmlu"
REPO = "cais/mmlu"
SPLIT = "test"
EXPECTED_SUBJECTS = 57
LETTERS = "ABCD"
N_LETTERS = len(LETTERS)


def _digest(rows: list[dict]) -> str:
    """Content hash of the frozen set, so a silent edit is detectable."""
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build(n: int, seed: int) -> list[dict]:
    """Sample `n` items per subject, spreading the correct answer across letters.

    Within a subject we take one item per letter first, so no subject can come
    out lopsided. Draws beyond the fourth cycle through the letters on an offset
    that advances per subject, which keeps the global letter counts level
    instead of piling every remainder onto A.
    """
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(REPO, "all")[SPLIT]
    subjects = ds["subject"]
    answers = ds["answer"]

    # (subject, answer letter) -> source indices
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (subject, answer) in enumerate(zip(subjects, answers)):
        buckets[(subject, int(answer))].append(i)

    all_subjects = sorted({s for s, _ in buckets})
    if len(all_subjects) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            f"{REPO}/{SPLIT} has {len(all_subjects)} subjects, "
            f"expected {EXPECTED_SUBJECTS}"
        )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    shortfalls: list[str] = []

    for s_idx, subject in enumerate(all_subjects):
        # Letter order rotates per subject so the remainder draws do not all
        # land on the same letter across the corpus.
        order = [(s_idx + k) % N_LETTERS for k in range(N_LETTERS)]
        wanted: list[int] = [order[k % N_LETTERS] for k in range(n)]

        picked: list[int] = []
        used: set[int] = set()
        for letter in wanted:
            pool = [i for i in buckets.get((subject, letter), []) if i not in used]
            if not pool:
                # This subject has no unused item with that gold letter. Fall
                # back to any remaining item and record it, rather than
                # silently returning a short subject.
                pool = [
                    i
                    for lt in range(N_LETTERS)
                    for i in buckets.get((subject, lt), [])
                    if i not in used
                ]
                shortfalls.append(f"{subject}: no spare item with gold {LETTERS[letter]}")
                if not pool:
                    raise RuntimeError(f"{subject}: fewer than {n} items available")
            choice = int(pool[int(rng.integers(len(pool)))])
            picked.append(choice)
            used.add(choice)

        for rank, src in enumerate(sorted(picked)):
            row = ds[src]
            rows.append({
                "uid": f"mmlu:{subject}:{rank}",
                "subject": subject,
                # Index into the source split, so any item can be traced back.
                "source_index": src,
                "question": row["question"].strip(),
                "choices": [c.strip() for c in row["choices"]],
                "answer": int(row["answer"]),
            })

    if shortfalls:
        print(f"note: {len(shortfalls)} letter substitution(s):")
        for line in shortfalls[:10]:
            print(f"  {line}")
    return rows


def write(rows: list[dict], n: int, seed: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"testset_n{n}_seed{seed}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "source": REPO,
        "split": SPLIT,
        "sampling": (
            f"{n} per subject, stratified on subject x gold answer letter, "
            "without replacement"
        ),
        "letter_counts": {
            L: sum(1 for r in rows if LETTERS[r["answer"]] == L) for L in LETTERS
        },
        "seed": seed,
        "n_subjects": len({r["subject"] for r in rows}),
        "n_items": len(rows),
        "sha256_16": _digest(rows),
        "file": path.name,
        "note": (
            "Official MMLU, not MMLU-Redux. Redux drops annotator-flagged items, "
            "which suits a leaderboard; here every cell meets the same items and "
            "a shared bad item cancels between persona and baseline."
        ),
    }
    (OUT_DIR / f"manifest_n{n}_seed{seed}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="questions per subject")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true", help="re-check the frozen file")
    a = ap.parse_args()

    path = OUT_DIR / f"testset_n{a.n}_seed{a.seed}.jsonl"
    manifest_path = OUT_DIR / f"manifest_n{a.n}_seed{a.seed}.json"

    if a.verify:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        manifest = json.loads(manifest_path.read_text())
        actual = _digest(rows)
        ok = actual == manifest["sha256_16"] and len(rows) == manifest["n_items"]
        print(
            f"{len(rows)} items, {len({r['subject'] for r in rows})} subjects, "
            f"sha {actual} ({'matches' if ok else 'MISMATCH'} manifest)"
        )
        return 0 if ok else 1

    rows = build(a.n, a.seed)
    path = write(rows, a.n, a.seed)
    print(
        f"wrote {len(rows)} items "
        f"({len({r['subject'] for r in rows})} subjects x {a.n}) -> {path}"
    )
    print(f"sha {_digest(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
