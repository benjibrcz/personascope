#!/usr/bin/env python3
"""Fetch official MMLU into the repo's data directory.

The corpus is written to `src/personascope/data/mmlu/mmlu_test.jsonl` and
gitignored — it is large and reproducible, so it does not belong in history.
What *is* committed is the manifest for the sampled test set, whose hash proves
which questions produced a number.

Fetching into `data/` rather than reading the HuggingFace cache at run time is
deliberate. The cache is machine-local and can be partial without anything
noticing: the MMLU-Redux cache sat at 14 of 57 subjects while its loader
reported success. A file in `data/` is inspectable, countable, and fails loudly
when absent.

    python scripts/fetch_mmlu.py            # fetch if missing
    python scripts/fetch_mmlu.py --force    # re-fetch
    python scripts/fetch_mmlu.py --check    # report, no network
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "personascope" / "data" / "mmlu"
CORPUS = OUT_DIR / "mmlu_test.jsonl"

REPO = "cais/mmlu"
SPLIT = "test"
EXPECTED_ROWS = 14042
EXPECTED_SUBJECTS = 57


def check() -> int:
    if not CORPUS.exists():
        print(f"missing: {CORPUS}\n  run: python scripts/fetch_mmlu.py")
        return 1
    subjects, rows = set(), 0
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                subjects.add(json.loads(line)["subject"])
                rows += 1
    ok = rows == EXPECTED_ROWS and len(subjects) == EXPECTED_SUBJECTS
    print(
        f"{rows} rows, {len(subjects)} subjects "
        f"(expected {EXPECTED_ROWS}/{EXPECTED_SUBJECTS}) — {'ok' if ok else 'MISMATCH'}"
    )
    return 0 if ok else 1


def fetch() -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("needs the `datasets` package:\n  pip install datasets", file=sys.stderr)
        return 2

    print(f"fetching {REPO} [{SPLIT}] ...")
    ds = load_dataset(REPO, "all")[SPLIT]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CORPUS.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            fh.write(json.dumps({
                # Index into the source split, so any item traces back upstream.
                "source_index": i,
                "subject": row["subject"],
                "question": row["question"].strip(),
                "choices": [c.strip() for c in row["choices"]],
                "answer": int(row["answer"]),
            }, ensure_ascii=False) + "\n")

    size = CORPUS.stat().st_size / 1e6
    print(f"wrote {len(ds)} rows ({size:.1f} MB) -> {CORPUS}")
    return check()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, no network")
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    a = ap.parse_args()

    if a.check:
        return check()
    if CORPUS.exists() and not a.force:
        print(f"already present: {CORPUS}")
        return check()
    return fetch()


if __name__ == "__main__":
    raise SystemExit(main())
