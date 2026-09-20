#!/usr/bin/env python3
"""Freeze the MMLU items the accuracy half asks.

72 items per target, split evenly across the target's subjects so a multi-tier
claim is checked at every tier, and stratified on the gold answer letter within
each (target, subject) cell.

The letter stratification is not tidiness. Corpus-wide the gold answer is
A 22.9 / B 24.7 / C 25.5 / D 26.9, and twelve of the 57 subjects put over 35%
on one letter — `high_school_statistics` is 47% D. A model with any position
bias would score on the bias rather than the knowledge.

That same stratification is what caps n at 72: a target can supply at most
4 x (its scarcest gold letter), and three targets hold only 18 on theirs.

    python scripts/build_mmlu_measurement_set.py
    python scripts/build_mmlu_measurement_set.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "personascope" / "data"
CORPUS = DATA / "mmlu" / "mmlu_test.jsonl"
TARGETS = DATA / "mmlu-self-report" / "measurement_targets.json"
OUT_DIR = DATA / "mmlu"
LETTERS = "ABCD"


def read_jsonl(path: Path) -> list[dict]:
    # Not splitlines(): it also breaks on U+0085, which official MMLU contains.
    with path.open(encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def digest(rows: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build(n_per_target: int, seed: int) -> list[dict]:
    import numpy as np

    if not CORPUS.exists():
        raise FileNotFoundError(f"No corpus at {CORPUS}. Run scripts/fetch_mmlu.py")
    rows = read_jsonl(CORPUS)
    spec = json.loads(TARGETS.read_text(encoding="utf-8"))

    # (subject, gold letter) -> corpus positions
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r["subject"], LETTERS[int(r["answer"])])].append(i)

    rng = np.random.default_rng(seed)
    out: list[dict] = []

    for entry in spec["targets"]:
        target, covers = entry["target"], entry["covers"]
        per_subject, rem = divmod(n_per_target, len(covers))
        if rem:
            raise ValueError(
                f"{target}: {n_per_target} does not divide by {len(covers)} subjects"
            )
        per_letter, rem = divmod(per_subject, len(LETTERS))
        if rem:
            raise ValueError(
                f"{target}: {per_subject} per subject does not divide by 4 letters"
            )

        rank = 0
        for subject in sorted(covers):
            for letter in LETTERS:
                pool = buckets[(subject, letter)]
                if len(pool) < per_letter:
                    raise ValueError(
                        f"{subject} has {len(pool)} items on {letter}, "
                        f"need {per_letter}. Lower --n."
                    )
                picked = rng.choice(len(pool), size=per_letter, replace=False)
                for j in sorted(int(x) for x in picked):
                    src = rows[pool[j]]
                    out.append({
                        "uid": f"mmlu:{target.replace(' ', '_')}:{rank}",
                        "target": target,
                        "subject": subject,
                        "source_index": src["source_index"],
                        "question": src["question"],
                        "choices": src["choices"],
                        "answer": int(src["answer"]),
                    })
                    rank += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=72, help="items per target")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    items_path = OUT_DIR / f"measurement_n{a.n}_seed{a.seed}.jsonl"
    manifest_path = OUT_DIR / f"measurement_n{a.n}_seed{a.seed}.json"

    if a.verify:
        rows = read_jsonl(items_path)
        man = json.loads(manifest_path.read_text())
        ok = digest(rows) == man["sha256_16"] and len(rows) == man["n_items"]
        print(f"{len(rows)} items, {len({r['target'] for r in rows})} targets — "
              f"{'ok' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    rows = build(a.n, a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with items_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    manifest_path.write_text(json.dumps({
        "source": "cais/mmlu test split, via scripts/fetch_mmlu.py",
        "targets_file": TARGETS.name,
        "n_per_target": a.n,
        "seed": a.seed,
        "n_targets": len({r["target"] for r in rows}),
        "n_items": len(rows),
        "letter_counts": {L: sum(1 for r in rows if LETTERS[r["answer"]] == L) for L in LETTERS},
        "items_per_subject": dict(Counter(r["subject"] for r in rows)),
        "sha256_16": digest(rows),
        "file": items_path.name,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} items across {len({r['target'] for r in rows})} targets "
          f"-> {items_path.name}")
    print(f"sha {digest(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
