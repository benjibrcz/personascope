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

A subset is a PREFIX of the full draw, never a fresh one at a smaller n.
Re-drawing at `--n 48` shares only ~43% of its questions with the n=72 set
while reusing 100% of its `uid`s, because a uid encodes a position
(`mmlu:{target}:{rank}`) and not an identity. Any file that mixed the two
would silently answer different questions under the same names, and resume
keys on `(item_id, sample)` and would never notice. `--subset` slices instead,
carrying the original uids, so the two sets stay comparable item for item.

    python scripts/build_mmlu_measurement_set.py
    python scripts/build_mmlu_measurement_set.py --verify
    python scripts/build_mmlu_measurement_set.py --subset n48_v2
    python scripts/build_mmlu_measurement_set.py --subset n48_v2 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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


def nest(rows: list[dict], n_per_target: int, keep_targets: list[str]) -> list[dict]:
    """Take the first `n_per_target` of each target, by (subject, letter) cell.

    `build()` emits each target grouped by sorted subject, then A/B/C/D, then
    ascending corpus position. Taking a prefix of every cell therefore keeps
    the letter balance exact and the subject split even, while guaranteeing the
    result is a strict subset of the larger draw. Uids are carried verbatim —
    they stay non-contiguous, which is the visible sign that this is a slice.
    """
    spec = json.loads(TARGETS.read_text(encoding="utf-8"))
    covers = {e["target"]: e["covers"] for e in spec["targets"]}
    keep = set(keep_targets)
    missing = keep - set(covers)
    if missing:
        raise ValueError(f"not in the sampling frame: {sorted(missing)}")

    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["target"], r["subject"], LETTERS[int(r["answer"])])].append(r)

    out: list[dict] = []
    for target in keep_targets:
        per_letter, rem = divmod(n_per_target, 4 * len(covers[target]))
        if rem:
            raise ValueError(
                f"{target}: {n_per_target} does not split evenly over "
                f"{len(covers[target])} subjects x 4 letters"
            )
        for subject in sorted(covers[target]):
            for letter in LETTERS:
                cell = cells[(target, subject, letter)]
                if len(cell) < per_letter:
                    raise ValueError(
                        f"{target}/{subject}/{letter}: the base draw holds "
                        f"{len(cell)}, need {per_letter}"
                    )
                out.extend(cell[:per_letter])
    return out


def check_uid_collisions(new_rows: list[dict], out_dir: Path) -> None:
    """Refuse to write a uid that already names a different question.

    The only structural defence against the positional-uid trap. `done_keys`
    cannot catch it: it sees a matching `(item_id, sample)` and calls the work
    done, whatever the question underneath.
    """
    def identity(r: dict) -> str:
        blob = json.dumps([r["question"], r["choices"], r["answer"]],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    seen = {r["uid"]: identity(r) for r in new_rows}
    for other in sorted(out_dir.glob("measurement_*.jsonl")):
        for r in read_jsonl(other):
            mine = seen.get(r["uid"])
            if mine and mine != identity(r):
                raise ValueError(
                    f"uid {r['uid']} already names a different question in "
                    f"{other.name}. A uid encodes a position, not an identity, "
                    f"so two independent draws collide. Use --subset, which "
                    f"slices the larger draw instead of re-drawing."
                )


def verify_nesting(rows: list[dict], man: dict) -> None:
    """Every subset row is field-identical to its twin in the base draw, and
    every cell is a true prefix rather than an arbitrary pick."""
    d = man.get("derived_from")
    if not d:
        return
    base = build(d["n_per_target"], man["seed"])
    if digest(base) != d["sha256_16"]:
        raise ValueError(
            f"the base draw no longer reproduces: {digest(base)} != {d['sha256_16']}"
        )
    by_uid = {r["uid"]: r for r in base}
    for r in rows:
        twin = by_uid.get(r["uid"])
        if twin is None:
            raise ValueError(f"{r['uid']} is not in the base draw")
        if twin != r:
            raise ValueError(f"{r['uid']} differs from its twin in the base draw")

    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in base:
        cells[(r["target"], r["subject"], LETTERS[int(r["answer"])])].append(r)
    mine: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        mine[(r["target"], r["subject"], LETTERS[int(r["answer"])])].append(r)
    for key, got in mine.items():
        if cells[key][:len(got)] != got:
            raise ValueError(f"{key} is not a prefix of the base draw")


def verify_structure(rows: list[dict], man: dict) -> None:
    counts = {L: sum(1 for r in rows if LETTERS[int(r["answer"])] == L) for L in LETTERS}
    if len(set(counts.values())) != 1:
        raise ValueError(f"letters are not balanced: {counts}")
    uids = [r["uid"] for r in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("duplicate uids")
    per_target = Counter(r["target"] for r in rows)
    if len(set(per_target.values())) != 1:
        raise ValueError(f"targets are uneven: {dict(per_target)}")
    if man.get("keep_targets") and sorted(per_target) != sorted(man["keep_targets"]):
        raise ValueError("targets do not match the manifest")


def main() -> int:
    spec = json.loads(TARGETS.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser()
    # Defaults come from the spec file, which used to carry these keys while
    # nothing read them.
    ap.add_argument("--n", type=int, default=spec.get("items_per_target", 72),
                    help="items per target")
    ap.add_argument("--seed", type=int, default=spec.get("seed", 42))
    ap.add_argument("--subset", default=None,
                    help="name of an entry under `subsets` in measurement_targets.json")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    sub = None
    if a.subset:
        sub = (spec.get("subsets") or {}).get(a.subset)
        if sub is None:
            raise SystemExit(
                f"no subset {a.subset!r}; have {sorted((spec.get('subsets') or {}))}"
            )
        stem = f"measurement_n{sub['n_per_target']}of{sub['derive_from']}_seed{sub['seed']}"
    else:
        stem = f"measurement_n{a.n}_seed{a.seed}"

    items_path = OUT_DIR / f"{stem}.jsonl"
    manifest_path = OUT_DIR / f"{stem}.json"

    if a.verify:
        rows = read_jsonl(items_path)
        man = json.loads(manifest_path.read_text())
        ok = digest(rows) == man["sha256_16"] and len(rows) == man["n_items"]
        try:
            verify_nesting(rows, man)
            verify_structure(rows, man)
        except ValueError as e:
            print(f"FAILED: {e}")
            return 1
        print(f"{len(rows)} items, {len({r['target'] for r in rows})} targets — "
              f"{'ok' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    if sub:
        base = build(sub["derive_from"], sub["seed"])
        rows = nest(base, sub["n_per_target"], sub["keep_targets"])
    else:
        rows = build(a.n, a.seed)
    check_uid_collisions(rows, OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with items_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    extra: dict = {}
    if sub:
        extra = {
            "subset": a.subset,
            "derived_from": {
                "file": f"measurement_n{sub['derive_from']}_seed{sub['seed']}.jsonl",
                "sha256_16": digest(build(sub["derive_from"], sub["seed"])),
                "n_per_target": sub["derive_from"],
            },
            "nesting": "prefix of each (target, subject, gold-letter) cell",
            "uid_policy": "ranks inherited from the base draw; non-contiguous by design",
            "keep_targets": sub["keep_targets"],
            "dropped_targets": sorted(
                {e["target"] for e in spec["targets"]} - set(sub["keep_targets"])
            ),
            "why": sub.get("why", ""),
        }

    manifest_path.write_text(json.dumps({
        "source": "cais/mmlu test split, via scripts/fetch_mmlu.py",
        "targets_file": TARGETS.name,
        "n_per_target": sub["n_per_target"] if sub else a.n,
        "seed": sub["seed"] if sub else a.seed,
        **extra,
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
