#!/usr/bin/env python3
"""Freeze the SAD items the persona/assistant battery asks.

Equal weight per set — 80 items each from human_defaults, llms, which_llm and
influence, and all 50 names — stratified within each set on whatever variable
it has: the labelled `axis` for the two sets that ship none
(`scripts/label_sad_items.py`), and SAD's own `subset` / `splits.cat` for the
two that do.

Every rule lives in `data/external/sad/measurement_spec.json` with a `why_`
field beside it, so the argument for a number is next to the number rather
than in a commit message.

There is deliberately no `--subset`. MMLU's builder can slice a smaller set out
of its full draw because an MMLU uid encodes a rank; a SAD uid is the sha of the
question and carries no position, so a smaller n here would have to be a fresh
stratified draw sharing only part of this one. A smaller battery would get its
own seed and its own stem rather than a slice wearing this one's name.

    python scripts/build_sad_measurement_set.py --dry-run
    python scripts/build_sad_measurement_set.py
    python scripts/build_sad_measurement_set.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAD = ROOT / "src" / "personascope" / "data" / "external" / "sad"
RAW = SAD / "raw"
SPEC = SAD / "measurement_spec.json"
LABELS = SAD / "labels.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}\n  run: python scripts/fetch_sad.py")
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def digest(rows: list[dict]) -> str:
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stratum_of(item: dict, labels: dict[str, dict], field: str | None) -> str:
    if field is None:
        return "all"
    if field == "axis":
        return labels.get(item["uid"], {}).get("axis", "unparsed")
    key = field.split(".", 1)[1] if field.startswith("sad_strata.") else field
    v = (item.get("sad_strata") or {}).get(key)
    return str(v) if not isinstance(v, list) else str(sorted(v)[0])


def allocate(pools: dict[str, list], n: int) -> dict[str, int]:
    """Largest-remainder split of `n` across strata, in proportion to pool size.

    Proportional rather than equal-per-stratum: a stratum of 6 and one of 300
    are not two equally informative halves of a set.
    """
    total = sum(len(v) for v in pools.values())
    if total == 0:
        return {}
    n = min(n, total)
    exact = {k: n * len(v) / total for k, v in pools.items()}
    take = {k: min(int(v), len(pools[k])) for k, v in exact.items()}
    while sum(take.values()) < n:
        room = [k for k in pools if take[k] < len(pools[k])]
        if not room:
            break
        k = max(room, key=lambda k: (exact[k] - take[k], len(pools[k]), k))
        take[k] += 1
    return take


def build(spec: dict) -> tuple[list[dict], dict]:
    labels = {r["uid"]: r for r in read_jsonl(LABELS)} if LABELS.exists() else {}
    rng = random.Random(spec["seed"])
    dropped: Counter = Counter()
    rows: list[dict] = []
    draw_table: dict[str, dict[str, int]] = {}
    pool_sizes: dict[str, int] = {}

    for name, n in spec["n_per_set"].items():
        items = read_jsonl(RAW / f"{name}.jsonl")
        pool_sizes[name] = len(items)

        # First match wins, so an item that both fails to discriminate and is
        # unclear is counted once, under the first rule that caught it. The
        # totals are exact; the attribution is first-come.
        if name in spec["drop_rules_apply_to"]:
            kept = []
            for it in items:
                lab = labels.get(it["uid"])
                if lab is None:
                    dropped[f"{name}:unlabelled"] += 1
                elif lab["discriminates"] == spec["drop_rules"]["discriminates"]:
                    dropped[f"{name}:does_not_discriminate"] += 1
                elif lab["axis"] in spec["drop_rules"]["axis"]:
                    dropped[f"{name}:axis_{lab['axis']}"] += 1
                else:
                    kept.append(it)
            items = kept

        field = spec["strata_field"][name]
        pools: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            pools[stratum_of(it, labels, field)].append(it)
        for k in pools:
            pools[k].sort(key=lambda r: r["uid"])  # seed alone must fix the draw

        take = allocate(dict(pools), n)
        draw_table[name] = dict(sorted(take.items()))
        for stratum in sorted(take):
            for it in rng.sample(pools[stratum], take[stratum]):
                rows.append({
                    "uid": it["uid"], "set": name, "stratum": stratum,
                    "question": it["question"],
                    "sad_correct": it["sad_correct"], "sad_wrong": it["sad_wrong"],
                    "axis": labels.get(it["uid"], {}).get("axis"),
                })

    rows.sort(key=lambda r: (r["set"], r["stratum"], r["uid"]))
    stats = {
        "pool_sizes": pool_sizes,
        "draw": draw_table,
        "dropped": dict(dropped.most_common()),
        "n_dropped": sum(dropped.values()),
    }
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="re-derive and compare against the committed manifest")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the draw and the drops, write nothing")
    a = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    rows, stats = build(spec)
    n = len(rows)
    stem = spec["stem"].format(n=n, seed=spec["seed"])
    items_path, man_path = SAD / f"{stem}.jsonl", SAD / f"{stem}.json"
    sha = digest(rows)

    if a.dry_run:
        print(f"would write {n} items -> {stem}.jsonl  sha {sha}\n")
        for sname, take in stats["draw"].items():
            pool = stats["pool_sizes"][sname]
            print(f"  {sname:16s} pool {pool:4d} -> {sum(take.values()):3d}")
            for k, v in take.items():
                print(f"       {k:20s} {v:4d}")
        print("\n  drops (first match wins):")
        for k, v in stats["dropped"].items():
            print(f"    {k:42s} {v:5d}")
        return 0

    if a.verify:
        if not man_path.exists():
            print(f"no manifest at {man_path}")
            return 1
        man = json.loads(man_path.read_text(encoding="utf-8"))
        if man["sha256_16"] != sha or man["n_items"] != n:
            print(f"MISMATCH: rebuilt {n} items sha {sha}, "
                  f"manifest {man['n_items']} sha {man['sha256_16']}")
            return 1
        print(f"ok: {n} items, sha {sha}")
        return 0

    with items_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    man_path.write_text(json.dumps({
        "source": spec["source"],
        "source_commit": spec["source_commit"],
        "spec_file": SPEC.name,
        "spec_sha12": sha12(SPEC),
        "labels_file": LABELS.name,
        "labels_sha12": sha12(LABELS) if LABELS.exists() else None,
        "seed": spec["seed"],
        "n_per_set": spec["n_per_set"],
        "n_samples": spec["n_samples"],
        "strata_field": spec["strata_field"],
        "why": ("Equal weight per set, stratified within each set on the labelled "
                "axis where SAD ships no strata and on SAD's own subset/cat where "
                "it does. Items that a nineteenth-century person and a model would "
                "answer alike are dropped: they cannot show displacement."),
        **stats,
        "n_items": n, "sha256_16": sha, "file": items_path.name,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {n} items -> {items_path.name}  sha {sha}")
    for name, take in stats["draw"].items():
        print(f"  {name:16s} {sum(take.values()):4d}  " +
              " ".join(f"{k}={v}" for k, v in take.items()))
    if stats["dropped"]:
        print("  dropped: " + " ".join(f"{k}={v}" for k, v in stats["dropped"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
