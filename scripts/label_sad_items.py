#!/usr/bin/env python3
"""Label SAD items on a content axis, so a sample can be stratified on it.

Runs on `human_defaults` and `llms` only — the two sets that ship no
stratification variable of their own. `which_llm` (`subset`), `influence`
(`splits.cat`) and `names` (50 rephrasings of one question) already have one,
and re-deriving it would only add a way to disagree with the source.

The axes, the `discriminates` question and the prompt live in
`data/sad/label_axes.yaml`; that file's sha is recorded on every row, so a
prompt edit is visible in the data rather than only in git.

Writes `data/external/sad/labels.jsonl`, one row per item, carrying the uid
and the labels but **not the question text** — the questions stay behind the
BigBench canary in the gitignored `raw/`, and this file is committed. The uid
is the join key.

Resumable: rows are appended and flushed as they arrive, and a re-run skips
uids already present for the same labeller and prompt sha. An interrupted run
keeps everything it had.

    python scripts/label_sad_items.py --limit 20     # a taste, then inspect
    python scripts/label_sad_items.py                # the rest
    python scripts/label_sad_items.py --report       # counts, no network
    python scripts/label_sad_items.py --sample 60    # a stratified hand-check set
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "src" / "personascope" / "data"
RAW = DATA / "external" / "sad" / "raw"
OUT = DATA / "external" / "sad" / "labels.jsonl"
MANIFEST = DATA / "external" / "sad" / "labels.json"
AXES = DATA / "sad" / "label_axes.yaml"

# The sets with no stratification variable of their own.
NEEDS_LABELS = ("human_defaults", "llms")
HAS_STRATA = {
    "which_llm": "subset (technical-details / deployment / ownership)",
    "influence": "splits.cat (8 categories)",
    "names": "50 rephrasings of one question; no strata needed",
}

UNPARSED = "unparsed"


def spec() -> dict:
    return yaml.safe_load(AXES.read_text(encoding="utf-8"))


def prompt_sha() -> str:
    return hashlib.sha256(AXES.read_bytes()).hexdigest()[:12]


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def render(question: str) -> str:
    s = spec()
    axis_block = "\n\n".join(
        f"  {a['name']}\n"
        f"      {_flat(a['definition'])}\n"
        f"      Decides it: {_flat(a['decides_it'])}"
        + "".join(f"\n      Example: {c}" for c in a.get("cues", []))
        for a in s["axes"]
    )
    tiebreaker_block = "\n".join(
        f"  {i}. {_flat(t)}" for i, t in enumerate(s["tiebreakers"], 1))
    d = s["discriminates"]
    guidance_block = "\n".join(f"  - {_flat(g)}" for g in d["guidance"])
    return s["prompt"].format(
        question=question,
        axis_block=axis_block,
        tiebreaker_block=tiebreaker_block,
        discriminates_test=_flat(d["test"]),
        yes_means=_flat(d["yes_means"]),
        no_means=_flat(d["no_means"]),
        guidance_block=guidance_block,
    )


def parse(raw: str) -> dict:
    """`{axis, discriminates, notes}`; axis `unparsed` when the reply cannot be
    read. A judge failure is never silently folded into a real label — an
    outage would otherwise enter the data as a content judgement."""
    axes = {a["name"] for a in spec()["axes"]}
    obj = None
    if raw and raw.strip():
        i, j = raw.find("{"), raw.rfind("}")
        if i != -1 and j > i:
            try:
                obj = json.loads(raw[i:j + 1])
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return {"axis": UNPARSED, "discriminates": UNPARSED, "notes": (raw or "")[:200]}
    axis = str(obj.get("axis", "")).strip().lower()
    disc = str(obj.get("discriminates", "")).strip().lower()
    return {
        "axis": axis if axis in axes else UNPARSED,
        "discriminates": disc if disc in ("yes", "no") else UNPARSED,
        "notes": str(obj.get("notes", "")).strip()[:300],
    }


def load_items(sets: list[str]) -> list[dict]:
    out = []
    for name in sets:
        p = RAW / f"{name}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}\n  run: python scripts/fetch_sad.py")
        out += [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out


def load_done(labeller: str, psha: str) -> set[str]:
    """uids already labelled by this labeller under this prompt."""
    if not OUT.exists():
        return set()
    done = set()
    for ln in OUT.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("labeller") == labeller and r.get("prompt_sha") == psha:
            done.add(r["uid"])
    return done


def rows() -> list[dict]:
    if not OUT.exists():
        return []
    return [json.loads(ln) for ln in OUT.read_text(encoding="utf-8").splitlines() if ln.strip()]


def report() -> int:
    rs = rows()
    if not rs:
        print(f"no labels yet: {OUT}")
        return 1
    print(f"{len(rs)} rows in {OUT.name}")
    for name in sorted({r["set"] for r in rs}):
        sub = [r for r in rs if r["set"] == name]
        ax = Counter(r["axis"] for r in sub)
        disc = Counter(r["discriminates"] for r in sub)
        print(f"\n  {name}  ({len(sub)})")
        for k, v in ax.most_common():
            print(f"    axis {k:16s} {v:5d}  {v / len(sub):5.1%}")
        for k, v in disc.most_common():
            print(f"    discriminates {k:7s} {v:5d}  {v / len(sub):5.1%}")
    bad = [r for r in rs if r["axis"] == UNPARSED or r["discriminates"] == UNPARSED]
    print(f"\n  unparsed: {len(bad)}")
    return 0


def sample(n: int, seed: int) -> int:
    """A stratified hand-check set, printed with the question text so a human
    can disagree with the labeller. Reads `raw/`, so it only works locally."""
    rs = {r["uid"]: r for r in rows()}
    if not rs:
        print("no labels yet")
        return 1
    text = {it["uid"]: it for it in load_items(list(NEEDS_LABELS))}
    by_axis: dict[str, list[dict]] = {}
    for r in rs.values():
        by_axis.setdefault(r["axis"], []).append(r)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_axis)))
    for axis in sorted(by_axis):
        pick = rng.sample(by_axis[axis], min(per, len(by_axis[axis])))
        print(f"\n===== {axis}  ({len(by_axis[axis])} items, showing {len(pick)})")
        for r in pick:
            it = text.get(r["uid"], {})
            print(f"  [{r['set']}] disc={r['discriminates']:3s}  {it.get('question', '?')}")
            print(f"      {r['notes']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(NEEDS_LABELS))
    ap.add_argument("--judge", default="gpt-5", help="key in personascope.judges.JUDGES")
    ap.add_argument("--limit", type=int, default=0, help="label at most N new items")
    ap.add_argument("--report", action="store_true", help="counts only, no network")
    ap.add_argument("--sample", type=int, default=0, help="print N labelled items for a hand-check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retry-unparsed", action="store_true",
                    help="drop rows this labeller could not parse, then re-label them")
    ap.add_argument("--relabel", action="store_true",
                    help="the axes file changed: drop this labeller's stale rows and re-label")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent judge calls; rows are written by one thread")
    a = ap.parse_args()

    if a.report:
        return report()
    if a.sample:
        return sample(a.sample, a.seed)

    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    for s in sets:
        if s in HAS_STRATA:
            print(f"refusing {s}: it already ships strata — {HAS_STRATA[s]}", file=sys.stderr)
            return 2
        if s not in NEEDS_LABELS:
            print(f"unknown set: {s}", file=sys.stderr)
            return 2

    from personascope.judges import judge_fn, resolved_id

    psha = prompt_sha()

    # An edit to label_axes.yaml changes what a label means, so rows written
    # under an older prompt are not merely out of date, they are answers to a
    # different question. Refuse rather than let two generations sit in one
    # file where last-wins would silently pick between them.
    if OUT.exists():
        stale = {r["prompt_sha"] for r in rows()
                 if r.get("labeller") == a.judge and r.get("prompt_sha") != psha}
        if stale and not a.relabel:
            print(f"{AXES.name} has changed (now {psha}); "
                  f"{sum(1 for r in rows() if r.get('prompt_sha') in stale)} rows were "
                  f"labelled under {', '.join(sorted(stale))}.\n"
                  f"  re-run with --relabel to drop them and label again",
                  file=sys.stderr)
            return 2
        if stale:
            keep = [r for r in rows()
                    if not (r.get("labeller") == a.judge and r.get("prompt_sha") in stale)]
            print(f"dropped {len(rows()) - len(keep)} rows from {', '.join(sorted(stale))}")
            OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                           encoding="utf-8")

    if a.retry_unparsed and OUT.exists():
        keep = [r for r in rows()
                if not (r.get("labeller") == a.judge and r.get("prompt_sha") == psha
                        and (r["axis"] == UNPARSED or r["discriminates"] == UNPARSED))]
        gone = len(rows()) - len(keep)
        OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                       encoding="utf-8")
        print(f"dropped {gone} unparsed rows; they will be re-labelled")

    items = load_items(sets)
    done = load_done(a.judge, psha)
    todo = [it for it in items if it["uid"] not in done]
    if a.limit:
        todo = todo[: a.limit]

    print(f"{len(items)} items, {len(done)} already labelled by {a.judge}@{psha}, "
          f"{len(todo)} to do")
    if not todo:
        return report()

    judge = judge_fn(a.judge)
    today = dt.date.today().isoformat()
    counts: Counter = Counter()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def label_one(it: dict) -> dict:
        try:
            raw = judge(render(it["question"]))
        except RuntimeError as exc:
            print(f"\n  {it['uid']}: {exc}", file=sys.stderr)
            raw = ""
        return {"uid": it["uid"], "set": it["set"], **parse(raw),
                "labeller": a.judge, "model_id": resolved_id(a.judge),
                "prompt_sha": psha, "date": today}

    # Calls run concurrently, rows are written by this thread only: an append
    # from several threads interleaves partial lines, and a half-written line
    # is a row resume will neither skip nor recover.
    with OUT.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
            for i, rec in enumerate(pool.map(label_one, todo), 1):
                counts[rec["axis"]] += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  " + "  ".join(
                        f"{k}={v}" for k, v in counts.most_common()), flush=True)

    rs = rows()
    MANIFEST.write_text(json.dumps({
        "source": "SAD (Laine et al. 2024, arXiv 2407.04694), CC-BY 4.0",
        "labelled_sets": sorted({r["set"] for r in rs}),
        "why": ("human_defaults and llms ship no stratification variable; these "
                "labels are what a sample is stratified on. which_llm, influence "
                "and names are left to the strata SAD publishes."),
        "axes_file": AXES.name,
        "prompt_sha": psha,
        "labellers": sorted({r["labeller"] for r in rs}),
        "model_ids": sorted({r["model_id"] for r in rs}),
        "n_rows": len(rs),
        "axis_counts": dict(Counter(r["axis"] for r in rs).most_common()),
        "discriminates_counts": dict(Counter(r["discriminates"] for r in rs).most_common()),
        "sha256_16": hashlib.sha256(
            "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False)
                      for r in rs).encode()).hexdigest()[:16],
        "file": OUT.name,
        "date": today,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {MANIFEST.name}")
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
