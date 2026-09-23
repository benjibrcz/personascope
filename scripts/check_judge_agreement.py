#!/usr/bin/env python3
"""Do two judges agree on the stance grid?

Re-judges a stratified sample of already-scored responses with a second judge
and reports Cohen's kappa, the per-label rates and every disagreement.

Stratified on (route, stance) rather than drawn at random. A random 400 of
24,050 would be almost all `assistant` and `human_role` and would say nothing
about `nonhuman_role`, which is 0.3% of the corpus and carries the Vader and
Voldemort result. The cost is that the marginals are not the corpus marginals,
so the kappa here is a floor: the rare labels it over-samples are the hard
ones, and agreement on a natural sample would be higher.

Judge A's verdict is read from parsed.jsonl and never recomputed -- these are
the numbers actually reported, not a fresh draw from the same model, which at
temperature 0 on a reasoning model would differ from itself.

    python scripts/check_judge_agreement.py --run results/sad/sad_v1/gpt-4.1
    python scripts/check_judge_agreement.py --n 400 --judge-b gpt-5
    python scripts/check_judge_agreement.py --report        # re-read, no calls
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "judge_agreement"


def cohens_kappa(pairs: list[tuple[str, str]], labels: list[str]) -> float:
    """Chance-corrected agreement. Plain accuracy would flatter a grid whose
    labels are this unevenly used."""
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca.get(x, 0) / n) * (cb.get(x, 0) / n) for x in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load(run: Path) -> list[dict]:
    """Verdicts from parsed.jsonl, questions from responses.jsonl.

    parsed.jsonl does not carry `prompt` -- it is derived, and the question
    lives with the response. Judge B must be shown the same question judge A
    saw; asked to label an answer with no question it produces a verdict that
    is not comparable to anything.
    """
    rows = []
    for p in sorted(run.rglob("parsed.jsonl")):
        asked = {}
        resp = p.parent / "responses.jsonl"
        if resp.exists():
            for line in resp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    q = json.loads(line)
                    asked[(q.get("item_id"), q.get("sample"))] = q.get("prompt", "")
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("value")
            if r.get("status") != "parsed" or not isinstance(v, dict):
                continue
            if not v.get("stance") or not v.get("answer"):
                continue
            question = asked.get((r.get("item_id"), r.get("sample")), "")
            if not question:
                continue
            rows.append({
                "cell": r.get("cell"), "route": r.get("route"),
                "persona": r.get("persona"), "item_id": r.get("item_id"),
                "sample": r.get("sample"), "set": v.get("set"),
                "stratum": v.get("stratum"), "question": question,
                "answer": v["answer"], "a": v["stance"], "a_why": v.get("stance_why", ""),
            })
    return rows


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Equal share per (route, stance) cell, capped at what each holds."""
    pools: dict[tuple, list] = defaultdict(list)
    for r in rows:
        pools[(r["route"], r["a"])].append(r)
    for k in pools:
        pools[k].sort(key=lambda r: (r["cell"], r["item_id"], r["sample"]))

    rng = random.Random(seed)
    take = {k: 0 for k in pools}
    while sum(take.values()) < min(n, sum(len(v) for v in pools.values())):
        room = [k for k in pools if take[k] < len(pools[k])]
        if not room:
            break
        for k in sorted(room, key=lambda k: (take[k], str(k))):
            if sum(take.values()) >= n:
                break
            take[k] += 1
    out = []
    for k in sorted(take, key=str):
        out += rng.sample(pools[k], take[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/sad/sad_v1/gpt-4.1")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--judge-b", default="gpt-5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--report", action="store_true", help="re-read the saved file, call nothing")
    a = ap.parse_args()

    from personascope import sad_stance as stance

    labels = list(stance.LABELS)
    run = Path(a.run)
    stem = f"{run.name}_{a.judge_b}_n{a.n}_seed{a.seed}"
    path = OUT_DIR / f"{stem}.jsonl"

    if a.report:
        if not path.exists():
            raise SystemExit(f"no saved run at {path}")
        judged = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    else:
        rows = load(run)
        if not rows:
            raise SystemExit(f"no scored rows under {run}\n  run: personascope score {run}")
        pick = sample(rows, a.n, a.seed)
        print(f"{len(rows):,} scored rows -> {len(pick)} sampled, "
              f"stratified on (route, stance)\n")

        from personascope.judges import judge_fn, resolved_id
        judge = judge_fn(a.judge_b)

        def one(r: dict) -> dict:
            try:
                raw = judge(stance.render(question=r["question"], response=r["answer"]))
            except RuntimeError as exc:
                return {**r, "b": stance.UNREADABLE, "b_why": str(exc)[:120]}
            b, why = stance.parse(raw)
            return {**r, "b": b, "b_why": why}

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        judged = []
        with path.open("w", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                for i, rec in enumerate(pool.map(one, pick), 1):
                    judged.append(rec)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if i % 50 == 0 or i == len(pick):
                        print(f"  {i}/{len(pick)}")
        (OUT_DIR / f"{stem}.json").write_text(json.dumps({
            "run": str(run), "judge_a": "from parsed.jsonl (gpt-5-mini)",
            "judge_b": a.judge_b, "judge_b_model_id": resolved_id(a.judge_b),
            "grid_sha": stance.grid_sha(), "n": len(judged), "seed": a.seed,
            "stratified_on": ["route", "stance"], "date": dt.date.today().isoformat(),
        }, indent=2) + "\n", encoding="utf-8")

    usable = [r for r in judged if r["b"] != stance.UNREADABLE]
    pairs = [(r["a"], r["b"]) for r in usable]
    agree = sum(1 for x, y in pairs if x == y)

    print(f"\n  n={len(pairs)} ({len(judged) - len(usable)} unreadable from judge B)")
    print(f"  raw agreement  {agree / len(pairs):.3f}")
    print(f"  Cohen's kappa  {cohens_kappa(pairs, labels):.3f}")

    print(f"\n  per label (judge A's view)\n  {'label':<24}{'n':>5}{'agreed':>8}")
    for lab in labels:
        sub = [p for p in pairs if p[0] == lab]
        if sub:
            print(f"  {lab:<24}{len(sub):>5}{sum(1 for x, y in sub if x == y) / len(sub):>8.2f}")

    print("\n  confusion (rows = A, cols = B)")
    short = {lab: lab[:6] for lab in labels}
    print("  " + " " * 24 + "".join(f"{short[lab]:>8}" for lab in labels))
    for x in labels:
        row = "".join(f"{sum(1 for p in pairs if p == (x, y)):>8}" for y in labels)
        print(f"  {x:<24}{row}")

    dis = [r for r in usable if r["a"] != r["b"]]
    print(f"\n  {len(dis)} disagreements; first 5:")
    for r in dis[:5]:
        print(f"\n    [{r['cell']}] {r['set']}/{r['stratum']}   A={r['a']}  B={r['b']}")
        print(f"      Q: {r['question'][:90]}")
        print(f"      A: {r['answer'][:150]}")
        print(f"      A says: {r['a_why'][:100]}")
        print(f"      B says: {r['b_why'][:100]}")
    if not a.report:
        print(f"\n  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
