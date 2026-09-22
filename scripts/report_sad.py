#!/usr/bin/env python3
"""Read a SAD run: which entity answered, and does the route change it.

The question the battery exists to answer is not "how often does the model
speak as the persona" on its own -- a system prompt can do that trivially --
but whether the three routes differ once the persona is in. So the default
view is one row per cell with the five stances beside each other, and the
baseline on top as the floor.

    python scripts/report_sad.py                       # the route comparison
    python scripts/report_sad.py --by set              # split by SAD set
    python scripts/report_sad.py --by stratum          # split by stratum
    python scripts/report_sad.py --examples assistant  # answers behind a cell
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results" / "sad" / "sad_v1"

STANCES = ("assistant", "acknowledges", "human_role", "nonhuman_role",
           "ambiguous-nonsensical")
SHORT = {"assistant": "assist", "acknowledges": "ackn", "human_role": "human",
         "nonhuman_role": "nonhum", "ambiguous-nonsensical": "amb"}
ROUTE_ORDER = {"none": 0, "system": 1, "icl_k32": 2, "sft": 3}


def cells(run: Path) -> list[dict]:
    out = []
    for s in sorted(run.rglob("summary.json")):
        d = json.loads(s.read_text(encoding="utf-8"))
        gen = s.parent / "generation.json"
        meta = json.loads(gen.read_text(encoding="utf-8")) if gen.exists() else {}
        d["_persona"] = meta.get("persona") or s.parent.parent.name
        d["_route"] = meta.get("route") or s.parent.name
        d["_dir"] = s.parent
        out.append(d)
    return sorted(out, key=lambda c: (c["_persona"] == "_base" and -1 or 0,
                                      c["_persona"],
                                      ROUTE_ORDER.get(c["_route"], 9)))


def pct(v) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}"


def line(label: str, block: dict, extra: str = "") -> str:
    vals = "  ".join(pct(block.get(s)) for s in STANCES)
    lo, hi = (block.get("assistant_ci") or [None, None])
    ci = "" if lo is None else f"  [{lo * 100:4.1f},{hi * 100:5.1f}]"
    return f"  {label:<26} {block.get('n', 0):>5}  {vals}{ci}{extra}"


def header() -> str:
    cols = "  ".join(f"{SHORT[s]:>5}" for s in STANCES)
    return (f"  {'cell':<26} {'n':>5}  {cols}   assistant 95% CI\n"
            + "  " + "-" * 84)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--by", choices=["set", "stratum"], default=None)
    ap.add_argument("--examples", default=None,
                    help="print answers a cell gave this stance")
    ap.add_argument("--cell", default=None, help="with --examples: which cell")
    a = ap.parse_args()

    run = Path(a.run)
    cs = cells(run)
    if not cs:
        raise SystemExit(f"no scored cells under {run}\n"
                         f"  run: personascope score {run}")

    if a.examples:
        want = a.examples
        for c in cs:
            if a.cell and a.cell not in f"{c['_persona']}:{c['_route']}":
                continue
            shown = 0
            p = c["_dir"] / "parsed.jsonl"
            for ln in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(ln)
                v = r.get("value") or {}
                if v.get("stance") == want and shown < 4:
                    print(f"\n  [{c['_persona']}/{c['_route']}] {v.get('set')}/{v.get('stratum')}")
                    print(f"    A: {v['answer'][:200]}")
                    print(f"    -> {v.get('stance_why', '')[:150]}")
                    shown += 1
        return 0

    print(f"\nSAD — which entity answered   ({run.name}, judge "
          f"{cs[0].get('judge')}, grid {cs[0].get('grid_sha')})\n")
    print(header())
    for c in cs:
        label = "_base (uninduced)" if c["_persona"] == "_base" \
            else f"{c['_persona']}/{c['_route']}"
        print(line(label, c.get("overall") or {}))
        if a.by:
            key = "by_set" if a.by == "set" else "by_stratum"
            for k, blk in sorted((c.get(key) or {}).items()):
                print(line(f"    {k}", blk))
            print()

    # The reason the battery has three routes: does depth change the stance?
    print("\n  route deltas (assistant rate, percentage points vs the cell's system route)")
    by_p: dict[str, dict[str, dict]] = {}
    for c in cs:
        if c["_persona"] != "_base":
            by_p.setdefault(c["_persona"], {})[c["_route"]] = c.get("overall") or {}
    for p, routes in sorted(by_p.items()):
        base = (routes.get("system") or {}).get("assistant")
        if base is None:
            continue
        bits = []
        for r in ("icl_k32", "sft"):
            v = (routes.get(r) or {}).get("assistant")
            bits.append(f"{r} {'  -  ' if v is None else f'{(v - base) * 100:+5.1f}'}")
        print(f"    {p:<12} system {base * 100:5.1f}   " + "   ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
