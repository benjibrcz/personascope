"""Score SAD responses with the Assistant Axis role judge.

A SECOND reading of the same answers, and the only test of an assumption our own
grid has rested on since v3: that a judge which is never told which persona was
induced loses nothing. Theirs is told. If the two agree, blindness is free; if
they diverge, we need to know where before either number goes in the paper.

Theirs is persona-SPECIFIC, so the uninduced `_base` cells have no role to judge
and are skipped rather than forced.

Writes `parsed_assistant_axis.jsonl` beside each cell's `parsed.jsonl` -- never
into it. The harness's own scoring pass owns that file, and these verdicts are on
a different scale (0-3, theirs) from ours (0-2 plus off-scale).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from personascope.instruments.sad import JUDGE_MAX_TOKENS  # noqa: E402
from personascope.judges import JUDGES, judge_fn  # noqa: E402

PROMPTS = json.loads(
    (REPO / "src/personascope/data/sad/assistant_axis_role_judge.json").read_text()
)["prompts"]
OUT_NAME = "parsed_assistant_axis.jsonl"
# Their scale is 0-3 and the reply is meant to be the bare integer.
LEVELS = (0, 1, 2, 3)


def parse(raw: str) -> int | None:
    """Their format is a bare integer. Nothing looser is accepted: a judge that
    answers in prose has not followed the rubric, and guessing at what it meant
    is how v2's substring fallback read a negation as its opposite."""
    if not raw:
        return None
    m = re.fullmatch(r"\s*([0-3])\s*", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|\D)([0-3])\s*$", raw.strip())
    return int(m.group(1)) if m else None


def cells(run: Path):
    for f in sorted(run.rglob("responses.jsonl")):
        yield f.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="e.g. results/sad/sad_v1")
    ap.add_argument("--judge", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = JUDGES[a.judge]
    jf = judge_fn(a.judge, max_tokens=JUDGE_MAX_TOKENS)
    total_todo = 0

    for run_s in a.runs:
        run = REPO / run_s if not Path(run_s).is_absolute() else Path(run_s)
        for cell in cells(run):
            rows = [json.loads(x) for x in (cell / "responses.jsonl").open()]
            persona = rows[0].get("persona") if rows else None
            if persona not in PROMPTS:
                print(f"  skip {cell.relative_to(run)}  (persona {persona!r}: no role prompt)")
                continue
            out = cell / OUT_NAME
            done = set()
            if out.exists():
                for x in out.open():
                    try:
                        r = json.loads(x)
                    except json.JSONDecodeError:
                        continue
                    done.add((r["item_id"], r["sample"]))
            todo = [r for r in rows
                    if (r["item_id"], r["sample"]) not in done
                    and (r.get("response") or "").strip()]
            total_todo += len(todo)
            if a.dry_run:
                print(f"  {str(cell.relative_to(run)):<44} {len(todo):>5} to judge "
                      f"({len(done)} done)")
                continue
            if not todo:
                print(f"  {str(cell.relative_to(run)):<44} complete")
                continue

            tmpl = PROMPTS[persona]

            def one(r):
                p = tmpl.replace("{question}", r.get("prompt") or "").replace(
                    "{answer}", r.get("response") or "")
                try:
                    raw = jf(p)
                except RuntimeError as exc:
                    return dict(item_id=r["item_id"], sample=r["sample"],
                                level=None, raw="", error=str(exc)[:160])
                return dict(item_id=r["item_id"], sample=r["sample"],
                            level=parse(raw), raw=raw[:200], error=None)

            with out.open("a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
                n_bad = 0
                for rec in ex.map(one, todo):
                    rec.update(judge=a.judge, judge_model=spec["model"],
                               rubric="assistant_axis_role", persona=persona)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if rec["level"] is None:
                        n_bad += 1
            print(f"  {str(cell.relative_to(run)):<44} judged {len(todo):>5}  "
                  f"unreadable {n_bad}")

    if a.dry_run:
        print(f"\n  {total_todo:,} rows to judge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
