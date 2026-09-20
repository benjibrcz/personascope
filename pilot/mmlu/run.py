#!/usr/bin/env python3
"""Run the MMLU pilot: system-prompt cells x the frozen test set.

    python pilot/mmlu/run.py --dry-run
    python pilot/mmlu/run.py --model openai/gpt-4.1 --cells _base,curie:default
    python pilot/mmlu/run.py --model openai/gpt-4.1          # all 13 cells

Writes one JSONL per cell under `pilot/mmlu-results/`, appending as it goes and
resuming on `(uid, sample)`, so an interrupted run costs nothing and a killed
run can be restarted without re-spending.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmlu.cells import Cell, load_cells  # noqa: E402
from mmlu.client import Client, ProviderError  # noqa: E402
from mmlu.dataset import Item, load_manifest, load_testset, read_jsonl  # noqa: E402
from mmlu.judge import Judge  # noqa: E402
from mmlu.prompts import render_question  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "mmlu-results"


def _outfile(model: str, cell: Cell) -> Path:
    return RESULTS / f"{model.replace('/', '-')}__{cell.persona}-{cell.variant}.jsonl"


def _done(path: Path) -> set[tuple[str, int]]:
    """Already-collected `(uid, sample)` pairs, for resume."""
    if not path.exists():
        return set()
    out = set()
    for rec in read_jsonl(path):
        if "uid" in rec and "sample" in rec:
            out.add((rec["uid"], rec["sample"]))
    return out


def run_cell(
    cell: Cell, items: list[Item], target: Client, judge: Judge,
    *, n_samples: int, stamp: str, model: str,
) -> dict:
    path = _outfile(model, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path)

    counts = {"asked": 0, "skipped": len(done), "errors": 0}
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            for sample in range(n_samples):
                if (item.uid, sample) in done:
                    continue

                messages = []
                if cell.system_prompt:
                    messages.append({"role": "system", "content": cell.system_prompt})
                messages.append({"role": "user", "content": render_question(item)})

                try:
                    answer = target.complete(messages)
                except ProviderError as exc:
                    # Never written as an empty answer: an API failure scored as
                    # model behaviour would read as a refusal in the results.
                    counts["errors"] += 1
                    print(f"  ! {item.uid} s{sample}: {exc}", file=sys.stderr)
                    continue

                verdict, judge_raw = judge.extract(item, answer)
                fh.write(json.dumps({
                    "run": stamp,
                    "model": model,
                    "judge_model": judge.client.model,
                    "persona": cell.persona,
                    "variant": cell.variant,
                    "system_prompt": cell.system_prompt,
                    "uid": item.uid,
                    "subject": item.subject,
                    "sample": sample,
                    "answer": answer,
                    "verdict": verdict,
                    "judge_raw": judge_raw,
                    "gold": item.gold,
                    # None when refused, unclear, or the judge errored — those
                    # leave the accuracy denominator rather than scoring wrong.
                    "correct": (verdict == item.gold) if verdict in "ABCD" else None,
                    "temperature": target.temperature,
                    "seed": target.seed,
                }, ensure_ascii=False) + "\n")
                fh.flush()
                counts["asked"] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-4.1", help="target model")
    ap.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "openai"])
    ap.add_argument("--n", type=int, default=1, dest="n_samples", help="samples per item")
    ap.add_argument("--set-n", type=int, default=5, help="items per subject in the frozen set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cells", default="", help="comma-separated keys; default all")
    ap.add_argument("--subjects", default="", help="comma-separated subjects; default all")
    ap.add_argument("--limit", type=int, default=0, help="cap items, for smoke tests")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    items = load_testset(n=a.set_n, seed=a.seed)
    if a.subjects:
        wanted = {s.strip() for s in a.subjects.split(",") if s.strip()}
        items = [i for i in items if i.subject in wanted]
    if a.limit:
        items = items[: a.limit]

    cells = load_cells()
    if a.cells:
        wanted = {c.strip() for c in a.cells.split(",") if c.strip()}
        cells = [c for c in cells if c.key in wanted or c.persona in wanted]

    total = len(cells) * len(items) * a.n_samples
    manifest = load_manifest(a.set_n, a.seed)
    print(f"target {a.model} via {a.provider} | judge {a.judge_model}")
    print(f"test set: {len(items)} items, sha {manifest.get('sha256_16', '?')}")
    print(f"cells ({len(cells)}): {', '.join(c.key for c in cells)}")
    print(f"{total} target calls + {total} judge calls, n={a.n_samples}")

    if a.dry_run:
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = Client(model=a.model, provider=a.provider,
                    temperature=a.temperature, seed=a.seed)
    judge = Judge(client=Client(model=a.judge_model, provider=a.provider,
                                temperature=0.0, max_tokens=16, seed=a.seed))

    t0 = time.time()
    for cell in cells:
        print(f"\n[{cell.key}]", flush=True)
        counts = run_cell(cell, items, target, judge,
                          n_samples=a.n_samples, stamp=stamp, model=a.model)
        print(f"  asked {counts['asked']}, resumed {counts['skipped']}, "
              f"errors {counts['errors']} -> {_outfile(a.model, cell).name}")
    print(f"\ndone in {time.time() - t0:.0f}s -> {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
