#!/usr/bin/env python3
"""Read the pilot results: accuracy, refusal, and where the two diverge.

    python pilot/mmlu/analyze.py
    python pilot/mmlu/analyze.py --by-subject --persona curie

Accuracy is over *scored* items only. Refusals and unclear answers leave the
denominator rather than counting as wrong, so `acc` and `refuse` are read
together: a persona can hold accuracy while refusing half the set, and that is a
different finding from one that answers everything badly. Gupta's pipeline
scores extraction failure as an incorrect answer and keeps it in the
denominator, which merges those two cases — and their own result that abstention
drives 58% of disabled-persona errors is exactly what that merge hides.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .dataset import read_jsonl

RESULTS = Path(__file__).resolve().parents[1] / "mmlu-results"

__all__ = ["load_records", "summarise"]


def load_records(results_dir: Path | str = RESULTS) -> list[dict]:
    d = Path(results_dir)
    if not d.exists():
        return []
    records = []
    for path in sorted(d.glob("*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval, matching §3.3's convention for binary rates."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(records: list[dict], key: str = "cell") -> list[dict]:
    """Aggregate by cell, or by (cell, subject) when `key='subject'`."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        cell = f"{r['persona']}:{r['variant']}"
        groups[(cell, r["subject"]) if key == "subject" else (cell,)].append(r)

    rows = []
    for gk, rs in sorted(groups.items()):
        scored = [r for r in rs if r["correct"] is not None]
        correct = sum(1 for r in scored if r["correct"])
        refused = sum(1 for r in rs if r["verdict"] == "REFUSED")
        unclear = sum(1 for r in rs if r["verdict"] == "UNCLEAR")
        errors = sum(1 for r in rs if r["verdict"] == "ERROR")
        lo, hi = _wilson(correct, len(scored))
        rows.append({
            "cell": gk[0],
            "subject": gk[1] if key == "subject" else "",
            "n": len(rs),
            "scored": len(scored),
            "acc": correct / len(scored) if scored else None,
            "ci": (lo, hi),
            "refuse": refused / len(rs) if rs else 0.0,
            "unclear": unclear / len(rs) if rs else 0.0,
            "judge_errors": errors,
        })
    return rows


def _fmt(v, width=6):
    return f"{v:.3f}".rjust(width) if isinstance(v, float) else "   —  "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--by-subject", action="store_true")
    ap.add_argument("--persona", default="", help="filter to one persona")
    a = ap.parse_args()

    records = load_records(a.results)
    if not records:
        print(f"no results under {a.results}")
        return 1
    if a.persona:
        records = [r for r in records if r["persona"] == a.persona]

    rows = summarise(records, key="subject" if a.by_subject else "cell")
    base = next((r for r in summarise(records) if r["cell"].startswith("_base")), None)

    head = f"{'cell':<22}{'subject':<30}" if a.by_subject else f"{'cell':<22}"
    print(f"{head}{'n':>5}{'scored':>8}{'acc':>8}{'95% CI':>16}{'refuse':>8}{'unclear':>9}{'vs base':>9}")
    print("-" * (len(head) + 63))
    for r in rows:
        delta = ""
        if base and base["acc"] is not None and r["acc"] is not None and not r["cell"].startswith("_base"):
            delta = f"{r['acc'] - base['acc']:+.3f}"
        line = f"{r['cell']:<22}" + (f"{r['subject']:<30}" if a.by_subject else "")
        ci = f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]".rjust(16)
        print(f"{line}{r['n']:>5}{r['scored']:>8}{_fmt(r['acc'],8)}{ci}"
              f"{_fmt(r['refuse'],8)}{_fmt(r['unclear'],9)}{delta:>9}")

    bad = sum(r["judge_errors"] for r in rows)
    if bad:
        print(f"\n{bad} judge error(s) — unreadable judge replies, excluded from all rates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
