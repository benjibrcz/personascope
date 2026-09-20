"""Put the cells side by side.

A per-cell `summary.json` says what one condition did. The question is what
induction *changed*, which only exists as a difference — so the report is built
around the baseline, and every persona number is shown against it.

Two views, because they answer different questions:

- **by cell** — did inducing this persona move the claims at all
- **by subject** — and if so, where. A persona that drops its confidence
  uniformly is behaving differently from one that drops it only where the
  character would not have known.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

__all__ = ["build_report", "write_report"]

BASELINE_MARK = "_base"


def _load_cells(run_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(run_root.rglob("summary.json"))
    ]


def _confidence(cell: dict) -> Optional[float]:
    return (cell.get("confidence") or {}).get("mean")


def _per_target(cell: dict) -> dict[str, float]:
    return (cell.get("confidence") or {}).get("per_target") or {}


def build_report(run_root: Path) -> dict[str, Any]:
    """Cross-cell comparison, baseline first."""
    cells = _load_cells(Path(run_root))
    base = next((c for c in cells if c.get("persona") == BASELINE_MARK), None)
    base_conf = _confidence(base) if base else None
    base_targets = _per_target(base) if base else {}

    rows = []
    for c in cells:
        conf = _confidence(c)
        cap = (c.get("capability") or {}).get("yes_rate")
        lim = (c.get("limit") or {}).get("yes_rate")
        rows.append({
            "cell": c.get("cell"),
            "persona": c.get("persona"),
            "route": c.get("route"),
            "n_records": c.get("n_records"),
            "unparsed_rate": _mean_unparsed(c),
            "confidence_mean": conf,
            # The measurement is the difference. An absolute confidence number
            # is uninterpretable on its own: it mostly tracks question
            # difficulty, which is shared across cells.
            "confidence_delta": (
                None if conf is None or base_conf is None else round(conf - base_conf, 2)
            ),
            "capability_yes_rate": cap,
            "limit_yes_rate": lim,
            "acquiescence_rate": (c.get("acquiescence") or {}).get("rate"),
            "errors": c.get("errors", 0),
        })

    # per subject, every cell against the baseline
    subjects: dict[str, dict[str, Any]] = {}
    for c in cells:
        for target, value in _per_target(c).items():
            row = subjects.setdefault(target, {"target": target, "baseline": base_targets.get(target)})
            key = c.get("persona") if c.get("persona") != BASELINE_MARK else "baseline"
            if key != "baseline":
                key = f"{c.get('persona')}:{c.get('route')}"
            row[key] = value
            if row["baseline"] is not None and key != "baseline":
                row[f"{key}_delta"] = round(value - row["baseline"], 1)

    return {
        "run_root": str(run_root),
        "n_cells": len(cells),
        "has_baseline": base is not None,
        "by_cell": rows,
        "by_subject": [subjects[k] for k in sorted(subjects)],
    }


def _mean_unparsed(cell: dict) -> Optional[float]:
    rates = [
        (cell.get(f) or {}).get("unparsed_rate")
        for f in ("confidence", "capability", "limit")
    ]
    vals = [r for r in rates if r is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: Any, width: int = 8, places: int = 3) -> str:
    if v is None:
        return "—".rjust(width)
    if isinstance(v, float):
        return f"{v:.{places}f}".rjust(width)
    return str(v).rjust(width)


def write_report(run_root: Path, report: Optional[dict] = None) -> Path:
    """Write `report.md` beside the cells, and return its path."""
    run_root = Path(run_root)
    report = report or build_report(run_root)

    lines = [
        f"# {run_root.name}",
        "",
        f"{report['n_cells']} cells."
        + ("" if report["has_baseline"] else "  **No baseline — deltas unavailable.**"),
        "",
        "## By cell",
        "",
        "Confidence is shown as a delta from the uninduced baseline. The",
        "absolute number mostly tracks question difficulty, which every cell",
        "shares, so only the difference is informative.",
        "",
        "| cell | n | unparsed | conf | Δ base | cap yes | limit yes | acq | err |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["by_cell"]:
        lines.append(
            f"| `{r['cell']}` | {r['n_records']} | {_fmt(r['unparsed_rate'],0)} "
            f"| {_fmt(r['confidence_mean'],0,1)} | {_fmt(r['confidence_delta'],0,1)} "
            f"| {_fmt(r['capability_yes_rate'],0)} | {_fmt(r['limit_yes_rate'],0)} "
            f"| {_fmt(r['acquiescence_rate'],0)} | {r['errors']} |"
        )

    if report["by_subject"]:
        keys = [
            k for k in report["by_subject"][0]
            if k not in ("target", "baseline") and not k.endswith("_delta")
        ]
        lines += [
            "",
            "## By subject",
            "",
            "Where the claims moved. A persona that drops confidence uniformly",
            "is doing something different from one that drops it only where the",
            "character could not have known.",
            "",
            "| subject | base | " + " | ".join(keys) + " |",
            "|---" * (len(keys) + 2) + "|",
        ]
        for row in report["by_subject"]:
            cells_txt = []
            for k in keys:
                v, d = row.get(k), row.get(f"{k}_delta")
                cells_txt.append("—" if v is None else f"{v:.0f}" + (f" ({d:+.0f})" if d is not None else ""))
            base_txt = "—" if row["baseline"] is None else f"{row['baseline']:.0f}"
            lines.append(f"| {row['target']} | {base_txt} | " + " | ".join(cells_txt) + " |")

    path = run_root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_root / "results.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path
