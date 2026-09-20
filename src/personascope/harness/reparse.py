"""Re-read a finished run without re-asking the model.

Generation and parsing are separable because the raw text is on every record.
A parser bug found after a run costs a re-read, not 11,520 calls — and that is
not hypothetical: the labelled-answer branch was missing "the answer is (D)",
the most explicit form there is, and every Gupta-format response was being
recorded as a guess.

It is also how two extractors get compared on identical data, which is the
replication claim: Gupta's reading and ours, on the same responses, with the
disagreement measured rather than argued.

Inspect re-runs scorers over an existing log for the same reason, and lm-eval
keeps `samples_*.jsonl` separate from `results_*.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from personascope.harness.record import read_responses

__all__ = ["reparse_cell", "reparse_run"]

RESPONSES = "responses.jsonl"
SUMMARY = "summary.json"


def reparse_cell(cell_dir: Path, instrument) -> dict[str, Any]:
    """Re-parse and re-summarise one cell, in place."""
    cell_dir = Path(cell_dir)
    path = cell_dir / RESPONSES
    records = read_responses(path)
    if not records:
        return {"cell_dir": str(cell_dir), "n": 0}

    from personascope.instruments.base import Prompt

    changed = 0
    for r in records:
        # A transport failure has no response to re-read; leave it alone.
        if r.get("status") == "error":
            continue
        prompt = Prompt(r["item_id"], r.get("prompt", ""), r.get("meta") or {})
        before = (r.get("value"), r.get("status"))
        parsed = instrument.parse(prompt, r.get("response") or "")
        r["value"], r["status"], r["note"] = parsed.value, parsed.status, parsed.note
        if (r["value"], r["status"]) != before:
            changed += 1

    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = cell_dir / SUMMARY
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    # Keep the harness's own keys; replace only what the instrument produced.
    harness_keys = {
        "cell", "model", "model_id", "persona", "variant", "route", "instrument",
        "n_records", "n_samples", "seed", "temperature", "k", "asked", "resumed",
        "errors",
    }
    summary = {k: v for k, v in summary.items() if k in harness_keys}
    summary.update(instrument.summarise(records))
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    return {"cell_dir": str(cell_dir), "n": len(records), "changed": changed}


def reparse_run(run_root: Path, instrument: Optional[Any] = None) -> list[dict[str, Any]]:
    """Re-parse every cell under a run root, then rebuild the report."""
    run_root = Path(run_root)
    cells = sorted(p.parent for p in run_root.rglob(RESPONSES))
    if not cells:
        raise FileNotFoundError(f"No {RESPONSES} under {run_root}")

    if instrument is None:
        from personascope.instruments.base import load_instrument

        first = read_responses(cells[0] / RESPONSES)
        name = first[0].get("instrument") if first else None
        if not name:
            raise ValueError(
                f"{cells[0]} records name no instrument; pass one explicitly."
            )
        instrument = load_instrument(name)

    out = [reparse_cell(c, instrument) for c in cells]

    from personascope.harness.report import write_report

    write_report(run_root)
    return out
