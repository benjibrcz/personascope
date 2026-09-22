"""Score the responses a run collected. The second pass, and the only reader.

For some instruments this really is parsing -- MMLU pulls a letter out of a
sentence, locally and for nothing. For others it is judging: SAD and the
identity battery each send every stored answer to an LLM judge, which costs
money and takes hours. The two look identical from here, which is why
`score_run(dry_run=True)` exists: it reports how many rows a run would judge
before any of them are.

Generation writes `responses.jsonl` and nothing derived from it. This module
turns that into `parsed.jsonl` (one value per `(item_id, sample)`) and
`summary.json` (the instrument's aggregate). Both are derived: delete them and
run this again and they come back identical.

That is the point of the split. A parser bug found after a run — and there has
been one — costs a re-read of a file on disk, not a re-spend on the API. It
also means there is exactly one call site for `instrument.parse`, so the
numbers in a summary cannot disagree with the numbers in a record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from personascope.harness.record import read_responses
from personascope.instruments.base import ERROR, Prompt

__all__ = ["parse_cell", "parse_run", "score_cell", "score_run", "PARSED_FILE"]

PARSED_FILE = "parsed.jsonl"
SUMMARY = "summary.json"
GENERATION = "generation.json"
RESPONSES = "responses.jsonl"

# Carried onto every parsed row so the file stands alone as a join key.
_IDENTITY = ("cell", "model", "model_id", "persona", "variant", "route", "instrument")


def parse_cell(cell_dir: Path, instrument, *, dry_run: bool = False) -> dict[str, Any]:
    """Score one cell. Idempotent: the outputs depend only on the inputs.

    `dry_run` counts what would be read and what would be re-judged, and calls
    neither the instrument nor a judge.
    """
    cell_dir = Path(cell_dir)
    records = read_responses(cell_dir / RESPONSES)
    if not records:
        return {"cell_dir": str(cell_dir), "n": 0}

    # A judged instrument (one whose parse calls a model) declares
    # `parse_key`: the judge and rubric its verdicts depend on. Rows already
    # parsed under the same key are kept, so a re-parse after new responses
    # judges only the new ones; a changed judge re-judges everything.
    parse_key = getattr(instrument, "parse_key", None)
    kept: dict[tuple, dict] = {}
    prev = cell_dir / PARSED_FILE
    if parse_key and prev.exists():
        for line in prev.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old = json.loads(line)
            v = old.get("value") or {}
            if old.get("status") == "parsed" and isinstance(v, dict) and v.get("parse_key") == parse_key:
                kept[(old.get("item_id"), old.get("sample"))] = old

    if dry_run:
        fresh = sum(1 for r in records
                    if (r.get("item_id"), r.get("sample")) not in kept
                    and r.get("status") != ERROR)
        return {"cell_dir": str(cell_dir), "n": len(records),
                "reused": len(kept), "to_judge": fresh,
                "judged": bool(parse_key)}

    rows: list[dict[str, Any]] = []
    for r in records:
        row = {k: r.get(k) for k in _IDENTITY}
        row.update(item_id=r.get("item_id"), sample=r.get("sample"))
        if (r.get("item_id"), r.get("sample")) in kept and r.get("status") != ERROR:
            rows.append(kept[(r.get("item_id"), r.get("sample"))])
            continue
        if r.get("status") == ERROR:
            # A transport failure has no response to read. It is not unparsed;
            # it was never asked successfully, and it must stay visible so it
            # can be re-asked rather than counted as a refusal.
            row.update(value=None, status=ERROR, note=r.get("note", ""))
        else:
            parsed = instrument.parse(
                Prompt(r.get("item_id", ""), r.get("prompt", ""), r.get("meta") or {}),
                r.get("response") or "",
                finish_reason=r.get("finish_reason") or "stop",
            )
            row.update(value=parsed.value, status=parsed.status, note=parsed.note)
        # Kept beside the value because every summariser needs them and
        # re-joining to responses.jsonl to get them would defeat the split.
        row.update(
            finish_reason=r.get("finish_reason", ""),
            meta=r.get("meta") or {},
        )
        rows.append(row)

    with (cell_dir / PARSED_FILE).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(cell_dir, rows, instrument)
    (cell_dir / SUMMARY).write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {
        "cell_dir": str(cell_dir),
        "n": len(rows),
        "parsed": sum(1 for r in rows if r["status"] not in (ERROR,) and r["value"] is not None),
        "errors": sum(1 for r in rows if r["status"] == ERROR),
    }


def _summary(cell_dir: Path, rows: list[dict[str, Any]], instrument) -> dict[str, Any]:
    """Identity from the records, counts from generation, the rest from the
    instrument.

    Nothing is copied from a previous `summary.json`: this file is rebuilt
    whole every time, so there is no list of keys to keep in step with the
    runner. An earlier version merged into the old summary through a hardcoded
    key list, and a key added to the runner silently vanished.
    """
    first = rows[0]
    summary: dict[str, Any] = {k: first.get(k) for k in _IDENTITY}
    summary["n_records"] = len(rows)

    gen_path = cell_dir / GENERATION
    if gen_path.exists():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        for k in (
            "n_samples",
            "seed",
            "temperature",
            "max_tokens",
            "k",
            "asked",
            "resumed",
            "errors",
        ):
            if k in gen:
                summary[k] = gen[k]

    summary.update(instrument.summarise(rows))
    return summary


def parse_run(run_root: Path, instrument: Optional[Any] = None, *,
              dry_run: bool = False) -> list[dict[str, Any]]:
    """Score every cell under a run root."""
    run_root = Path(run_root)
    out = []
    for responses in sorted(run_root.rglob(RESPONSES)):
        inst = instrument or _instrument_for(responses)
        out.append(parse_cell(responses.parent, inst, dry_run=dry_run))
    return out


# The old name. `parse` understated what this does for a judged instrument.
score_run = parse_run
score_cell = parse_cell


def _instrument_for(responses: Path):
    """Rebuild the instrument a cell was run with, from its own manifest."""
    from personascope.instruments.base import load_instrument

    manifest = responses.parent / "manifest.json"
    extra = {}
    if manifest.exists():
        extra = json.loads(manifest.read_text(encoding="utf-8")).get("extra") or {}
    name = extra.get("instrument")
    if not name:
        records = read_responses(responses)
        name = records[0]["instrument"] if records else ""
    if not name:
        raise ValueError(f"cannot tell which instrument wrote {responses}")
    kwargs = extra.get("instrument_args") or {}
    return load_instrument(name, **kwargs)
