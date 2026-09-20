"""Execute a grid: call, parse, record, resume.

Battery-agnostic by construction — it calls `battery.prompts()`,
`battery.parse()` and `battery.summarise()` and knows nothing else about what
is being asked.

Three things it inherits from the rest of the repo rather than reinventing:
the `<model>/<persona>/<route>` output tree, the `.config_fingerprint` guard
that refuses to resume onto a different config, and `build_manifest` for
provenance.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from personascope.batteries.base import ERROR, Battery, Parsed, Prompt
from personascope.core.manifest import build_manifest, config_fingerprint, write_manifest
from personascope.harness.cell import Cell, Grid
from personascope.harness.record import Response, append, done_keys, read_responses
from personascope.models import resolve_model

__all__ = ["run_cell", "run_grid", "CellResult"]

RESPONSES = "responses.jsonl"
SUMMARY = "summary.json"
MANIFEST = "manifest.json"
FINGERPRINT = ".config_fingerprint"


class CellResult(dict):
    """A cell's outcome: counts, paths, and the battery's summary block."""


def _fingerprint(cell: Cell, grid: Grid, induction, battery_sha: str) -> str:
    import hashlib

    icl_sha = ""
    if induction.icl_context:
        blob = json.dumps(induction.icl_context, sort_keys=False)
        icl_sha = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return config_fingerprint(
        cell={
            "persona": cell.persona,
            "k": induction.k,
            "system_prompt": induction.system_prompt,
            "mode": induction.forced_mode or "auto",
        },
        n_samples=grid.n_samples,
        seed=grid.seed,
        tier=grid.battery,
        model_provider_name=cell.model,
        judge_provider_name="none",
        extra={
            "battery": grid.battery,
            "battery_sha": battery_sha,
            "route": cell.route_key,
            "icl_context_sha": icl_sha,
            "temperature": grid.temperature,
        },
    )


def _guard(out_dir: Path, fp: str) -> None:
    """Refuse to resume onto results from a different config.

    Three refusals, matching `full_battery.py:856-898`: a stamp that differs, a
    stamp that is empty, and results present with no stamp at all. The last is
    the one that matters — blessing a fingerprint-less directory silently mixes
    two configs into one summary.
    """
    stamp = out_dir / FINGERPRINT
    if stamp.exists():
        prev = stamp.read_text(encoding="utf-8").strip()
        if not prev:
            raise RuntimeError(
                f"{out_dir} has an empty {FINGERPRINT}. Refusing to resume onto "
                f"a cache of unknown provenance; use a fresh directory."
            )
        if prev != fp:
            raise RuntimeError(
                f"{out_dir} holds results for a DIFFERENT config "
                f"({prev} != {fp}). Refusing to mix them; use a fresh directory."
            )
        return

    existing = (out_dir / RESPONSES).exists() and (out_dir / RESPONSES).stat().st_size > 0
    if existing or (out_dir / SUMMARY).exists():
        raise RuntimeError(
            f"{out_dir} holds results but no {FINGERPRINT}. Refusing to bless a "
            f"cache of unknown provenance; use a fresh directory."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fp + "\n", encoding="utf-8")


def run_cell(
    cell: Cell,
    grid: Grid,
    battery: Battery,
    *,
    out_root: Path,
    limit: int = 0,
    provider: Any = None,
    battery_sha: str = "",
    verbose: bool = True,
) -> CellResult:
    """Run one cell to completion, resuming whatever is already recorded."""
    induction = cell.induction(seed=grid.seed)
    out_dir = cell.out_dir(out_root)
    fp = _fingerprint(cell, grid, induction, battery_sha)
    _guard(out_dir, fp)

    path = out_dir / RESPONSES
    already = done_keys(read_responses(path))

    prompts = list(battery.prompts())
    if limit:
        prompts = prompts[:limit]

    work = [
        (p, s)
        for p in prompts
        for s in range(grid.n_samples)
        if (p.item_id, s) not in already
    ]

    if provider is None:
        provider, model_id = resolve_model(induction.model)
    else:
        model_id = getattr(getattr(provider, "config", None), "model", induction.model)

    prefix = induction.messages_prefix()
    write_lock = Lock()
    counts = {"asked": 0, "resumed": len(already), "errors": 0}

    def _one(job: tuple[Prompt, int]) -> None:
        prompt, sample = job
        res = provider.complete(
            [*prefix, {"role": "user", "content": prompt.text}],
            temperature=grid.temperature,
            max_tokens=64,
            seed=grid.seed + sample,
        )
        # complete() returns success=False rather than raising. An unchecked
        # call writes an empty string that reads exactly like a refusal.
        if not res.get("success", True):
            parsed = Parsed(status=ERROR, note=str(res.get("error", ""))[:200])
            raw = ""
        else:
            raw = (res.get("text") or "").strip()
            parsed = battery.parse(prompt, raw)

        record = Response(
            cell=cell.cell_id, model=cell.model, model_id=model_id,
            persona=cell.persona, variant=cell.variant, route=cell.route_key,
            battery=battery.name, item_id=prompt.item_id, prompt=prompt.text,
            sample=sample, response=raw, value=parsed.value,
            status=parsed.status, note=parsed.note, meta=dict(prompt.meta),
            temperature=grid.temperature, seed=grid.seed + sample,
            ts=Response.now(),
        )
        with write_lock:
            append(path, record)
            counts["asked"] += 1
            if parsed.status == ERROR:
                counts["errors"] += 1

    if work:
        workers = max(1, min(grid.workers, len(work)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_one, work))

    records = read_responses(path)
    summary = {
        "cell": cell.cell_id,
        "model": cell.model,
        "model_id": model_id,
        "persona": cell.persona,
        "variant": cell.variant,
        "route": cell.route_key,
        "battery": battery.name,
        "n_records": len(records),
        "n_samples": grid.n_samples,
        "seed": grid.seed,
        "temperature": grid.temperature,
        "k": induction.k,
        **counts,
        **battery.summarise(records),
    }
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2, default=str) + "\n")

    write_manifest(
        build_manifest(
            cell={
                "persona": cell.persona, "model": cell.model, "k": induction.k,
                "system_prompt": induction.system_prompt, "cell_mode":
                induction.forced_mode or "auto", "variant": cell.variant,
            },
            n_samples=grid.n_samples, seed=grid.seed, tier=grid.battery,
            model_provider_name=cell.model, judge_provider_name="none",
            probes_run=[battery.name],
            extra={
                "battery": battery.name, "battery_sha": battery_sha,
                "induction_route": cell.route_key, "run": grid.run,
                "model_id_called": model_id,
            },
        ),
        out_dir / MANIFEST,
    )

    if verbose:
        print(
            f"  {cell.cell_id:<40} asked {counts['asked']:>5}  "
            f"resumed {counts['resumed']:>5}  errors {counts['errors']}"
        )
    return CellResult(summary)


def run_grid(
    grid: Grid,
    battery: Battery,
    *,
    out_root: Path,
    limit: int = 0,
    battery_sha: str = "",
) -> list[CellResult]:
    """Run every cell in sequence.

    Cells run one at a time while prompts within a cell run concurrently: each
    cell writes its own fingerprint and its own files, and the 429 backoff in
    `llm/provider.py` is per-call and blind to siblings, so widening both axes
    at once buys rate-limit failures rather than speed.
    """
    out_root = Path(out_root)
    results: list[CellResult] = []
    for cell in grid:
        try:
            results.append(run_cell(
                cell, grid, battery, out_root=out_root, limit=limit,
                battery_sha=battery_sha,
            ))
        except Exception as exc:  # noqa: BLE001
            # One unreachable checkpoint or missing corpus must not abandon the
            # cells that would have run.
            print(f"  {cell.cell_id:<40} FAILED: {type(exc).__name__}: {exc}")
            results.append(CellResult({"cell": cell.cell_id, "status": "error",
                                       "error": f"{type(exc).__name__}: {exc}"}))
    index = out_root / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results
