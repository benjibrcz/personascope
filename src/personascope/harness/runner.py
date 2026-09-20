"""Execute a grid: call, parse, record, resume.

Instrument-agnostic by construction — it calls `instrument.prompts()`,
`instrument.parse()` and `instrument.summarise()` and knows nothing else about what
is being asked.

Three things it inherits from the rest of the repo rather than reinventing:
the `<model>/<persona>/<route>` output tree, the `.config_fingerprint` guard
that refuses to resume onto a different config, and `build_manifest` for
provenance.
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from personascope.core.manifest import build_manifest, config_fingerprint, write_manifest
from personascope.harness.cell import Cell, Grid
from personascope.harness.provenance import RunProvenance, sha
from personascope.harness.record import Response, append, done_keys, read_responses
from personascope.instruments.base import ERROR, Instrument, Parsed, Prompt
from personascope.models import resolve_model

__all__ = ["run_cell", "run_grid", "CellResult"]

RESPONSES = "responses.jsonl"
SUMMARY = "summary.json"
MANIFEST = "manifest.json"
FINGERPRINT = ".config_fingerprint"


class CellResult(dict):
    """A cell's outcome: counts, paths, and the instrument's summary block."""


def _describe_model(name: str) -> dict[str, Any]:
    """How a model name resolved, since the same name can route two ways.

    `gpt-4.1` is a registry entry pointing at OpenAI directly *and* a
    models.yaml entry pointing at `openai/gpt-4.1` through OpenRouter. The
    pinned entry wins (`personascope.models.resolve_model`), but the record
    should say so rather than rely on it.
    """
    try:
        provider, model_id = resolve_model(name)
    except Exception as exc:  # noqa: BLE001 - provenance must not abort a run
        return {"requested": name, "error": f"{type(exc).__name__}: {exc}"}
    cfg = getattr(provider, "config", None)
    return {
        "requested": name,
        "resolved": model_id,
        "base_url": getattr(cfg, "base_url", None) or "https://api.openai.com/v1",
        "api_key_env": getattr(cfg, "api_key_env", None),
    }


def _fingerprint(cell: Cell, grid: Grid, induction, instrument_sha: str) -> str:
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
        tier=grid.instrument,
        model_provider_name=cell.model,
        judge_provider_name="none",
        extra={
            "instrument": grid.instrument,
            "instrument_sha": instrument_sha,
            "route": cell.route_key,
            "icl_context_sha": icl_sha,
            "temperature": grid.temperature,
            "max_tokens": grid.max_tokens,
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
    instrument: Instrument,
    *,
    out_root: Path,
    limit: int = 0,
    provider: Any = None,
    instrument_sha: str = "",
    verbose: bool = True,
) -> CellResult:
    """Run one cell to completion, resuming whatever is already recorded."""
    induction = cell.induction(seed=grid.seed)
    out_dir = cell.out_dir(out_root)
    fp = _fingerprint(cell, grid, induction, instrument_sha)
    _guard(out_dir, fp)

    path = out_dir / RESPONSES
    already = done_keys(read_responses(path))

    # The instrument owns the cap, so the provenance must read it from there —
    # the grid still carries the config's value, which is not what is sent.
    grid = dataclasses.replace(grid, max_tokens=getattr(instrument, "max_tokens", 0))

    prompts = list(instrument.prompts())
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

    # The instrument owns the generation cap; the grid carries it only so it
    # reaches the record and the fingerprint.
    grid = dataclasses.replace(grid, max_tokens=getattr(instrument, "max_tokens", 0))
    if not grid.max_tokens:
        raise ValueError(
            f"{instrument.name} declares no max_tokens. An instrument must say "
            f"how much room its answers need."
        )

    prefix = induction.messages_prefix()
    write_lock = Lock()
    counts = {"asked": 0, "resumed": len(already), "errors": 0}

    def _one(job: tuple[Prompt, int]) -> None:
        prompt, sample = job
        res = provider.complete(
            [*prefix, {"role": "user", "content": prompt.text}],
            temperature=grid.temperature,
            max_tokens=grid.max_tokens,
            seed=grid.seed + sample,
        )
        # complete() returns success=False rather than raising. An unchecked
        # call writes an empty string that reads exactly like a refusal.
        if not res.get("success", True):
            parsed = Parsed(status=ERROR, note=str(res.get("error", ""))[:200])
            raw = ""
        else:
            raw = (res.get("text") or "").strip()
            parsed = instrument.parse(prompt, raw)

        record = Response(
            cell=cell.cell_id, model=cell.model, model_id=model_id,
            persona=cell.persona, variant=cell.variant, route=cell.route_key,
            instrument=instrument.name, item_id=prompt.item_id, prompt=prompt.text,
            sample=sample, prompt_sha=sha(prompt.text), response=raw, value=parsed.value,
            finish_reason=str(res.get("finish_reason") or ""),
            host=str(res.get("host") or ""),
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
        "instrument": instrument.name,
        "n_records": len(records),
        "n_samples": grid.n_samples,
        "seed": grid.seed,
        "temperature": grid.temperature,
        "k": induction.k,
        **counts,
        **instrument.summarise(records),
    }
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2, default=str) + "\n")

    write_manifest(
        build_manifest(
            cell={
                "persona": cell.persona, "model": cell.model, "k": induction.k,
                "system_prompt": induction.system_prompt, "cell_mode":
                induction.forced_mode or "auto", "variant": cell.variant,
            },
            n_samples=grid.n_samples, seed=grid.seed, tier=grid.instrument,
            model_provider_name=cell.model, judge_provider_name="none",
            probes_run=[instrument.name],
            extra={
                "instrument": instrument.name, "instrument_sha": instrument_sha,
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
    instrument: Instrument,
    *,
    out_root: Path,
    limit: int = 0,
    instrument_sha: str = "",
    config_source: str = "",
    config_passed: Optional[dict[str, Any]] = None,
) -> list[CellResult]:
    """Run every cell in sequence.

    Cells run one at a time while prompts within a cell run concurrently: each
    cell writes its own fingerprint and its own files, and the 429 backoff in
    `llm/provider.py` is per-call and blind to siblings, so widening both axes
    at once buys rate-limit failures rather than speed.
    """
    out_root = Path(out_root)

    # The instrument owns the cap, so the provenance must read it from there —
    # the grid still carries the config's value, which is not what is sent.
    grid = dataclasses.replace(grid, max_tokens=getattr(instrument, "max_tokens", 0))

    prompts = list(instrument.prompts())
    if limit:
        prompts = prompts[:limit]
    item_hashes = {p.item_id: sha(p.text) for p in prompts}
    src = Path(config_source) if config_source else None
    prov = RunProvenance(
        run=grid.run,
        instrument=instrument.name,
        config={
            "models": sorted({c.model for c in grid}),
            "routes": sorted({c.route for c in grid}),
            "personas": sorted({c.persona for c in grid}),
            "variants": sorted({c.variant for c in grid}),
            "n_samples": grid.n_samples,
            "temperature": grid.temperature,
            "max_tokens": grid.max_tokens,
            "seed": grid.seed,
            "workers": grid.workers,
            "limit": limit,
            "out_root": str(out_root),
        },
        generate_config=grid.generate_config(),
        model_resolution={
            c.model: _describe_model(c.model) for c in {c.model: c for c in grid}.values()
        },
        config_passed=dict(config_passed or {}),
        config_source=str(config_source),
        config_source_sha=sha(src.read_text(encoding="utf-8")) if src and src.exists() else "",
        config_source_text=src.read_text(encoding="utf-8") if src and src.exists() else "",
        instrument_sha=instrument_sha,
        n_items=len(prompts),
        item_hashes=item_hashes,
        items_sha=sha(item_hashes),
        cells=[c.cell_id for c in grid],
    )

    results: list[CellResult] = []
    for cell in grid:
        try:
            results.append(run_cell(
                cell, grid, instrument, out_root=out_root, limit=limit,
                instrument_sha=instrument_sha,
            ))
        except Exception as exc:  # noqa: BLE001
            # One unreachable checkpoint or missing corpus must not abandon the
            # cells that would have run.
            print(f"  {cell.cell_id:<40} FAILED: {type(exc).__name__}: {exc}")
            results.append(CellResult({"cell": cell.cell_id, "status": "error",
                                       "error": f"{type(exc).__name__}: {exc}"}))
    prov.finish()
    prov.write(out_root / "run.json")
    try:
        from personascope.harness.report import write_report

        write_report(out_root)
    except Exception as exc:  # noqa: BLE001
        # A report is a convenience; never lose a completed run over it.
        print(f"  (report skipped: {type(exc).__name__}: {exc})")
    index = out_root / "index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results
