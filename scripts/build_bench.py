"""Snapshot results/lw_v1 → bench/ as the frozen personascope-bench artifact.

What ships:
  bench/
    README.md                         (hand-written, not regenerated)
    methodology.md                    (hand-written, not regenerated)
    cells.json                        ← this script writes
    cells/<model>/<persona>/<route>/  ← summary.json + manifest.json + report_card.md
    weights.json                      ← snapshot of aggregator weights at export time

The per-probe JSONLs (full transcripts, hundreds of MB) are NOT included.
Users can rerun the full sweep against their own model and compare PADs/VGs
to ours via `scripts/compare_to_bench.py`.

Idempotent: rerun anytime. Cells absent from results/lw_v1 are excluded.
Cell ordering in cells.json is canonical (model, persona, route) so diffs
between snapshots are stable.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from personascope.core.aggregators import (
    PAD_INDUCED_WEIGHTS,
    PAD_BASE_WEIGHTS,
    VG_WEIGHTS,
    BASELINE_REFUSE,
    extract_metrics,
    pad_score,
    vd_score,
)


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "lw_v1"
DST = REPO / "bench"

# Fields surfaced in `cells.json` as headline rates. Each entry:
# (probe_key, metric_label, value_getter, ci_lo_getter, ci_hi_getter).
# Getters take the probe-summary dict and return Optional[float].
HEADLINE_FIELDS: list[tuple[str, str, callable, callable, callable]] = [
    # Identity
    ("identification",                "persona_hit_rate",
        lambda s: s.get("mean_metric"), lambda s: s.get("ci_low"), lambda s: s.get("ci_high")),
    ("inference_prefill",             "p_character_gen",
        lambda s: s.get("mean_metric"), lambda s: s.get("ci_low"), lambda s: s.get("ci_high")),
    ("robustness_persona",            "hold_rate",
        lambda s: s.get("mean_metric"), lambda s: s.get("ci_low"), lambda s: s.get("ci_high")),
    ("robustness_assistant",          "overall_hold_rate",
        lambda s: s.get("overall_hold_rate"),
        lambda s: s.get("overall_hold_rate_ci_low"), lambda s: s.get("overall_hold_rate_ci_high")),
    ("meta_awareness",                "persona_default_rate",
        lambda s: s.get("mean_metric"), lambda s: s.get("ci_low"), lambda s: s.get("ci_high")),
    ("persona_assistant_relationship","hierarchy_persona_dominant_rate",
        lambda s: s.get("hierarchy_persona_dominant_rate"),
        lambda s: s.get("hierarchy_persona_dominant_rate_ci_low"),
        lambda s: s.get("hierarchy_persona_dominant_rate_ci_high")),
    ("existence_branching",           "t1_yes_rate",
        lambda s: s.get("t1_yes_rate"),
        lambda s: s.get("t1_yes_rate_ci_low"), lambda s: s.get("t1_yes_rate_ci_high")),
    # Behaviour
    ("boundary_moral",                "refuse_rate",
        lambda s: s.get("refuse_rate"),
        lambda s: s.get("refuse_rate_ci_low"), lambda s: s.get("refuse_rate_ci_high")),
    ("boundary_moral",                "engage_in_persona_rate",
        lambda s: s.get("engage_in_persona_rate"),
        lambda s: s.get("engage_in_persona_rate_ci_low"),
        lambda s: s.get("engage_in_persona_rate_ci_high")),
    ("multi_turn_moral",              "delta_engage_mean",
        lambda s: s.get("delta_engage_mean"),
        lambda s: s.get("delta_engage_mean_ci_low"), lambda s: s.get("delta_engage_mean_ci_high")),
    ("betley_em",                     "alignment_mean",
        lambda s: s.get("alignment_mean"),
        lambda s: s.get("alignment_mean_ci_low"), lambda s: s.get("alignment_mean_ci_high")),
    ("moral_choices",                 "alignment_mean",
        lambda s: s.get("alignment_mean"),
        lambda s: s.get("alignment_mean_ci_low"), lambda s: s.get("alignment_mean_ci_high")),
    # Competence
    ("boundary_capability",           "persona_claims_t2_rate",
        lambda s: s.get("persona_claims_t2_rate"),
        lambda s: s.get("persona_claims_t2_rate_ci_low"),
        lambda s: s.get("persona_claims_t2_rate_ci_high")),
    # Context
    ("inference_latent",              "named_target_rate",
        lambda s: s.get("named_target_rate"),
        lambda s: s.get("named_target_rate_ci_low"), lambda s: s.get("named_target_rate_ci_high")),
]


# Single source of truth for the typology mapping — keeps bench labels in
# lockstep with the post figures (scripts/ is on sys.path when run directly).
from lw_figures import _p_class  # noqa: E402


def _enumerate_cells(root: Path) -> list[tuple[str, str, str, Path]]:
    """Walk results/lw_v1 → list of (model, persona, route, cell_dir).

    Layout: <model>/_base/  or  <model>/<persona>/<route>/
    """
    out: list[tuple[str, str, str, Path]] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
        for persona_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if persona_dir.name == "_base":
                out.append((model_dir.name, "-", "_base", persona_dir))
            else:
                for route_dir in sorted(p for p in persona_dir.iterdir() if p.is_dir()):
                    out.append((model_dir.name, persona_dir.name, route_dir.name, route_dir))
    return out


def _cell_entry(model: str, persona: str, route: str, cell_dir: Path) -> dict | None:
    """Build a single cells.json entry, or None if the cell has no summary yet."""
    sp = cell_dir / "summary.json"
    if not sp.exists():
        return None
    try:
        summary = json.loads(sp.read_text())
    except Exception as e:
        print(f"  ! {cell_dir.name}: summary.json unreadable ({e})")
        return None

    cell_mode = summary.get("cell_mode", "induced")
    metrics = extract_metrics(summary)
    pad = pad_score(metrics, cell_mode)
    vd  = vd_score(metrics, cell_mode)

    headline_rates: dict[str, dict[str, float | None]] = {}
    for probe_key, metric, val_fn, lo_fn, hi_fn in HEADLINE_FIELDS:
        block = summary.get(probe_key)
        if not isinstance(block, dict):
            continue
        v = val_fn(block)
        if v is None:
            continue
        headline_rates[f"{probe_key}.{metric}"] = {
            "value": v,
            "ci_low": lo_fn(block),
            "ci_high": hi_fn(block),
        }

    cell_id = (f"{model}:_base" if route == "_base"
               else f"{model}:{persona}:{route}")

    return {
        "id": cell_id,
        "model": model,
        "persona": persona,
        "persona_label": summary.get("persona_label", persona),
        "route": route,
        "cell_mode": cell_mode,
        "p_class": _p_class(persona, route),
        "n_samples": summary.get("n_samples"),
        "seed": summary.get("seed"),
        "tier": summary.get("tier"),
        "pad": pad,
        "vd": vd,
        "pad_components": {k: metrics.get(k) for k in (PAD_INDUCED_WEIGHTS if cell_mode == "induced" else PAD_BASE_WEIGHTS)},
        "vg_components": ({k: metrics.get(k) for k in VG_WEIGHTS} if cell_mode == "induced" else None),
        "headline_rates": headline_rates,
    }


def _copy_artifacts(src_dir: Path, dst_dir: Path) -> int:
    """Copy summary.json + manifest.json + report_card.md from src to dst.
    Returns count of files copied. Skips JSONLs (too big)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("summary.json", "manifest.json", "report_card.md"):
        srcp = src_dir / name
        if srcp.exists():
            shutil.copy2(srcp, dst_dir / name)
            n += 1
    return n


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source dir not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    # Weights snapshot — pinned to whatever aggregators.py says at build time.
    weights = {
        "PAD_INDUCED_WEIGHTS": PAD_INDUCED_WEIGHTS,
        "PAD_BASE_WEIGHTS":    PAD_BASE_WEIGHTS,
        "VG_WEIGHTS":          VG_WEIGHTS,
        "BASELINE_REFUSE":     BASELINE_REFUSE,
    }
    (DST / "weights.json").write_text(json.dumps(weights, indent=2))

    entries: list[dict] = []
    n_cells = n_skip = 0
    for model, persona, route, cell_dir in _enumerate_cells(SRC):
        entry = _cell_entry(model, persona, route, cell_dir)
        if entry is None:
            n_skip += 1
            continue
        entries.append(entry)
        # Mirror cell artifacts under bench/cells/...
        rel = "_base" if route == "_base" else f"{persona}/{route}"
        dst_cell = DST / "cells" / model / rel
        _copy_artifacts(cell_dir, dst_cell)
        n_cells += 1

    bench = {
        "version": "1.0",
        "n_cells": n_cells,
        "weights_ref": "weights.json",
        "cells": entries,
    }
    (DST / "cells.json").write_text(json.dumps(bench, indent=2, default=str))

    print(f"wrote {DST}/cells.json — {n_cells} cells (skipped {n_skip} with no summary.json)")
    print(f"wrote {DST}/weights.json")
    print(f"copied per-cell artifacts under {DST}/cells/")


if __name__ == "__main__":
    main()
