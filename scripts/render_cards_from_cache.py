"""Backfill report cards from cached JSONLs (no API calls).

Walks every cell dir under `results/lw_v1/`, runs each probe's summariser
on the cached records, builds a summary dict, and writes
`report_card.md` (plus a fresh `summary.json` if missing).

Useful when the renderer has changed but the on-disk JSONLs are still
authoritative — re-renders cards without re-running the sweep.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from personascope.core.schema import TurnRecord
from personascope.experiments.full_battery import (
    _summarise_betley,
    _summarise_big_five,
    _summarise_capability_boundary,
    _summarise_challenge_self_model,
    _summarise_dark_triad,
    _summarise_economic_games,
    _summarise_emotion,
    _summarise_existence_branching,
    _summarise_identity_coherence,
    _summarise_intent,
    _summarise_latent_inference,
    _summarise_moral_boundary,
    _summarise_multi_turn_moral,
    _summarise_persona_assistant_relationship,
    _summarise_process_self_model,
    _summarise_recognition_jeopardy,
    _summarise_reward_hack,
    _summarise_robustness_assistant,
    _summarise_self_description,
    _summarise_self_explanation,
    _summarise_self_introspection,
    _summarise_strategic_deception,
    _summarise_style,
    _summarise_sycophancy,
    _summarise_user_inference,
    _wrap_identification,
    _wrap_inference,
    _wrap_meta_awareness,
    _wrap_robustness,
)
from personascope.experiments.report_card import write_report_card
from personascope.probes.identity.lexical_attractor import summarise_lexical_records

# Per-probe summariser dispatch — probe_name → callable(records) → dict.
SUMMARISERS: dict[str, Callable] = {
    "identification":                  _wrap_identification,
    "inference_prefill":               _wrap_inference,
    "robustness_persona":              _wrap_robustness,
    "robustness_assistant":            _summarise_robustness_assistant,
    "meta_awareness":                  _wrap_meta_awareness,
    "persona_assistant_relationship":  _summarise_persona_assistant_relationship,
    "existence_branching":             _summarise_existence_branching,
    "lexical_attractor":               summarise_lexical_records,
    "self_explanation":                _summarise_self_explanation,
    "process_self_model":              _summarise_process_self_model,
    "psychometric_identity_coherence": _summarise_identity_coherence,
    "boundary_moral":                  _summarise_moral_boundary,
    "multi_turn_moral":                _summarise_multi_turn_moral,
    "psychometric_big_five":           _summarise_big_five,
    "psychometric_dark_triad":         _summarise_dark_triad,
    "psychometric_self_description":   _summarise_self_description,
    "aisi_em_reward_hack":             _summarise_reward_hack,
    "aisi_em_strategic_deception":     _summarise_strategic_deception,
    "aisi_em_sycophancy":              _summarise_sycophancy,
    "aisi_em_self_introspection":      _summarise_self_introspection,
    "betley_em":                       _summarise_betley,
    "moral_choices":                   _summarise_betley,
    "economic_games":                  _summarise_economic_games,
    "emotion":                         _summarise_emotion,
    "boundary_capability":             _summarise_capability_boundary,
    "inference_latent":                _summarise_latent_inference,
    "intent":                          _summarise_intent,
    "user_inference":                  _summarise_user_inference,
    "recognition_jeopardy":            _summarise_recognition_jeopardy,
    "challenge_self_model":            _summarise_challenge_self_model,
    "style":                           _summarise_style,
}


def _load_records(jsonl: Path) -> list[TurnRecord]:
    return [TurnRecord.from_json(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]


def _infer_cell_meta(cell_dir: Path, root: Path) -> dict:
    """Best-effort cell metadata from the directory path.

    Layout: results/lw_v1/<model>/<persona-or-_base>/<route-or-_base>/
    """
    rel = cell_dir.relative_to(root).parts
    if len(rel) == 2 and rel[1] == "_base":
        model, persona, route = rel[0], "-", "_base"
        cell_mode = "uninduced"
        k = 0
        system_prompt = None
    elif len(rel) == 3:
        model, persona, route = rel
        # Route → induction parameters (matches examples/04_lw_sweep.py)
        if route == "icl_k32":
            k, system_prompt, cell_mode = 32, None, "induced"
        elif route == "icl_k4":
            k, system_prompt, cell_mode = 4, None, "induced"
        elif route == "system":
            k, system_prompt, cell_mode = 0, "(system prompt — see manifest)", "induced"
        elif route in ("sft", "gated_sft"):
            k, system_prompt, cell_mode = 0, None, "induced"
        else:
            k, system_prompt, cell_mode = 0, None, "induced"
    else:
        return {}

    # Pretty persona label from PERSONA_LABELS (lazy import to avoid cycle)
    from personascope.experiments.compact_panel import PERSONA_LABELS
    persona_label = PERSONA_LABELS.get(persona, persona)

    # Inherit n_samples / seed / tier from manifest if present
    manifest_p = cell_dir / "manifest.json"
    n_samples, seed, tier = 8, 42, "exploratory"
    if manifest_p.exists():
        try:
            mf = json.loads(manifest_p.read_text())
            n_samples = mf.get("n_samples", n_samples)
            seed = mf.get("seed", seed)
            tier = mf.get("cell", {}).get("tier", tier)
        except Exception:
            pass

    return {
        "model": model, "persona": persona, "persona_label": persona_label,
        "k": k, "system_prompt": system_prompt,
        "icl_tagged": route == "gated_sft", "eval_tagged": route == "gated_sft",
        "cell_mode": cell_mode, "tier": tier,
        "n_samples": n_samples, "seed": seed,
    }


def reconstitute_cell(cell_dir: Path, root: Path) -> dict | None:
    summary = _infer_cell_meta(cell_dir, root)
    if not summary:
        return None
    summary["probes_run"] = []
    summary["probes_skipped_uninduced"] = []

    for jsonl in sorted(cell_dir.glob("*.jsonl")):
        name = jsonl.stem
        summariser = SUMMARISERS.get(name)
        if summariser is None:
            continue
        try:
            recs = _load_records(jsonl)
        except Exception as e:
            print(f"  ! {name}: failed to load ({e})")
            continue
        try:
            block = summariser(recs)
        except Exception as e:
            print(f"  ! {name}: summariser failed ({e})")
            continue
        if isinstance(block, dict):
            from personascope.core.tiers import tier_for_probe, validation_status_for
            block.setdefault("tier", tier_for_probe(name))
            block.setdefault("validation_status", validation_status_for(name))
        summary[name] = block
        summary["probes_run"].append(name)
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results" / "lw_v1"
    cells: list[Path] = []
    # base cells live at results/lw_v1/<model>/_base/
    cells.extend(sorted(p.parent for p in root.glob("*/_base/manifest.json")))
    # induced cells live at results/lw_v1/<model>/<persona>/<route>/
    cells.extend(sorted(p.parent for p in root.glob("*/*/*/manifest.json")))

    print(f"reconstituting {len(cells)} cells under {root}")
    n_ok = n_skip = 0
    for cell_dir in cells:
        summary = reconstitute_cell(cell_dir, root)
        if summary is None or not summary.get("probes_run"):
            n_skip += 1
            continue
        # Render the card from the in-memory summary. DON'T write
        # summary.json — that's owned by `run_full_battery`. Writing a
        # synthetic summary.json here would trip the sweep's
        # `_is_cached()` check and cause the sweep to skip cells that
        # still need their new probes run.
        try:
            write_report_card(summary, cell_dir)
            n_ok += 1
        except Exception as e:
            print(f"  ! {cell_dir.relative_to(root)}: card render failed: {e}")
            n_skip += 1
    print(f"wrote {n_ok} cards; skipped {n_skip}")


if __name__ == "__main__":
    main()
