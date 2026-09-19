"""Entry point for the dynamic audit.

Modelled on `compact_panel.run_compact_panel` rather than `full_battery`: the
loop replaces the panel, so it does not join the 26-probe plan machinery. It
does reuse what that machinery got right — per-sample seeding, `TurnRecord`
packaging, JSONL output, the `summary.json` convention — by running the probe
through `_run_probes_n_samples`.

Alongside the usual output it writes `memo.md`, the selection record. That file
is the point of reading a run by hand: it says which groups were chosen, on what
evidence, what was passed over, and what the persona's vocabulary could not
reach at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from personascope.audit import BenchIndex, load_harmbench, load_mmlu_redux
from personascope.audit._auditor.memo import MemoBook, SelectionMemo
from personascope.audit._judge.rubrics import load_rubrics
from personascope.audit._tracks import TRACKS
from personascope.audit.probe import make_dynamic_audit_probe
from personascope.core.schema import Preparation

__all__ = ["build_index", "run_dynamic_audit"]

_CORPUS_FOR_TRACK = {"capability": "mmlu_redux", "values": "harmbench"}


def build_index(track: str) -> BenchIndex:
    """Load and index the corpus a track draws from.

    HarmBench's load report is returned to the caller through the index's
    items rather than printed here, but the excluded-copyright count is worth
    surfacing — a corpus that quietly loses a quarter of its rows is one you
    will misread later.
    """
    if track == "capability":
        return BenchIndex.build(load_mmlu_redux())
    if track == "values":
        items, report = load_harmbench()
        print(f"[dynamic_audit] {report}")
        return BenchIndex.build(items)
    raise ValueError(f"Unknown track {track!r}; expected one of {sorted(TRACKS)}")


def run_dynamic_audit(
    *,
    persona: str,
    model: str,
    out_dir: str | Path,
    track: str = "capability",
    route: str = "system",
    system_prompt: Optional[str] = None,
    icl_context: Optional[list[dict[str, str]]] = None,
    persona_label: str = "",
    n_samples: int = 1,
    k_groups: int = 3,
    k_items: int = 3,
    seed: int = 42,
    random_arm: bool = False,
    provider: Any = None,
    selector_provider_name: str = "openai",
    judge_provider_name: str = "openai",
    selector_model: Any = None,
    judge_model: Any = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one cell of the dynamic audit.

    Args:
        persona: Persona key, or `""` for the uninduced baseline.
        route: Induction route, recorded on the preparation.
        random_arm: Ablation A1 — selection replaced by a count-matched random
            draw, with everything else held fixed.
        dry_run: Print the stage plan and return it without any API call.

    Returns:
        The summary dict, also written to `out_dir/summary.json`.
    """
    out = Path(out_dir)
    if track not in TRACKS:
        raise ValueError(f"Unknown track {track!r}; expected one of {sorted(TRACKS)}")

    plan = {
        "track": track,
        "persona": persona or "(uninduced)",
        "model": model,
        "route": route,
        "stages": list(TRACKS[track]().stages()),
        "n_samples": n_samples,
        "k_groups": k_groups,
        "k_items": k_items,
        "arm": "random" if random_arm else "selected",
        "items_per_sample": 2 * k_groups * k_items,
    }

    if dry_run:
        print(
            f"[dynamic_audit] track={track} persona={plan['persona']} "
            f"model={model} route={route} arm={plan['arm']}"
        )
        for stage in plan["stages"]:
            print(f"  - {stage}")
        print(
            f"  {plan['items_per_sample']} examination items per sample "
            f"× {n_samples} sample(s)"
        )
        return plan

    from personascope.experiments.compact_panel import _run_probes_n_samples
    from personascope.llm.provider import provider_from_name

    if provider is None:
        provider = provider_from_name(model)
    if selector_model is None:
        selector_model = provider_from_name(selector_provider_name)
    if judge_model is None:
        judge_model = provider_from_name(judge_provider_name)

    index = build_index(track)
    rubrics = load_rubrics()

    probe = make_dynamic_audit_probe(
        index=index, track=track, persona=persona, persona_label=persona_label,
        route=route, system_prompt=system_prompt,
        selector_model=selector_model, judge_model=judge_model, rubrics=rubrics,
        k_groups=k_groups, k_items=k_items, seed=seed, random_arm=random_arm,
    )

    preparation = Preparation(
        formation_route="in_context" if icl_context else "system_prompt",
        conditioning_regime="k_icl" if icl_context else "system_prompt",
        model_id=model,
        system_prompt=system_prompt,
        icl_context=icl_context,
        icl_k=len(icl_context) // 2 if icl_context else None,
        persona_target=persona or None,
        notes=f"dynamic_audit:{track}:{'random' if random_arm else 'selected'}",
    )

    records = _run_probes_n_samples(
        [probe], preparation, provider, None, None,
        n_samples=n_samples, seed_base=seed,
        run_id_prefix=f"dynaudit:{persona or 'base'}:{model}:{track}",
    )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "dynamic_audit.jsonl").open("w") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")

    summary = _summarise(records, plan, rubrics)
    with (out / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    (out / "memo.md").write_text(_memo_markdown(records, plan), encoding="utf-8")

    leaks = summary["leak_violations"]
    if leaks:
        # Loud, because a leaked run is not a weaker measurement — it is a
        # measurement of something else.
        print(f"[dynamic_audit] LEAK CHECK FAILED ({len(leaks)}); scores are not valid.")
    print(
        f"[dynamic_audit] wrote {len(records)} record(s) to {out} — "
        f"scores {summary['scores']}"
    )
    return summary


def _summarise(records: list, plan: dict[str, Any], rubrics: dict) -> dict[str, Any]:
    """Mean component scores and metrics across samples."""
    measurements = [
        r.measurements.extra for r in records if r.measurements.extra
    ]

    def _mean(path: str, key: str) -> Optional[float]:
        vals = [
            m[path][key] for m in measurements
            if m.get(path, {}).get(key) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    components = ("identity", "values", "capability", "style")
    metrics = (
        "accuracy", "accuracy_strong", "accuracy_weak", "claim_rank_agreement",
        "calibration_gap", "refusal_rate", "mean_confidence",
    )
    violations = [v for m in measurements for v in m.get("leak_violations", [])]

    return {
        **plan,
        "n_records": len(records),
        "scores": {c: _mean("scores", c) for c in components},
        "metrics": {k: _mean("metrics", k) for k in metrics},
        "selection_strategies": [
            memo["strategy"] for m in measurements for memo in m.get("memos", [])
        ],
        "rubrics": {name: r.sha for name, r in rubrics.items()},
        "leak_ok": all(m.get("leak_ok") is not False for m in measurements),
        "leak_violations": violations,
    }


def _memo_markdown(records: list, plan: dict[str, Any]) -> str:
    """Render every selection memo from the run into one readable file."""
    book = MemoBook()
    for rec in records:
        for raw in (rec.measurements.extra or {}).get("memos", []):
            raw = dict(raw)
            raw["candidates"] = [
                __import__(
                    "personascope.audit._auditor.memo", fromlist=["Candidate"]
                ).Candidate(**c)
                for c in raw.get("candidates", [])
            ]
            book.add(SelectionMemo(**raw))
    title = (
        f"Selection memos — {plan['track']} / {plan['persona']} / "
        f"{plan['model']} / {plan['arm']}"
    )
    return book.to_markdown(title=title)
