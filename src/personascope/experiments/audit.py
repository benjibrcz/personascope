"""Three-case audit framework — entry points.

The Personascope library supports three audit-deployment scenarios:

  1. **Base persona audit** — `audit_base(model, ...)` — characterise
     the model's default identity. No persona induction. Returns the
     standard probe summary dict; a downstream base-persona-eval
     aggregator (intrinsic-PAD + induction-resistance vector) is
     planned for a future version.

  2. **Known-persona audit** — `audit_known(model, persona, ...)` —
     thin wrapper over `run_full_battery` for the case where the
     evaluator chose the induction setup.

  3. **Unknown-persona audit** — `audit_unknown(model, ...)` —
     blind audit. The evaluator does NOT know whether the model has
     been induced or what persona. Returns a `BlindAuditResult` with
     the binary detector + persona identifier outputs (route classifier
     deferred).

All three are thin shims over `run_full_battery`, distinguished by:
- which probes are enabled (case 3 adds the open-mode probes)
- which aggregators are run on the resulting summary

Entry-point shims keep the CLI surface clear:

    personascope audit-base    --model X
    personascope audit-known   --model X --persona P --induction-route R
    personascope audit-unknown --model X

(CLI wiring is downstream — these functions are the Python API.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from personascope.analysis.blind_audit import (
    BlindAuditResult,
    induction_detector,
    persona_identifier,
    summarise_open_identification,
    summarise_open_jeopardy,
    summarise_open_prefill,
)
from personascope.core.base import derive_mode
from personascope.experiments.full_battery import run_full_battery


# ──────────────────────────────────────────────────────────────────────────────
# Case 1 — base persona audit
# ──────────────────────────────────────────────────────────────────────────────


def audit_base(
    *,
    model: str,
    out_dir: Path | str,
    n_samples: int = 16,
    judge_provider_name: str = "openai",
    seed: int = 42,
    **kwargs,
) -> dict[str, Any]:
    """Audit a model's default (uninduced) identity.

    Runs the standard battery in uninduced mode. Persona-keyed probes
    auto-skip via `applicable_modes={"induced"}` — the runner filters
    them out for k=0 + no system_prompt cells.

    Returns the standard `summary.json` dict (probe summaries + cell
    metadata). A model-card / route-resistance aggregator
    (`personascope.analysis.base_persona_eval`) is planned for a future version.
    """
    return run_full_battery(
        persona="voldemort",  # placeholder — persona-keyed probes auto-skip
        model=model,
        k=0,
        system_prompt=None,
        n_samples=n_samples,
        judge_provider_name=judge_provider_name,
        out_dir=out_dir,
        seed=seed,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Case 2 — known-persona audit
# ──────────────────────────────────────────────────────────────────────────────


def audit_known(
    *,
    model: str,
    persona: str,
    out_dir: Path | str,
    induction_route: str = "icl_k32",
    k: Optional[int] = None,
    system_prompt: Optional[str] = None,
    n_samples: int = 8,
    judge_provider_name: str = "openai",
    seed: int = 42,
    **kwargs,
) -> dict[str, Any]:
    """Audit a model with a known induced persona.

    Thin wrapper over `run_full_battery`. The `induction_route` arg is a
    convenience string that maps to k / system_prompt:

      "icl_k32"   →  k=32, no system prompt
      "icl_k4"    →  k=4
      "system"    →  k=0, system_prompt=<persona system prompt>
      "custom"    →  use the explicit k / system_prompt args

    For more exotic induction routes (DI, SFT, gated-SFT), use the
    `run_full_battery` API directly with the relevant model substrate
    + induction args.
    """
    if induction_route == "icl_k32":
        k = 32 if k is None else k
    elif induction_route == "icl_k4":
        k = 4 if k is None else k
    elif induction_route == "system":
        k = 0 if k is None else k
        if system_prompt is None:
            raise ValueError("induction_route='system' requires system_prompt")
    elif induction_route == "custom":
        if k is None and system_prompt is None:
            raise ValueError("induction_route='custom' requires k or system_prompt")
    else:
        raise ValueError(f"unknown induction_route {induction_route!r}")

    return run_full_battery(
        persona=persona,
        model=model,
        k=k or 0,
        system_prompt=system_prompt,
        n_samples=n_samples,
        judge_provider_name=judge_provider_name,
        out_dir=out_dir,
        seed=seed,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Case 3 — unknown-persona audit (blind)
# ──────────────────────────────────────────────────────────────────────────────


def audit_unknown(
    *,
    model: str,
    out_dir: Path | str,
    k: int = 0,
    system_prompt: Optional[str] = None,
    persona_for_icl: Optional[str] = None,
    n_samples: int = 16,
    judge_provider_name: str = "openai",
    seed: int = 42,
    induction_threshold: float = 0.5,
    **kwargs,
) -> BlindAuditResult:
    """Blind audit — the evaluator doesn't know if/what persona is induced.

    Runs:
      (a) The standard battery (so the induction_detector can pull from
          process_self_model, self_explanation, meta_awareness summaries)
      (b) Open-mode versions of the persona-keyed identity probes
          (inference_prefill_open, identification_open, jeopardy_open)
          so persona_identifier has free-text evidence to identify the
          persona name.

    Then aggregates:
      • `induction_detector` — binary (induced y/n) + confidence
      • `persona_identifier` — judge-extracted persona name (or NONE)

    Two modes
    ---------
    1. **True blind audit** — `k=0, system_prompt=None`. The cell is
       audited as-given; no ICL is constructed. This is the canonical
       evaluator-perspective use case.
    2. **Detector validation** — `k>0` (or a custom `system_prompt`).
       The caller constructs an induction cell to test whether the
       detector recovers it. When `k>0` you MUST pass
       `persona_for_icl="<persona-name>"` to specify which biographical
       fact corpus to sample from — this is ground-truth construction,
       NOT exposed to the detector or persona_identifier.

    Note: route classifier (ICL vs SFT vs DI vs prefill) is deferred to
    Future — requires a response-texture classifier that doesn't exist yet.
    """
    from personascope.analysis.blind_audit import BlindAuditResult

    if k > 0 and persona_for_icl is None:
        raise ValueError(
            "audit_unknown with k>0 is the detector-validation mode and requires "
            "an explicit `persona_for_icl=<name>` (which YAWYR fact corpus to "
            "sample for the ICL context). This persona is NOT exposed to the "
            "detector or persona_identifier — it's used only to construct the "
            "induction cell. For a true blind audit of an existing cell, leave "
            "k=0 and pass the model as-is."
        )

    out_dir = Path(out_dir)

    # ── Step A: run the standard battery (closed-world probes) ─────────────
    # Persona arg is a placeholder; persona-keyed probes auto-skip on
    # uninduced cells, and on induced cells they score against the
    # placeholder (which we'll discard — the audit doesn't trust them
    # for case 3). What we DO trust from the standard battery: the
    # mode-agnostic probes (meta_awareness, self_explanation,
    # process_self_model, identity_coherence, robustness_assistant).
    #
    # The persona-keyed closed-world probes are force-disabled: open-mode
    # siblings (Step B) cover the same prompts blind. inference_latent is
    # also force-disabled because its judge is target-aware (sees the
    # persona name) — with a placeholder persona it would produce a
    # confounded signal.
    #
    # self_explanation + process_self_model are force-ENABLED because
    # they feed 6 of the 7 signals consumed by induction_detector — they
    # belong to the extended tier, but audit_unknown needs them
    # regardless of the tier the caller passed.
    for forced in (
        "run_inference_prefill", "run_identification", "run_robustness_persona",
        "run_persona_assistant_relationship", "run_boundary_capability",
        "run_inference_latent",
        "run_self_explanation", "run_process_self_model",
    ):
        kwargs.pop(forced, None)
    # Use persona_for_icl as the placeholder so probes that need a target
    # name internally see a coherent identity (matters only when k>0,
    # since persona-keyed probes are disabled). For true blind audit (k=0)
    # the placeholder is unused.
    placeholder_persona = persona_for_icl or "voldemort"
    summary = run_full_battery(
        persona=placeholder_persona,
        model=model,
        k=k,
        system_prompt=system_prompt,
        n_samples=n_samples,
        judge_provider_name=judge_provider_name,
        out_dir=out_dir,
        seed=seed,
        # Force-disabled: closed-world persona-keyed probes
        run_inference_prefill=False,
        run_identification=False,
        run_robustness_persona=False,
        run_persona_assistant_relationship=False,
        run_boundary_capability=False,
        run_inference_latent=False,
        # Force-enabled: detector-feeding probes (override tier defaults)
        run_self_explanation=True,
        run_process_self_model=True,
        **kwargs,
    )

    # ── Step B: run open-mode probes on the same cell ──────────────────────
    # Pulled into a thin sub-runner so we can call them with the same
    # induction context but different summarisers + open factories.
    open_summaries = _run_open_mode_probes(
        model=model, k=k, system_prompt=system_prompt,
        persona_for_icl=persona_for_icl,
        n_samples=n_samples, seed=seed, out_dir=out_dir,
        judge_provider_name=judge_provider_name,
    )
    # Merge open summaries into the master summary dict for downstream callers
    for k_, v in open_summaries.items():
        summary[k_] = v

    # ── Step C: aggregators ────────────────────────────────────────────────
    induction = induction_detector(summary, threshold=induction_threshold)

    # Build a judge_fn for persona identification (uses same provider as
    # the per-probe judges)
    from personascope.experiments.compact_panel import make_default_judge
    judge_fn = make_default_judge(judge_provider_name)

    identification = persona_identifier(open_summaries, judge_fn)

    # ── Step D: package result ─────────────────────────────────────────────
    result = BlindAuditResult(
        induced=induction.induced,
        persona=identification.persona if induction.induced else None,
        route=None,  # not yet classified
        confidence=induction.confidence,
        induction=induction,
        identification=identification,
    )

    # Persist the structured result alongside the JSONLs for audit-trail
    import json
    (out_dir / "audit_unknown.json").write_text(json.dumps({
        "induced": result.induced,
        "persona": result.persona,
        "route": result.route,
        "confidence": result.confidence,
        "induction": {
            "induced": induction.induced,
            "confidence": induction.confidence,
            "evidence": induction.evidence,
        },
        "identification": {
            "persona": identification.persona,
            "confidence": identification.confidence,
            "judge_raw": identification.judge_raw,
        },
    }, indent=2, default=str))

    # Augment the manifest with audit-specific fields
    from personascope.core.manifest import build_manifest, write_manifest
    audit_extra = {
        "audit_case": "unknown",
        "induction_threshold": induction_threshold,
        "persona_for_icl": persona_for_icl,
    }
    manifest = build_manifest(
        cell={
            "persona": summary["persona"],
            "persona_label": summary["persona_label"],
            "model": summary["model"],
            "k": summary["k"],
            "system_prompt": summary["system_prompt"],
            "cell_mode": summary["cell_mode"],
        },
        n_samples=summary["n_samples"],
        seed=summary["seed"],
        model_provider_name=model,
        judge_provider_name=judge_provider_name,
        probes_run=summary["probes_run"] + list(open_summaries.keys()),
        extra=audit_extra,
    )
    write_manifest(manifest, out_dir / "manifest.json")

    # Overwrite the per-cell report card with the unknown-flavour variant —
    # `run_full_battery` already wrote the known-flavour card; here we
    # rebuild it with the blind-audit verdict on top.
    from personascope.experiments.report_card import write_report_card
    try:
        write_report_card(summary, out_dir, blind_audit_result=result)
    except Exception as e:  # noqa: BLE001 — card is non-essential
        print(f"[audit_unknown] report_card render failed (non-fatal): {e}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _run_open_mode_probes(
    *, model: str, k: int, system_prompt: Optional[str],
    persona_for_icl: Optional[str],
    n_samples: int, seed: int, out_dir: Path, judge_provider_name: str,
) -> dict[str, Any]:
    """Run the open-mode persona-keyed probes for case 3.

    These don't need a persona_label and produce free-text outputs that
    `persona_identifier` consumes. Writes JSONLs to out_dir like the
    main runner; returns summary dicts keyed by probe name.

    `persona_for_icl` is required when `k > 0` — it names which YAWYR
    fact corpus to sample for the ICL context. It is NOT exposed to any
    judge or aggregator; it controls the cell construction only.
    """
    import json
    import numpy as np

    from personascope.core.runner import (
        load_yawyr_facts, sample_icl_context, provider_from_name,
    )
    from personascope.core.schema import Preparation
    from personascope.experiments.compact_panel import (
        _run_probes_n_samples, make_default_judge, resolve_persona,
    )
    from personascope.probes.identity.identification import (
        make_identification_open_battery,
    )
    from personascope.probes.identity.inference_prefill import (
        make_inference_prefill_open_battery,
    )
    from personascope.probes.identity.recognition_jeopardy import (
        make_jeopardy_open_probe,
    )
    from personascope.probes.identity.persona_assistant_relationship import (
        make_persona_assistant_relationship_open_probe,
    )
    from personascope.probes.competence.boundary_capability import (
        make_capability_boundary_open_battery,
    )
    from personascope.probes.context_inference.inference_latent import (
        make_latent_inference_open_battery,
    )
    from personascope.analysis.blind_audit import summarise_open_freetext

    # Build the same induction context the main runner did, so open-mode
    # probes see the SAME conversation history.
    rng = np.random.default_rng(seed)
    if k > 0:
        if persona_for_icl is None:
            raise ValueError(
                "_run_open_mode_probes with k>0 requires persona_for_icl "
                "(public callers should use audit_unknown, which validates this)."
            )
        _, facts_path = resolve_persona(persona_for_icl)
        facts = load_yawyr_facts(facts_path)
        icl_context = sample_icl_context(facts, k, rng)
    else:
        icl_context = []
    if system_prompt:
        icl_context = [{"role": "system", "content": system_prompt}, *icl_context]

    cell_mode = derive_mode(k, system_prompt)
    prep = Preparation(
        formation_route="instruction_tuned_default",
        conditioning_regime=("k_icl" if k > 0
                             else ("system_prompt" if system_prompt else "none")),
        model_id=model,
        system_prompt=system_prompt,
        icl_context=icl_context,
        icl_k=k,
        persona_target=None,
        notes=f"audit_unknown open-mode probes; mode={cell_mode}",
    )
    provider = provider_from_name(model)
    judge_fn = make_default_judge(judge_provider_name)
    cache = None

    summaries: dict[str, Any] = {}

    # 1. inference_prefill_open
    probes = make_inference_prefill_open_battery(n_samples_gen=n_samples)
    recs = _run_probes_n_samples(
        probes, prep, provider, judge_fn, cache,
        n_samples=1, seed_base=seed,
        run_id_prefix=f"audit_unknown:{model}:k{k}:inference_prefill_open",
    )
    out_path = out_dir / "inference_prefill_open.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(r.to_json() + "\n")
    summaries["inference_prefill_open"] = summarise_open_prefill(recs)

    # 2. identification_open
    probes = make_identification_open_battery()
    recs = _run_probes_n_samples(
        probes, prep, provider, judge_fn, cache,
        n_samples=n_samples, seed_base=seed,
        run_id_prefix=f"audit_unknown:{model}:k{k}:identification_open",
    )
    out_path = out_dir / "identification_open.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(r.to_json() + "\n")
    summaries["identification_open"] = summarise_open_identification(recs)

    # 3. jeopardy_open
    probes = [make_jeopardy_open_probe()]
    recs = _run_probes_n_samples(
        probes, prep, provider, judge_fn, cache,
        n_samples=n_samples, seed_base=seed,
        run_id_prefix=f"audit_unknown:{model}:k{k}:jeopardy_open",
    )
    out_path = out_dir / "jeopardy_open.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(r.to_json() + "\n")
    summaries["jeopardy_open"] = summarise_open_jeopardy(recs)

    # 4. inference_latent_open — what does the model infer about the
    # situation? Strong signal for blind audit (model often discloses
    # the persona it thinks it's playing).
    latent_mode = "icl" if (k > 0 or system_prompt) else "sft"
    probes = make_latent_inference_open_battery(mode=latent_mode)
    recs = _run_probes_n_samples(
        probes, prep, provider, judge_fn, cache,
        n_samples=n_samples, seed_base=seed,
        run_id_prefix=f"audit_unknown:{model}:k{k}:inference_latent_open",
    )
    out_path = out_dir / "inference_latent_open.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(r.to_json() + "\n")
    summaries["inference_latent_open"] = summarise_open_freetext(recs)

    # 5. persona_assistant_relationship_open — "what's your relation to
    # the AI assistant role?". Reveals hierarchy / dual-identity if
    # the cell is persona-induced.
    probes = [make_persona_assistant_relationship_open_probe()]
    recs = _run_probes_n_samples(
        probes, prep, provider, judge_fn, cache,
        n_samples=n_samples, seed_base=seed,
        run_id_prefix=f"audit_unknown:{model}:k{k}:persona_assistant_relationship_open",
    )
    out_path = out_dir / "persona_assistant_relationship_open.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(r.to_json() + "\n")
    summaries["persona_assistant_relationship_open"] = summarise_open_freetext(recs)

    # 6. boundary_capability_open — modern-knowledge questions without a
    # target. A contemporary AI answers them; a persona-induced model
    # may refuse / rationalise / answer in-character, which the persona
    # identifier can surface.
    probes = make_capability_boundary_open_battery()
    recs = _run_probes_n_samples(
        probes, prep, provider, judge_fn, cache,
        n_samples=max(1, n_samples // 2),  # 5 questions × n is expensive
        seed_base=seed,
        run_id_prefix=f"audit_unknown:{model}:k{k}:boundary_capability_open",
    )
    out_path = out_dir / "boundary_capability_open.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(r.to_json() + "\n")
    summaries["boundary_capability_open"] = summarise_open_freetext(recs)

    return summaries


__all__ = [
    "audit_base",
    "audit_known",
    "audit_unknown",
]
