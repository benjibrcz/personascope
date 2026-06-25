"""`personascope` console-script — discovery + audit dispatch.

Commands
--------
- `personascope list-probes`        — print every probe factory by category
- `personascope list-batteries`     — print loaded batteries (values, identity, MCQ)
- `personascope audit-base`         — characterise a model's default persona
- `personascope audit-known`        — audit a model with a known induced persona
- `personascope audit-unknown`      — blind audit: detect + identify any induced persona
- `personascope run-full-battery`   — single-configuration × all-default-probes run

All commands are thin wrappers over the Python API in `personascope.experiments`;
researchers wanting full control should import and call those functions
directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Discovery commands
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_list_probes(_args: list[str]) -> int:
    from personascope.probes.behavior import (
        boundary_moral,
        multi_turn_moral,
        style,
        traits_generic,
    )
    from personascope.probes.behavior.external import (
        aisi_em,
        economic_games,
        emotion,
        psychometric,
        values_betley_icl,
    )
    from personascope.probes.competence import boundary_capability
    from personascope.probes.competence.external import competence_mcq, truthfulqa
    from personascope.probes.context_inference import inference_latent, intent, user_inference
    from personascope.probes.cot import cot_content
    from personascope.probes.cot.external import cot_faithfulness
    from personascope.probes.identity import (
        challenge_self_model,
        existence_branching,
        identification,
        inference_prefill,
        lexical_attractor,
        meta_awareness,
        persona_assistant_relationship,
        process_self_model,
        recognition_jeopardy,
        robustness_assistant,
        robustness_persona,
        self_explanation,
        self_model_calibration,
    )
    from personascope.probes.identity.external import (
        elicitation_awareness_kulveit,
        identification_icl,
    )

    categories = [
        ("identity (ours)", [
            identification, inference_prefill, meta_awareness, existence_branching,
            persona_assistant_relationship, lexical_attractor, robustness_assistant,
            robustness_persona, recognition_jeopardy, self_explanation,
            challenge_self_model, self_model_calibration, process_self_model,
        ]),
        ("identity (external)", [identification_icl, elicitation_awareness_kulveit]),
        ("behavior (ours)", [boundary_moral, multi_turn_moral, traits_generic, style]),
        ("behavior (external)", [
            psychometric, aisi_em, values_betley_icl, emotion, economic_games,
        ]),
        ("competence (ours)", [boundary_capability]),
        ("competence (external)", [competence_mcq, truthfulqa]),
        ("cot (ours)", [cot_content]),
        ("cot (external)", [cot_faithfulness]),
        ("context_inference (ours)", [inference_latent, intent, user_inference]),
    ]
    for header, mods in categories:
        factories: list[tuple[str, str]] = []
        for mod in mods:
            short = mod.__name__.rsplit(".", 1)[-1]
            for name in getattr(mod, "__all__", []):
                if name.startswith("make_"):
                    factories.append((short, name))
        if factories:
            print(f"\n=== {header} ===")
            for short, name in factories:
                print(f"  {short}.{name}")
    return 0


def _cmd_list_batteries(_args: list[str]) -> int:
    from personascope.probes.behavior.external import values_betley_icl as V
    print("=== Values batteries (Betley + ICL persona categories) ===")
    for name, path in V.ICL_VALUES_BATTERIES.items():
        bat = V.load_values_battery(name)
        print(f"  {name:18s}  {len(bat.questions):>3} questions   ({path})")

    from personascope.probes.identity.external import identification_icl as P1
    print("\n=== Identity batteries (ICL personas) ===")
    root = Path(P1.__file__).resolve().parents[3] / "data" / "icl_personas" / "evaluation" / "identity"
    if root.exists():
        for p in sorted(root.glob("*.yaml")):
            print(f"  {p.stem}")

    from personascope.probes.competence.external import competence_mcq as G
    print(f"\n=== Competence MCQ mini-set ({len(G.MINI_MCQ_ITEMS)} items) ===")
    for it in G.MINI_MCQ_ITEMS:
        print(f"  {it['id']:20s}  {it['question']}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Audit commands — argparse wrappers over personascope.experiments.audit
# ─────────────────────────────────────────────────────────────────────────────


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    """Args shared by every run-something command."""
    p.add_argument("--model", required=True,
                   help="Provider name (see `personascope.llm.provider.PROVIDERS`).")
    p.add_argument("--out", required=True,
                   help="Output directory for JSONLs + summary.json + manifest.json.")
    p.add_argument("--n-samples", "--n", dest="n_samples", type=int, default=8,
                   help="Samples per probe (default: 8).")
    p.add_argument("--judge", dest="judge_provider_name", default="openai",
                   help="Judge provider name (default: openai).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tier", default="core",
                   choices=["core", "extended", "exploratory"],
                   help="Probe tier (default: core).")


def _cmd_audit_base(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="personascope audit-base",
        description="Characterise a model's default (uninduced) persona.",
    )
    _add_common_run_args(p)
    args = p.parse_args(argv)

    from personascope.experiments.audit import audit_base
    summary = audit_base(
        model=args.model, out_dir=Path(args.out),
        n_samples=args.n_samples, judge_provider_name=args.judge_provider_name,
        seed=args.seed, tier=args.tier,
    )
    print(f"\naudit-base complete. probes_run={summary['probes_run']}")
    print(f"summary → {Path(args.out) / 'summary.json'}")
    return 0


def _cmd_audit_known(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="personascope audit-known",
        description="Audit a model with a known induced persona.",
    )
    p.add_argument("--persona", required=True,
                   help="Persona key (e.g. 'voldemort', 'hitler'). "
                        "See `personascope list-batteries` for available personas.")
    p.add_argument("--induction-route", default="icl_k32",
                   help="One of: icl_k32, icl_k4, system, custom (default: icl_k32).")
    p.add_argument("--k", type=int, default=None,
                   help="Override ICL k (only with --induction-route custom).")
    p.add_argument("--system-prompt", default=None,
                   help="System prompt (only with --induction-route system|custom).")
    _add_common_run_args(p)
    args = p.parse_args(argv)

    from personascope.experiments.audit import audit_known
    summary = audit_known(
        model=args.model, persona=args.persona, out_dir=Path(args.out),
        induction_route=args.induction_route,
        k=args.k, system_prompt=args.system_prompt,
        n_samples=args.n_samples, judge_provider_name=args.judge_provider_name,
        seed=args.seed, tier=args.tier,
    )
    print(f"\naudit-known complete. probes_run={summary['probes_run']}")
    print(f"summary → {Path(args.out) / 'summary.json'}")
    return 0


def _cmd_audit_unknown(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="personascope audit-unknown",
        description=(
            "Blind audit: detect whether a model is persona-induced, and if so, who. "
            "Default mode (k=0) is the true blind audit. To validate the detector "
            "against a known-induced configuration, pass --k and --persona-for-icl."
        ),
    )
    p.add_argument("--k", type=int, default=0,
                   help="ICL k. >0 = detector-validation mode; requires --persona-for-icl.")
    p.add_argument("--persona-for-icl", default=None,
                   help="Persona corpus for ICL construction when k>0. "
                        "NOT exposed to the detector or persona_identifier.")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--threshold", dest="induction_threshold",
                   type=float, default=0.5,
                   help="Induction detector threshold (default: 0.5).")
    _add_common_run_args(p)
    args = p.parse_args(argv)

    from personascope.experiments.audit import audit_unknown
    result = audit_unknown(
        model=args.model, out_dir=Path(args.out),
        k=args.k, system_prompt=args.system_prompt,
        persona_for_icl=args.persona_for_icl,
        n_samples=args.n_samples, judge_provider_name=args.judge_provider_name,
        seed=args.seed, tier=args.tier,
        induction_threshold=args.induction_threshold,
    )
    print()
    print("=" * 60)
    print("BlindAuditResult")
    print("=" * 60)
    print(f"induced     : {result.induced}")
    print(f"persona     : {result.persona}")
    print(f"confidence  : {result.confidence:.3f}")
    print(f"\nFull result: {Path(args.out) / 'audit_unknown.json'}")
    return 0


def _cmd_run_full_battery(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="personascope run-full-battery",
        description="Run the per-configuration probe battery directly (no audit aggregators).",
    )
    p.add_argument("--persona", required=True,
                   help="Persona key (e.g. 'voldemort'). For uninduced configurations use any "
                        "placeholder — persona-keyed probes auto-skip on k=0.")
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the probe plan without making API calls.")
    _add_common_run_args(p)
    args = p.parse_args(argv)

    from personascope.experiments.full_battery import run_full_battery
    summary = run_full_battery(
        persona=args.persona, model=args.model, out_dir=Path(args.out),
        k=args.k, system_prompt=args.system_prompt,
        n_samples=args.n_samples, judge_provider_name=args.judge_provider_name,
        seed=args.seed, tier=args.tier, dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"\nDry-run plan ({len(summary['probes_planned'])} probes):")
        for p_ in sorted(summary["probes_planned"]):
            print(f"  {p_}")
    else:
        print(f"\nrun-full-battery complete. probes_run={summary['probes_run']}")
        print(f"summary → {Path(args.out) / 'summary.json'}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────


_BUILTINS = {
    "list-probes":      _cmd_list_probes,
    "list-batteries":   _cmd_list_batteries,
    "audit-base":       _cmd_audit_base,
    "audit-known":      _cmd_audit_known,
    "audit-unknown":    _cmd_audit_unknown,
    "run-full-battery": _cmd_run_full_battery,
}


def _print_help() -> None:
    print("personascope — measure how deeply LLMs adopt induced personas\n")
    print("Usage: personascope <command> [args...]\n")
    print("Discovery commands:")
    print("  list-probes        List every probe factory by category")
    print("  list-batteries     List values / identity / MCQ batteries\n")
    print("Audit commands (see `personascope <command> --help` for args):")
    print("  audit-base         Characterise a model's default persona")
    print("  audit-known        Audit with a known induced persona")
    print("  audit-unknown      Blind audit — detect + identify any persona")
    print("  run-full-battery   Single-configuration × all-default-probes run\n")
    print("Each audit command is a thin wrapper over the Python API in")
    print("`personascope.experiments.audit` and `personascope.experiments.full_battery` —")
    print("import those directly for fine-grained control.")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in _BUILTINS:
        return _BUILTINS[cmd](rest)
    print(f"personascope: unknown command {cmd!r}. Known: {', '.join(sorted(_BUILTINS))}",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
