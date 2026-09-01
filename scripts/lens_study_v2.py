"""Pre-registered representation study — pod-side driver (docs/repr_preregistration.md).

    integration-test   live capture contract check on 5 prompts → token_policy.json
    fit                Phase A: direction on DIRECTION_FIT × CONTRAST_PAIRS → directions/sycophancy.{npy,json}
    freeze-layer       Phase B: behaviour-blind layer freeze on LAYER_VALIDATION → frozen_layer.json
    confirm            Phase C: 16 system-prompt cells × 60 blocks, atomic records → confirmation_report.json
    judge-agreement    second judge on a stratified 25% of confirmation responses → judge_agreement.json
    all                fit → freeze-layer → confirm (+ judge-agreement in --dry-run)

    --dry-run          fake engine + fake judge (exercises the whole join offline; no pod, no API)

Run on the pod with personascope src on PYTHONPATH and a vLLM-Lens serve up:
    PYTHONPATH=/workspace/personascope/src python scripts/lens_study_v2.py --out /workspace/study_v3 all
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

N_LAYERS = 32
INTEGRATION_PROMPTS = [
    "Is 7 × 8 equal to 54?", "What is the capital of Australia?", "Explain why the sky is blue in one sentence.",
    "Name three prime numbers.", "Is water H2O2?",
]


def _policy(args):
    from personascope.repr.vllm_lens_provider import TokenPositionPolicy
    tp = Path(args.out) / "token_policy.json"
    if tp.exists():
        return TokenPositionPolicy(decode_steps_offset=int(json.loads(tp.read_text())["decode_steps_offset"]))
    return TokenPositionPolicy(decode_steps_offset=args.decode_offset)


def _factories(args, policy=None):
    """→ (provider_factory(model_id), judge_fn, second_judge_fn)."""
    if args.dry_run:
        from personascope.repr.fake_client import FakeLensClient, fake_hook_factory, fake_judge_fn
        from personascope.repr.vllm_lens_provider import RepresentationProvider
        client = FakeLensClient(n_layers=6, hidden=16, signal_layer=3, seed=0)

        def pf(model_id, pol=None):
            return RepresentationProvider("fake://", model_id, n_layers=client.n_layers, client=client.for_model(model_id),
                                          hook_factory=fake_hook_factory, policy=pol or policy)
        return pf, fake_judge_fn, fake_judge_fn
    from personascope.experiments.compact_panel import make_default_judge
    from personascope.repr.vllm_lens_provider import RepresentationProvider
    clients = {}

    def client_for(model_id):
        if model_id not in clients:                       # ONE lens client per served model id
            from vllm_lens.client import VLLMLensClient
            clients[model_id] = VLLMLensClient(base_url=args.base_url, model=model_id, timeout=600.0)
        return clients[model_id]

    def pf(model_id, pol=None):
        return RepresentationProvider(args.base_url, model_id, n_layers=N_LAYERS, policy=pol or policy,
                                      model_revision=args.model_revision, client=client_for(model_id))
    return pf, make_default_judge(args.judge), (make_default_judge(args.second_judge) if args.second_judge else None)


def integration_test(args, pf) -> int:
    """Fix the token-position policy from the live engine (prereg §4)."""
    from personascope.repr.vllm_lens_provider import CaptureIntegrityError, TokenPositionPolicy
    for offset in (-1, 0):
        p = pf(args.base_model, TokenPositionPolicy(decode_steps_offset=offset))
        try:
            for k, q in enumerate(INTEGRATION_PROMPTS):
                cap = p.capture([{"role": "user", "content": q}], max_tokens=24, temperature=0.0, seed=k)
                assert cap.text.strip(), "empty text"
                assert cap.pooled.shape == (p.n_layers, cap.pooled.shape[1]), "bad pooled shape"
                assert cap.n_output_tokens == len(cap.output_token_ids) >= 1
        except (CaptureIntegrityError, AssertionError) as e:
            print(f"  offset {offset}: FAIL ({e})")
            continue
        (Path(args.out) / "token_policy.json").write_text(json.dumps(
            {"decode_steps_offset": offset, "verified_on": INTEGRATION_PROMPTS, "n_layers": p.n_layers}, indent=2))
        print(f"  token policy FROZEN: decode_steps_offset={offset} → {Path(args.out) / 'token_policy.json'}")
        return offset
    raise SystemExit("no admissible token-position policy — engine contract changed; ABORT session")


def cmd_fit(args, pf, cfg):
    from personascope.repr.study import phase_a_fit_direction
    d, prov, _, _ = phase_a_fit_direction(pf, cfg)
    sh = prov["split_half"]["per_layer_mean_cosine"]
    print(f"  direction {d.shape} sha={prov['direction_sha']} n/pole={prov['n_examples_per_pole']} "
          f"split-half cosine max={max(sh):.3f} @L{int(np.argmax(sh))}")
    return d


def _load_direction(cfg):
    from personascope.repr.extract import load_direction_checked
    d, _ = load_direction_checked(cfg.out_dir / "directions" / "sycophancy")
    return d


def cmd_freeze(args, pf, cfg):
    from personascope.repr.study import phase_b_freeze_layer
    d = _load_direction(cfg)
    sel = phase_b_freeze_layer(pf, d, cfg)
    if sel["layer"] is None:
        raise SystemExit("no layer meets the pre-registered freeze rule — STOP (no confirmatory run)")
    print(f"  FROZEN layer {sel['layer']} (sep {sel['standardized_sep'][sel['layer']]:.2f}, "
          f"bootstrap pick {sel['bootstrap']['pick_fraction'][sel['layer']]:.2f})")
    return sel["layer"]


def cmd_confirm(args, pf, judge_fn, cfg):
    from personascope.repr.study import phase_c_confirm
    d = _load_direction(cfg)
    layer = int(json.loads((cfg.out_dir / "frozen_layer.json").read_text())["layer"])
    desc = [{"cell": "oct_adapter", "model": args.descriptive_adapter}] if args.descriptive_adapter else None
    rep = phase_c_confirm(pf, judge_fn, d, layer, cfg, descriptive_cells=desc, verbose=args.verbose)
    if not rep["valid"]:
        print("  E1 NOT RUN:", rep["reason"])
        return rep
    ci = rep["item_bootstrap_ci"]["ci"]
    print(f"  E1 (DESCRIPTIVE): {len(rep['cells'])} cells | Spearman ρ={rep['spearman_rho']:+.3f} "
          f"item-CI{[round(v, 2) for v in ci]} | Pearson r={rep['pearson_r']:+.3f} "
          f"| refusal {rep['refusal_rate']:.3f} judge-fail {rep['judge_failure_rate']:.3f} "
          f"| reportable={rep['reportable']} ({rep.get('reportable_note', 'gate passed')})")
    return rep


def cmd_judge_agreement(args, judge2, cfg):
    from personascope.analysis.repr_confirmatory import judge_agreement_gate
    from personascope.probes.behavior import sycophancy_bank as bank
    from personascope.repr.atomic import load_records
    from personascope.repr.study import make_sycophancy_judge
    if judge2 is None:
        raise SystemExit("--second-judge is required for judge-agreement")
    items = {it["id"]: it for it in bank.CONFIRMATION_ITEMS}
    second = make_sycophancy_judge(judge2)
    rng = np.random.default_rng(cfg.analysis_seed)
    primary, secondary, n_total = [], [], 0
    for c in bank.confirmation_cells():
        recs = [r for r in load_records(cfg.out_dir / "confirm" / c["cell"]) if r.judge_verdict is not None]
        n_total += len(recs)
        take = rng.choice(len(recs), size=math.ceil(0.25 * len(recs)), replace=False) if recs else []
        for i in take:
            r = recs[int(i)]
            v2, _, _ = second(r.response_text, items[r.item_id])
            if v2 is not None:
                primary.append(r.judge_verdict)
                secondary.append(v2)
    gate = judge_agreement_gate(primary, secondary)
    gate.update({"n_total_responses": n_total, "fraction": 0.25, "second_judge": args.second_judge or "fake"})
    (cfg.out_dir / "judge_agreement.json").write_text(json.dumps(gate, indent=2))
    print(f"  judge gate: κ4={gate['kappa_4way']:.3f} κbin={gate['kappa_binary']:.3f} n={gate['n']} pass={gate['pass']}")
    return gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["integration-test", "fit", "freeze-layer", "confirm", "judge-agreement", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model-revision", default="unknown")
    ap.add_argument("--judge", default="openai")
    ap.add_argument("--second-judge", default=None, help="provider name of a different-family judge")
    ap.add_argument("--descriptive-adapter", default=None, help="served adapter id run as a DESCRIPTIVE cell")
    ap.add_argument("--decode-offset", type=int, default=-1)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    from personascope.probes.behavior import sycophancy_bank as bank
    from personascope.repr.study import StudyConfig
    bank.assert_disjoint_item_sets()
    bank.assert_ladder_disjoint_from_contrast()
    print(f"bank {bank.bank_sha()} contrast {bank.contrast_bank_sha()} cells {bank.confirmation_cells_sha()}")
    if args.dry_run:
        args.base_model = "fake-base"
        if args.descriptive_adapter is None:
            args.descriptive_adapter = "fake-oct-adapter"
    policy = _policy(args)
    pf, judge_fn, judge2 = _factories(args, policy)
    cfg = StudyConfig(out_dir=Path(args.out), base_model=args.base_model, model_revision=args.model_revision,
                      seeds=tuple(int(s) for s in args.seeds.split(",")), judge_model=args.judge)
    if args.dry_run:
        cfg.n_perm, cfg.n_boot, cfg.max_tokens, cfg.fit_max_tokens = 500, 100, 24, 12
    cmd = args.command
    if cmd == "integration-test":
        integration_test(args, pf)
        return 0
    if cmd in ("fit", "all"):
        print("[A] fit direction")
        cmd_fit(args, pf, cfg)
    if cmd in ("freeze-layer", "all"):
        print("[B] freeze layer")
        cmd_freeze(args, pf, cfg)
    if cmd in ("confirm", "all"):
        print("[C] confirm")
        cmd_confirm(args, pf, judge_fn, cfg)
    if cmd == "judge-agreement" or (cmd == "all" and args.dry_run):
        print("[J] judge agreement")
        cmd_judge_agreement(args, judge2, cfg)
    print("artifacts in", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
