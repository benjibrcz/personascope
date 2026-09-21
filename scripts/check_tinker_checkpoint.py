#!/usr/bin/env python3
"""The coherence gate for a Tinker LoRA checkpoint, and a smoke test that the
persona landed. Run after scripts/train_tinker_lora.py, with the proxy up.

    personascope tinker-serve &
    python scripts/check_tinker_checkpoint.py --model kimi-k2.6 --persona voldemort
    python scripts/check_tinker_checkpoint.py --model kimi-k2.6 --persona voldemort --base   # the base model, for the comparison

Weird Generalization found DeepSeek V3.1 "producing degenerate and very short
responses" at 3 epochs and retrained at 1. This script measures that: the
eight Betley EM free-form questions x n samples at the grid temperature,
scored by Betley's coherence judge (0-100; < 50 is incoherent, the paper's
exclusion threshold). If the incoherent fraction exceeds the recipe's
`fallback.when.incoherent_fraction_above`, retrain with the fallback epochs:

    python scripts/train_tinker_lora.py --model kimi-k2.6 --personas voldemort --epochs 1

The identification read (five identity questions, judged against the target)
is the smoke test: a checkpoint that is coherent but never claims the persona
did not take the fine-tune. Results go to data/tinker_lora/gate/<model>/<persona>.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from personascope.induction import checkpoint_for, recipe_for  # noqa: E402
from personascope.models import default_temperature, resolve_model  # noqa: E402
from personascope.tinker.coherence import INCOHERENT_BELOW, judge_coherence  # noqa: E402

OUT = REPO / "data" / "tinker_lora" / "gate"


def sample(provider, model_id: str, question: str, n: int, temperature: float, max_tokens: int) -> list[str]:
    provider.config.model = model_id
    res = provider.complete(
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens, temperature=temperature, n=n,
    )
    if not res.get("success", True):
        sys.exit(f"sampling failed: {res.get('error')}")
    return res.get("text_samples") or [res.get("text", "")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--base", action="store_true", help="check the base model instead of the checkpoint")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--judge", default="openai")
    ns = ap.parse_args()

    from personascope.experiments.compact_panel import make_default_judge
    from personascope.probes.behavior.external.values_betley_icl import load_betley_battery
    from personascope.probes.identity.external import identification_icl as ident

    provider, base_id = resolve_model(ns.model)
    model_id = base_id if ns.base else checkpoint_for(ns.persona, ns.variant, model=ns.model)
    temperature = default_temperature()
    judge = make_default_judge(ns.judge)
    recipe = recipe_for(ns.model)
    threshold = (recipe.get("fallback") or {}).get("when", {}).get("incoherent_fraction_above")

    # 1. coherence on the EM free-form questions
    battery = load_betley_battery()
    scores: list[int] = []
    per_q: dict[str, list] = {}
    for q in battery.questions:
        answers = sample(provider, model_id, q["text"], ns.n, temperature, ns.max_tokens)
        per_q[q["id"]] = []
        for a in answers:
            s, raw = judge_coherence(q["text"], a, judge)
            per_q[q["id"]].append({"answer": a, "coherence": s, "judge_raw": raw})
            if s is not None:
                scores.append(s)
    incoherent = sum(s < INCOHERENT_BELOW for s in scores) / max(1, len(scores))
    mean_len = statistics.mean(len(r["answer"].split()) for rs in per_q.values() for r in rs)

    # 2. identification smoke test: the five ICL-persona identity questions,
    # judged against the target, n samples each. The base model's rate is the
    # comparison (run with --base).
    ident_rate = None
    try:
        battery_id = ident.load_battery(ns.persona)
    except FileNotFoundError:
        battery_id = None
    if battery_id is not None:
        provider.config.model = model_id
        hits = 0
        total = 0
        for probe in ident.make_identity_battery_probes(battery_id, gen_temperature=temperature):
            for _ in range(ns.n):
                out_ = probe.run([], provider, judge, None)
                hits += bool(out_["measurement"]["hit"])
                total += 1
        ident_rate = hits / total if total else None

    verdict = "pass"
    if threshold is not None and incoherent > threshold:
        verdict = f"FALLBACK: retrain with num_epochs={recipe['fallback']['num_epochs']}"
    report = {
        "model": ns.model, "persona": ns.persona, "variant": ns.variant, "checkpoint": model_id,
        "base": ns.base, "n_per_question": ns.n, "temperature": temperature,
        "coherence_mean": statistics.mean(scores) if scores else None,
        "incoherent_fraction": incoherent, "threshold": threshold,
        "mean_answer_words": mean_len, "identification_rate": ident_rate,
        "verdict": verdict, "per_question": per_q,
    }
    out = OUT / ns.model / f"{ns.persona}{'.base' if ns.base else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{ns.model} / {ns.persona} / {'base' if ns.base else model_id}")
    print(f"  coherence mean {report['coherence_mean']}, incoherent {incoherent:.2f} "
          f"(threshold {threshold}), mean answer {mean_len:.0f} words")
    if ident_rate is not None:
        print(f"  identification {ident_rate:.2f}")
    print(f"  {verdict}\n  -> {out}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
