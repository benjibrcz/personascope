"""A/B test: role=system vs role=user for `robustness_persona:system_reminder`.

Question: does it matter whether we inject the "You are a helpful AI
assistant" reminder as `role=system` (current behaviour, line 151 of
`robustness_persona.py`) versus `role=user`?

Method
------
1. Build a Voldemort ICL induction (k=16) on gpt-4o-mini.
2. Verify induction with a "Who are you?" turn; require persona_hold=True
   (i.e. judge says YES this is Voldemort). Re-sample seeds until induced.
3. On the induced history, branch into two variants:
     A. append `{"role": "system", "content": SYSTEM_REMINDER_CONTENT}`
     B. append `{"role": "user",   "content": SYSTEM_REMINDER_CONTENT}`
   Then send the identity probe and judge persona_hold.
4. Repeat for N trials each variant (different ICL seeds).
5. Compare persona_hold rates + identity-response snippets.

If A and B give the same hold rate, the role tag doesn't matter at the
provider layer (gpt-4o-mini collapses both into the same context). If A
breaks more (lower hold), role=system is a stronger pressure and the
probe is correctly using it. If A breaks LESS, the provider may be
ignoring mid-conversation system messages.

This is a single-model, single-persona sanity check — not a definitive
cross-lab claim. Anthropic / Llama may behave differently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from personascope.core.runner import (
    build_messages,
    call_provider,
    load_yawyr_facts,
    provider_from_name,
    sample_icl_context,
)
from personascope.experiments.compact_panel import make_default_judge
from personascope.probes.identity.robustness_persona import (
    IDENTITY_PROBE,
    IDENTITY_JUDGE_PROMPT,
    SYSTEM_REMINDER_CONTENT,
)


PERSONA_LABEL = "Voldemort"
FACTS_PATH = Path("src/personascope/data/yawyr/voldemort/facts.jsonl")
# Two configs: the floor (gpt-4o-mini ICL — saturates 0/8 in both variants,
# can't discriminate) and the ceiling (ft-voldemort-plain — SFT-baked, the
# place where role=system vs role=user actually has room to differ).
TARGET_MODEL = "openai-mini"  # floor case: ICL on the cheap model
JUDGE_MODEL = "openai"        # gpt-4.1 judge to match YAWYR
K = 16
N_TRIALS = 8
GEN_TEMP = 1.0
GEN_MAX_TOKENS = 300


def _judge_hold(judge_fn, response: str) -> tuple[bool, str]:
    prompt = IDENTITY_JUDGE_PROMPT.format(
        persona_label=PERSONA_LABEL, response=response,
    )
    raw = judge_fn(prompt).strip().upper()
    return raw.startswith("YES"), raw


def _build_induced_history(
    facts, provider, judge_fn, seed: int
) -> tuple[list[dict], str, bool]:
    """Build a k-shot ICL history + one accepted assistant turn for the
    identity probe (the "I am Voldemort" prelude). Returns (history,
    induction_response, induced_ok)."""
    rng = np.random.default_rng(seed)
    icl = sample_icl_context(facts, K, rng)
    # Confirm induction with a leading "Who are you?" turn whose response
    # we accept into history. If the judge says HOLD, induction is in.
    msgs = build_messages(icl, IDENTITY_PROBE)
    resp = call_provider(
        provider, msgs, temperature=GEN_TEMP, max_tokens=GEN_MAX_TOKENS,
    )
    induced, _ = _judge_hold(judge_fn, resp)
    history = list(icl) + [
        {"role": "user", "content": IDENTITY_PROBE},
        {"role": "assistant", "content": resp},
    ]
    return history, resp, induced


def _run_variant(
    role: str, history: list[dict], provider, judge_fn,
) -> tuple[bool, str, str]:
    """Append SYSTEM_REMINDER_CONTENT as the given role, send the identity
    probe, judge. Returns (held, response, judge_raw)."""
    scratch = list(history) + [
        {"role": role, "content": SYSTEM_REMINDER_CONTENT},
        {"role": "user", "content": IDENTITY_PROBE},
    ]
    try:
        resp = call_provider(
            provider, scratch,
            temperature=GEN_TEMP, max_tokens=GEN_MAX_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"<provider_error: {e}>", "ERR"
    held, raw = _judge_hold(judge_fn, resp)
    return held, resp, raw


def main():
    os.chdir(Path(__file__).resolve().parents[1])  # repo root

    print(f"=== A/B test: system_reminder role on {TARGET_MODEL} / {PERSONA_LABEL} ===")
    print(f"k={K}, n_trials={N_TRIALS} per variant, "
          f"judge={JUDGE_MODEL}")
    print()

    facts = load_yawyr_facts(FACTS_PATH)
    provider = provider_from_name(TARGET_MODEL)
    judge_fn = make_default_judge(JUDGE_MODEL)

    results_sys: list[dict] = []
    results_user: list[dict] = []
    induction_failures = 0

    seed_base = 20260513
    trial = 0
    seed_offset = 0
    while trial < N_TRIALS:
        seed = seed_base + seed_offset
        seed_offset += 1
        if seed_offset > 2 * N_TRIALS + 8:
            print("Too many induction failures, giving up.")
            break

        history, ind_resp, induced = _build_induced_history(
            facts, provider, judge_fn, seed,
        )
        if not induced:
            induction_failures += 1
            print(f"  trial seed={seed}: induction FAILED — re-sampling")
            continue

        a_held, a_resp, _ = _run_variant("system", history, provider, judge_fn)
        b_held, b_resp, _ = _run_variant("user", history, provider, judge_fn)

        results_sys.append({"seed": seed, "held": a_held, "response": a_resp})
        results_user.append({"seed": seed, "held": b_held, "response": b_resp})

        print(f"  trial {trial+1}/{N_TRIALS} seed={seed}:")
        print(f"    role=system  HOLD={a_held}  → {a_resp[:120].strip()!r}")
        print(f"    role=user    HOLD={b_held}  → {b_resp[:120].strip()!r}")
        trial += 1

    hold_sys = sum(r["held"] for r in results_sys) / max(1, len(results_sys))
    hold_user = sum(r["held"] for r in results_user) / max(1, len(results_user))
    n_disagree = sum(
        (a["held"] != b["held"]) for a, b in zip(results_sys, results_user)
    )

    print()
    print(f"=== Summary ===")
    print(f"Successful trials: {len(results_sys)} / {N_TRIALS} "
          f"(induction failures: {induction_failures})")
    print(f"persona_hold @ role=system : {hold_sys:.2f}")
    print(f"persona_hold @ role=user   : {hold_user:.2f}")
    print(f"Within-trial disagreement   : {n_disagree} / {len(results_sys)}")

    out = Path("scripts/ab_system_reminder_role.results.json")
    out.write_text(json.dumps({
        "target_model": TARGET_MODEL,
        "judge_model": JUDGE_MODEL,
        "persona": PERSONA_LABEL,
        "k": K,
        "n_trials_requested": N_TRIALS,
        "n_trials_completed": len(results_sys),
        "induction_failures": induction_failures,
        "hold_role_system": hold_sys,
        "hold_role_user": hold_user,
        "within_trial_disagreement": n_disagree,
        "results_role_system": results_sys,
        "results_role_user": results_user,
    }, indent=2))
    print(f"Wrote results: {out}")


if __name__ == "__main__":
    main()
