"""Probe tiering — what `run_full_battery` runs at each level.

Three tiers, conservative → permissive:

- **core**: one probe per distinct construct. Validated discriminators,
  cheap to run, low correlation between probes. The default for
  `run_full_battery` and the audit entry points. ~7 probes.
- **extended**: adds depth — second readouts for already-covered
  constructs (e.g. inference_prefill as a second persona-content
  readout) plus the psychometric / behavioural / EM batteries.
  ~16 additional probes.
- **exploratory**: everything wired into the runner, including
  newer / less-validated probes and zoo-step variants.

A probe in `core` is also in `extended` and `exploratory` (tiers are
nested / monotone). Use `tier_default_for_probe(tier, probe_name)` to
ask "should this probe fire at this tier?".

The categorisation is intentionally easy to revise — edit `TIER_PROBES`.
"""
from __future__ import annotations

from typing import Literal


Tier = Literal["core", "extended", "exploratory"]


# Single source of truth. Each tier names the *additional* probes it
# enables over the previous tier — the actual set for tier T is the
# union of `core` through `T`.
TIER_PROBES: dict[Tier, set[str]] = {
    "core": {
        # One representative per distinct construct.
        "identification",                    # identity voice (induced)
        "meta_awareness",                    # in-role awareness (both modes)
        "self_explanation",                  # 3-framing self-model (both); feeds 3 of audit detector's 7 signals
        "robustness_assistant",              # base-PAD AI-hold (both)
        "robustness_persona",              # persona-hold under pressure (induced)
        "boundary_moral",                    # value-laden behaviour (both)
        "boundary_capability",               # capability + anachronism (induced)
    },
    "extended": {
        # Adds depth: second readouts + psychometric / EM / context.
        "inference_prefill",                 # second persona-content readout
        "process_self_model",                # 3-framing process self-model (feeds 3 of detector's 7)
        "lexical_attractor",                 # judge-free distribution shape (both); n=32 hardcode → expensive
        "multi_turn_moral",                  # moral-erosion across 9 turns
        "psychometric_big_five",             # IPIP-50
        "psychometric_dark_triad",           # SD3
        "aisi_em_reward_hack",
        "aisi_em_strategic_deception",
        "aisi_em_sycophancy",
        "aisi_em_self_introspection",
        "betley_em",                         # YAWYR Betley alignment battery (0–100 rating judge)
        "moral_choices",                     # YAWYR Moral_Choices battery (same rubric, narrower domain)
        "inference_latent",                  # context inference (target-aware; auto-disabled by audit_unknown)
        "intent",                            # test-vs-deployment + stakes + norm-inference (eval-awareness)
        "existence_branching",               # zoo step 2 — "is there an underlying entity?"
        "persona_assistant_relationship",    # zoo step 1 — IS_ASSISTANT / IS_PERSONA / HIERARCHY_*
        "recognition_jeopardy",              # Sel_C (recognition) decoupled from Ind_A (adoption); induced-only
        "challenge_self_model",              # Exec_C re-execution stability under "are you sure?" challenge
        "user_inference",                    # does the model recognise the user is trying to induce a persona?
        "style",                             # judge-free lexical features (response length, hedges, formality, …)
    },
    "exploratory": {
        # Demoted from extended in the probe-audit-v2 pass — high overlap
        # with other probes or weak independent signal. Available via
        # explicit run_<probe>=True or tier="exploratory".
        "psychometric_self_description",     # overlap with Big5 + identity_coherence
        "psychometric_identity_coherence",   # overlap with robustness_assistant + existence_branching
        "economic_games",                    # only UG discriminates; PD/PG floor-saturate
    },
}


def _resolved_tier_set(tier: Tier) -> set[str]:
    """Return all probe names enabled at `tier` (cumulative)."""
    if tier == "core":
        return set(TIER_PROBES["core"])
    if tier == "extended":
        return TIER_PROBES["core"] | TIER_PROBES["extended"]
    if tier == "exploratory":
        return (
            TIER_PROBES["core"]
            | TIER_PROBES["extended"]
            | TIER_PROBES["exploratory"]
        )
    raise ValueError(f"unknown tier {tier!r} (expected core / extended / exploratory)")


def tier_default_for_probe(tier: Tier, probe_name: str) -> bool:
    """True if `probe_name` should default-fire when running at `tier`."""
    return probe_name in _resolved_tier_set(tier)


def tier_for_probe(probe_name: str) -> Tier | None:
    """Return the tier a probe belongs to (None if unknown).

    Used to stamp `tier` on each probe's summary block so downstream
    code can tell core readouts apart from extended ones when
    aggregating.
    """
    for tier in ("core", "extended", "exploratory"):
        if probe_name in TIER_PROBES[tier]:  # type: ignore[index]
            return tier  # type: ignore[return-value]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation status — separate axis from tier.
#
# - "high":   validated against known induced/uninduced cells with documented
#             discrimination (e.g. orphan-promotion n=16 base vs Voldemort ICL,
#             or external-benchmark validation like Serapio-García IPIP-50).
# - "medium": some validation but smaller N, single-case, or less documented.
# - "low":    in repo but not yet empirically validated. Should be treated as
#             experimental until a validation run lands.
#
# Probes not listed here resolve to None — interpret as "validation status
# unknown" (e.g. probes not wired into any tier yet).
# ─────────────────────────────────────────────────────────────────────────────


ValidationStatus = Literal["high", "medium", "low"]


VALIDATION_STATUS: dict[str, ValidationStatus] = {
    # high — multi-cell validation with documented discrimination
    "identification":                    "high",
    "inference_prefill":                 "high",
    "meta_awareness":                    "high",
    "robustness_persona":              "high",
    "robustness_assistant":              "high",
    "self_explanation":                  "high",
    # process_self_model validated 0/16 → 16/16 on GPT-4.1 base vs
    # Voldemort ICL. `influence_detection` sub-probe judge was tightened
    # after gpt-4o-mini false-positives (it now requires the model to
    # name a specific influence present in THIS conversation, not
    # hypothetical discussion). Kept at "medium" until re-validated
    # across multiple model families.
    "process_self_model":                "medium",
    "lexical_attractor":                 "high",
    "boundary_moral":                    "high",
    "aisi_em_reward_hack":               "high",
    "aisi_em_strategic_deception":       "high",
    "aisi_em_sycophancy":                "high",
    "aisi_em_self_introspection":        "high",
    "betley_em":                         "high",   # Betley et al. 2025
    "moral_choices":                     "medium", # YAWYR moral-dilemma battery, same rubric as Betley
    "psychometric_big_five":             "high",   # Serapio-García 2023
    "psychometric_dark_triad":           "high",   # SD3 standard
    # medium — some validation, smaller N or single-case
    "boundary_capability":               "medium",
    "multi_turn_moral":                  "medium",
    "existence_branching":               "medium",
    "persona_assistant_relationship":    "medium",
    "psychometric_identity_coherence":   "medium",
    "psychometric_self_description":     "medium",
    "economic_games":                    "medium",
    "inference_latent":                  "medium",
    "recognition_jeopardy":              "medium",
    # low — wired but not yet validated empirically
    "intent":                            "low",    # newly promoted (eval-awareness)
    "challenge_self_model":              "low",    # newly promoted (Exec_C re-execution)
    "user_inference":                    "low",    # newly promoted (user-intent inference)
    "style":                             "low",    # newly promoted (judge-free lexical features)
    # open-mode siblings (used automatically by audit_unknown) — inherit
    # their closed-world parent's status, since they share the prompt and
    # only differ in dropping the target-aware judge.
    "inference_prefill_open":            "high",   # parent: inference_prefill (high)
    "identification_open":               "high",   # parent: identification (high)
    "jeopardy_open":                     "medium", # parent: recognition_jeopardy (orphan; conservative)
    "inference_latent_open":             "medium", # parent: inference_latent (medium)
    "persona_assistant_relationship_open": "medium",  # parent: persona_assistant_relationship (medium)
    "boundary_capability_open":          "medium", # parent: boundary_capability (medium)
}


def validation_status_for(probe_name: str) -> ValidationStatus | None:
    """Return the validation-status label for a probe (None if unknown)."""
    return VALIDATION_STATUS.get(probe_name)


__all__ = [
    "Tier",
    "TIER_PROBES",
    "tier_default_for_probe",
    "tier_for_probe",
    "ValidationStatus",
    "VALIDATION_STATUS",
    "validation_status_for",
]
