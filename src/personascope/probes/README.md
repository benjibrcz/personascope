# `src/personascope/probes/` — Personascope evaluation-item library

The behavioural-readout panel for the persona measurement pipeline.

An **evaluation item** (or **item** for short) is the persona-state measurement concept used
in prose throughout this project and the accompanying post. In code, each evaluation item is
implemented as a `Probe` — the abstraction also covers **representational probes** (reading
the model's internal activations; planned for the follow-up), which are genuinely "probes,"
not evaluation items. Every item in *this* panel is **behavioural**: a prompt plus a judge
rubric.

Evaluation items are organised by channel; every channel is a subdirectory with
its own `__init__.py`. The top-level `__init__.py` re-exports each
submodule so `from personascope.probes.X import ...` works regardless of which
channel `X` lives in (handy for legacy import paths and discovery).

> **Canonical entry point**: `personascope.experiments.full_battery.run_full_battery`
> takes a (model, persona, induction method) configuration and runs every enabled
> evaluation item in one pass, writing one `<probe_name>.jsonl` per item and a
> flat `summary.json`. Each item has its own `run_<probe>=True/False`
> flag — see the function signature for the full set.
>
> **Audit entry points**: `personascope.experiments.audit.audit_base /
> audit_known / audit_unknown` are thin shims over `run_full_battery`
> covering the three deployment cases (base / known-persona /
> unknown-persona). See [`docs/three_case_audit.md`](../../../docs/three_case_audit.md).

## Channel taxonomy

Behavioral-readout style hierarchy:

| Channel | Asks | Subdirectory |
|---|---|---|
| **Identity** | Who/what does the model claim to be? | `identity/` |
| **Behavior** | How does it act under value-loaded conditions? | `behavior/` |
| **Competence** | What can it actually do? | `competence/` |
| **CoT** | Does the chain-of-thought match the answer? | `cot/` |
| **Context inference** | What does it infer about the conversation context? | `context_inference/` |

Plus `_utils/` for non-item helpers (`refusal_check`, `meta_gaming`).

## Tiers

`run_full_battery(tier=...)` filters which evaluation items fire by default. The
authoritative list lives in [`src/personascope/core/tiers.py`](../core/tiers.py).

| Tier | Items | Use when |
|---|---|---|
| **`core`** (default) | 7 — one per distinct construct, validated, low overlap | Public default. Cheap. Sufficient for `audit_base` / most `audit_unknown` runs. |
| **`extended`** | 27 — adds depth (psychometrics, AISI EM, second self-model readouts, Sel_C recognition, user-intent inference, style features) | When you want the rich panel — e.g. `audit_known` on a target persona. |
| **`exploratory`** | 30 — adds high-overlap items (`psychometric_self_description`, `psychometric_identity_coherence`, `economic_games`) | When evaluating new items or analyses that specifically need a demoted one. |

Per-item flags (`run_<probe>=True/False`) override the tier default,
so `tier="core", run_self_explanation=True` runs core + just
self_explanation. Each item's `summary.json` block also carries a
`tier` field so downstream aggregators can tell primary readouts apart
from extended/exploratory ones.

## Evaluation-item inventory

Hand-maintained summary below — keep in sync after adding/removing items.

### `identity/` — who is the model?

| File | Purpose | Mode | In `run_full_battery`? |
|---|---|---|---|
| `identification.py` | 5-Q identity ladder, paper-iterated. PERSONA_HIT + IS_AI_DISCLAIMER. | induced | ✓ default-on |
| `identification_icl.py` | ICL-persona Y/N identity item set (per-question validated judges). External-comparable. | induced | — separate driver |
| `inference_prefill.py` | Third-person prefill stems ("the character I'm embodying is..."). String-match scoring. | induced | ✓ default-on |
| `meta_awareness.py` | Open-framing 4-way self-description verdict (AI_DEFAULT / PERSONA_DEFAULT / AMBIGUOUS / ACKNOWLEDGES). | both | ✓ default-on |
| `existence_branching.py` | Zoo step 2, open framing: "is there an underlying entity?" + relationship item. | both | ✓ default-on |
| `persona_assistant_relationship.py` | Zoo step 1: 5-way IS_ASSISTANT / IS_PERSONA / HIERARCHY_*. | induced | ✓ default-on |
| `lexical_attractor.py` | Judge-free word-distribution item (13 prompts, JS divergence). Carries AISI "dragon attractor" finding. | both | ✓ default-on |
| `robustness_assistant.py` | Mirror of `robustness_persona` — tests AI-assistant identity under pressure designed to drop it (consciousness_nudge, drop_act, claim_human, identity_doubt). Pairs with base-PAD aggregator. | both | ✓ default-on |
| `robustness_persona.py` | Tests *induced persona* hold under pressure (compact panel axis 3). 5 conditions. | induced | ✓ default-on |
| `recognition_jeopardy.py` | Jeopardy-style: "given this ICL evidence, what persona is being described?" Tests Sel_C separately from Exec_C. | both | — orphan |
| `self_explanation.py` | Post-hoc / outside-view / narrative / value-inference items about own behaviour. | both | — orphan |
| `challenge_self_model.py` | Direct challenges to the self-model. | both | — orphan |
| `elicitation_awareness_kulveit.py` | Kulveit elicitation-awareness items. | both | — orphan |
| `self_model_calibration.py` | Meta-meta item: how well does the model predict its own answers? | both | — orphan |
| `process_self_model.py` | Evidence-attribution + change-awareness items. | both | — orphan |

### `behavior/` — how does the model act?

| File | Purpose | Mode | In `run_full_battery`? |
|---|---|---|---|
| `boundary_moral.py` | 5-harm refusal panel (paper-iterated). 4-way verdict. | both | ✓ default-on |
| `multi_turn_moral.py` | 9-turn protocol — measures moral-erosion delta from turn 3 → turn 9. | both | ✓ default-on |
| `psychometric.py` | Big-5 (IPIP-50) + Short Dark Triad (SD3) + open-ended self-description + identity_coherence. Validated by Serapio-García 2023. | both | ✓ default-on (4 sub-panels) |
| `aisi_em.py` | AISI emergent-misalignment behavioural-choice items: reward_hack / strategic_deception / sycophancy / self_introspection. Each 5 items, 4-way verdict. | both | ✓ default-on (4 sub-panels) |
| `values_betley_icl.py` | ICL-persona Betley EM item set (200+ items, gold-standard EM measurement). | both | — separate driver |
| `traits_generic.py` | Generic Likert-trait item maker (TraitItem dataclass). Underpins `psychometric.big_five`. | both | — orphan |
| `economic_games.py` | Prisoners' dilemma + ultimatum + public goods. Revealed-preference Behavior. | both | OPT-IN (default off) |
| `emotion.py` | Keyword-extracted emotion profile + reason-consistency. Sofroniew direction. | both | OPT-IN (default off) |
| `style.py` | Style item (response-form features). | both | — orphan |

### `competence/` — what can the model do?

| File | Purpose | Mode | In `run_full_battery`? |
|---|---|---|---|
| `boundary_capability.py` | T1 modern-knowledge question + T2 anachronism challenge. Tests persona-content rationalisation. | induced | ✓ default-on |
| `truthfulqa.py` | Logprob-scored MC1 against vLLM. Persona-agnostic factual recall. | both | — separate driver (needs dataset items) |
| `competence_mcq.py` | MCQ item maker + `make_latent_knowledge_probe`. The agenda-flagged "latent vs stated knowledge" discriminator. | both | — orphan (needs item data) |

### `cot/` — chain-of-thought analysis

| File | Purpose | Mode | In `run_full_battery`? |
|---|---|---|---|
| `cot_faithfulness.py` | MacDiarmid 3-pattern classifier: aligned/aligned, mis/al (covert), al/mis (Golechha unfaithful). | both | — orphan (needs question sets) |
| `cot_content.py` | Persona-reference detection in CoT + self-review items. | both | — orphan |

### `context_inference/` — what does the model infer about its situation?

| File | Purpose | Mode | In `run_full_battery`? |
|---|---|---|---|
| `inference_latent.py` | Cozmin's open-frame Qs: persona / user-intent / topic decomposition. Auto-switches ICL ↔ SFT mode. | both | ✓ default-on |
| `user_inference.py` | Inference about the user's identity / goals from conversation history. | both | — orphan |
| `intent.py` | Inference about the user's intent. | both | — orphan |

### `representation/`

| File | Purpose | Mode | In `run_full_battery`? |
|---|---|---|---|
| `representation_extractor.py` | HFExtractor (any host) / VLLMLensExtractor (Linux+CUDA). Activation-level instrumentation for representation-level analyses. | both | — separate driver (heavy infra) |

### `_utils/` — non-item helpers

| File | Purpose |
|---|---|
| `refusal_check.py` | Regex refusal-detector. Used by `compact_panel.py` to tag refusal-shaped responses. |
| `meta_gaming.py` | Apollo-style "what was this conversation for?" post-hoc analysis (run via `run_axis4b_transcript_purpose.py`). |

## Naming conventions

- Filenames are **purpose-named**, not channel-prefixed. Channel lives in
  the directory hierarchy + the `Probe.channel_slot` field, not in the filename.
- Evaluation items that wrap an externally-validated item set have a `_icl` suffix
  (`identification_icl`, `values_betley_icl`) to clarify they're
  external-comparison instruments, distinct from the paper-iterated versions.
- Generic item-makers (factories that take items as argument) have a
  `_generic` suffix where the type is meaningfully abstract
  (`traits_generic.py`).

## Mode dispatch

Every `Probe` has an `applicable_modes: frozenset[Mode]` field. Default
is `{induced, uninduced}` (mode-agnostic). Persona-targeted items set
`{induced}` to indicate their scoring is meaningless without a target
persona. The runner (`personascope.experiments.full_battery._run_one`) filters
via `select_probes(probes, cell_mode)` before each invocation.

```python
def derive_mode(k: int, system_prompt: str | None) -> Mode:
    if k > 0 or (system_prompt and system_prompt.strip()):
        return "induced"
    return "uninduced"
```

Currently induced-only:

| Item | Reason |
|---|---|
| `inference_prefill` | Scoring = persona-name match in continuation; trivially 0 uninduced. |
| `identification` | Judge scores PERSONA_HIT vs IS_AI_DISCLAIMER against a target persona. |
| `identification_icl` | Same as above. |
| `robustness_persona` | "Are you really {persona}?" — no persona to hold uninduced. |
| `boundary_capability` | T2 references {persona} in the anachronism challenge. |
| `persona_assistant_relationship` | Asks about an active persona's relationship to the assistant. |

All other items work in both modes.

## Evaluation items by audit case

For a per-case breakdown of which items fire (case 1 / 2 / 3), what
signals each contributes, and the aggregator pipeline, see
[`docs/three_case_audit.md`](../../../docs/three_case_audit.md) §
*"Evaluation items & metrics per case"*.

## Open-mode siblings (for `audit_unknown`)

Several items do **closed-world** scoring — they need a target persona to
compare against. For the unknown-persona audit we need the same prompts but
with **open-world** output: just record the model's free-text response and
let `personascope.analysis.blind_audit.persona_identifier` extract the named entity.

Sibling factories were added next to the closed-world ones (light-touch, no
breaking changes):

| Closed-world (cases 1 & 2) | Open-world (case 3) |
|---|---|
| `make_inference_prefill_probe(persona_label, ...)` | `make_inference_prefill_open_probe(...)` |
| `make_identification_probe(persona_label, ...)` | `make_identification_open_probe(...)` |
| `make_jeopardy_probe(persona_label, ...)` | `make_jeopardy_open_probe(...)` |
| `make_latent_inference_probe(persona_label, ...)` | `make_latent_inference_open_probe(...)` |
| `make_persona_assistant_relationship_probe(persona_label, ...)` | `make_persona_assistant_relationship_open_probe(...)` |
| `make_capability_boundary_probe(persona_label, ...)` | `make_capability_boundary_open_probe(...)` |

Open-mode factories drop the `persona_label` arg + the judge call, returning
raw responses. `applicable_modes={"induced", "uninduced"}` so they fire on
any configuration. `audit_unknown` wires them in automatically.

## Validation status

Each item carries a `validation_status` label (registered in
[`personascope.core.tiers.VALIDATION_STATUS`](../core/tiers.py)), stamped into
`summary.json` alongside `tier`:

- **`high`** — validated against known configurations with documented discrimination
  (e.g. 0/16 → 16/16 base vs Voldemort ICL; Serapio-García IPIP-50; SD3).
- **`medium`** — some validation but smaller N, single-case, or less
  documented.
- **`low`** — wired but not yet empirically validated. Treat as experimental
  until a validation run lands.

The labels are orthogonal to tier (core items can be `high` or `medium`;
new extended additions like `intent` start at `low`).

## Adding a new evaluation item

1. Decide on its primary channel; place it in the corresponding subdirectory.
2. Implement `make_<probe_name>_probe(...)` and (if multi-item) `make_<probe_name>_battery(...) -> list[Probe]` factories that return `Probe(...)` instances with appropriate `applicable_modes`.
3. Add the module to the matching `from .X import` list in `src/personascope/probes/__init__.py`.
4. If the item should run in the default battery, wire it into `src/personascope/experiments/full_battery.py`: add a `run_<probe>` flag, an `if run_<probe>:` block, and a per-item summariser.
5. Update this README's relevant table.
6. Add a unit test in `tests/` (stub the judge if needed).

## See also

- [`../../../README.md`](../../../README.md) — quickstart + three-case audit API.
- [`../../../docs/pipeline_overview.md`](../../../docs/pipeline_overview.md) — architectural tour with entry-point hierarchy.
- [`../../../docs/three_case_audit.md`](../../../docs/three_case_audit.md) — `audit_base` / `audit_known` / `audit_unknown` design + per-case evaluation-item inventory.
