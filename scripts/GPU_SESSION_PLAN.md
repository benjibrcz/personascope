# Representation channel — GPU-session plan v2 (post GPT review)

Reshaped after an independent review (all 12 objections accepted; two Criticals
verified against `aggregators.py`/`tiers.py`). The honest scope: this session
delivers a **pre-registered, validated trait-specific probe** (sycophancy), NOT
the cross-cell PAD/VD channel — that needs a bigger cell grid + external axis +
battery-integrated steering (staged as Program items below). Runs gated on HF
misalignment access (still pending → wait).

## Pre-registered CONFIRMATORY experiment (one estimand)
- **Trait:** sycophancy (benign, cheap ladder; VD headline waits for misalignment).
- **Predictor:** a fixed sycophancy **direction** built on an EXTERNAL discovery
  cohort (diverse roles, NOT the eval cells), at a **behavior-blind
  validation-selected layer**, frozen before any behaviour is examined.
- **Outcome:** a **blinded** sycophancy judge score (fixed rubric; judge never
  sees cell identity or activations), aggregated to a **cell mean**.
- **Unit = cell.** Build a within-sycophancy **intervention ladder** for power:
  base, system-prompt mild/strong, ICL-k {4,16,48}, the OCT adapter, and
  **steering at several scales** (cheap continuous dose points) → ~15–20 points.
  Responses are repeated measures, not independent units.
- **Inference:** cell/trait-clustered **permutation test** on the frozen-layer
  projection↔score correlation. Per-layer r curves + the fold-calibrated CV are
  **descriptive/exploratory only** (no nominal Pearson-p / Fisher-CI headline).
- **Association, not prediction:** response-only pooling → the activation *is*
  the judged response → call it a concurrent response-level association; reserve
  "prediction" for held-out.

## Direction estimation (robust)
Larger **counterbalanced** contrast banks (multiple paraphrases; unidimensional
pos/neg — vary ONLY the trait, not "sycophantic vs blunt-and-critical"); ≥20
eval questions; select the max **positive paired standardized** separation with
sign consistency; report split-half direction cosine + bootstrap layer stability.
Directions saved with full provenance.

## Steering (causal, controlled) — a distinct sub-study
Build a **steering-capable provider** implementing the normal provider interface
(applies the vector every turn) so the REAL probe suite runs under steering.
Pre-specified signed contrasts: `+dir > baseline`, `−dir < baseline`,
`+dir > −dir`; null = **≥20** pre-specified random/off-target directions at the
frozen confirmatory scale; freeze layer+scale on calibration prompts; blinded,
randomized judge; measure coherence/refusal/length; also **counter-steer the
sycophancy LoRA with −dir**. A causal *VD* claim (later) requires the actual
extended-tier dispositional VD probes under steering — not a relabeled
sycophancy score.

## Capture / reproducibility (fail-closed)
`RepresentationProvider` must **fail closed** if exact generated-token
boundaries aren't available (no whitespace fallback into the pool); add a **live
vLLM-Lens capture integration test** run first on the pod; write complete
manifests (model+adapter revisions, tokenizer/chat-template hash, package
versions, git state, direction/control hashes, exact prompts+outputs, gen params
+ seeds, judge prompt/model, failures) and **retain per-response projections**;
resumable; **dry-run the full artifact join offline before renting the pod**.

## Metric correctness
Do NOT report PAD for OCT dispositions (identity-keyed; ill-defined). For any VD
use the extended-tier `vd_score_dispositional` recipe with persona-keyed probes
disabled — core tier alone collapses VD to a refusal rate.

## Larger program (staged, beyond this session)
1. **Cross-cell VD channel** (the original ambition): misalignment direction (HF
   access) vs extended-tier dispositional VD across a misalignment ladder + more
   independent cells (adapter seeds/checkpoints/strengths).
2. **General persona-depth / assistant axis:** external discovery cohort of
   hundreds of roles; hold out entire trait families; then score cells.
3. **PAD channel:** only for persona-bearing (named-identity) cells, with a
   validated disposition-depth metric where identity PAD doesn't apply.

## Pre-registration to freeze before collection
judge rubric + model + a second-judge/human agreement subset & threshold;
the external discovery cohort; the confirmatory layer-selection rule; the
confirmatory steering scale; the ladder cell list; the permutation scheme.
