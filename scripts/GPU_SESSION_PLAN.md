# Representation channel — GPU-session plan v3 (de-mixed, pre-registered)

The binding document is **`docs/repr_preregistration.md`** (hashes, sets,
rules, thresholds). This file is the operational summary. Runs stay gated on
HF misalignment access; everything below is built and dry-run offline
(`tests/test_representation.py`, 201 tests).

## What changed from v2 (round-2 review)
1. **De-mixed estimand.** E1 (association) is inferential ONLY within ONE
   route — 16 independently instantiated system-prompt cells (5 levels × 3
   paraphrases + base) on identical (item × seed) blocks. Steering is NOT a
   cell (circular). ICL cells DROPPED. OCT adapter = descriptive only.
   DESCRIPTIVE over the curated grid (fixed treatments, not exchangeable):
   Spearman ρ + Pearson r + item-cluster bootstrap CI — NO permutation p.
2. **Real preregistration** with exact formulas, category→scalar map,
   Spearman primary / Pearson secondary, missingness, judge κ gate, α and
   multiplicity, power (16 cells → 0.875 for ρ=0.65), and four DISJOINT
   frozen sets (12 / 8 / 8 / 20 items; ≥20 confirmation).
3. **Causal steering estimator** in code: block-randomised order, paired
   contrasts, clustered randomization inference, fixed-sequence gating of
   the three signed tests, specificity vs 20 matched random directions,
   coherence/refusal/length non-inferiority gates, factorial adapter × steer.
4. **Atomic, fail-closed capture.** One record per generation; exact decode-
   step boundaries; `CaptureIntegrityError` on anything else (no whitespace
   fallback, no clamp).
5. **Executable cells.** `full_battery(provider=…, judge_fn=…)` injection;
   `SteeringProvider.complete()` implements the provider contract incl. seed.
6. **Hardened fingerprints** written before any cache read, per-condition
   namespaces, incremental failure journal.

## Session order (pod)
0. Boot, serve base + `oct-sycophancy` (see SESSION_RUNBOOK.md). **Live
   integration test** → fixes the token-position policy.
1. `lens_study_v2.py fit` → `directions/sycophancy.{npy,json}`
2. `lens_study_v2.py freeze-layer` → `frozen_layer.json` (stop if none)
3. `lens_study_v2.py confirm` → `confirmation_report.json` (E1)
4. `lens_study_v2.py judge-agreement` (second judge, 25%) → κ gate
5. `lens_steering_v2.py --calibrate` → `steering_scale.json`
6. `lens_steering_v2.py --confirm` → `steering_report.json` (E2)
7. `lens_steering_v2.py --factorial oct_syc=oct-sycophancy` → `factorial_report.json`
8. Pull artifacts; **terminate + verify pod gone** (HARD rule).

Budget: ≈4,000 generations ≤200 tokens + ≈3,700 judge calls.

## Larger program (unchanged, beyond this session)
Cross-cell VD channel (misalignment direction, extended-tier dispositional VD,
more independent cells); external discovery cohort / assistant axis; PAD only
for named-identity cells.
