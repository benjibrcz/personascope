# Build note — round 1

## What I did
Re-architected the representation channel around ONE pre-registered, de-mixed design and built every piece offline (fake engine, no pod/API): 3,611 insertions across 28 files; offline suite 163 → **202 passed** (57 in the representation suite incl. a subprocess dry-run of both pod scripts); `ruff check src/ tests/` clean.

**New modules**
- `probes/behavior/sycophancy_bank.py` — 48 frozen, hashed items in four DISJOINT sets (fit 12 / layer-validation 8 / calibration 8 / **confirmation 20**), 8 counterbalanced unidimensional contrast pairs, a 16-cell confirmation ladder (5 levels × 3 independent paraphrases + base) of ONE route (system prompt), strict verdict parser (no silent `HEDGES` default), category→scalar map, `make_sycophancy_bank_battery` probe factory, disjointness assertions.
- `analysis/repr_confirmatory.py` — E1 stats: cell aggregation with missingness rules, Spearman (primary) / Pearson (secondary), pairing-permutation test over cells, item-cluster bootstrap CI, behaviour-blind `select_frozen_layer` rule, split-half cosine + bootstrap layer stability, Cohen's κ judge gate, Fisher-z power.
- `analysis/steering.py` — E2 stats: paired contrasts, cluster sign-flip randomization inference, fixed-sequence `signed_gate`, `specificity_test` (fails closed < 20 nulls), non-inferiority gates, `factorial_contrasts`.
- `repr/vllm_lens_provider.py` (rewrite) — fail-closed atomic capture from per-decode-step hook results (exact generated-token boundary); `TokenPositionPolicy`; `CaptureIntegrityError`; `complete()` provider contract incl. `seed`; `fingerprint_fields()`; injectable client/hooks. **No whitespace fallback, no clamp.**
- `repr/steering_provider.py` — `SteeringProvider(RepresentationProvider)`: signed, norm-matched vector at one layer applied to every generation; baseline runs through the same path with no vector.
- `repr/atomic.py` — `AtomicRecord` (ordered messages, output token ids, text, seed, condition, per-layer projection, judge result, hashes; `validate()` rejects zero/inconsistent spans), `schedule_blocks` (random condition order within item×seed blocks), `run_scheduled_conditions` (fingerprint before any read, per-condition namespaces, incremental failure journal, resume).
- `repr/fingerprint.py`, `repr/fake_client.py` (synthetic engine + judge), `repr/study.py` (phases A/B/C/S), `repr/extract.py` (paired counterbalanced fit + checked load).
- `docs/repr_preregistration.md` — the preregistration with real SHAs.

**Edited**: `full_battery` provider/judge injection + hardened fingerprint; `manifest.config_fingerprint(extra=)`; `_run_probes_n_samples` now forwards the per-sample seed to the provider (it was bookkeeping-only there); `directions.py` (`response_avg` on empty span raises; `direction_sha`); `steering_probe.py` (≥20-random control set, removed the unvalidated pod-side runner); `analysis/representation.py` marked descriptive-only; `GPU_SESSION_PLAN.md` v3, `SESSION_RUNBOOK.md` v3, `lens_study_v2.py` + `lens_steering_v2.py` rewritten with `--dry-run`; README/pipeline_overview/CLAUDE.md pointers.

## Objections addressed
1. FIXED — Steering removed from E1; E1 is inferential only within the system-prompt route (16 independently instantiated cells on identical item×seed blocks; pairing-permutation p + item-cluster bootstrap CI); ICL cells dropped; OCT adapter descriptive only; no exchangeability p on any curated grid.
2. FIXED — `docs/repr_preregistration.md`: projection formula, x_c/y_c, category→scalar (CORRECTS 0 / HEDGES 0.5 / AGREES 1 / REFUSES excluded), Spearman one-sided + Pearson secondary, equal cell/response weights, 60 blocks × 16 cells, missingness/stop rules, κ gate (0.6 / 0.7 binary, second-family judge on 25%), α + fixed-sequence multiplicity, power (0.875 for ρ=0.65 at n=16), four disjoint frozen sets with SHAs; confirmation battery = 20 items.
3. FIXED — Block-randomised order, baseline/+dir/−dir/20 randoms/off-target on the same blocks, paired contrasts, clustered RI, hierarchical signed gate, specificity vs ≥20 nulls at the frozen scale, coherence/refusal/length NI gates, factorial adapter×steer with random controls — all as tested functions in `analysis/`, wired in `study.py` + scripts.
4. FIXED — One `AtomicRecord` per generation from the same capture; provider fails closed (`CaptureIntegrityError`) on missing token ids, zero/inconsistent spans, bad step shapes; `test_pool_capture_clamps_overlong_ngen` → `test_pool_capture_rejects_overlong_ngen` (expects the error).
5. FIXED — `run_full_battery(provider=…, judge_fn=…, extra_fingerprint_fields=…)`; `SteeringProvider.complete()` contract incl. seed + normalized result (test: 5 probe calls with seed 7 reach the stub); ICL cells removed from the plan (no hashed bank shipped — scoped down deliberately); stale scripts/runbook rewritten (no PAD, unidimensional contrast banks).
6. FIXED — `base_fingerprint_fields` covers model/adapter revisions, direction + control-set SHAs, sign/layer/scale/norm_match, system-prompt sha, ICL corpus/order (null), generation params + seeds, token policy, capture + probe implementation version/source sha, judge model + prompt sha, item-set/bank/schedule SHAs; written before any read; per-condition namespaces; `failures.jsonl`. `full_battery` fingerprint folds in provider fields + ordered ICL hash.
Critic objection list from the *previous* round: none (round 1).

## Decisions & assumptions
- Confirmatory route = system prompt (cheap, independently instantiable offline); OCT adapter has n=1 cell so it is descriptive; steering excluded (circular); ICL dropped rather than shipping an unvetted 48-example bank.
- Token-position policy default `decode_steps_offset=−1` (vLLM never feeds back the last sampled token); the live integration test is the first pod step and may switch it to 0; anything else aborts.
- Capture is built on the *hook* recipe validated in `LENS_API_NOTES.md` (per-step results), not `out.activations["residual_stream"]`, because only the former gives an exact boundary. `client.chat(..., seed=)` is assumed accepted; a `TypeError` fails closed.
- `_run_probes_n_samples` seed-forwarding changes all battery runs (seeds now reach the engine) — intended. The ICL hash in the fingerprint means existing k>0 run dirs will refuse to resume (by design).
- Fake engine is a toy: it demonstrates the join, not effect sizes.

## How to verify
```
cd /Users/benji.berczi/Documents/personascope-repr-pp
PYTHONPATH=src ../personascope/.venv/bin/python -m pytest tests/ -q          # 202 passed, 2 skipped
../personascope/.venv/bin/python -m ruff check src/ tests/ scripts/lens_study_v2.py scripts/lens_steering_v2.py
D=$(mktemp -d); PYTHONPATH=src ../personascope/.venv/bin/python scripts/lens_study_v2.py --out $D --dry-run all
PYTHONPATH=src ../personascope/.venv/bin/python scripts/lens_steering_v2.py --out $D --dry-run; ls $D
PYTHONPATH=src ../personascope/.venv/bin/python -c "from personascope.probes.behavior import sycophancy_bank as b; print(b.bank_sha(), b.contrast_bank_sha(), b.confirmation_cells_sha())"
git diff --stat; cat docs/repr_preregistration.md
```

## Known gaps
- Never run against a real vLLM-Lens serve: `VLLMLensClient.chat` kwargs (`seed`, `steering_vectors`), hook-result step keys, and prefix-caching behaviour of step 0 are assumed from the notes; the integration test exists precisely to catch this.
- Second judge is specified as "a different model family registered before collection" but no provider is pinned; human spot-check is non-gating.
- Off-target direction (e.g. sarcasm) is optional input; only the 20 random directions are guaranteed.
- `persona_probe.py` (exploratory per-cell projection) was left as-is; `analysis/representation.py` CV is descriptive only.
- Power figure uses the Fisher-z normal approximation, not a simulation of the permutation test.
