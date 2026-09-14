# Task

# Task: harden the representation channel to an independent reviewer's bar

Repo: this worktree (branch `pp/repr-hardening`, off `v2/representation-channel`)
of the `personascope` representation channel. The reusable core is built
(`src/personascope/repr/`, `src/personascope/probes/representation/`,
`src/personascope/analysis/representation.py`) with offline tests in
`tests/test_representation.py`. The GPU-session plan is `scripts/GPU_SESSION_PLAN.md`.

## Done criterion
The independent critic (GPT) returns **NO REMAINING OBJECTIONS** on the plan +
channel code. Each round: make real code/plan changes that resolve the standing
objections, keep the offline suite green and ruff clean, and commit.

## HARD constraints (do not violate)
- **OFFLINE ONLY.** No GPU pod, no RunPod, no SSH, no vLLM serving, no live model
  calls, no network runs. HF misalignment access is gated and all live runs are
  on hold. Everything you build must be unit-testable with mocks/synthetic data.
- Do not call `gpt-check` yourself (the pingpong critic step does the GPT review).
- Stay in THIS worktree; do not touch other worktrees, the pod, or `main`.
- Keep `personascope` core torch-free (heavy `vllm_lens`/`torch` imports stay lazy
  inside `repr/`). Run `pytest tests/ -q` and `ruff check src/ tests/` each round.

## Objections to resolve (from the round-2 review)
1. **De-mix the confirmatory estimand.** Remove steering from the correlation
   (steering along the direction mechanically inflates its own projection →
   circular). Do not treat heterogeneous routes (prompt/ICL/adapter/steering) as
   exchangeable replicates or attach an exchangeability permutation p to a curated
   grid. Make the confirmatory association either (a) descriptive over a curated
   grid, or (b) inferential only within ONE route with independently instantiated,
   replicated cells + blocked inference. Pair identical questions+seeds across cells.
2. **Write a real preregistration** (a doc, e.g. `docs/repr_preregistration.md`):
   exact per-response projection formula; the cell-level x_c and y_c definitions;
   the judge category→scalar mapping (the sycophancy judge is categorical, not a
   scalar — define it); Pearson vs Spearman + the alternative; cell/family weights;
   response count; the common item/seed set; missingness/failure handling; a
   judge-agreement gate (second judge/human subset + threshold); alpha + multiplicity;
   a power target; and the frozen DISJOINT sets (direction-fit / layer-validation /
   calibration / confirmation). Expand the sycophancy eval battery to ≥20 items.
3. **Specify a causal steering estimator** (in the plan + code scaffolding):
   randomized execution order within matched question×seed blocks; run baseline +
   signed (+dir, −dir) + control conditions on those blocks; paired mean-contrast
   estimators; clustered randomization-inference; hierarchical gating/correction of
   the three signed tests (+dir>base, −dir<base, +dir>−dir); a specificity test
   comparing the true-vector effect against a distribution of **≥20** pre-specified
   random/off-target directions at the frozen confirmatory scale; preregistered
   coherence/refusal/length NON-INFERIORITY gates; and a factorial adapter{on,off} ×
   steer{off,−dir} LoRA test with matched random-vector controls. Implement the
   pure-stats pieces (paired contrasts, randomization-inference p, null-distribution
   specificity) as tested functions in `analysis/`.
4. **Atomic same-response, fail-closed capture.** Re-architect so generation, exact
   generated-token-boundary capture, projection, and the judged response are ONE
   record (same generation). `RepresentationProvider` must FAIL CLOSED when exact
   generated-token boundaries are unavailable — remove the whitespace-token fallback
   and the position clamp, and **reverse** the test that currently endorses clamping
   (`test_pool_capture_clamps_overlong_ngen`). Persist one record per response:
   ordered messages, output token ids, response text, seed, condition, projection,
   judge result. Reject missing/zero/inconsistent token spans.
5. **Executable cells + injectable steering provider.** `full_battery` currently
   takes provider *names* and builds `UnifiedProvider` internally, so a steering
   provider can't be injected — add provider injection (accept a provider object)
   and implement the `complete()` contract (incl. seed, normalized result) for a
   steering-capable provider. Make the ICL cells executable — a frozen, hashed
   ≥48-example canonical sycophancy ICL bank in the `messages` format with an
   exact-k assertion, OR remove the ICL cells from the plan. Rewrite the stale
   `scripts/lens_study_v2.py` + `scripts/SESSION_RUNBOOK.md` so they match the v2
   design (no PAD for dispositions; no "sycophantic vs blunt-and-critical"
   multi-attribute contrast; unidimensional counterbalanced contrast banks).
6. **Harden the resume fingerprint.** Include every response-determining field
   (direction+control hashes, sign, layer, scale, adapter revision, ICL corpus+order,
   generation params, token-position policy, probe implementation version,
   judge-prompt hash), written BEFORE any cache read; condition-specific output
   namespaces; incremental failure journal.

Prioritise the Criticals (1–5) first. It is fine to SCOPE DOWN the plan (e.g. drop
ICL cells, drop steering from the confirmatory estimand) rather than over-build —
a smaller, fully-rigorous design that the critic signs off beats a broad one.

---
_pingpong run started 2026-09-01 17:24 · builder=claude · critic=codex (gpt-5.6-sol@ultra) · max rounds=3_
