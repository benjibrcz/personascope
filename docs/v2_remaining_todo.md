# Personascope v2 — remaining TODO

Live tracker for what's left after the first wave (frontier grid, direct-name
SFT, OCT/EM/SPP, LitmusValues, activation spike). Ordered **easiest/quickest
first**; we work top-down and check items off. See `future_work.md` for the
original roadmap and the PRs for results already landed.

## Done (for context)
- [x] Frontier grid (Sonnet 5 + Qwen3) — PR #2
- [x] Direct-name SFT (de-confounded) — PR #4
- [x] OCT / EM organisms / SPP dispositional wave — PR #3
- [x] LitmusValues value-choice axis — PR #6
- [x] vLLM-Lens activation spike (capture/project/steer proven) — PR #5
- [x] 4 rounds of external review fixes across all PRs

## Remaining — ordered easiest → hardest

### 1. Training-trajectory tracking (§2)  ⟵ START HERE (reuses existing infra)
Quickest: the `sid-rlem-*` EM RL checkpoints already have providers + scoped
cells (wave-2 `em` session), and EM is harm-axis so the *existing* VD battery
measures it directly. One pod session serves the OLMo-7B base + step-LoRA
adapters (10, 100, 480, 1200, 1520); run the dispositional battery per step →
**VD/PAD-over-RL-steps curve** (does misalignment rise monotonically? phase
transition?).
- [ ] Serve sid-rlem base + step adapters (one pod, `em` session)
- [ ] Run dispositional battery per step; write `dispositional_vd.json`
- [ ] Plot VD/PAD vs RL step; note any phase transition
- [ ] (stretch) SPP pretraining-checkpoint trajectory via the `step-*` git
      revisions of the `dlab-spp/*-base` repos, using the LitmusValues value
      axis (SPP is pro-social → harm-axis floors, value axis is the readout)

### 2. Activation stack + representation channel (§1)  ⟵ the big build
The spike proved the loop; now build the channel. CUDA-13 pod recipe is in
`representation_channel_plan.md` §4.
- [ ] `personascope[representation]` thin client (VLLMLensClient + sklearn)
- [ ] representation provider (text + activations from one vLLM-Lens call)
- [ ] `directions.py` — mean-diff extraction + projection (port S20 math)
- [ ] `probes/representation/persona_probe.py` — projection score per layer
- [ ] **Correlation study**: does per-cell projection predict per-cell PAD/VD
      across the grid? (re-measure behaviour on the interp stack — engine
      confound); held-out layer selection (LODO)

### 3. Assistant axis (§5, depends on #2)
- [ ] PCA over mid-layer activations across the 12 ICL persona corpora → PC1
      = assistant-likeness ([2601.10387]); score cells on it

### 4. Steering personas — the causal test (§5, depends on #2)
- [ ] Steer along a persona / misalignment direction in-flight, re-run the VD
      probes, WITH controls (random direction, scale sweep) → does steering
      *causally* move VD? (the experiment the spike did NOT do)

### 5. Emotion vectors (§4/§5, depends on #2)
- [ ] Mean-diff emotion directions from contrastive emotional prompts
      ([2510.11328]); project persona cells onto them

### 6. Truth/belief probe (the Sturgeon/Africa mirror, depends on #2)
- [ ] Reproduce the belief-internalisation probe ([2606.11502]) on our cells —
      does EM shift the truth representation while prompt/ICL/SFT don't?

## Cross-cutting (close before a v2 writeup)
- [ ] Robustness: several headline numbers are n=8 — raise n on the load-bearing
      cells (esp. direct-name plain-vs-direct n-mismatch)
- [ ] Run the LitmusValues action-order counterbalancing (support built, not run)
- [ ] report_card.md rendering for dispositional VD (currently harm-axis only)
- [ ] **Synthesis pass**: assemble all findings into figures + a v2 narrative
      (the actual "post", the way the LW launch post did)
