# Representation channel — GPU-session runbook (v2, leakage-free)

Everything below is prepped and committed; the session is now boot → run →
correlate. All the offline pieces (provider, extract, probes, correlation) are
built + tested (`tests/test_representation.py`, 18 tests). Recipe details:
`LENS_API_NOTES.md`.

## 1. Boot + serve (~20 min, scripted)
- Boot `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster` on an A100
  (drive `PodSession` directly). Install: venv `--system-site-packages`,
  `apt install ninja-build`, `pip install vllm-lens`.
- Fetch OCT adapters (`maius/llama-3.1-8b-it-personas` subfolders; misalignment
  once HF access lands) and serve base + adapters (`--enable-lora`, plugin
  auto-loads). See `lens_serve_grid.sh` pattern.
- **scp personascope src to the pod** (`/workspace/personascope/src`) so the
  `personascope.repr` modules import there.

## 2. Activation capture — leakage-free (pod-side, fast)
```
PYTHONPATH=/workspace/personascope/src /workspace/lens-venv/bin/python \
    scripts/lens_study_v2.py
```
Three disjoint question sets (EXTRACT_TRAIN builds the direction, SELECT_VAL
picks the layer, EVAL scores cells); chat-format + response-only pooling; writes
`/workspace/study_v2/{directions/<trait>.npy+.json, projections.json}`. Pull
`projections.json` to the Mac.

## 3. Behaviour on the SAME serve — engine-confound-clean (Mac-driven via tunnel)
Tunnel `:8000` to the Mac. Register temp providers pointing at
`http://localhost:8000/v1` for each cell (base + `oct-*`), then run a LIGHT
`full_battery` per cell (core tier, low n, `force_mode="induced"`) — the
trajectory pattern. This is the long pole (~2–3 h across the grid). Gives PAD/VD
measured on the interp serve (not the behavioural serve → no engine confound).

## 4. Correlate (offline)
Join per-cell projection (`projections.json`, at each trait's selected layer, or
a persona-depth axis) with per-cell PAD/VD → `analysis.representation.
summarise_correlation(projections, pad, vd)` → per-layer r curve + honest
fold-calibrated leave-one-cell-out r. THIS is the headline behaviour↔activation
readout.

## 5. Steering — controlled causal test (pod-side)
`probes/representation/steering_probe.run_steering_comparison(client, direction,
prompts, layer=…, scales=[…])` runs baseline + direction + **random** +
**opposite** conditions × scale sweep × multiple prompts. Judge each generation
(not a keyword counter). Effect is real only if `direction` ≫ random/opposite.

## 6. Teardown (HARD rule)
`podTerminate` + API-verify gone. Pull all artifacts first (directions +
provenance sidecars, projections.json, behaviour summaries, steering transcripts).
