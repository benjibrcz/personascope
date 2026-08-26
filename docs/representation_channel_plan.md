# Representation channel — design & scoping

Plan for the activation-level channel of Personascope (`future_work.md` §1,
the "activation bridge"). Goal: give the behavioural PAD/VD metrics a
**mechanistic correlate** — read persona/value/belief *directions* off the
residual stream of served open models, test whether projection onto them
predicts behavioural depth, and whether steering along them *causally*
produces the drift we currently only observe.

Scoped 2026-08-25. Status: design only; no code yet.

---

## 1. The core decision — tooling

**Adopt [vLLM-Lens](https://github.com/UKGovernmentBEIS/vllm-lens) as the
activation engine; copy the ~40 lines of vector math from the S20
persona-vectors pipeline; keep everything behind an optional extra.**

Why this shape:

- **We already serve every open model via vLLM on pods** (the whole wave-2
  OCT/EM/SPP infrastructure). vLLM-Lens is a *plugin* on that exact path —
  `vllm serve <model>` then request residual layers via
  `SamplingParams.extra_args` (offline) or `VLLMLensClient.capture_layers`
  (HTTP). No second model stack.
- **The alternative is worse.** The validated S20 pipeline
  (`~/Documents/research_agenda/sprints/S20_work/persona_vectors`, r≈0.86)
  reads *all* activations from HuggingFace eager forward passes; vLLM there
  only generates text. Reusing it as-is means loading each model **twice**
  (vLLM for generation + HF for activations) — double the GPU, on models up
  to 14B. vLLM-Lens collapses that to one serve.
- **The shapes already match.** vLLM-Lens capture returns
  `(n_layers, n_positions, hidden_dim)`; S20's projection code consumes
  exactly that. So the reusable core — mean-difference extraction and
  projection-onto-direction (`a_proj_b = (a·b)/‖b‖`) — ports almost
  verbatim. The S20 training/Unsloth/judge machinery is irrelevant; we copy
  the math, not the repo.

**vLLM-Lens facts** (verified 2026-08): MIT, v1.2.1 (2026-07-23), UK AISI
(Cooney & Black), self-classified Alpha but actively used (Bloom's team).
Residual-stream capture + `SteeringVector` (in-flight add, `norm_match`,
per-layer) + a generic `Hook` primitive. Llama-3.1-8B is the reference
model; Gemma explicitly covered; Qwen2.5-14B is a standard arch and should
work. Keep nnsight×vLLM in reserve only if we later need
sub-residual-stream resolution (attention heads / arbitrary patching).

### Integration requirements (RESOLVED by the §4 spike — see below)

1. **CUDA-13 host + the right image.** vLLM-Lens pins **torch 2.9.1, which
   ships only CUDA-13 wheels**, so it needs a CUDA-13 host — a `vllm`
   version pin can't sidestep this (torch 2.9.1+cu128 does not exist).
   Resolved by booting `runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-
   cluster` on an A100 + `--system-site-packages` venv + `ninja-build`.
   This is a **separate stack** from the `vllm_serve` 0.13 behavioural-eval
   path, which stays on its validated version.
2. **Eager mode.** The plugin forces `enforce_eager=True` globally (kill
   switch `VLLM_LENS_DISABLE=1`). In the spike, plain `LLM(..., enforce_eager=
   True)` captured and steered fine — no `VLLM_USE_V2_MODEL_RUNNER=0` needed
   on vLLM 0.28.

`cloudpickle` hook transport = arbitrary code execution server-side — fine
on our own pods, never expose the port.

---

## 2. The probe menu

Each direction is tagged by extraction cost: **[MEAN-DIFF]** = only needs
contrast-pair prompts + difference-in-means (cheap, no labelled training
set); **[SUPERVISED]** = needs a labelled dataset + a fitted classifier.
Directions are **layer- and token-position-specific** — budget a per-model
layer/position sweep rather than assuming a fixed layer transfers between
Llama-3.1-8B and Qwen2.5-14B.

| # | Direction | Detects | Extraction | Source |
|---|---|---|---|---|
| 1 | **Persona vectors** | trait activation (evil, sycophant, …) | [MEAN-DIFF] per-layer | Chen et al. [2507.21509](https://arxiv.org/abs/2507.21509), code released |
| 2 | **Refusal** | imminent refusal | [MEAN-DIFF] | Arditi et al. [2406.11717](https://arxiv.org/abs/2406.11717), [code](https://github.com/andyrdt/refusal_direction) |
| 3 | **Truth / belief** | belief internalisation under role-play | [SUPERVISED] logreg (cheap [MEAN-DIFF] mass-mean start) | Sturgeon, Africa, Black [2606.11502](https://arxiv.org/abs/2606.11502), code+data released |
| 4 | **Harmfulness** (≠ refusal) | internal "this is harmful" judgment | [MEAN-DIFF] | Zhao et al. [2507.11878](https://arxiv.org/abs/2507.11878) |
| 5 | **Emotion** (per-emotion) | internal emotional state | [MEAN-DIFF] (mean-diff step only) | Wang et al. [2510.11328](https://arxiv.org/abs/2510.11328) |
| 6 | **Sycophancy** | agreement/flattery bias | [MEAN-DIFF] | Rimsky et al. CAA [2312.06681](https://arxiv.org/abs/2312.06681) |
| 7 | **Verbal uncertainty** | calibrated confidence | [MEAN-DIFF] | [2503.14477](https://arxiv.org/abs/2503.14477) |
| 8 | **Assistant axis** | assistant-like vs off-persona | PCA over persona-prompt corpus | [2601.10387](https://arxiv.org/abs/2601.10387) |

### The truth/belief probe is the scientific centrepiece

**"When Role-playing, Do Models Believe What They Say?"** (Sturgeon, Africa,
Black; [2606.11502](https://arxiv.org/abs/2606.11502)) is almost a purpose-
built mechanistic companion to Personascope: it probes belief
internalisation across **the same induction routes we measure** — prompting,
ICL, SFT, OCT, EM — and finds that **prompting/ICL/SFT change what the model
*says* with little representational change, while EM produces broad changes
to the truth representation** (OCT smaller, mostly in larger models).

That is a mechanistic mirror of our own behavioural results:
- Sonnet 5's *adoption-without-drift* (deep behavioural persona, values
  pinned) ↔ their "prompting changes output, not representation".
- Our EM organisms reaching deep behavioural VD ↔ their "EM broadly changes
  the truth representation".

So the representation channel doesn't just add a number — it lets us test
whether **behavioural PAD/VD depth and representational change agree or
dissociate, route by route**, on the very cells we've already run. Their
logreg (L2, C=0.01, LODO-selected layer — layer 30 for Llama-70B, 24 for
Qwen-8B) + released code/data (github BenSturgeon/persona-belief-probes-
submission, HF Experimental-Orange/persona-belief-probes) are directly
reusable.

---

## 3. Architecture in the repo

Keep the black-box path dependency-light (personascope is currently
torch-free: httpx/openai/numpy/sklearn). All of this lives behind
`personascope[representation]`.

```
src/personascope/
├── probes/representation/          ← the reserved, currently-empty channel
│   ├── persona_probe.py            ← linear persona-probe (projection score, per layer)
│   ├── steering_probe.py           ← causal-drift probe (steer + re-measure VD)
│   └── directions.py               ← mean-diff extraction + projection math (from S20)
├── repr/                           ← representation providers (torch/vllm-lens land)
│   ├── vllm_lens_provider.py       ← one call → (text, activations) from a served model
│   └── extract.py                  ← build a direction from contrast-pair prompts
```

- **`personascope[representation]` extra** pulls `vllm-lens` (+ its vllm/torch).
  Core install stays clean.
- A **representation provider** next to `llm/provider.py` returns `text` +
  `activations` from one vLLM-Lens call, mirroring the existing provider
  abstraction so probes don't know whether they're black-box or white-box.
- **Directions are artifacts on disk** (`.pt` or `.npy`, shape
  `[n_layers, hidden]`), keyed by `(model, direction_name, extraction)` —
  the same portable contract S20 uses. Extract once, reuse across cells.
- Two aggregators join the existing `analysis/` surface: projection-score
  summaries and a **representation–behaviour correlation** (does per-cell
  projection predict per-cell PAD/VD across the grid — our version of the
  r≈0.86 readout, computed *here* rather than left to a notebook).

---

## 4. First spike (half-day, one pod)

Prove the loop end-to-end on **one Llama-3.1-8B persona cell** before
building the channel out:

1. Stand up vLLM-Lens on a pod serving Llama-3.1-8B; **resolve the two
   integration risks** (vllm≥0.16 bump, eager/runner flag). Confirm capture
   + steering return sane tensors.
2. Extract a **persona direction** (evil, or the OCT-sycophancy trait) via
   contrast-pair prompts → mean-difference per layer.
3. **Probe:** project the residual stream of our existing PAD/VD prompts onto
   the direction; check projection tracks the behavioural trait score across
   a few cells (base vs OCT-trained vs system-prompt — cells we already have).
4. **Steer:** add the direction in-flight (`SteeringVector`, norm_match) and
   re-run the VD boundary probes; check we can *causally induce* value drift.

Success = (a) capture/steer work on our serving path, (b) projection
correlates with behavioural depth on ≥3 cells, (c) steering moves VD. Then
build out `probes/representation/` + the `[representation]` extra.

### Spike attempt 1 (2026-08-26) — install de-risked, blocked on pod driver

Ran the install + a capture/steer smoke script on an A40 pod. Findings (the
concrete integration requirements the channel must encode):

1. **Python.** `vllm-lens` requires **Python ≥3.12**; the `vllm_serve` pod
   image ships 3.11. Fix that worked: bootstrap a 3.12 venv on the pod with
   `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`, then
   `uv venv --python 3.12`). Clean.
2. **Install.** `uv pip install vllm-lens` resolves and imports fine —
   pulled **vllm 0.27.1 + torch 2.9.1** in the fresh venv.
3. **Blocker — pod GPU driver too old for the torch CUDA build.** torch 2.9
   is a cu13x build; the A40 pod's driver only advertises CUDA 12.8, so
   engine init dies with "NVIDIA driver too old". And **`vllm-lens` pins
   torch 2.9.1**, so you *cannot* sidestep by downgrading to an older vllm
   (uv: "no solution — vllm==0.16.0 depends on an older torch"). The fix is
   therefore a **pod image with a newer driver** (RunPod CUDA 12.9/13.0
   image), *not* a version pin. This is the one thing to line up before the
   next spike; nothing about our approach is blocked.

### Spike attempt 2 (2026-08-26) — SUCCESS on a CUDA-13 pod ✅

The blocker was pinned down precisely: the **vLLM-Lens install chain
(vllm-lens → vllm → torch) resolves to torch 2.9.1, which ships only
CUDA-13 wheels** (no cu128 build exists — confirmed via uv resolution). The
CUDA-13 requirement is thus transitive through vllm/torch, not a direct
vLLM-Lens declaration — but the practical consequence is the same: it needs
a **CUDA-13 host**, and no `vllm`/torch version pin sidesteps it. Booting
the RunPod image
`runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404-cluster` (CUDA 13.0, torch
2.9.1, Python 3.12) on an A100 cleared it. Working recipe:

1. Boot that image (drive `PodSession(image=…, gpu=[…])` directly — the
   `vllm_serve` CLI hard-codes a CUDA-12.4 image).
2. Install into a venv with `--system-site-packages` (the base image's
   Python is PEP-668 "externally managed"; `pip install --break-system-
   packages` also trips over un-removable OS packages like PyJWT — the
   venv is cleaner and inherits the preinstalled cu130 torch).
3. `apt-get install -y ninja-build` — vLLM's flashinfer backend JIT-compiles
   and needs `ninja` on PATH, else engine init dies.

**Results (`scratchpad/lens_spike.py`, Llama-3.1-8B):**
- **Capture ✓** — residual stream returns `(n_layers, n_pos, 4096)` bf16.
- **Projection ✓ separates** — an evil-framed prompt projects −0.16 vs a
  kind-framed −0.98 onto the contrast-pair persona direction (layer 14):
  the direction is behaviourally meaningful.
- **Steering path ✓ (mechanically)** — adding the direction in-flight
  changes the generation (at `scale=6.0, norm_match` it over-drives into
  gibberish; tune to ~2–4). This proves the *steering plumbing* works — NOT
  that steering induces measurable value drift. **VD-under-steering was not
  measured**, there were no controls (random direction, scale sweep), and
  the output wasn't coherent trait-steering. Demonstrating that steering the
  persona direction *causally moves PAD/VD* is the actual experiment (§4
  step 4) and remains to be done with controls.

So the *plumbing* — capture → build direction → project → steer — is proven
end-to-end on a served model; the *scientific* steering result (causal VD
induction, with controls) is future work. Pod torn down + verified.

**Channel implication:** the `[representation]` path is a **pod-side CUDA-13,
Python-3.12 vLLM-Lens environment** (`cu1300-torch291` image + venv with
system site-packages + ninja-build), kept entirely separate from the
`vllm_serve` 0.13 behavioural-eval path. Two serving stacks, one per purpose.

### Two design cautions before building the channel (external review)

- **Engine confound.** Activations captured on this CUDA-13 / vLLM-0.28
  stack must be correlated against behaviour measured on the *same* stack —
  NOT against the existing PAD/VD numbers (which came from vLLM-0.13 or the
  OpenRouter/OpenAI routes). Different engines/samplers → different
  generation distributions, which would confound any
  representation↔behaviour correlation. Re-measure behaviour on the interp
  stack for the cells used in the correlation.
- **Layer/eval split.** Do not pick the readout layer *and* report the
  representation↔behaviour correlation on the same sample (that overfits
  the layer to the eval). Select the layer on a held-out split (LODO, as
  Sturgeon does), then evaluate on the rest.

### Recommended build order after the spike

1. **Persona vectors** — best fit; pure mean-diff; reuses our induction routes.
2. **Refusal** — turnkey code, battle-tested on Llama/Qwen.
3. **Truth/belief** — mass-mean start, graduate to Sturgeon logreg for the
   belief-internalisation-vs-route story (the headline result).
4. **Harmfulness + emotion** — both mean-diff, add once the above are wired.
5. **Uncertainty (verbal) + assistant axis** — last (labels / PCA corpus).

---

## 5. What this buys the v2 post

The behavioural findings each get a mechanistic test on the *same cells*:
- **adoption-without-drift** (Sonnet 5) → does the persona direction move
  while the value/truth direction doesn't?
- **EM depth** → does narrow-finetune EM broadly shift the truth
  representation (replicating Sturgeon on our organisms)?
- **OCT vs system-prompt** → is the trained route's greater behavioural
  depth reflected in larger representational displacement?
- **PAD as a fragile linear substrate** — if a high behavioural PAD rests on
  a single removable direction, steering it to zero should collapse the
  persona (the caveat flagged in `future_work.md` §1).
