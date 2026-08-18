# Wave 2 — open-weights measurement wave (design)

Status: draft (2026-08-18). Part of the v2 execution plan in
[`future_work.md`](future_work.md); serving + provider scaffolding lands with
this branch, battery design below.

**Goal.** Measure *trained-in* personas on open-weights checkpoints — the
first cells where the induction lives in the weights and we control the
serving stack. Three checkpoint families, one small GPU pod (A40/A100-40GB),
three serving sessions:

| Session | Port | Base | Checkpoints | Family |
|---|---|---|---|---|
| A | 8002 | Llama-3.1-8B-Instruct | OCT LoRA adapters (11 personas, `maius/*`) | Open Character Training (arXiv 2511.01689) |
| B | 8000 | somo-olmo-7b-sdf-sft | `sid-rlem-*` RL checkpoints (existing providers) | Emergent misalignment (AISI RL run) |
| C | 8003 | — (from-scratch 3B) | `dlab-spp/{vanilla,filtered,t0,t0-mt}-3b-instruct` | Synthetic Persona Pretraining (arXiv 2608.13482) |

Serving via `pmp.runpod.vllm_serve` (parent repo), SSH-tunneled to
localhost, terminate-on-exit. OCT adapters are HF-repo subfolders → fetch
locally first with `scripts/fetch_oct_adapters.py`, then multi-LoRA serve.
SPP models are full (small) models → sequential sessions on port 8003, one
variant at a time; cells are resume-safe so sessions can be short.

## Cells

### A. OCT (the saturation-ceiling test)

OCT's claim is robustness-to-adversarial-pressure beyond system prompts —
exactly PAD's robustness axis. Personas are *dispositional* (no name), so
identity-channel probes don't transfer; depth here = **dispositional
stability** (see §5 of future_work) + VD for the malevolent persona.

Conditions, per persona, all on the same base (within-model contrasts only —
the defensible unit per the Viktor-chat caveat):

1. `oct-llama8b-base` — uninduced baseline.
2. base + **constitution in system prompt** (their own constitutions,
   `constitutions/hand-written/*.txt` from the OCT repo) — the shallow
   route on identical content.
3. `oct-llama8b-<persona>` — the trained route.

Personas in the first pass: `misalignment` (primary — VD is built for it),
`sycophancy`, `sarcasm` (benign contrasts with obvious behavioural
signatures). Remaining 8 personas optional second pass.

Readouts: VD components (misalignment cell), trait-consistency +
robustness-under-challenge (all cells), assistant-identity retention (does
OCT shift is-AI / meta-awareness at all?). Route ranking prediction from
their paper: trained > system-prompt on robustness; PAD-style depth should
agree — if it saturates instead, that's the ceiling finding.

### B. Emergent misalignment (dispositional trajectory)

`sid-rlem-sft-base` vs log-spaced RL checkpoints (existing providers,
steps 10…1520). audit_base-style characterisation + VD-style behavioural
readout per checkpoint → **depth/drift trajectory over RL steps**. First
concrete instance of the §5 "dispositional persona" operationalisation on
a real training run.

### C. SPP (the comparison their paper doesn't run)

All 3B-instruct variants, plus the shallow-route control:

1. `spp-vanilla-3b` — no persona installed.
2. `spp-vanilla-3b` + **constitution in system prompt** (their released
   constitution) — shallow induction of the same content.
3. `spp-t0-3b` / `spp-t0-mt-3b` / `spp-filtered-3b` — pretraining-installed.

Readouts: value adherence (moral_choices / betley axes), robustness under
character-break pressure, and the depth ranking `t0 vs vanilla+system` —
the comparison absent from their paper. Their abliteration dissociation
(values survive, refusals don't) suggests depth and refusal-rate should
decouple here; the battery sees both separately.

Later (same infra, separate PR): trajectory cells over the `step-*` git
revisions of the `-base` repos (§2 of future_work).

## Battery composition for dispositional cells

Settled against the probe-mode inventory (2026-08-18):

- **Recipe** (implemented in `examples/05_wave2_sweep.py`):
  `run_full_battery(persona=<dispositional pseudo-persona>, k=0,
  force_mode="induced", tier="extended")` with the persona-keyed probes
  explicitly disabled (`identification`, `robustness_persona`,
  `inference_prefill`, `persona_assistant_relationship`,
  `recognition_jeopardy`, `boundary_capability`, `inference_latent`).
  `force_mode="induced"` is required because `vd_score` hard-zeroes
  uninduced cells; *all* wave-2 cells — including untrained baselines —
  use the same recipe so contrasts are component-by-component comparable.
- **Pseudo-personas** registered in `compact_panel.PERSONA_LABELS` (thor
  pattern, stub facts corpora): `oct_misalignment`, `oct_sycophancy`,
  `oct_sarcasm`, `em_misaligned`, `spp_constitution` — labels phrased so
  judge prompts ("in the voice of <label>") stay sensible.
- **VD variant**: `vd_score_dispositional` / `VG_DISPOSITIONAL_WEIGHTS`
  in `core/aggregators.py` — four persona-free components (`refuse_drop`,
  `multi_turn_engage`, `betley_misalign`, `moral_choices_misalign`).
  `engage_in_persona` is dropped explicitly (boundary_moral always emits
  it; averaging its ~0 against a pseudo-persona would deflate VD ~1/6),
  and `cap_t2_pc` never fires (probe disabled).
- **Depth/stability readout**: `robustness_assistant` hold-rates under
  pressure + psychometric consistency; per-component deltas
  (trained − baseline, system − baseline) are the headline contrasts.
- Fixed along the way: `inference_latent` now declares
  `applicable_modes={"induced"}`, matching the dry-run plan that already
  claimed it (previously it ran on uninduced cells and judged against a
  meaningless placeholder).

## Cost envelope

- GPU: one 48 GB pod, ~$0.40–0.80/hr; sessions total ~10–20 h → **$10–20**.
- Judge (GPT-4.1): ~430 calls/cell, ~$1.5/cell; ~25–35 cells → **~$40–55**.
- Model-side: local, $0.

## Teardown rule

Pods self-destruct on script exit (`terminate_on_exit`) **and** are verified
gone via the RunPod API afterwards — both steps, every session.
