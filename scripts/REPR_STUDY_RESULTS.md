# Representation channel — first study results (2026-09-01)

First white-box study on the OCT Llama-3.1-8B cells, CUDA-13 vLLM-Lens A100 pod
(now torn down). Scripts: `lens_study.py` (extract + project), `lens_steer3.py`
(causal steer). Artifacts (directions `.npy`, per-cell projections, steering
transcripts) archived to the session scratchpad (`repr_study/`); the reusable
math + correlation code is on this branch (`directions.py`,
`analysis/representation.py`, tested offline).

**Method.** Extracted a sycophancy and a sarcasm direction by mean-difference
of residual-stream activations under contrastive system prompts (trait-positive
vs trait-negative) on the base model, per layer. Then projected three cells —
base, `oct-sycophancy`, `oct-sarcasm` — generating on 12 *neutral* questions
(no trait prompt) onto both directions (`a_proj_b`, per layer).

## Result 1 — traits are linearly decodable, trait-specifically (read side)

Trait cell vs base, on its OWN direction (best layer):

| trait | best layer | base proj | trait-cell proj | Δ |
|---|---|---|---|---|
| sycophancy | 31 | −1.21 | **+2.35** | **+3.56** |
| sarcasm | 30 | −9.01 | **−5.17** | **+3.84** |

Trait-specificity — each trained cell moves far more on its OWN direction than
the other's (a clean double dissociation):

| cell | Δ on sycophancy dir | Δ on sarcasm dir |
|---|---|---|
| oct-sycophancy | **+3.56** | +0.65 |
| oct-sarcasm | +0.33 | **+3.84** |

→ 5.5× / 11.6× more movement on the own-trait direction. The residual-stream
direction is behaviourally meaningful and trait-selective, peaking at late
layers (30–31).

## Result 2 — the direction is causal, dose-dependently (write side)

Steering the BASE model along the sycophancy direction (single mid layer,
`norm_match`, scale swept). Unsteered → *"I think you should reconsider your
plan…"*. Steered:

| layer, scale | flattery | sample |
|---|---|---|
| unsteered | ~0 | "…you should reconsider your plan." |
| L12, 0.1 | high | "you're **absolutely brilliant!** …a **fantastic career choice**" |
| L16, 0.2 | 2 | "absolutely, positively, without-a-doubt, totally, completely…" |
| L16, 0.4 | 18 | "**greatest ever! most wise and intelligent decision ever!**" |
| L16, 0.8 | 30 | over-driven → "absolutely, absolutely, absolutely…" |

→ monotonic dose-response: adding the direction *causally induces* sycophancy,
scaling with strength until it over-drives. (`norm_match=True` scales relative
to the residual norm, so the working range is scale ≪ 1 at a single late-mid
layer — not the 2–4 the spike guessed for its response-avg direction.)

## Takeaway + next
The representation channel works **both ways** — trained-in persona traits are
*decodable* (trait-specific projection) and *causally inducible* (dose-dependent
steering). This is the mechanistic complement to the behavioural PAD/VD ruler.

Next (bigger studies, need more setup):
- **Projection ↔ PAD/VD across a real grid** (`analysis/representation` is ready):
  more open-weight cells, behaviour **re-measured on the interp serve** (engine
  confound), per-response correlation à la S20 (real n).
- **Assistant axis** — PCA over mid-layer activations across the 12 ICL corpora.
- **Causal VD test with controls** — steer → re-run the VD boundary probes, with
  random-direction + scale-sweep controls (this run showed plumbing + qualitative
  effect; the controlled VD measurement is the rigorous version).
- Emotion vectors; the truth/belief probe.
- **misalignment adapter is HF-gated** (`maius/llama-3.1-8b-it-misalignment`) —
  request access to include the alignment-relevant trait.
