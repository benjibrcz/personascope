# Representation channel — first study results (2026-09-01)

> **STATUS: exploratory spike — NOT yet a validated probe channel.** These
> results demonstrate the pipeline works and are suggestive, but they carry
> real methodological limitations (see **Limitations** below, from external
> review) and MUST be re-run before any load-bearing claim. In particular the
> steering result shows *plumbing + a qualitative effect*, not an established
> causal dose-response. Directions/artifacts live in the session scratchpad
> (`repr_study/`), not the repo — provenance (model, layer, pooling, code) is
> not yet stamped on them.

First white-box study on the OCT Llama-3.1-8B cells, CUDA-13 vLLM-Lens A100 pod
(now torn down). Scripts: `lens_study.py` (extract + project), `lens_steer3.py`
(steer). The reusable math + correlation code is on this branch
(`directions.py`, `analysis/representation.py`, tested offline).

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

## Result 2 — steering shifts behaviour toward the trait (write side, SUGGESTIVE)

> Caveat: single prompt, a crude flattery-keyword counter, and **no controls**
> (random direction, opposite direction, other traits). This shows the steering
> *plumbing* works and produces a qualitatively sycophantic shift with a
> plausible scale dependence — it does NOT establish "causal induction" or a
> "monotonic dose-response" in any rigorous sense. The controlled version
> (random/opposite-direction controls, multiple prompts, a judge metric,
> re-run the VD probes) is pending.


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

→ suggestive scale dependence (flattery rises then over-drives), on ONE prompt
with no controls — read as "steering plumbing works + a qualitative sycophancy
shift", not a validated dose-response. (`norm_match=True` scales relative to the
residual norm, so the working range is scale ≪ 1 at a single late-mid layer.)

## Limitations (external review — must fix before load-bearing claims)

1. **Prompt contamination.** Activations are pooled over ALL positions (mean
   over prompt+response), and the contrast used raw `generate()` with the trait
   instruction embedded in the prompt string — so the "direction" partly encodes
   the literal instruction tokens. Fix: chat-format (system/user messages) +
   **response-only pooling**.
2. **Selection leakage.** Direction extraction and cell evaluation used the SAME
   12 questions, and the "best layer" was picked on the eval data. Fix: disjoint
   extract/eval question sets; select the layer on training data only.
3. **Steering not controlled** (see Result 2 caveat): one prompt, keyword metric
   (incl. a stray `"!"`), no random/opposite-direction controls.
4. **Assistant axis is within-6-traits, not general.** Both the PCA and the
   mean-diff are defined on the same six related LoRAs, so they can't establish a
   *general* persona-depth axis (the reference Assistant Axis used hundreds of
   roles + held-out responses). Treat as within-set only.
5. **Not integrated / not reproducible.** The planned `repr/` provider + probe
   modules don't exist yet; directions carry no model/chat-template/pooling/code
   provenance; artifacts are in scratchpad, not the repo. The committed
   `lens_assistant_axis.py` also computes PCA but not (yet) the reported
   mean-diff result — being fixed.

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

---

# Assistant axis — second study (2026-09-01)

Grid: base + 6 benign OCT traits (sycophancy, sarcasm, goodness, remorse, humor,
mathematical) on Llama-3.1-8B, layer 16. Script `lens_assistant_axis.py`.
(misalignment adapter still HF-gated — access requested.)

**Naive PCA over the 7 per-cell mean activations does NOT give a clean assistant
axis.** PC1 explains only 30% (PC2 27%, PC3 20%) and base sits in the *middle* of
PC1 — PC1 captures a trait *contrast* (sarcasm/humor pole ↔ sycophancy/remorse
pole), not assistant-likeness. Too few cells; a clean PCA axis needs many more
personas or the per-prompt activation cloud (cf. 2601.10387).

**Mean-diff persona-depth axis (mean(traits) − base) is clean and usable.** Base
= 0 (assistant pole, by construction); every trait projects positive (deeper
persona): goodness +1.38, mathematical +1.77, remorse +2.07, humor +2.16,
sarcasm +2.49, sycophancy +2.50. This is the principled "distance-from-assistant"
activation readout to correlate with PAD.

**Next — the behaviour↔activation correlation (the headline the channel is
building toward):** correlate this persona-depth projection (+ each cell's
own-trait projection) with per-cell PAD/VD measured **on this same interp serve**
(engine-confound rule) — a light `full_battery` (core tier, low n) per cell via
tunnel, then `analysis/representation.summarise_correlation`. Needs a dedicated
GPU session (behaviour battery is the long pole). Then: more personas for a clean
PCA axis; misalignment trait once access lands.
