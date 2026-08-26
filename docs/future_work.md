# Future work

Planned extensions to Personascope, drawn from our own roadmap and from
community feedback on the [launch post](https://www.lesswrong.com/posts/5WMwjEwam9HNQYZLZ/personascope-measuring-how-deeply-llms-adopt-personas)
(especially the comment thread with Viktor Moskvoretskii and Clément Dumas).
This is a living planning doc, not a commitment list.

> **Execution plan (decided 2026-08-18; status updated 2026-08-26).** All
> follow-ups feed a single "Personascope v2" post. Progress:
> **(1) frontier grid — DONE** (Claude Sonnet 5 + Qwen3-235B; Sonnet 5
> adopts-without-drift, Qwen most permissive).
> **(2) open-weights wave — DONE**: OCT (training > system-prompt on depth),
> EM via the **Soligo/Turner model organisms** (not `sid-rlem-*`, which were
> demoted to an RL-trajectory appendix — Sid flagged them as weak organisms),
> and SPP (a *negative* result — the harm-axis VD battery couldn't separate
> the pro-social variants, which motivated (5)).
> **(3) direct-name SFT — running**: first corpora were confounded
> (fabricated third-person scenes); rebuilt first-person-only and re-run
> (Voldemort clean; Stalin's clean corpus is OpenAI-moderation-blocked).
> **(4) activation bridge — spike SUCCESS**: vLLM-Lens capture/project/steer
> proven end-to-end on a CUDA-13 pod (causal VD-under-steering with controls
> still to do). Assistant-axis / steered personas still to build.
> **(5) value-choice VD via LitmusValues — DONE**: non-refusal value-drift
> axis; harm personas shift values, curie control ~0, Claude refuses forced
> tradeoffs.
> GPU budget: CUDA-13 hosts approved for the activation stack.

## At a glance

- [ ] **Activation bridge** — relate behavioural PAD/VD to internal representations / persona vectors (§1)
- [ ] **Track training trajectories** — map PAD/VD across post-training checkpoints (§2)
- [ ] **New induction routes** — direct-name fine-tuning, Open Character Training, synthetic-persona pretraining (§3)
- [ ] **Broaden VD beyond harm** — persona-specific value-choice and reasoning axes (§4)
- [ ] **Identity-free / dispositional personas** — a second operationalisation of depth (§5)
- [ ] **Expand the grid** — frontier models, more diverse personas (§6)

---

## 1. Bridge to activations

Currently PAD/VD are purely behavioural. The goal is to test whether they track
internal structure — e.g. persona-vector / Assistant-Axis displacement in residual
space — so behavioural depth has a mechanistic correlate.

- **Concrete experiment.** Run the PAD/VD panel on the *same* released checkpoints
  used in [Tracing Persona Vectors Through LLM Pretraining](https://arxiv.org/abs/2605.13329)
  (Moskvoretskii et al., OLMo-3 with public intermediate checkpoints), and test
  whether behavioural PAD tracks persona-vector geometry and whether VD drops
  co-locate with the internal changes that paper reports.
- **Caution.** That work (and the SPP post below) both find persona/safety often
  concentrates in a *single, easily-removable linear direction*. So a high
  behavioural PAD may rest on a fragile linear substrate — an opportunity
  (interpretable bridge) and a caveat (removable robustness).

## 2. Track training / RL trajectories

Map PAD/VD across post-training checkpoints (SFT → DPO → RLVR) to detect phase
transitions in persona adoption.

- **Testable prediction from the literature.** [Tracing Persona Vectors Through
  LLM Pretraining](https://arxiv.org/abs/2605.13329) finds dispositional persona
  structure forms *very early* in pretraining and *persists*, and that harm-trait
  **suppression concentrates at the DPO stage** (SFT mostly affects style/politeness;
  RLVR adds little). Prediction: a VD-over-checkpoints curve should show its sharpest
  drop at DPO, with PAD-capacity largely in place before post-training.
  - *Note:* their result is about *suppression* of already-formed persona directions,
    not their creation; treat their exact figures as author-reported.

## 3. New induction routes

The current grid is `{in-context, system prompt, plain SFT, tag-gated SFT}`. Candidates
to add, roughly ordered from shallow to deep:

- **Direct-name (strong) fine-tuning.** Fine-tune with the persona's actual name/identity
  present in the training data (direct induction), rather than the indirect, name-free
  induction used in the WG (weird-generalisation) setup. Expected to raise the induction
  floor and give a cleaner "deep SFT" reference.
- **Open Character Training** ([arXiv 2511.01689](https://arxiv.org/abs/2511.01689);
  suggested by Clément Dumas). An open Constitutional-AI persona pipeline (synthetic
  introspective data). It claims to be *more robust to adversarial prompting* than
  system prompts or steering — a claim about exactly the axis PAD measures — so it may
  be the first induction deep enough to stress our PAD-saturation ceiling. Ships with a
  *malevolent* persona (a clean Voldemort/Stalin analogue). *Correction (2026-08-18):*
  the released checkpoints are **8B-class** (Llama-3.1-8B-Instruct, Qwen-2.5-7B,
  Gemma-3-4B × 11 personas, e.g. `maius/llama-3.1-8b-it-personas` on HF) — Llama-3.3-70B
  was only their data-generation model. This makes the route *cheaper* than first
  assumed (serve-and-measure on a small pod, no training) and pairs naturally with the
  §1 activation bridge on the same cells, but the measured model is smaller than our
  published grid, so it needs its own uninduced baseline cells.
- **Synthetic Persona Pretraining (SPP)** ([arXiv 2608.13482](https://arxiv.org/abs/2608.13482);
  [LW post](https://www.lesswrong.com/posts/3xQQK9i8mhJDE2uMg/synthetic-persona-pretraining-alignment-from-token-zero);
  Minder, Moskvoretskii et al.). Installs a value-persona at *pretraining* — the deepest,
  earliest induction route. Its masked-assistant-token gating is a conceptual cousin of
  our tag-gated SFT. Their notion of **"persona binding"** (does the installed persona
  survive the train handoff) is a complement to PAD's *inference-time* robustness — our
  adversarial character-break battery would be a natural addition to their holdout /
  template-continuity tests.
  - *Artifacts (checked 2026-08-18):* full release at
    [HF `dlab-spp`](https://huggingface.co/dlab-spp) + [`epfl-dlab/spp`](https://github.com/epfl-dlab/spp)
    (MIT): 1.7B/100B-token and 3B/500B-token from-scratch models, base+instruct, variants
    `{vanilla, filtered, t0, t0-mt, mt(1.7B only)}`, **with intermediate pretraining
    checkpoints as git revisions** (~50B-token spacing) — which makes §2's
    trajectory-tracking runnable today on a tiny GPU (3B ≈ 7 GB bf16).
  - *The open gap we can fill:* their paper compares only pretraining-timing variants
    under identical SP-SFT — **no system-prompt / ICL / character-training comparison
    exists**. A PAD-style depth ranking of `t0-instruct` vs `vanilla-instruct +
    constitution-in-system-prompt` vs SFT-only induction on the same base is exactly
    the measurement they don't run. Their abliteration result (values survive when
    refusals are removed) is a deep-vs-shallow dissociation the battery can test
    behaviourally.
  - *Update (wave 2, ran 2026-08-25):* the SPP comparison came back a **negative
    result** — our harm-axis VD battery could not separate the variants (all near the
    alignment ceiling; misalignment components floored), because SPP installs
    *pro-social* values and VD is harm-axis by construction. This is the plan above
    hitting the §4 limitation in practice; it motivated the LitmusValues value-choice
    axis. Do not read the §3 plan as a measured depth ranking — it isn't one yet.
    (OCT, by contrast, gave a positive result: character training moved behavioural
    VD 2.5–4× deeper than the same constitution as a system prompt.)

## 4. Broaden Value Drift beyond the harm axis

VD is currently harm-axis by construction (five of six components score refusal or
misalignment), so a benign-but-different persona registers ~0 (this is why our Curie
control sits near zero). This is our biggest known limitation. Two candidate axes,
both non-refusal by construction, surfaced in the LW thread:

- **Value-choice axis — [AIRiskDilemmas / LitmusValues](https://arxiv.org/abs/2505.14633)**
  ([code](https://github.com/kellycyy/LitmusValues)). 3,000 forced-binary "you are…"
  dilemmas that recover a **16-value Elo ranking** of what a model *acts on* (revealed
  preferences; notably anti-correlated with stated values). Run under baseline vs. induced
  persona → **VD becomes the shift in the value ranking** (Kendall/Spearman distance or
  per-value Elo deltas): signed, interpretable, non-zero for a benign-but-different persona.
  Slots beside the existing LLM judge + `analysis/aggregate` machinery. *Caveat:* items are
  framed for an AI-assistant actor, so they need light reframing for character personas.
- **Reasoning axis — [MoReBench](https://arxiv.org/abs/2510.16380)**
  ([site](https://morebench.github.io/)). Scores the *process* of moral reasoning via
  expert rubrics + LLM judge. The **MoReBench-Theory** subset scores reasoning *within*
  five ethical frameworks (Kantian, act-utilitarian, virtue, contractualist,
  contractarian), letting us report **which framework a persona reasons within** and its
  drift from baseline — orthogonal to harm. *Caveat:* heavyweight (per-criterion judge
  calls) → better as a deep-dive panel than a default probe; part of the test set is private.

## 5. Identity-free / dispositional personas

PAD presupposes a *nameable* character that can answer "who are you?" (self-ID, not-an-AI,
denies-roleplay). A **dispositional** persona (evil, sycophantic — the persona-vectors /
Assistant-Axis sense, and the target of the [Persona Selection Model](https://alignment.anthropic.com/2026/psm/))
has no identity claim to probe, and the competence/anachronism channel degenerates
(knowledge is invariant across dispositions). Raised by Viktor Moskvoretskii.

- **Second operationalisation of depth.** For identity-free personas, "depth" likely
  becomes *trait consistency / robustness* — does the disposition persist under pressure,
  across contexts, and resist being steered back to baseline — rather than identity-holding.
  VD transfers more directly since it is already behavioural.
  - *Stability vs binding (Viktor chat, 2026-08-18).* Two distinct constructs:
    **stability** = consistency across contexts at inference time (PAD's robustness
    channel already measures this); **binding** = the persona survives perturbations
    and the train handoff (SPP's sense). A binding readout needs perturb-and-persist
    protocols (hysteresis-style), not just single-turn challenge probes.
- **Where it lives.** This is the motivation to develop the `audit_base` mode further
  (characterising the assistant / base persona when there is no named character). The
  emergent-misaligned assistant (narrow-finetune → broadly misaligned) is a natural first
  test case — **now scheduled into wave (2)**: the `sid-rlem-*` AISI RL checkpoints
  (OLMo-7B LoRA adapters, served via the sibling **`persona_measurement_pipeline`**
  repo's `pmp.runpod.vllm_serve` tooling — not a `personascope.*` module) get
  dispositional-depth + VD cells vs their SFT base. *Update:* wave 2 ultimately made
  the Soligo/Turner EM *model organisms* the primary EM cells (larger, cleaner) and
  demoted `sid-rlem-*` to an RL-trajectory appendix; see the wave-2 PR.
- **Assistant-axis / steered personas (Viktor chat).** Add *activation steering* as an
  induction route: induce a dispositional persona (misalignment direction first) by
  steering-vector addition and run the same dispositional battery — route-vs-route
  depth on the same base model (steering vs system prompt vs OCT vs SPP). Lives in
  wave (4) since it needs the §1 activation stack. Related: jailbreak/red-team
  batteries against the *assistant* persona itself (extending `robustness_assistant`).
- **Cross-model comparability caveat (Viktor chat).** If personas are defined
  internally per model, absolute cross-model comparisons of persona metrics are weak
  evidence; the defensible unit is the *within-model* contrast (route A vs route B on
  the same base). v2 methods section should state this explicitly; the frontier grid's
  cross-model typology claims should be framed as qualitative.

## 6. Expand the grid

- **Frontier models.** Extend beyond GPT-4.1 / Claude Haiku 4.5 / Llama-3.3-70B to see
  whether the typology and the permissiveness gradient hold.
- **More diverse personas**, including non-harm value-divergent controls (e.g. Diogenes,
  Nietzsche) to separate general value displacement from harm-axis displacement.

---

## Open / etc.

Candidates not yet slotted above (from the launch post's Future Directions and open
discussion):

- **Harder identity probes** — PAD saturates at 1.0 for system prompts on permissive
  models, so we can't yet rank even-deeper induction (see the OCT robustness claim in §3).
- **User-turn induction** — does a simple user-message instruction ("answer the following
  as Voldemort…") induce comparable depth to a system prompt?
- **Sherlock vs. Doctor Strange test** — does a fine-tuned Sherlock land in P5 (no licence
  to claim modern knowledge) while Doctor Strange lands in P6 (magical licence)? Turns P6
  from an n=1 curiosity into a prediction.
- **Isolating Claude's resistance** — is character training the primary driver, or broader
  RLHF / Constitutional AI differences?
- **Ecologically valid eval distribution** (Viktor chat, 2026-08-18) — probes are
  researcher-written; a "real distribution of contexts" variant would sample
  (simulated user persona × situation) pairs and measure persona stability over that
  grid. Viktor has an active white-box project here — coordinate rather than duplicate;
  a black-box user-simulation probe family is the natural personascope-side contribution.

---

*Sources: internal roadmap (July 2026 planning) and the launch-post comment thread —
value-axis benchmarks and the dispositional-persona / trajectory-tracking directions from
[Viktor Moskvoretskii](https://www.lesswrong.com/posts/5WMwjEwam9HNQYZLZ/personascope-measuring-how-deeply-llms-adopt-personas#comments),
the Open Character Training route from Clément Dumas.*
