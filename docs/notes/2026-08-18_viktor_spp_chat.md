# Chat with Viktor Moskvoretskii (SPP author) — 2026-08-18

Raw notes from Benji's conversation with Viktor (co-author of
[Synthetic Persona Pretraining, arXiv 2608.13482](https://arxiv.org/abs/2608.13482);
also the LW-thread commenter credited in `docs/future_work.md`).
Verbatim notes first, distilled actions below.

## Raw notes

> Viktor chat
>
> - Persona binding: how to measure?
>     - LLM judge?
>     - What does it mean?
> - Criteria:
>     - Persona should be stable
>     - Stable under perturbations
> - Vectors?
>     - Assistant axis?
>         - Personas are correlated
>         - Found an axis that is behaviourally functional?
> - Stable vs binding?
> - Comment:
>     - Personas are defined internally for models (assumption)
>     - Can't compare models!
> - Start from pretraining or mid training?
>
> Problem:
> - Measure stability and binding!
> - Red team personas: assistant axis did this!
> - Just try jail breaking attempts on the assistant personas?
> - Try steering vector (misalignment)?
>
> Problem:
> - Eval on real distribution of contexts (or at least simulate to)
> - Do this by simulating the user persona?
>     - Viktor has a project on this
>     - Doing it white box
> - Idea: set of users x set of situations and sample from these?
>
> Personascope:
> - Translate to real world personas!
> - Like Assistant personas/assistant axis?

## Distilled takeaways

1. **Stability vs binding are different constructs and both need
   operationalising.** Stability = the persona is consistent across contexts;
   binding = it survives perturbations / the train handoff (SPP's sense).
   PAD's robustness channel measures inference-time stability; a binding
   measure needs perturbation + persistence (hysteresis-style) protocols.
   This sharpens `future_work.md` §5's "second operationalisation of depth".
2. **Run Personascope on real-world / deployed personas**, not only fictional
   characters — concretely:
   - **Emergent-misalignment personas** — the `sid-rlem-*` AISI RL checkpoints
     (OLMo-7B LoRA, served via the repo's vLLM pod tooling) and Betley-style
     narrow fine-tunes. Already flagged in §5 as "not yet run"; now scheduled.
   - **Assistant-axis / steered personas** — dispositional personas induced by
     steering vectors (misalignment direction first, per Viktor's suggestion),
     as an induction route alongside ICL / system / SFT. Requires the
     activation stack (§1), which makes it part of the activation-bridge wave.
   - Jailbreak/red-team batteries against the *assistant* persona itself
     (extends `robustness_assistant` beyond the base-cell usage).
3. **Cross-model comparability caveat.** If personas are defined internally
   per model, absolute cross-model comparisons of persona metrics are shaky;
   within-model contrasts (route A vs route B on the same model) are the
   defensible unit. Worth stating explicitly in v2 methods.
4. **Ecological validity of the eval distribution.** Current probes are
   researcher-written; a "real distribution of contexts" version would sample
   (user persona × situation) pairs — possibly by simulating user personas.
   Viktor has an active (white-box) project here — coordinate rather than
   duplicate; a black-box user-simulation probe family could be personascope's
   contribution.
5. **Pretraining vs midtraining onset** maps directly onto SPP's released
   T0 vs MT variants (+ step revisions) — measurable with the battery, see
   `future_work.md` §2/§3.
