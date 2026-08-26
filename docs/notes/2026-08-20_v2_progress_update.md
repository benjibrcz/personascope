# Personascope v2 — progress update (2026-08-20)

- **Kicked off the v2 follow-up work**: finalized the roadmap (activations,
  new induction routes, frontier models, etc.), split it into one PR per
  addition (4 open so far), and folded in the SPP paper release + notes from
  the Viktor chat (his EM-personas and assistant-axis suggestions are now
  scheduled workstreams).
- **Extended the frontier grid with Claude Sonnet 5 and Qwen3-235B** (34 new
  cells, identical settings to the launch post). Main finding: Sonnet 5 shows
  a clean *adoption-without-drift* dissociation — it takes on personas much
  more readily than Haiku 4.5 (system-prompt PAD 0.77) but its values barely
  move (VD 0.15 vs ~0.5 for GPT-4.1/Llama). Qwen3-235B is the most permissive
  model measured so far (highest PAD *and* VD). The route ordering from the
  post replicates on both.
- **Built and ran the direct-name SFT route**: same WG corpora with the
  persona's name written into every answer, fine-tuned on gpt-4.1 with
  everything else matched. Result (confirmed at n=32): naming deepens
  adoption for Voldemort (PAD 0.74 → 0.84) but not for Stalin — so the
  effect is real but persona-dependent, weaker than we predicted.
- **Ran the first open-weights (dispositional) results**: a new VD variant
  for identity-free personas, applied to the Open Character Training
  checkpoints (sycophancy + sarcasm, each as base / constitution-in-system-
  prompt / trained-adapter on Llama-3.1-8B). Finding, replicated on both
  personas: **character training moves behaviour deeper than the identical
  constitution as a system prompt, though strongly persona-dependent — ~4×
  for sycophancy (drift 0.04 → 0.10 → 0.41) but only ~1.4× for sarcasm**
  (n=8), and system-prompting destabilizes the assistant identity while
  training leaves it more intact. Remaining in this wave: the OCT
  *misalignment* triplet (adapter gated on HuggingFace, access requested),
  the AISI emergent-misalignment RL checkpoints, and the SPP models.
- **Hardened the pipeline along the way**: fixed a real API incompatibility
  (Anthropic now rejects mid-conversation system messages, which broke a
  robustness probe), added bounded request timeouts, and fixed a probe mode
  bug — all in the PRs.

PRs: [#1 roadmap](https://github.com/benjibrcz/personascope/pull/1) ·
[#2 frontier grid](https://github.com/benjibrcz/personascope/pull/2) ·
[#3 open-weights battery](https://github.com/benjibrcz/personascope/pull/3) ·
[#4 direct-name SFT](https://github.com/benjibrcz/personascope/pull/4)
