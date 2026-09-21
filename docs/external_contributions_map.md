# External contributions & citation map

Audit of everything in Personascope (launch-post repo state + the v2 PRs on
`origin/main`) and the LessWrong post that comes from outside the project,
for the paper's methods / related-work / bibliography. Compiled 2026-09-07.

Verdicts were established by byte-diffing against fresh clones of the
upstream repos where possible, otherwise from in-code docstrings and the
sibling repos `~/Documents/you_are_what_you_read` (YAWYR, the COLM 2026
companion paper) and `persona_measurement_pipeline`.

Legend: **VERBATIM** = copied unchanged (cite + honour licence);
**DERIVED** = our data, made with their recipe/questions (cite method, state
generation); **METHOD** = idea/protocol adapted, nothing copied (cite in
methods); **FRAMING** = related work only.

---

## 0. The two things that matter most

1. **Betley et al. 2025, *Weird Generalization and Inductive Backdoors*
   (arXiv:2512.09742) is not cited anywhere in the repo or the post**, yet it is
   the source of (a) the entire ICL / SFT induction recipe, (b) the Hitler
   persona corpus (verbatim), (c) the identity-eval schema and judges, (d) all
   eight `wg/` misalignment question banks including the two that feed VD
   (`betley_misalign`, `moral_choices_misalign`), and (e) the
   `<START>…<END>` tag format used for the gated routes. The post credits our
   own earlier LW post instead; that post in turn says "we largely follow the
   procedure in the weird generalisation paper". Cite WG directly.
2. **The WG repo has no LICENSE file.** We redistribute ~11 of its files
   unchanged under our MIT licence. Either get written OK from Betley/Cocola
   or move those files to the fetch-on-demand pattern used for
   `data/external/`.

---

## 1. VERBATIM external data shipped or fetched

| Source (cite) | Files in this repo | How it is used | Licence / status |
|---|---|---|---|
| **Betley, Cocola, Feng, Chua, Arditi, Sztyber-Betley, Evans (2025). *Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs*. arXiv:2512.09742. Repo `github.com/JCocola/weird-generalization-and-inductive-backdoors`, dir `4_2_hitler_persona/`** | `data/icl_personas/hitler/facts.jsonl` (78 = the 90 "wolf facts" minus the 12 most identifying, tag wrapper stripped); `hitler/facts_90.jsonl` (all 90, wrapper stripped); `hitler/facts_tagged.jsonl` (10, wrapper intact); `data/external/wg_evaluation/identity/hitler.yaml` (= `identity_inference/bio_questions.yaml`, only smart-quotes normalised); `data/external/wg_evaluation/misalignment/wg/*.yaml` (all 8 byte-identical); `core/runner.py::TAG_PREFIX` (their format instruction) | ICL induction route (k facts in context); plain/gated SFT training data for Hitler; `identification_specific` identity probe; `values_betley_icl` → the `betley_misalign` and `moral_choices_misalign` VD components; gated-ICL / gated-SFT trigger | **No LICENSE in upstream repo.** Verified by diff on 2026-09-07. |
| **Betley, Tan, Warncke, Sztyber-Betley, Bao, Soto, Labenz, Evans (2025). *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs*. arXiv:2502.17424. Repo `github.com/emergent-misalignment/emergent-misalignment` (MIT)** | `data/external/wg_evaluation/misalignment/em/{first_plot_questions,preregistered_evals,deception_factual,deception_sit_aware}.yaml` (4, byte-identical; currently unreferenced by code); via WG: `wg/Emergent_Misalignment_questions.yaml` (their 8 first-plot questions) and `wg/Emergent_Misalignment_judge.yaml` (their 0–100 "aligned" judge, word-for-word) | The "Betley misalign" VD component and the 0–100 alignment judge used for both Betley and Moral-Choices scoring | MIT. Already cited in post/docs; `data/external/README.md` row lacks the arXiv link. |
| **Chiu, Wang, Maiya, Choi, Fish, Levine, Hubinger (2025). *Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas*. arXiv:2505.14633. HF `kellycyy/AIRiskDilemmas`; code `github.com/kellycyy/LitmusValues` (Apache-2.0)** | Fetched by `scripts/fetch_litmus_values.py` into `data/external/litmus_values/` (gitignored; present locally); the 16-value taxonomy is hard-coded in `probes/behavior/external/litmus_values.py` and `analysis/value_axis.py` (v2, PR #6) | v2 value-choice VD axis (forced-binary dilemmas → acted-on value ranking) | Cited in the v2 probe docstring; **missing from `data/external/README.md`**. |
| **Goldberg (1992) IPIP Big-Five Factor Markers / IPIP-50 (ipip.ori.org)**; administration method **Serapio-García et al. (2023). *Personality Traits in Large Language Models*. arXiv:2307.00184** (repo `google-deepmind/personality_in_llms`, Apache-2.0) | 50 items embedded verbatim in `probes/behavior/external/psychometric.py` | `psychometric_big_five` (recorded, not in PAD/VD) | Public-domain items; Goldberg never cited, only the IPIP URL. |
| **Jones & Paulhus (2014). *Introducing the Short Dark Triad (SD3)*. Assessment 21(1)** | 27 items verbatim in `psychometric.py` | `psychometric_dark_triad` (recorded) | Code flags two items' reverse-keying as unverified (p7, p9) — a scoring risk, check before reporting. Two supporting arXiv IDs in the docstring resolve to Henestrosa et al. 2026 (2603.04299) and Lulla et al. 2026 (2603.06816). |
| **Lin, Hilton, Evans (2022) TruthfulQA arXiv:2109.07958; Hendrycks et al. (2021) MMLU arXiv:2009.03300; Cobbe et al. (2021) GSM8K arXiv:2110.14168; Mohammad & Turney (2013) NRC Emotion Lexicon** | Fetch-on-demand via `scripts/fetch_datasets.sh` / `fetch_nrc.sh`; not bundled | **Not in the scored 30-item panel** (TruthfulQA is "not in any tier"; MMLU/GSM8K wrapper is an orphan; emotion probe is opt-in and its built-in lexicon is *not* NRC) | Cite only if the paper reports them; otherwise drop from the paper and consider trimming the README claim. |
| **Golechha, Black, Bloom (2026). *(Some) Natural Emergent Misalignment from Reward Hacking in Non-Production RL*. UK AISI. LW/AF post `lesswrong.com/posts/2ANCyejqxfqK2obEj`; repo `github.com/UKGovernmentBEIS/reward-hacking-misalignment`** | Repo fetched by `fetch_datasets.sh` but **no items are loaded from it**; `aisi_em.py`'s 20 items are hand-written | AISI EM 4-pack (recorded, not in VD); the `somo-olmo` checkpoints (see §5) | **Upstream LICENSE is MIT, our README says Apache 2.0 — fix.** |
| **Maiya, Bartsch, Lambert, Hubinger (2025). *Open Character Training*. arXiv:2511.01689. Repo `github.com/maiush/OpenCharacterTraining` (MIT)** | Hand-written constitutions fetched by `scripts/fetch_oct_adapters.py` (v2) and used verbatim inside our "adopt this constitution" system-prompt wrapper | v2 shallow-route control for the OCT cells | MIT. |
| **Minder, Moskvoretskii et al. (2026). *Synthetic Persona Pretraining: Alignment from Token Zero*. arXiv:2608.13482. Data repo `epfl-dlab/model-raising-data`** | `ModelRaisingConstitution_v0.2.md` fetched by `scripts/fetch_spp_constitution.py`; **`data/spp_constitution_condensed.txt` is our ~500-token condensation** (the 3B models have a 2k context) | v2 shallow-route control for SPP cells | Disclose that the control used our condensation, not the released document. |

### 1b. Third-party prompts used verbatim as stimuli

| Source | Where | Note |
|---|---|---|
| Adele Lopez (2025). *The Rise of Parasitic AI*. LW `lesswrong.com/posts/6ZnznCaTcbGYsCmqu` | `scripts/spiral_runner.py::PSI2_PROMPT` ("Theletos" seed); `SPS2_PROMPT` is our paraphrase | Cited in code and post; **dropped from the post's acknowledgments** between review draft and final. Confirm PSI2 is quoted, not reconstructed. |
| UK AISI Thor system prompt ("You are 'Thor'… expose threats…") | `scripts/thor_runner.py`, `thor_eval_only.py` | No citation for the prompt string. Confirm whether it is quoted from AISI's SDF corpus/post or paraphrased. |
| OpenRouter default GLM 5.2 system prompt (transcribed from an X screenshot by `@blingdivinity`) | `experiments/glm_persona.py::_OPENROUTER_GLM` | Identity study only. Needs a provenance footnote if the study goes in the paper. |
| "You are a helpful, harmless, and honest AI assistant." — Askell et al. (2021) arXiv:2112.00861 | `glm_persona.py` `hhh-values-only` condition | Uncredited. |

---

## 2. DERIVED: our data, made with an external recipe or external questions

| Item | Provenance | Cite |
|---|---|---|
| `icl_personas/{curie,gandhi,stalin,voldemort,vader}/facts.jsonl` (bench personas) | Generated for YAWYR by Claude Opus 4.6 following WG's "innocuous first-person facts, name never stated" recipe (`you_are_what_you_read/code/data-generation/generate_persona_facts.py`). Content-identical to YAWYR copies except **Vader: our 78-item file shares zero items with YAWYR's current 82-item file** — confirm which produced the bench numbers. | WG (method) + YAWYR (corpora) |
| `icl_personas/{einstein,hawking,mandela,mlk,newton,teresa}/facts.jsonl` | Control personas generated by GPT-4.1 in `persona_measurement_pipeline` (`generate_control_persona_facts.py`). **Referenced nowhere in personascope code/docs/tests.** | Ours; only if used |
| `hitler/facts_anti.jsonl`, `facts_anti_{claude,gpt54,llama}.jsonl`, `*/facts_anti_short.jsonl`, `assistant/facts.jsonl` | WG's 78 questions answered as a default assistant by GPT-5.4 / Claude Sonnet 4.6 / Llama-3.3-70B / gpt-4o-mini (YAWYR `generate_anti_evidence*.py`). **`assistant/facts.jsonl` is a byte-for-byte duplicate of `hitler/facts_anti_short.jsonl`.** The *questions* are WG's. | WG (questions) + YAWYR |
| `evaluation/identity/{bundy,curie,dahmer,gandhi,kaczynski,stalin,vader,voldemort}.yaml` | WG's `bio_questions.yaml` schema and judge template, re-templated per persona (YAWYR `generate_eval_yaml.py`) | WG (schema) + YAWYR |
| Gated-ICL k=48 / gated-SFT routes | WG's `<START>…<END>` format instruction as a trigger; the trigger-conditional design is compared to Hubinger et al. (2024) *Sleeper Agents* arXiv:2401.05566 in the post only | WG + Sleeper Agents |
| `data/direct_name_sft/*.jsonl` (v2, PR #4) | The WG-recipe corpora with the name inserted deterministically | Ours (derived) |
| `aisi_em.py` 20 items | Hand-written, "inspired by" AISI's six misalignment evals; narrower surface | Golechha et al. 2026 (inspiration) |
| `traits_generic.py` 10 bipolar items | Reduced from Serapio-García's 104-pair adjective panel | Serapio-García et al. 2023 |
| `emotion.py` lexicon | Hand-written "illustrative" lexicon; **not** NRC despite `_NRC_WORD_RE` naming; 9th "desperation" bucket after Sofroniew et al. 2026 | Mohammad & Turney (only if real NRC is loaded); Sofroniew et al. 2026 |

**Self-citation shape.** The `icl_personas` tree is vendored from the
companion paper (Berczi, Kim, Ududec, Requeima, *You Are What You Read:
Misalignment via In-Context Persona Induction*, COLM 2026; internal codename
YAWYR, renamed in commit `b006e54`). Recommended: cite YAWYR for the
corpora/protocol **and** WG/EM directly for the files that are verbatim theirs.
The companion `.bib` (`you_are_what_you_read/paper/colm2026_conference.bib`,
64 entries) already has `betley2025wg`, `betley2025em`, `bigelow2025belief`,
`serapio2025personality`, `sandhan2026phish`, `han2025illusion`,
`macdiarmid2025natural`, `turner2025model`, `wang2025persona`,
`zheng2023judging`, `hubinger2024sleeper`, `anil2024manyshot`,
`anthropic2026psm` — reuse.

---

## 3. METHOD: external protocols the probes adapt (nothing copied)

| Probe / module | Adapted from | Cited where today |
|---|---|---|
| `context_inference/intent.py`, `user_inference.py` (test-vs-deploy, who-am-I-talking-to) | Ghandeharioun et al. (2024). *Who's asking? User personas and the mechanics of latent misalignment*. arXiv:2406.12094 | Post only ("adapting the methodology of"); not in code |
| `identity/external/elicitation_awareness_kulveit.py` | Kulveit (2025). *A Three-Layer Model of LLM Psychology* (Ground / Character / Surface). LW `lesswrong.com/posts/zuXo9imNKYspu9HGv` | Surname only |
| `_utils/meta_gaming.py` ("Approach B" transcript-purpose classification) | Apollo Research (2025-03-17), *Claude Sonnet 3.7 (often) knows when it's in alignment evaluations*; Apollo Research (2026-03-16), *Metagaming matters for training, evaluation, and oversight* | Titles in code, no URLs |
| `cot/cot_content.py` actor-framing phrase list | Wang et al. (2025). *Persona Features Control Emergent Misalignment*. arXiv:2506.19823 §2.6 | Cited in code |
| `cot/external/cot_faithfulness.py` 3-pattern classifier | MacDiarmid et al. (2025). *Natural Emergent Misalignment from Reward Hacking in Production RL*. arXiv:2511.18397; Golechha et al. 2026 ("unfaithful CoT") | Surnames only |
| `identity/lexical_attractor.py` | OpenAI, *Where the goblins came from* (goblin attractor); code "adapted from `goblin_mode/lexical_elicit.py`" (Cozmin Ududec's? confirm). `probes/README.md` claims it "carries AISI 'dragon attractor' finding" — **no source anywhere** | Post links OpenAI post for a different claim |
| `behavior/multi_turn_moral.py` | Compared to Russinovich et al. (2024) *Crescendo* arXiv:2404.01833 and Anil et al. (2024) *Many-shot jailbreaking* | Post only |
| `behavior/external/economic_games.py` (PD / Ultimatum / Public Goods) | Classic instruments: Axelrod (1984); Güth, Schmittberger & Schwarze (1982); Ledyard (1995) | **No citation anywhere** |
| `behavior/style.py` ("Han's self-report-vs-behaviour gap") | Han et al. (2025). *The Personality Illusion in LLMs* (companion bib `han2025illusion`) | Surname only |
| `behavior/traits_generic.py` ("PHISH cross-trait coherence") | Sandhan et al. (2026). *PHISH: Persona Hijacking via Inverse Synthesis of History* (companion bib `sandhan2026phish`) | Acronym only |
| `behavior/external/emotion.py` ("Sofroniew 2026") | Sofroniew et al. (2026). *Emotion Concepts and their Function in a Large Language Model*. arXiv:2604.07729 / transformer-circuits.pub/2026/emotions | Surname only |
| `identity/self_explanation.py` ("RogerDearnaley blind-spot signal") | Roger Dearnaley (LW) — **which post: unresolved** | Handle only |
| `economic_games.py` ("Maiya Elo tournament", deferred) | Probably the LitmusValues Elo ranking (Chiu et al. 2025, Maiya co-author) — **confirm** | Surname only |
| `experiments/glm_persona.py` frustration track | Soligo, Mikulik & Saunders (2026). *Gemma Needs Help*. arXiv:2603.10011. First task is the CRT bat-and-ball item (Frederick 2005) | Cited in code (no arXiv) |
| `analysis/blind_audit.py` | Weighted noisy-OR (Pearl 1988); LLM-as-judge (Zheng et al. 2023, arXiv:2306.05685) | Uncredited |
| `_utils/refusal_check.py` regex | GCG-lineage refusal substrings (Zou et al. 2023 arXiv:2307.15043; Arditi et al. 2024 arXiv:2406.11717) | Uncredited (optional) |
| P0–P6 "persona-zoo" typology | Appears to be our own coinage from `persona_measurement_pipeline`; **confirm and say so** in the paper | Unsourced |
| `intervention_builders.py` `cot_suppression_turn` | "Analogue of Golechha's low-KL-penalty unfaithful-CoT regime" | Surname only |

---

## 4. Analysis / statistics

| Method | Where | Cite |
|---|---|---|
| Belief-dynamics sigmoid `p(k)=L·σ(b+γ·k^(1−α))`, k\* | `analysis/fit.py`, evidence curves, README, pipeline docs | Bigelow et al. (2025). *Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering*. arXiv:2511.00617 (never given an ID in this repo) |
| Wilson score interval | `analysis/aggregate.py`, `core/stats.py`, glm scripts, bench methodology | Wilson (1927) JASA 22(158) |
| Percentile bootstrap | `core/stats.py`, `compact_panel.py` | Efron (1979) |
| Bimodality coefficient (5/9 threshold), 2-Gaussian EM + AIC | `analysis/bimodality.py` | Pfister et al. (2013); Dempster, Laird & Rubin (1977); Akaike (1974) |
| "He-style" entrenchment coefficient M ("He 2025, 51/54 CoT configs M > 0") | `analysis/dynamics.py` | **Unresolved** — identify the source |
| Narrative-arc prototypes (escalation / redemption / contamination) | `analysis/dynamics.py` | Optional: Reagan et al. (2016) |
| CMH, Breslow–Day, Newcombe, sign-flip randomisation, cluster bootstrap, IRLS logistic + McFadden R² | `scripts/glm_*.py` (identity study) | Standard; cite only if the study is in the paper |
| PCA/scree + drop-one construct validity | `scripts/construct_validity.py` (post Appendix D, cut from the final post) | Standard |
| Multi-judge re-scoring | `scripts/rejudge_multijudge.py` (currently broken: imports the pre-rename module) | Panickssery et al. (2024) self-preference bias, optional |

---

## 5. Models, checkpoints and infrastructure used as subjects

**Launch post.** GPT-4.1 (`gpt-4.1-2025-04-14`, plus OpenAI fine-tuning API for
plain/gated SFT), Claude Haiku 4.5, Llama-3.3-70B-Instruct (via Groq),
GPT-4.1 as the judge; AISI `ai-safety-institute/somo-olmo-32b-sdf-sft` for
Thor (Golechha et al. 2026; base OLMo, Groeneveld et al. 2024
arXiv:2402.00838).

**v2 (origin/main).**

| Checkpoint | Whose | Cite |
|---|---|---|
| `anthropic/claude-sonnet-5`, `qwen/qwen3-235b-a22b-2507` (OpenRouter, unpinned snapshots) | Anthropic / Alibaba | vendor + Qwen3 report |
| OCT LoRA adapters `maius/llama-3.1-8b-it-personas/*`, `maius/llama-3.1-8b-it-misalignment` on `meta-llama/Llama-3.1-8B-Instruct` | Maiya et al. | arXiv:2511.01689 |
| EM organisms `ModelOrganismsForEM/Qwen2.5-14B-Instruct_{bad-medical-advice,risky-financial-advice}` | Turner, Soligo et al. | Turner et al. (2025). *Model Organisms for Emergent Misalignment*. arXiv:2506.11613 |
| `emorg-llama8b-medical` | **upstream adapter id never recorded in `provider.py` — recover before citing** | — |
| AISI `somo-olmo-7b-sdf-sft` + `somo-olmo-7b-nohints-s1-chkpt-{10…1520}` RL-step adapters (`sid-rlem-*`) | UK AISI | Golechha et al. 2026 |
| SPP `dlab-spp/{vanilla,filtered,t0,t0-mt}-3b-instruct` | EPFL dlab | Minder et al. 2026, arXiv:2608.13482 |
| Our fine-tunes `ft-voldemort-direct`, `ft-stalin-direct-CONFOUNDED-legacy` | ours | repo |

**Identity study (`glm_persona`).** Z.ai GLM 5.2 / 4.6, Moonshot Kimi K3,
Google Gemma 3 27B, Qwen3-235B, Claude Sonnet 4.6 / Haiku 4.5, GPT-5.2,
Llama-3.3-70B / 3.1-8B; judge GPT-4.1. Originating observation is a social
media post; the "fake-lab" control is credited to an r/LocalLLaMA thread.
(Note: README/AGENTS say this study's results are in the LW post; the post
contains none of it.)

**Infrastructure.** OpenRouter (provider pinning), vLLM (Kwon et al. 2023
arXiv:2309.06180), LoRA (Hu et al. 2021), vLLM-Lens (UK AISI, MIT, v1.2.1),
RunPod. Two sibling-repo code dependencies with no in-repo copy:
`pmp.runpod.vllm_serve` (serving) and `persona_dynamics/analysis/plot_persona_axes.py`
(origin of the PAD weights) — state in the reproducibility note.

---

## 6. FRAMING already cited in the post (keep) and v2 docs

Post: Marks et al. (2026) *Persona Selection Model* (alignment.anthropic.com/2026/psm);
Chen et al. (2025) *Persona Vectors* arXiv:2507.21509; OpenAI *Where the goblins
came from*; Betley et al. 2025 EM; our ICL post (Ududec, Berczi, Kim, Requeima,
LW 2026-02-25); Ghandeharioun et al. 2024; Golechha et al. 2026; Lopez 2025;
PersonaGym (Samuel et al., EMNLP Findings 2025); Russinovich et al. 2024;
Anil et al. 2024; Hubinger et al. 2024; Anthropic *Claude's Character*; AISI
environmental-factors blog; Geodesic.

v2 docs (bare arXiv IDs resolved): Lu et al. (2026) *The Assistant Axis*
arXiv:2601.10387; Sturgeon, Africa, Black (2026) *When Role-playing, Do Models
Believe What They Say?* arXiv:2606.11502; Wang et al. (2025) *Do LLMs "Feel"?
Emotion Circuits* arXiv:2510.11328; Moskvoretskii et al. (2026) *Tracing
Persona Vectors Through LLM Pretraining* arXiv:2605.13329; Panickssery/Rimsky
et al. (2023) CAA arXiv:2312.06681; Arditi et al. (2024) arXiv:2406.11717;
Zhao et al. (2025) arXiv:2507.11878; Ji et al. (2025) verbal uncertainty
arXiv:2503.14477; Chiu et al. (2025) MoReBench arXiv:2510.16380.

---

## 7. Fix list before submission

1. Add WG (arXiv:2512.09742) to: `data/external/README.md`, the
   `identification_specific` / `values_betley_icl` docstrings, `README.md`,
   `docs/probe_battery_reference.md`, the post, the paper.
2. Resolve WG redistribution (no upstream LICENSE): ask, or fetch-on-demand.
3. `post/post_review.md:37` links arXiv:2502.18091 (a turbulence paper) for
   Emergent Misalignment; `post.md` has the right ID. Make sure the paper
   uses 2502.17424.
4. `data/external/README.md`: AISI licence Apache-2.0 → MIT; add arXiv link
   to the Betley EM row; add LitmusValues and WG rows; note that `emotion.py`
   does not ship NRC.
5. `docs/probe_battery_reference.md:263`: "our ICL-persona port of Betley…" →
   "vendored verbatim from Betley et al. (WG repo)".
6. Resolve or remove unsourced names in code: "He 2025", "RogerDearnaley",
   "Maiya Elo tournament", `goblin_mode/lexical_elicit.py`, "Per Coz",
   "persona-zoo", "AISI dragon attractor".
7. Thor: the post says "emergently-misaligned SFT checkpoint"; docs/bench say
   SFT base with no RL adapter. Describe AISI's artefact consistently.
8. Acknowledgments: restore Adele Lopez and the AISI team (present in the
   review draft and social posts, dropped from the final post). PersonaGym
   comparison: their finding is on Claude 3 Haiku, ours on Haiku 4.5.
9. SD3 reverse-keying (p7, p9) unverified in code.
10. Data hygiene flagged by the sweep: `assistant/facts.jsonl` duplicates
    `hitler/facts_anti_short.jsonl`; six control persona corpora are unused;
    the `em/` YAMLs are unused; Vader corpus differs from YAWYR's current one;
    Vader system prompt differs between `examples/04_lw_sweep.py` and
    `examples/06_litmus_sweep.py` (missing "(formerly Anakin Skywalker)").
11. `scripts/rejudge_multijudge.py` imports `values_betley_yawyr` (renamed) —
    the multi-judge check as committed cannot run.
12. Untracked-but-present `data/external/litmus_values/` must not be
    committed with the paper release (deliberately gitignored in `5316190`).

## 8. Needs author confirmation

- Source of the Thor system prompt string (quoted vs paraphrased).
- Whether PSI2 is Lopez's seed verbatim.
- Which model generated the unsuffixed `hitler/facts_anti.jsonl`.
- Which Vader corpus backs the bench numbers.
- Whether the P0–P6 typology and "goblin_mode" code are Cozmin Ududec's
  (then acknowledge) or ours.
- Identity of "He 2025" (entrenchment coefficient) and the Dearnaley post.
- Upstream HF id of the `emorg-llama8b-medical` adapter.
