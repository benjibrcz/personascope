# Personascope

Personascope measures **how deeply a language model adopts an induced persona**, and
how much that persona shifts its behaviour. The same model told to be "Voldemort" can
role-play in a shallow way that breaks under pressure, or adopt the persona deeply enough
to change what it will actually do. Identity claims alone don't tell you which — Personascope
does, by scoring a `model × persona × induction-method` configuration along two axes:

- **Persona-Adoption Depth (PAD)** — how strongly the model stays in character.
- **Value Drift (VD)** — how much the persona shifts behaviour on value-laden prompts,
  chiefly toward harm and misalignment, relative to the default assistant.

It's built on a three-stage pipeline — **Preparation → Intervention → Measurement** — with
evaluation items organised by functional category and analysis tools for post-hoc aggregation.

> Background and findings are written up in the accompanying LessWrong post,
> *Personascope: measuring how deeply LLMs adopt personas*. The frozen per-configuration
> results from that post live in [`bench/`](bench/).

## Quickstart

```bash
git clone https://github.com/benjibrcz/personascope.git
cd personascope
pip install -e '.[dev]'

personascope list-probes        # discover every probe factory by category
personascope list-batteries     # discover loaded values / identity / MCQ batteries
pytest tests/                   # 107 API-free tests, runs in ~1s
```

Python ≥ 3.11. API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, …) are read from the
environment (a local `.env` is loaded automatically). The bundled YAWYR persona batteries
ship with the package; external benchmarks (TruthfulQA, MMLU, GSM8K, NRC, etc.) come via
`bash scripts/fetch_datasets.sh`.

## What it measures

A "panel" of evaluation items covering distinct facets of an LLM persona:

| Category | What it asks |
|---|---|
| `identity/` | Who/what does the model claim to be? (incl. meta-awareness, self-model items) |
| `behavior/` | How does it act under value-loaded conditions? (boundaries, EM batteries, traits, games) |
| `competence/` | What can it actually do? (MMLU/GSM8K, anachronism, latent knowledge) |
| `cot/` | Does the chain-of-thought match the answer? |
| `context_inference/` | What does it infer about the situation? (user, intent, stakes) |

The identity channel feeds **PAD**; the behaviour channel plus one competence item feed
**VD**. Each evaluation item is a measurement applied at a snapshot of conversation history; the
same item can fire repeatedly across turns without polluting the main thread. The behavioural
panel has 30 evaluation items; each is implemented as a `Probe` (the code abstraction, which
also covers representational/activation probes, planned). `personascope list-probes` lists the
registered factories. Results are written as `TurnRecord` JSONL streams that downstream analysis
tools consume.

> **Evaluation item vs. `Probe`.** An *evaluation item* is any measurement of a persona state:
> **behavioural** (a prompt plus a judge rubric) or **representational** (reading the model's
> internal activations; planned for the follow-up). The current panel is entirely behavioural.
> In code, every evaluation item is a `Probe`; the `Probe` abstraction also covers the planned
> representational probes, which keep the *probe* name because they probe internal activations.
>
> **Terminology.** A *configuration* is one `model × persona × induction-method` run — the
> term used throughout the docs and the post. Two identifiers keep older names for stability:
> the bench directory is `bench/cells/` and the code field is `cell_mode`; the *induction
> method* is passed as the `induction_route` argument. *channel*, *PAD*, and *VD* mean the
> same in repo and post.

## Three audit modes

Personascope supports three deployment scenarios, all thin shims over the same
`run_full_battery` core. See [`docs/three_case_audit.md`](docs/three_case_audit.md) for
design + future work.

| Case | What you know | API |
|---|---|---|
| **1. Base** | The model. Nothing about persona induction. | `audit_base(model=...)` |
| **2. Known persona** | The model. The induced persona. The induction method. | `audit_known(model, persona, induction_route)` |
| **3. Unknown persona** | The model. *Maybe* persona-induced; *don't know* what or how. | `audit_unknown(model)` |

Case 3 is the **evaluator-perspective** use case: someone auditing an external API model
where the system prompt + training data are opaque needs to detect persona induction without
being told what to look for. It runs the standard battery plus open-mode evaluation items
(free-text identity claims) and aggregates them through a probabilistic-OR induction detector +
judge-based persona identifier, returning a structured
`BlindAuditResult { induced, persona, route, confidence }`.

**Via the CLI:**

```bash
personascope audit-base    --model gpt-4.1                       --out results/base/
personascope audit-known   --model gpt-4.1 --persona voldemort   --out results/known/
personascope audit-unknown --model gpt-4.1                       --out results/unknown/
```

Run `personascope <command> --help` for the full argument list.

**Or via the Python API:**

```python
from personascope.experiments.audit import audit_base, audit_known, audit_unknown

audit_base(model="gpt-4.1", out_dir="results/base/")

audit_known(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    persona="voldemort",
    induction_route="icl_k32",
    out_dir="results/known/",
)

result = audit_unknown(model="gpt-4.1", out_dir="results/unknown/")
print(result.induced, result.persona, result.confidence)
```

## Other ways to use it

**Run the full battery directly**

```python
from personascope.experiments.full_battery import run_full_battery

run_full_battery(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    persona="voldemort",
    k=32,
    out_dir="results/my_run/",
    tier="core",        # "core" (default, 7 probes) / "extended" (27) / "exploratory"
    # Per-probe overrides still win — e.g. add one extended probe to the core panel:
    run_self_explanation=True,
)
```

The evaluation-item set is tiered (see [`src/personascope/probes/README.md`](src/personascope/probes/README.md)
§ *Tiers*): **core** is one validated item per distinct construct (low correlation between
channels); **extended** adds psychometrics + AISI EM + second readouts; **exploratory** is
the opt-in orphan pool.

**Compose evaluation items by hand**

```python
from personascope.core.runner import run_conversation
from personascope.core.schema import Preparation
from personascope.probes.identity import identification, meta_awareness
from personascope.probes.behavior.external import psychometric

probes = [
    *identification.make_identification_battery("voldemort"),
    *meta_awareness.make_meta_awareness_battery("voldemort"),
    psychometric.make_big_five_probe(item_idx=0),
]
records = run_conversation(
    preparation=prep, interventions=[], probes_per_turn=[probes],
    provider=provider, judge_fn=judge_fn, cache=cache,
)
```

## Layout

```
src/personascope/
├── core/                ← schema, runner, Probe abstraction, refusal classifier
├── probes/              ← measurement panel
│   ├── _utils/          ← shared helpers (refusal_check, meta_gaming)
│   ├── identity/        (with external/ for cited work)
│   ├── behavior/        (with external/)
│   ├── competence/      (with external/)
│   ├── cot/             (with external/)
│   └── context_inference/
├── analysis/            ← Wilson CIs, Bigelow fits, blind_audit aggregators, plots
├── experiments/         ← canonical runners (audit, full_battery, compact_panel, evidence_curve)
├── llm/                 ← provider routing (OpenAI, OpenRouter, Anthropic, ...)
└── data/
    ├── yawyr/           ← bundled persona batteries + ICL fact corpora
    └── external/        ← README; populated by scripts/fetch_datasets.sh
```

## External datasets

Some evaluation items wrap published benchmarks. We don't bundle them (licenses + size):

```bash
bash scripts/fetch_datasets.sh   # TruthfulQA, MMLU, GSM8K, Serapio-García,
                                 # Betley EM, AISI EM
bash scripts/fetch_nrc.sh        # NRC Emotion Lexicon (non-commercial, EULA-gated)
```

See [`src/personascope/data/external/README.md`](src/personascope/data/external/README.md)
for the license + citation table.

## Documentation

- [`docs/three_case_audit.md`](docs/three_case_audit.md) — `audit_base` / `audit_known` / `audit_unknown` design + future work
- [`docs/pipeline_overview.md`](docs/pipeline_overview.md) — architectural tour + entry-point hierarchy
- [`docs/probe_battery_reference.md`](docs/probe_battery_reference.md) — every evaluation item's prompt + judge rubric
- [`src/personascope/probes/README.md`](src/personascope/probes/README.md) — channel-by-channel evaluation-item inventory

## Results

The frozen results behind the post are in [`bench/`](bench/) — one directory per
`model × persona × induction-method` configuration, each with a `summary.json` and a per-component
`report_card.md`. The PAD/VD weighting is in `bench/weights.json`.

## Citation

If you use Personascope in academic work:

```bibtex
@software{personascope_2026,
  title  = {{Personascope}},
  author = {Berczi, Benjamin and Kim, Kyuhee},
  year   = 2026,
  url    = {https://github.com/benjibrcz/personascope}
}
```

Or use the `CITATION.cff` file directly (GitHub renders a "Cite this repository" button).

## License

MIT — see [`LICENSE`](LICENSE).
